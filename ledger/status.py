"""`ledger/status.py` — the bounded L4 status screen (`theseus status [--budget N]`).

Exactly the view the driver lacked: claims with state + the missing obligations that cap them,
cell counts (measured/predicted/unavailable), lease state read from `m1/work/gpu.lock/owner`
(with dead-pid detection via /proc — I6 steals dead leases), disk free, and the ranked next
actions from `plan` with their cost. Rendering is bounded to ≤ 40 lines.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from . import claims
from .plan import plan
from .rules import superseded_ids


def _lease_info(work: Path) -> str:
    owner = work / "gpu.lock" / "owner"
    if not owner.exists():
        return "no lease (GPU free)"
    try:
        text = owner.read_text().strip()
    except OSError as exc:
        return f"lease unreadable: {exc}"
    m = re.search(r"pid=(\d+)", text)
    if not m:
        return f"lease present, pid unparseable: {text[:80]}"
    pid = int(m.group(1))
    if os.path.exists(f"/proc/{pid}"):
        return f"lease pid={pid} LIVE ({text[m.end():].strip()[:70]})"
    return f"lease pid={pid} DEAD (absent from /proc) — stealable (I6)"


def _disk(where: Path) -> str:
    try:
        free = shutil.disk_usage(str(where)).free
        return f"{free / 1e9:.1f} G free"
    except OSError:
        return "disk free UNAVAILABLE"


def status_screen(ledger, *, budget: float = 60.0, work: Path | None = None,
                  allow_mixed_env: bool = False, max_lines: int = 40) -> str:
    work = work or Path("/home/admin/theseus/m1/work")
    lines = []
    counts = ledger.counts()
    total_cells = counts["cell"]
    env_unknown = len([c for c in ledger.all("cell")
                       if (c.get("environment") or {}).get("unknown")])

    lines.append(f"theseus ledger @ {ledger.root.absolute()}")
    lines.append(f"records: artifacts={counts['artifact']} cells={total_cells} "
                 f"claims={counts['claim']} incidents={counts['incident']} | "
                 f"env.unknown cells={env_unknown} (excluded from joins, I3)")
    lines.append(f"disk: {_disk(ledger.root)}  |  lease: {_lease_info(work)}")

    lines.append("claims:")
    for key in claims.all_keys():
        try:
            r = claims.explain(ledger, key, allow_mixed_env=allow_mixed_env)
        except Exception as exc:  # noqa: BLE001
            lines.append(f"  {key:<6} RENDER REFUSED ({exc})")
            continue
        missing = "; ".join(r["missing"]) or ("—" if r["state"] in
                                              ("controlled", "confirmed", "refuted") else "—")
        blocked = ("; blocked: " + "; ".join(n for n, _ in r["blocked_by"])) if r["blocked_by"] else ""
        lines.append(f"  {key:<6} {r['state'].upper():<12} missing: {missing}{blocked}")

    # verdict tally: measured denominators only (I8)
    tally = {}
    for c in ledger.all("cell"):
        res = c.get("result") or {}
        s = res.get("status")
        tally[s] = tally.get(s, 0) + 1
    lines.append(f"cells: measured={tally.get('measured', 0)} predicted={tally.get('predicted', 0)} "
                 f"unavailable={tally.get('unavailable', 0)}")

    # stale: superseded by an `invalidates` correction (I1) or verdict stale/invalidated (I7),
    # never tallied in either direction (I8) — surfacing them is what makes honesty cheap.
    all_cells = ledger.all("cell")
    stale_ids = sorted(c.get("id") for c in all_cells
                       if c.get("id") in superseded_ids(all_cells)
                       or (c.get("result") or {}).get("verdict") in ("stale", "invalidated"))
    lines.append(f"stale/invalidated: {len(stale_ids)}"
                 + (f" ({', '.join(stale_ids[:6])})" if stale_ids else " (I8: never tallied)"))

    lines.append(f"next (budget {budget:g} gpu-min):")
    p = plan(ledger, budget_gpu_min=budget)
    for i, a in enumerate(p["ranked"], 1):
        lines.append(f"  {i} {a['action']}  [{a['cost']}] belief {a['belief']}")
    for a in p["refused"][:2]:
        unblock = f" — unblock: {a['unblocking']}" if a.get("unblocking") else ""
        lines.append(f"  REFUSED {a['action']}{unblock} (I4)")
    return "\n".join(lines[:max_lines])
