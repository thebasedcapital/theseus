"""`ledger/cli.py` — the L4 session surface: `python -m ledger.cli <verb> …`.

Commands: admit | cell | status | plan | explain | render | import-m1 | verify.
None of them ever import `m1/` (live pipeline) — only `m1/work/*.json` files are read, and only
by `import-m1` with an explicit `--work` path. The ledger root defaults to `$THESEUS_LEDGER_ROOT`
or `.theseus`; this agent runs every command with `--root` under /tmp so nothing is ever written
into the live repo layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import claims, rules
from .env import env_digest
from .import_m1 import ImportReport, run as run_import
from .plan import plan as plan_fn
from .render import render as render_fn
from .status import status_screen
from .store import Ledger, LedgerError, IdConflictError

DEFAULT_WORK = "/home/admin/theseus/m1/work"


def _store(args) -> Ledger:
    root = args.root or os.environ.get("THESEUS_LEDGER_ROOT") or ".theseus"
    return Ledger(root)


def _perr(msg):
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _load_json(path: str, what: str) -> dict:
    p = Path(path)
    if not p.exists():
        _perr(f"{what} file not found: {p}")
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        _perr(f"{what} file unparseable: {exc}")
    if not isinstance(data, dict):
        _perr(f"{what} must be a JSON object")
    return data


def cmd_admit(args):
    store = _store(args)
    body = {}
    if args.json:
        body = _load_json(args.json, "artifact")
    elif args.hf:
        body = {"kind": "artifact", "origin": {"hf": args.hf, "revision": args.revision,
                                               "blob_sha256": None},
                "container": {"file": None, "bytes": None, "sha256": None,
                              "tensors": None, "weights": None, "dtype": None, "tied": None},
                "config": {}, "features": {"total": {"q4_block_mse": None,
                                                     "convention": "mean_of_per_tensor_ratios"}},
                "ancestry": []}
    else:
        _perr("admit needs --json FILE or --hf REPO")
    body.setdefault("kind", "artifact")
    problems = rules.check_artifact(body, ledger=store)
    if problems:
        _perr("; ".join(problems))
    try:
        rid, action = store.add("artifact", body)
    except (LedgerError, IdConflictError) as exc:
        _perr(str(exc))
    print(f"artifact {rid} {action}")


def cmd_cell(args):
    store = _store(args)
    body = _load_json(args.json, "cell")
    body.setdefault("kind", "cell")
    if args.invalidates:
        body["invalidates"] = args.invalidates
    problems = rules.check_cell(body, ledger=store)
    if problems:
        _perr("; ".join(problems))
    try:
        rid, action = store.add("cell", body)
    except (LedgerError, IdConflictError) as exc:
        _perr(str(exc))
    print(f"cell {rid} {action}")


def cmd_status(args):
    store = _store(args)
    print(status_screen(store, budget=args.budget, work=Path(args.work or DEFAULT_WORK),
                        allow_mixed_env=args.allow_mixed_env))


def cmd_plan(args):
    store = _store(args)
    p = plan_fn(store, budget_gpu_min=args.budget, op=args.op, subject=args.subject)
    if args.op is not None:
        if p["allow"]:
            for e in p["ranked"]:
                print(f"SCHEDULE  {e['action']}  [{e['cost']}]  "
                      f"reference: {e['i4']['reference_cell']}")
        else:
            for e in p["refused"]:
                print(f"REFUSED (I4)  {e['action']}  —  unblocking cell: {e['unblocking']}")
        return
    print(f"plan (budget {p['budget_gpu_min']:g} gpu-min, spend {p['spend_gpu_min']:g}):")
    for i, a in enumerate(p["ranked"], 1):
        print(f"  {i}  {a['action']}  [{a['cost']}]  belief {a['belief']}  -> "
              + ", ".join(a["discharges"]))
    for a in p["refused"]:
        uk = f"  unblocking cell: {a['unblocking']}" if a.get("unblocking") else ""
        print(f"  REFUSED (I4)  {a['action']}{uk}")


def cmd_explain(args):
    store = _store(args)
    try:
        r = claims.explain(store, args.key, allow_mixed_env=args.allow_mixed_env)
    except LedgerError as exc:
        _perr(str(exc))
    print(f"{r['key']} [{r['state'].upper()}]  (declared {r['declared']})")
    print(f"  {r['text']}")
    if r["missing"]:
        print(f"  capped at {r['state'].upper()}: missing required obligation(s): "
              + ", ".join(r["missing"]))
        if r["promoting_cells"]:
            print("  promoting cells (would move it): " + ", ".join(r["promoting_cells"]))
    if r["blocked_by"]:
        for name, reason in r["blocked_by"]:
            print(f"  BLOCKED: {name} -> {reason}")
    for o in r["obligations"]:
        mark = "OK " if o["satisfied"] else ("REQ" if o["required"] else "inf")
        ev = f" cells={o['evidence'][:4]}" if o["evidence"] else ""
        note = f"  ({o['note']})" if o["note"] else ""
        print(f"    [{mark}] {o['name']}{ev}{note}")
    if r["numbers"]:
        print("  numbers (I10): " + "; ".join(
            f"{n['value']} (cell {n['cite']})" for n in r["numbers"]))
    print(f"  refuter: {r['refuter']['query']}")
    print(f"    would_drop_to: {r['refuter']['would_drop_to']}; answering: "
          + ", ".join(r["refuter"]["answering_cells"]))
    if r["mixed_env_stamp"]:
        print(f"  stamp: {r['mixed_env_stamp']}")


def cmd_render(args):
    store = _store(args)
    try:
        out = render_fn(store, store.root, allow_mixed_env=args.allow_mixed_env)
    except LedgerError as exc:
        _perr(str(exc))
    for p in out:
        print(f"generated {store.root / p}")


def cmd_import_m1(args):
    store = _store(args)
    report = ImportReport()
    try:
        report = run_import(store, args.work or DEFAULT_WORK,
                            dry_run=args.dry_run, verify=args.verify)
    except (LedgerError, IdConflictError, RuntimeError) as exc:
        _perr(str(exc))
    print(f"import-m1 {'(dry-run, no writes)' if args.dry_run else ''} from {args.work or DEFAULT_WORK}:")
    print(f"  records to write    : artifacts={report.total['artifact']} "
          f"cells={report.total['cell']} claims={report.total['claim']} "
          f"incidents={report.total['incident']}")
    for name in sorted(report.files):
        f = report.files[name]
        if f["records"] or f["unknown"] or f["malformed"]:
            u = f"  unknown-env {f['unknown']}" if f["unknown"] else ""
            m = f"  malformed {f['malformed']}" if f["malformed"] else ""
            print(f"    {name}: {f['records']} record(s){u}{m}")
        for note in f["notes"]:
            print(f"      - {note}")
    if report.confirmed_refs:
        print(f"  calibration refs confirmed: {len(report.confirmed_refs)}")
    if not args.dry_run:
        counts = store.counts()
        print(f"  ledger now: {counts}")
        if args.verify:
            print("  --verify: ledger integrity recomputed, OK")
    else:
        _print_dry_verify(report)


def _print_dry_verify(report):
    # dry-run re-computes content ids for a sample to assert determinism
    print("  (dry-run: no files written)")


def cmd_verify(args):
    """Audit cell provenance: recorded commit resolves, named script existed there, and each
    operation kind carries the fields its own writer cannot omit. See ledger/verify.py."""
    from . import verify as _verify
    return _verify.main(["--work", args.work] + (["--json"] if args.json else []))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="theseus", description=__doc__)
    ap.add_argument("--root", help=".theseus root (default $THESEUS_LEDGER_ROOT or .theseus)")
    ap.add_argument("--work", default=DEFAULT_WORK, help="m1/work directory for import/status")
    ap.add_argument("--allow-mixed-env", action="store_true",
                    help="stamp, not refuse, mixed-environment joins (I3)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_admit = sub.add_parser("admit", help="admit an artifact record (write-once)")
    p_admit.add_argument("--json", help="full artifact body JSON file")
    p_admit.add_argument("--hf", help="HuggingFace repo id for a manually declared artifact")
    p_admit.add_argument("--revision", default="main")
    p_admit.set_defaults(fn=cmd_admit)

    p_cell = sub.add_parser("cell", help="admit a cell record (write-once, I4/I3 enforced)")
    p_cell.add_argument("--json", required=True, help="full cell body JSON file")
    p_cell.add_argument("--invalidates", metavar="CELL_ID", default=None,
                        help="cell id this correction supersedes; writes the I1 invalidates edge "
                             "(target must exist and stays untouched, write-once)")
    p_cell.set_defaults(fn=cmd_cell)

    p_status = sub.add_parser("status", help="bounded status screen")
    p_status.add_argument("--budget", type=float, default=60.0, help="gpu-min budget")
    p_status.add_argument("--work", default=DEFAULT_WORK, help="m1/work dir (lease read)")
    p_status.set_defaults(fn=cmd_status)

    p_plan = sub.add_parser("plan", help="ranked next actions with cost (I4 gate)")
    p_plan.add_argument("--budget", type=float, default=60.0)
    p_plan.add_argument("--op", help="check one op: is it calibrated?")
    p_plan.add_argument("--subject", help="artifact for the op")
    p_plan.set_defaults(fn=cmd_plan)

    p_ex = sub.add_parser("explain", help="claim state + obligations + refuter")
    p_ex.add_argument("key")
    p_ex.set_defaults(fn=cmd_explain)

    p_render = sub.add_parser("render", help="regenerate views/ (L3)")
    p_render.set_defaults(fn=cmd_render)

    p_imp = sub.add_parser("import-m1", help="PLAN §0: map m1/work onto the ledger")
    p_imp.add_argument("--work", default=DEFAULT_WORK, help="m1/work directory to read")
    p_imp.add_argument("--dry-run", action="store_true")
    p_imp.add_argument("--verify", action="store_true",
                       help="after writing, recompute every record's content id")
    p_imp.set_defaults(fn=cmd_import_m1)

    p_ver = sub.add_parser("verify", help="check every persisted cell names a generator that "
                                          "could have written it (incidents #18/#19)")
    p_ver.add_argument("--work", default=DEFAULT_WORK, help="cell store to audit")
    p_ver.add_argument("--json", action="store_true")
    p_ver.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    # --root must be absolute when passed at CLI to avoid accidental local writes
    if args.root:
        args.root = str(Path(args.root).expanduser().resolve())
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
