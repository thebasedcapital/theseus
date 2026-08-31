#!/usr/bin/env python3
"""EXPLORATORY screen (not evidence): does natural-history order divergence ever
clear the frozen K-8 present-match gate while reserve still differs?

The v1 attempt failed at merge alpha 0.30: relative PPL 0.00054 (9.3x INSIDE the 0.005
limit) but mean KL 0.032311 (16.2x OUTSIDE the 2e-3 limit) and top-1 0.88235. This script
sweeps merge alpha, the knob that controls how far the two orders diverge, to find whether a
separating window exists: small enough KL to pass the gate, still-large-enough static reserve
features to make a future-reserve difference worth probing.

Writes only to a scratch dir under /tmp. Never touches m3/results.json: per I1/I4 and the
"predicted rows are never tallied" rule, screening output is not evidence and is not admitted
to the ledger. If a window exists here, K-8 v2 is then run properly under the frozen contract.
"""
from __future__ import annotations

import json, os, shutil, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m1"))
import common          # noqa: E402
import history_pair as hp   # noqa: E402

ALPHAS = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0.30", "0.15", "0.06", "0.02"])]
GATE = hp.CONTRACT["present_match"]
KEYS = ("q4_block_mse", "dyn_range_log10", "row_energy_imbalance", "frac_below_f16_normal")


def relgap(a, b, k):
    va, vb = a.get(k), b.get(k)
    if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
        return None
    return abs(va - vb) / max(abs(va), abs(vb), 1e-12)


def one(alpha: float, device: str):
    hp.CONTRACT["merge"]["alpha"] = alpha          # construct() reads this live
    work = Path(tempfile.mkdtemp(prefix="theseus-m3screen-", dir="/tmp"))
    cmds = []
    t0 = time.time()
    try:
        _, qa, metaA = hp.construct("A", work, device, cmds)
        _, qb, metaB = hp.construct("B", work, device, cmds)
        pm = hp.q4_present_metrics(qa, qb, work, cmds)
        fa = hp.feature_map(hp.inspect(qa, work / "A.json"))
        fb = hp.feature_map(hp.inspect(qb, work / "B.json"))
        kl, top, rp = pm.get("mean_kl"), pm.get("top1_agreement"), pm.get("relative_ppl")
        ok = (kl is not None and kl <= GATE["mean_kl_max"]
              and top is not None and top >= GATE["top1_min"]
              and rp is not None and rp <= GATE["relative_ppl_max"])
        gaps = {k: relgap(fa, fb, k) for k in KEYS}
        return {"alpha": alpha, "gate_pass": bool(ok), "mean_kl": kl, "top1": top,
                "relative_ppl": rp, "feature_gap": gaps, "adapt_capture":
                {"A": metaA["adapt"].get("capture"), "B": metaB["adapt"].get("capture")},
                "wall_s": round(time.time() - t0, 1), "commands": len(cmds)}
    finally:
        shutil.rmtree(work, ignore_errors=True)
        hp.CONTRACT["merge"]["alpha"] = 0.30


def main():
    device = "cpu" if os.environ.get("TSX_CPU") else common.pick_device(2.4)
    print(f"# EXPLORATORY screen | device={device} | gate KL<={GATE['mean_kl_max']} "
          f"top1>={GATE['top1_min']} relPPL<={GATE['relative_ppl_max']}", flush=True)
    out = []
    ctx = common.lock("gpu") if device == "cuda" else hp._nullcontext()
    with ctx:
        for a in ALPHAS:
            r = one(a, device)
            out.append(r)
            fg = " ".join(f"{k.split('_')[0]}={r['feature_gap'][k]:.4f}" if r['feature_gap'][k] is not None
                          else f"{k}=NA" for k in KEYS)
            print(f"alpha={a:<5} gate={'PASS' if r['gate_pass'] else 'fail':<4} "
                  f"KL={r['mean_kl'] if r['mean_kl'] is None else round(r['mean_kl'],6)} "
                  f"top1={r['top1']} relPPL={round(r['relative_ppl'],6) if r['relative_ppl'] is not None else None} "
                  f"| gaps {fg} | {r['wall_s']}s", flush=True)
    Path("/tmp/theseus_m3_screen.json").write_text(json.dumps(
        {"kind": "EXPLORATORY_screen_not_evidence", "gate": GATE, "device": device,
         "results": out}, indent=2))
    print("wrote /tmp/theseus_m3_screen.json (screening only, never admitted to the ledger)")


if __name__ == "__main__":
    main()
