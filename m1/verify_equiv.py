#!/usr/bin/env python3
"""M1 equivalence gate: prove two checkpoints compute the same function.

The gate below is FROZEN before any result is read. It is deliberately not "max logit diff
== 0": the artifacts are bf16, so any gauge re-expression carries storage rounding. What must
not move is the *predictive* behaviour: predictive distribution, teacher-forced argmax
agreement, and perplexity. max_dlogit is reported and gated only as a gross-bug tripwire
(a wrong-family transform, e.g. a non-commuting rotation, produces KL >> 1, not 1e-4).

    <venv python> m1/verify_equiv.py --a <dir> --b <dir> [--out out.json] [--ntokens 8192]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("TSX_THREADS", "4")))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import canonicalize as canon  # noqa: E402

GATE = {
    "kl_mean_nats_max": 2e-3,     # mean KL(P_a || P_b) over every scored position
    "top1_agree_min": 0.995,      # teacher-forced next-token argmax agreement
    "rel_ppl_max": 2e-3,          # |ppl_b/ppl_a - 1|
    "max_dlogit_max": 0.5,        # gross-error tripwire only
}


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(common.REPO), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def verify(a: Path, b: Path, ntokens: int, seqlen: int) -> dict:
    t0 = time.time()
    rep = common.equivalence_report(Path(a), Path(b), ntokens=ntokens, seqlen=seqlen)
    sa, sb = common.load_state(Path(a)), common.load_state(Path(b))
    rel = abs(rep["ppl_b"] / rep["ppl_a"] - 1.0)
    checks = {
        "kl": rep["kl_mean_nats"] <= GATE["kl_mean_nats_max"],
        "top1": rep["top1_agree"] >= GATE["top1_agree_min"],
        "ppl": rel <= GATE["rel_ppl_max"],
        "dlogit": rep["max_dlogit"] <= GATE["max_dlogit_max"],
    }
    return {
        "script": "verify_equiv.py", "a": str(Path(a).resolve()), "b": str(Path(b).resolve()),
        "gate": GATE, "metrics": rep, "rel_ppl": rel, "checks": checks,
        "verdict": "EQUIVALENT" if all(checks.values()) else "NOT_EQUIVALENT",
        "cond_a": canon.quant_condition(sa), "cond_b": canon.quant_condition(sb),
        "git_head": git_head(), "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "duration_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(common.REF_MODEL))
    ap.add_argument("--b", default=str(common.REF_MODEL))
    ap.add_argument("--ntokens", type=int, default=4096)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = verify(Path(a.a), Path(a.b), a.ntokens, a.seqlen)
    print(json.dumps(res, indent=2, sort_keys=True))
    if a.out:
        common.wjson(Path(a.out), res)
    if res["verdict"] != "EQUIVALENT":
        sys.exit(2)


if __name__ == "__main__":
    main()
