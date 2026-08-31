#!/usr/bin/env python3
"""Does the equivalence claim survive the compute dtype people actually run?

M1's equivalence gate compares fp32 forwards of the stored bf16 artifacts. A local user runs a
0.5B model in bf16 compute (or f16), where activation magnitudes and reduced-mantissa dot products
are no longer the same arithmetic. This script measures the same pair under both compute dtypes and
writes m1/work/compute_dtype_check.json, so any statement about "the same function" can carry the
qualifier the evidence supports.

    <venv python> m1/compute_dtype_check.py --b <dir> [--tokens 2048]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

torch.set_num_threads(int(__import__("os").environ.get("TSX_THREADS", "4")))


def run_pair(a: Path, b: Path, tokens: int, seqlen: int, dt: torch.dtype, dev: str) -> dict:
    x = common.corpus_tokens(ntokens=tokens)[: tokens - (tokens % seqlen)].view(-1, seqlen)
    batches = [x[i:i + 1] for i in range(x.size(0))]
    la = common.cached_logits(a, batches, dt, dev)
    lb = common.cached_logits(b, batches, dt, dev)
    out = common.compare_logits(la, lb, "cpu")
    out["ppl_a"] = common.ppl_from_logits(la, x, dev)
    out["ppl_b"] = common.ppl_from_logits(lb, x, dev)
    out["logit_scale"] = float(la.abs().max())
    del la, lb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=str(common.REF_MODEL))
    ap.add_argument("--b", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--out", default=str(common.WORK / "compute_dtype_check.json"))
    a = ap.parse_args()
    dev = common.pick_device(2.4)
    tag = Path(a.b).name
    res = {"pair": [str(a.a), str(a.b)], "tokens": a.tokens, "seqlen": a.seqlen, "device": dev,
           "runs": {}}
    for name, dt in (("fp32_compute", torch.float32), ("bf16_compute", torch.bfloat16)):
        r = run_pair(Path(a.a), Path(a.b), a.tokens, a.seqlen, dt, dev)
        r["max_dlogit_over_scale"] = r["max_dlogit"] / max(1e-9, r["logit_scale"])
        res["runs"][name] = r
        print(f"{tag:16s} {name:13s} max|dlogit|={r['max_dlogit']:.3e} "
              f"({r['max_dlogit_over_scale']:.2e} of scale) KL={r['kl_mean_nats']:.3e} "
              f"top1={r['top1_agree']:.5f} ppl {r['ppl_a']:.4f} -> {r['ppl_b']:.4f}")
    out = Path(a.out)
    allv = {}
    if out.exists():
        allv = json.loads(out.read_text())
    allv[tag] = res
    out.write_text(json.dumps(allv, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
