#!/usr/bin/env python3
"""Control: is a gate failure a broken gauge, or bf16 storage rounding?

Gauges are computed in fp64 but stored in the artifact's bf16. Every touched entry therefore
re-rounds, which is a REAL change to the artifact (it is the same size as the noise already in
any released bf16 checkpoint) and it is not a symmetry violation. This script separates the two
by running the identical transform with NO bf16 round-trip, in memory, on the same tokens.

    <venv python> m1/control_precision_floor.py [--tokens 1024] [--specs g3_rand,g1_haar,g2_rand]

Readout:
  dlogit_bf16  = |f(artifact after gauge) - f(base artifact)|   (what verify_equiv measures)
  dlogit_fp32  = |f(fp32-exact gauge)     - f(base in fp32))|   (algebra only, no re-storage)
If dlogit_fp32 ~ 1e-6 while dlogit_bf16 ~ 1, the transform is exact and bf16 re-rounding is the
whole story; the checkpoint's own precision caps how far you may travel along an orbit for free.
A power-of-two diagonal is bf16-exact by construction, so it is reported as the free case.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import gauge  # noqa: E402
from common import log  # noqa: E402

torch.set_num_threads(int(__import__("os").environ.get("TSX_THREADS", "4")))

SPECS = {"g3_rand": lambda sd, a: gauge.g3_norm_diag(sd, a, "random", 1, 3.0),
         "g1_haar": lambda sd, a: gauge.g1_vo_orth(sd, a, "haar", 1),
         "g2_rand": lambda sd, a: gauge.g2_rope_pairs(sd, a, "random", 1),
         "g7_rand": lambda sd, a: gauge.g7_up_diag(sd, a, "random", 1, 3.0)}


def logits_of(sd: dict, dtype: torch.dtype, x: torch.Tensor, ref_dir: Path, dev: str) -> torch.Tensor:
    m = common.state_to_model({k: v.to(dtype) for k, v in sd.items()}, ref_dir, dtype, dev)
    out = []
    with torch.no_grad():
        for i in range(0, x.size(0), 1):
            out.append(m(input_ids=x[i:i + 1].to(dev)).logits.float().cpu())
    del m
    common.release(dev)
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--specs", default="g3_rand,g1_haar,g2_rand,g7_rand")
    ap.add_argument("--out", default=str(common.WORK / "precision_floor.json"))
    a = ap.parse_args()
    dev = common.pick_device(2.6)
    arch = common.read_arch(common.REF_MODEL)
    base64 = {k: v.to(torch.float64) for k, v in common.load_state(common.REF_MODEL).items()}
    x = common.corpus_tokens(ntokens=a.tokens)[: a.tokens - (a.tokens % 512)].view(-1, 512)
    lg_bf16_base = logits_of({k: v.to(torch.bfloat16) for k, v in base64.items()},
                             torch.bfloat16, x, common.REF_MODEL, dev)
    lg_fp32_base = logits_of(base64, torch.float32, x, common.REF_MODEL, dev)
    res = {"device": dev, "tokens": int(x.numel()), "cases": {}}
    for name in [s.strip() for s in a.specs.split(",") if s.strip()]:
        fn = SPECS[name]
        sd_exact, man = fn({k: v.clone() for k, v in base64.items()}, arch)
        d_fp32 = float((logits_of(sd_exact, torch.float32, x, common.REF_MODEL, dev)
                        - lg_fp32_base).abs().max())
        d_bf16 = float((logits_of({k: v.to(torch.bfloat16) for k, v in sd_exact.items()},
                                  torch.bfloat16, x, common.REF_MODEL, dev)
                        - lg_bf16_base).abs().max())
        scale = float(lg_bf16_base.abs().max())
        res["cases"][name] = {"dlogit_fp32_exact_algebra": d_fp32, "dlogit_bf16_artifact": d_bf16,
                              "logit_scale": scale,
                              "rel_bf16": d_bf16 / scale, "rel_fp32": d_fp32 / scale,
                              "verdict_bf16_gate": "PASS" if d_bf16 <= 0.5 else "FAIL"}
        log(f"{name:9s} algebra-only max|dlogit|={d_fp32:.2e}   via bf16 artifact={d_bf16:.2e} "
            f"({d_bf16 / scale:.2%} of logit scale)   gate(bf16)={res['cases'][name]['verdict_bf16_gate']}")

    # free case: a power-of-two diagonal is exactly representable in bf16
    d = torch.tensor([2.0 ** int(e) for e in torch.randint(-8, 9, (arch.hidden,))], dtype=torch.float64)
    sd_pow, _ = gauge.g3_norm_diag({k: v.clone() for k, v in base64.items()}, arch, "random",
                                   decades=0.0, d_attn=d, d_mlp=1.0 / d)
    d_pow = float((logits_of({k: v.to(torch.bfloat16) for k, v in sd_pow.items()}, torch.bfloat16,
                             x, common.REF_MODEL, dev) - lg_bf16_base).abs().max())
    res["cases"]["g3_pow2"] = {"dlogit_bf16_artifact": d_pow, "logit_scale": scale,
                               "rel_bf16": d_pow / scale,
                               "verdict_bf16_gate": "PASS" if d_pow <= 0.5 else "FAIL"}
    log(f"{'g3_pow2':9s} bf16-exact diagonal (d = 2^k, +-8): max|dlogit|={d_pow:.2e} "
        f"({d_pow / scale:.2%})  gate={res['cases']['g3_pow2']['verdict_bf16_gate']}")
    common.wjson(Path(a.out), res)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
