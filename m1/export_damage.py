#!/usr/bin/env python3
"""M1 export-dtype decomposition: how much of the "quantization" damage is really the f16 export?

`convert_hf_to_gguf.py --outtype f16` is the default path every quantization script on the
internet uses. It is NOT a no-op for gauged artifacts: an RMSNorm-diagonal gauge moves tens of
millions of weights below f16's normal range (bf16, with fp32's exponent range, holds them
exactly). So a Q4_K_M damage number measured from an f16 source conflates two operations:

  export damage   = ppl(bf16 GGUF of the artifact) -> ppl(f16 GGUF of the same artifact)
  quant damage    = ppl(f16 GGUF) -> ppl(Q4_K_M from that same f16 source)

This script measures the bf16 (and optionally f32) export per checkpoint and writes
m1/work/export_damage.json, which is what `theseus preflight quantize` should consult before
letting a user download 400 MB of noise.

    <venv python> m1/export_damage.py [--tags base,g3_pow2,...] [--outtype bf16]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

LLAMA = Path("/home/admin/tools/llama.cpp-vulkan/llama-b9851")
PERPLEXITY, CONVERT_BIN = LLAMA / "llama-perplexity", Path("/home/admin/tools/llama.cpp-cuda-src/convert_hf_to_gguf.py")
EXTRA_SITE = Path("/home/admin/laps/benchmarks/swebench/.venv/lib/python3.12/site-packages")
F16_NORMAL = 6.103515625e-5


def convert(src: Path, dst: Path, outtype: str) -> bool:
    env = dict(os.environ)
    if EXTRA_SITE.exists():
        env["PYTHONPATH"] = str(EXTRA_SITE) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, CONVERT_BIN, "--outfile", dst, "--outtype", outtype, src]
    r = subprocess.run([str(x) for x in cmd], env=env, capture_output=True, text=True, timeout=1800)
    log("  conv " + " ".join(shlex.quote(str(x)) for x in cmd[-4:]))
    return r.returncode == 0 and dst.exists()


def ppl_of(model: Path, corpus: Path) -> float | None:
    cmd = [PERPLEXITY, "-m", model, "-f", corpus, "-c", "512", "--temp", "0", "--seed", "0",
           "-ngl", "0", "--chunks", "4"]
    r = subprocess.run([str(x) for x in cmd], capture_output=True, text=True, timeout=3600)
    vals = re.findall(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", r.stdout + r.stderr)
    return float(vals[-1]) if vals else None


def subnormal_census(sd: dict) -> dict:
    tot = below = 0
    mn = float("inf")
    for k, t in sd.items():
        if not k.endswith(".weight") or "layers" not in k:
            continue
        a = t.float().abs()
        nz = a[a > 0]
        if nz.numel() == 0:
            continue
        tot += nz.numel()
        below += int((nz < F16_NORMAL).sum())
        mn = min(mn, float(nz.min()))
    return {"weights": tot, "below_f16_normal": below, "frac_below": round(below / max(1, tot), 5),
            "min_abs_weight": mn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--outtypes", default="bf16")
    ap.add_argument("--out", default=str(common.WORK / "export_damage.json"))
    a = ap.parse_args()
    reg = common.rjson(common.WORK / "VARIANTS.json") if (common.WORK / "VARIANTS.json").exists() else {}
    tags = [t.strip() for t in a.tags.split(",") if t.strip()] or sorted(reg)
    corpus = common.WORK / "gguf" / f"eval_ppl_{32768}.txt"
    if not corpus.exists():
        corpus.write_bytes(common.EVAL_TEXT.read_bytes()[:32768])
    work = common.WORK / "export_test"
    work.mkdir(parents=True, exist_ok=True)
    res = common.rjson(Path(a.out)) if Path(a.out).exists() else {}
    # No GPU lock: everything here runs -ngl 0 (CPU), so it cannot collide with the panel's
    # Vulkan contexts; taking the lock would just serialize a CPU job behind GPU work.
    try:
        for tag in tags:
            src = common.REF_MODEL if tag == "base" else common.WORK / tag
            if not (src / "model.safetensors").exists():
                subprocess.run([sys.executable, str(common.M1 / "make_variants.py"), "--only", tag,
                                "--out", str(common.WORK)], check=False)
            if not (src / "model.safetensors").exists():
                res[tag] = {"status": "UNAVAILABLE", "reason": "variant dir missing"}
                continue
            sd = common.load_state(src)
            row = {"status": "OK", "census": subnormal_census(sd), "ppl": {}}
            f16 = work / f"{tag}-f16.gguf"
            if convert(src, f16, "f16"):
                row["ppl"]["f16"] = ppl_of(f16, corpus)
                f16.unlink(missing_ok=True)
            for ot in [x.strip() for x in a.outtypes.split(",") if x.strip()]:
                p = work / f"{tag}-{ot}.gguf"
                if convert(src, p, ot):
                    row["ppl"][ot] = ppl_of(p, corpus)
                    p.unlink(missing_ok=True)
            if row["ppl"].get("f16") and row["ppl"].get("bf16"):
                row["export_damage_ratio"] = round(row["ppl"]["f16"] / row["ppl"]["bf16"], 4)
            res[tag] = row
            common.wjson(Path(a.out), res)
            log(f"{tag:16s} f16={row['ppl'].get('f16')} bf16={row['ppl'].get('bf16')} "
                f"below_f16_normal={row['census']['below_f16_normal']:,} "
                f"frac={row['census']['frac_below']}")
    finally:
        pass
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
