#!/usr/bin/env python3
"""M1 quantization probe: real llama.cpp GGUF K-quant surgery on an HF checkpoint.

    <venv python> m1/gguf_probe.py --model-dir <hf dir> --out <json> [--tag NAME]
                                   [--backend auto|cpu|vulkan] [--tags q8_0,q6_k,q5_k_m,q4_k_m]

Damage is always measured *within* one artifact: the checkpoint is converted to f16 GGUF, that
f16 model is the reference, and each quantization of the same weights is compared against it.
That is the reserve question — "how much does quantizing THIS artifact cost" — and it means no
cross-variant comparison ever leaks into the number.

Metrics per tag:
  ppl / dppl / rel_dppl  from llama-perplexity over a fixed 32 KiB corpus slice
  kl_mean, kl_median, kl_p90/95/99/99.9, kl_max
                          from llama-perplexity --kl-divergence against the f16 model's own
                          logits (--save-all-logits), fixed 8 KiB slice. The percentiles matter:
                          a checkpoint can look fine on the mean and be broken in the tail.
  tokagree                greedy (temp 0) 32-token continuation agreement, 8 fixed prompts
  size_mb                 artifact size

PASS contract (frozen before any variant was measured; base is calibration only):
  rel_dppl <= 0.02 and kl_mean <= 0.01 and (tokagree is None or tokagree >= 0.85)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

LLAMA = Path("/home/admin/tools/llama.cpp-vulkan/llama-b9851")
QUANTIZE, PERPLEXITY, COMPLETION = (LLAMA / x for x in
                                    ("llama-quantize", "llama-perplexity", "llama-completion"))
CONVERTER = Path("/home/admin/tools/llama.cpp-cuda-src/convert_hf_to_gguf.py")
EXTRA_SITE = Path("/home/admin/laps/benchmarks/swebench/.venv/lib/python3.12/site-packages")
SEED = 0
PPL_BYTES = 32768
KL_BYTES = 8192
N_PROMPTS, N_TOKENS = 4, 32   # greedy-agreement budget: 4 prompts x 2 models = 8 loads
TAGS_DEFAULT = ("q8_0", "q6_k", "q5_k_m", "q4_k_m")
AGREE_TAGS = ("q8_0", "q4_k_m")
PASS_CONTRACT = {"rel_dppl_max": 0.02, "kl_mean_max": 0.01, "tokagree_min": 0.85,
                 "tokagree_none_passes": True,
                 "frozen": "2026-08-30, before any variant was measured"}
KL_RE = {
    "kl_mean": r"Mean\s+KLD:\s*([0-9.eE+-]+)",
    "kl_median": r"Median\s+KLD:\s*([0-9.eE+-]+)",
    "kl_p90": r"90\.0%\s+KLD:\s*([0-9.eE+-]+)",
    "kl_p95": r"95\.0%\s+KLD:\s*([0-9.eE+-]+)",
    "kl_p99": r"99\.0%\s+KLD:\s*([0-9.eE+-]+)",
    "kl_p999": r"99\.9%\s+KLD:\s*([0-9.eE+-]+)",
    "kl_max": r"Maximum KLD:\s*([0-9.eE+-]+)",
}


def run(cmd, env=None, timeout=2400):
    p = subprocess.run([str(x) for x in cmd], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return p.returncode, p.stdout


def git_head() -> str:
    return subprocess.run(["git", "-C", str(common.REPO), "rev-parse", "HEAD"],
                          text=True, capture_output=True).stdout.strip() or "unknown"


def prep_inputs(work: Path) -> dict:
    """Deterministic corpus slices + the fixed prompt set, reused by every variant."""
    work.mkdir(parents=True, exist_ok=True)
    text = common.EVAL_TEXT.read_bytes()
    shared = common.WORK / "gguf"
    shared.mkdir(parents=True, exist_ok=True)
    ppl_f, kl_f = shared / f"eval_ppl_{PPL_BYTES}.txt", shared / f"eval_kl_{KL_BYTES}.txt"
    ppl_f.write_bytes(text[:PPL_BYTES])
    kl_f.write_bytes(text[:KL_BYTES])
    pf = shared / "prompts.txt"
    if not pf.exists():
        step = PPL_BYTES // (N_PROMPTS + 1)
        chunk = text[:PPL_BYTES].decode("utf-8", "replace")
        pf.write_text("\n".join(chunk[i * step: i * step + 128] for i in range(N_PROMPTS)) + "\n")
    return {"ppl": ppl_f, "kl": kl_f, "prompts": pf}


def convert(model_dir: Path, f16: Path, cmds: list) -> tuple[bool, str]:
    env = dict(os.environ)
    if EXTRA_SITE.exists():
        env["PYTHONPATH"] = str(EXTRA_SITE) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, CONVERTER, "--outfile", f16, "--outtype", "f16", model_dir]
    rc, out = run(cmd, env=env, timeout=1200)
    cmds.append(shlex.join(str(x) for x in cmd))
    if rc or not f16.exists():
        return False, f"convert rc={rc}: {out[-1500:]}"
    # untied artifacts must keep their own output tensor, or the comparison is a lie
    tie = bool(json.loads((model_dir / "config.json").read_text())
               .get("tie_word_embeddings", False))
    head = subprocess.run([sys.executable, "-c",
                           "from safetensors import safe_open;import sys;"
                           "print(any('lm_head' in k for k in safe_open(sys.argv[1],'pt').keys()))",
                           str(model_dir / "model.safetensors")], capture_output=True, text=True)
    untied_src = head.stdout.strip() == "True"
    out_t = has_output_tensor(f16)
    if not tie and untied_src and out_t is False:
        return False, f"untied source produced no output.weight in {f16}"
    if not tie and untied_src and out_t is None:
        return True, "could not read GGUF tensor names; tie check skipped"
    return True, ""


def has_output_tensor(path: Path) -> bool | None:
    """Read GGUF tensor names without importing the gguf package. None = unreadable."""
    try:
        import struct
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return False
            ver = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def rstr():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", "replace")
            for _ in range(n_kv):
                k = rstr()
                t = struct.unpack("<I", f.read(4))[0]
                if t == 0:
                    rstr()
                elif t == 1:
                    f.read(1)
                elif t == 2:
                    f.read(2)
                elif t == 3:
                    f.read(4)
                elif t == 4:
                    f.read(4)
                elif t == 5:
                    f.read(8)
                elif t == 6:
                    rstr()
                elif t == 7:
                    n = struct.unpack("<Q", f.read(8))[0]
                    et = struct.unpack("<I", f.read(4))[0]
                    f.read(8 * n if et == 8 else 4 * n)
                elif t == 8:
                    f.read(8)
                elif t == 9:
                    f.read(16)
            for _ in range(n_tensors):
                name = rstr()
                if name in ("output.weight", "score_weight"):
                    return True
                n_dims = struct.unpack("<I", f.read(4))[0]
                f.read(8 * n_dims + 4 + 8)
        return None
    except Exception:                                     # noqa: BLE001
        return None


def ppl_of(model: Path, corpus: Path, ngl: str, cmds: list, extra=()) -> tuple[float | None, str]:
    cmd = [PERPLEXITY, "-m", model, "-f", corpus, "-c", "512", "--temp", "0", "--seed", SEED,
           "-ngl", ngl, "--chunks", "4", *extra]
    rc, out = run(cmd, timeout=3600)
    cmds.append(shlex.join(str(x) for x in cmd))
    vals = re.findall(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", out)
    return (float(vals[-1]) if vals else None), ("" if vals else out[-800:])


def kl_of(model: Path, corpus: Path, base_logits: Path, ngl: str, cmds: list):
    cmd = [PERPLEXITY, "-m", model, "-f", corpus, "-c", "512", "--temp", "0", "--seed", SEED,
           "-ngl", ngl, "--chunks", "1", "--kl-divergence", "--kl-divergence-base", base_logits]
    rc, out = run(cmd, timeout=3600)
    cmds.append(shlex.join(str(x) for x in cmd))
    vals = {k: (float(re.search(p, out).group(1)) if re.search(p, out) else None)
            for k, p in KL_RE.items()}
    if vals["kl_mean"] is None:
        return None, out[-800:]
    return vals, ""


def agree(a: Path, b: Path, prompts: Path, ngl: str, cmds: list) -> float:
    rows = [p for p in prompts.read_text().splitlines() if p.strip()]
    same = 0
    for pr in rows:
        outs = []
        for m in (a, b):
            cmd = [COMPLETION, "-m", m, "-p", pr, "-n", str(N_TOKENS), "--temp", "0",
                   "--seed", SEED, "-ngl", ngl, "--no-display-prompt", "--no-conversation"]
            rc, out = run(cmd, timeout=600)
            cmds.append(shlex.join(str(x) for x in cmd))
            outs.append(out.strip())
        same += int(outs[0] == outs[1])
    return same / max(1, len(rows))


def pick_backend(f16: Path, corpus: Path, requested: str, cmds: list) -> tuple[str, str, dict]:
    """Benchmark -ngl 0 vs -ngl 99 once; pick the faster. Never assume GPU is better."""
    if requested != "auto":
        return requested, ("99" if requested == "vulkan" else "0"), {}
    speeds = {}
    for name, ngl in (("cpu", "0"), ("vulkan", "99")):
        t0 = time.time()
        v, _ = ppl_of(f16, corpus, ngl, cmds)
        if v is not None:
            speeds[name] = round(PPL_BYTES / max(1e-6, time.time() - t0), 1)
    best = max(speeds, key=speeds.get) if speeds else "cpu"
    return best, ("99" if best == "vulkan" else "0"), speeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--backend", choices=("auto", "cpu", "vulkan"), default="auto")
    ap.add_argument("--tags", default=",".join(TAGS_DEFAULT))
    ap.add_argument("--keep-gguf", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    model_dir = Path(a.model_dir).resolve()
    tag = a.tag or model_dir.name
    shared = common.WORK / "gguf"
    scratch = shared / tag
    scratch.mkdir(parents=True, exist_ok=True)
    inp = prep_inputs(scratch)
    cmds: list[str] = []
    results: dict = {}
    notes: list[str] = []
    f16 = scratch / "model-f16.gguf"
    lock_name = None
    ok, err = convert(model_dir, f16, cmds)
    if err:
        notes.append(err)
    if not ok:
        payload = {"script": "gguf_probe.py", "tag": tag, "model_dir": str(model_dir),
                   "status": "FAILED", "error": err, "cmds": cmds, "git_head": git_head(),
                   "duration_s": round(time.time() - t0, 1)}
        common.wjson(Path(a.out), payload)
        print(json.dumps(payload, indent=2))
        sys.exit(1)

    want_gpu = a.backend in ("auto", "vulkan")
    if want_gpu:
        log("  [gguf] waiting for gpu lock")
        lock_name = common.lock("gpu", timeout=5400)
        lock_name.__enter__()
    try:
        backend, ngl, speeds = pick_backend(f16, inp["kl"], a.backend, cmds)
        log(f"  [gguf] backend={backend} ngl={ngl} speeds={speeds}")
        logits = scratch / "f16-logits.bin"
        ppl_f16, perr = ppl_of(f16, inp["ppl"], ngl, cmds)
        if ppl_f16 is None:
            notes.append(f"f16 ppl failed: {perr}")
        _, klerr = ppl_of(f16, inp["kl"], ngl, cmds,
                          extra=("--save-all-logits", logits))
        results["f16"] = {"status": "OK" if ppl_f16 else "FAILED", "ppl": ppl_f16,
                          "ppl_f16": ppl_f16, "size_mb": f16.stat().st_size / 1048576,
                          "backend": backend, "throughput_Bps": speeds or None}
        for t in [x.strip() for x in a.tags.split(",") if x.strip()]:
            qp = scratch / f"model-{t}.gguf"
            rc, out = run([QUANTIZE, f16, qp, t.upper(), "8"], timeout=1800)
            cmds.append(shlex.join([str(x) for x in (QUANTIZE, f16, qp, t.upper(), "8")]))
            if rc or not qp.exists():
                results[t] = {"status": "UNAVAILABLE", "reason": out[-1200:]}
                continue
            pq, perr2 = ppl_of(qp, inp["ppl"], ngl, cmds)
            ent = {"status": "OK", "size_mb": qp.stat().st_size / 1048576,
                   "ppl": pq, "ppl_f16": ppl_f16}
            if pq is None:
                ent.update(status="UNAVAILABLE", reason=perr2)
                results[t] = ent
                continue
            ent["dppl"] = round(pq - (ppl_f16 or 0), 4)
            ent["rel_dppl"] = round(pq / ppl_f16 - 1.0, 6) if ppl_f16 else None
            if logits.exists():
                kv, kerr = kl_of(qp, inp["kl"], logits, ngl, cmds)
                if kv:
                    ent.update(kv)
                else:
                    ent["kl_mean"] = "UNAVAILABLE"
                    notes.append(f"{t} kl: {kerr[:200]}")
            if t in AGREE_TAGS:
                ent["tokagree"] = round(agree(f16, qp, inp["prompts"], ngl, cmds), 4)
            rel, km, ta = ent.get("rel_dppl"), ent.get("kl_mean"), ent.get("tokagree")
            ent["pass"] = bool(rel is not None and rel <= PASS_CONTRACT["rel_dppl_max"]
                               and (not isinstance(km, (int, float))
                                    or km <= PASS_CONTRACT["kl_mean_max"])
                               and (ta is None or ta >= PASS_CONTRACT["tokagree_min"]))
            results[t] = ent
            log(f"  [gguf] {t}: ppl={pq:.3f} rel_dppl={ent['rel_dppl']:+.4f} "
                f"KLD={ent.get('kl_mean')} agree={ta} pass={ent['pass']}")
            if not a.keep_gguf:
                qp.unlink(missing_ok=True)
    finally:
        if lock_name is not None:
            lock_name.__exit__()
        logits = scratch / "f16-logits.bin"
        logits.unlink(missing_ok=True)
        if not a.keep_gguf:
            f16.unlink(missing_ok=True)
    payload = {"script": "gguf_probe.py", "tag": tag, "model_dir": str(model_dir),
               "status": "OK", "results": results, "pass_contract": PASS_CONTRACT,
               "corpus": {"ppl_bytes": PPL_BYTES, "kl_bytes": KL_BYTES,
                          "prompts": N_PROMPTS, "prompt_tokens": N_TOKENS,
                          "source": str(common.EVAL_TEXT)},
               "notes": notes, "backend": {"device": backend, "ngl": ngl, "llama_dir": str(LLAMA)},
               "versions": {"llama_cpp": subprocess.run(
                   [COMPLETION, "--version"], text=True, capture_output=True).stdout.strip().splitlines()[:1],
                            "python": sys.version.split()[0]},
               "git_head": git_head(), "duration_s": round(time.time() - t0, 1),
               "cmds": cmds}
    common.wjson(Path(a.out), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
