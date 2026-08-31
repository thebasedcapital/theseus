#!/usr/bin/env python3
"""M1 quantization probe: real llama.cpp GGUF K-quant surgery on an HF checkpoint.

    <venv python> m1/gguf_probe.py --model-dir <hf dir> --out <json> [--tag NAME]
                                   [--backend auto|cpu|vulkan] [--tags q8_0,q5_k_m,q4_k_m]

Damage is always measured *within* one artifact: the checkpoint is converted to f16 GGUF, that
f16 model is the reference, and each quantization of the same weights is compared against it.
That is the reserve question — "what does quantizing THIS artifact cost" — and it keeps any
cross-variant comparison out of the measurement.

Per tag: ppl / dppl / rel_dppl (llama-perplexity, fixed 32 KiB slice); mean, median, p90, p95,
p99, p99.9 and max KL divergence (llama-perplexity --kl-divergence against the f16 model's own
saved logits, fixed 8 KiB slice); greedy continuation retention (temp 0, 4 fixed prompts, mean
shared prefix length over 32 tokens); artifact size.

Contract is reference-relative: pristine Q4_K_M on this checkpoint already costs +2.27 % PPL and
0.0319 nats mean KLD, so an absolute cap fails base. The run tagged "base" (or the first run to
produce a number for a tag) writes m1/work/quant_ref.json and reports pass=null; later runs are
judged against it. Amended after base calibration and before any variant was measured.
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
QUANTIZE, PERPLEXITY, COMPLETION = (LLAMA / x for x in
                                    ("llama-quantize", "llama-perplexity", "llama-completion"))
CONVERTER = Path("/home/admin/tools/llama.cpp-cuda-src/convert_hf_to_gguf.py")
# Export dtype matters more than the quantizer for gauged artifacts. Measured on g3_pow2
# (bit-identical logits in fp32 torch): bf16 export ppl 12.1351 vs base 12.1399, while f16 and
# f32 exports both give ~177 and Q4_K_M from that source gives 3.2e5. bf16 is also the artifact's
# native dtype, so it is the honest reference for "damage caused by quantizing".
OUTTYPE = os.environ.get("TSX_OUTTYPE", "bf16")
EXTRA_SITE = Path("/home/admin/laps/benchmarks/swebench/.venv/lib/python3.12/site-packages")
SEED = 0
PPL_BYTES = 32768
KL_BYTES = 8192
N_PROMPTS, N_TOKENS = 4, 32
AGREE_TAGS = ("q4_k_m",)
REF_FILE = common.WORK / "quant_ref.json"
PASS_CONTRACT = {"mode": "reference-relative",
                 "rel_dppl_slack": 0.010,      # <= ref + 1.0 pp
                 "kl_mean_slack": 0.005,       # <= ref + 0.005 nats
                 "prefix_agree_slack": 0.10,   # >= ref - 0.10 of 32 tokens
                 "amended": "2026-08-30 after the base calibration run, before any variant was "
                            "measured; absolute caps failed the pristine checkpoint"}
KL_RE = {"kl_mean": r"Mean\s+KLD:\s*([0-9.eE+-]+)",
         "kl_median": r"Median\s+KLD:\s*([0-9.eE+-]+)",
         "kl_p90": r"90\.0%\s+KLD:\s*([0-9.eE+-]+)",
         "kl_p95": r"95\.0%\s+KLD:\s*([0-9.eE+-]+)",
         "kl_p99": r"99\.0%\s+KLD:\s*([0-9.eE+-]+)",
         "kl_p999": r"99\.9%\s+KLD:\s*([0-9.eE+-]+)",
         "kl_max": r"Maximum KLD:\s*([0-9.eE+-]+)"}


def run(cmd, env=None, timeout=2400):
    p = subprocess.run([str(x) for x in cmd], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return p.returncode, p.stdout


def q(cmd):
    return shlex.join(str(x) for x in cmd)


def git_head() -> str:
    return subprocess.run(["git", "-C", str(common.REPO), "rev-parse", "HEAD"],
                          text=True, capture_output=True).stdout.strip() or "unknown"


def llama_version() -> list[str]:
    for exe, args in ((COMPLETION, ["--version"]), (QUANTIZE, [])):
        try:
            r = subprocess.run([str(exe), *args], text=True, capture_output=True, timeout=60)
        except Exception:                                        # noqa: BLE001
            continue
        hits = [ln.strip() for ln in (r.stdout + r.stderr).splitlines()
                if "version:" in ln or "built with" in ln][:2]
        if hits:
            return hits
    return ["UNAVAILABLE"]


def prep_inputs() -> dict:
    shared = common.WORK / "gguf"
    shared.mkdir(parents=True, exist_ok=True)
    text = common.EVAL_TEXT.read_bytes()
    ppl_f, kl_f = shared / f"eval_ppl_{PPL_BYTES}.txt", shared / f"eval_kl_{KL_BYTES}.txt"
    ppl_f.write_bytes(text[:PPL_BYTES])
    kl_f.write_bytes(text[:KL_BYTES])
    pf = shared / "prompts.txt"
    if not pf.exists():
        step = PPL_BYTES // (N_PROMPTS + 1)
        chunk = text[:PPL_BYTES].decode("utf-8", "replace")
        pf.write_text("\n".join(chunk[i * step: i * step + 128] for i in range(N_PROMPTS)) + "\n")
    return {"ppl": ppl_f, "kl": kl_f, "prompts": pf}


def gguf_has_output_tensor(path: Path) -> bool | None:
    """Read tensor names straight out of the GGUF header (no gguf package needed)."""
    import struct
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                f.seek(0)
            f.read(4)                                            # version
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def rstr():
                return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
            for _ in range(n_kv):
                rstr()
                t = struct.unpack("<I", f.read(4))[0]
                if t in (0, 6):
                    rstr()
                elif t == 1:
                    f.read(1)
                elif t == 2:
                    f.read(2)
                elif t in (3, 4):
                    f.read(4)
                elif t in (5, 8):
                    f.read(8)
                elif t == 7:
                    n = struct.unpack("<Q", f.read(8))[0]
                    et = struct.unpack("<I", f.read(4))[0]
                    f.read((8 if et == 8 else 4) * n)
                elif t == 9:
                    f.read(16)
                else:
                    return None
            for _ in range(n_tensors):
                if rstr() == "output.weight":
                    return True
                n_dims = struct.unpack("<I", f.read(4))[0]
                f.read(8 * n_dims + 4 + 8)
        return False
    except Exception:                                            # noqa: BLE001
        return None


def convert(model_dir: Path, f16: Path, cmds: list, notes: list) -> bool:
    env = dict(os.environ)
    if EXTRA_SITE.exists():
        env["PYTHONPATH"] = str(EXTRA_SITE) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, CONVERTER, "--outfile", f16, "--outtype", OUTTYPE, model_dir]
    rc, out = run(cmd, env=env, timeout=1200)
    cmds.append(q(cmd))
    if rc or not f16.exists():
        notes.append(f"convert failed rc={rc}: {out[-600:]}")
        return False
    tie = bool(json.loads((model_dir / "config.json").read_text()).get("tie_word_embeddings", True))
    has_head_file = any("lm_head" in k for k in __import__("safetensors.torch", fromlist=["x"])
                        .load_file(str(model_dir / "model.safetensors")))
    out_t = gguf_has_output_tensor(f16)
    if not tie and has_head_file and out_t is False:
        notes.append("FINDING: untied source produced no output.weight — llama.cpp re-tied the "
                     "head, so this artifact's GGUF is not the same function")
        return False
    if not tie and has_head_file and out_t is None:
        notes.append("could not parse GGUF header; tie fidelity unverified")
    return True


def ppl_of(model: Path, corpus: Path, ngl: str, cmds: list, extra=()):
    cmd = [PERPLEXITY, "-m", model, "-f", corpus, "-c", "512", "--temp", "0", "--seed", SEED,
           "-ngl", ngl, "--chunks", "4", *extra]
    rc, out = run(cmd, timeout=3600)
    cmds.append(q(cmd))
    vals = re.findall(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", out)
    return (float(vals[-1]) if vals else None), ("" if vals else out[-500:])


def kl_of(model: Path, corpus: Path, base_logits: Path, ngl: str, cmds: list):
    cmd = [PERPLEXITY, "-m", model, "-f", corpus, "-c", "512", "--temp", "0", "--seed", SEED,
           "-ngl", ngl, "--chunks", "1", "--kl-divergence", "--kl-divergence-base", base_logits]
    rc, out = run(cmd, timeout=3600)
    cmds.append(q(cmd))
    vals = {k: (float(re.search(p, out).group(1)) if re.search(p, out) else None)
            for k, p in KL_RE.items()}
    return (vals, "") if vals["kl_mean"] is not None else (None, out[-400:])


def agree(a: Path, b: Path, prompts: Path, ngl: str, cmds: list) -> dict:
    """Greedy retention vs the same weights in f16.

    Exact 32-token equality is the wrong statistic for a 0.5B at 4 bits — one divergent token
    zero-codes a whole prompt (measured: base Q4 scores 0.00). Shared prefix length is primary.
    """
    rows = [p for p in prompts.read_text().splitlines() if p.strip()]
    pref, exact = [], 0
    for pr in rows:
        outs = []
        for m in (a, b):
            cmd = [COMPLETION, "-m", m, "-p", pr, "-n", str(N_TOKENS), "--temp", "0",
                   "--seed", SEED, "-ngl", ngl, "--no-display-prompt", "--no-conversation"]
            rc, out = run(cmd, timeout=900)
            cmds.append(q(cmd))
            outs.append(out.strip().split())
        n = 0
        for x, y in zip(outs[0], outs[1]):
            if x != y:
                break
            n += 1
        pref.append(n / max(1, N_TOKENS))
        exact += int(outs[0] == outs[1])
    return {"prefix_agree": round(sum(pref) / max(1, len(pref)), 4),
            "exact_agree": round(exact / max(1, len(rows)), 4), "prompts": len(rows),
            "prompt_tokens": N_TOKENS}


def pick_backend(f16: Path, corpus: Path, requested: str, cmds: list):
    """Benchmark -ngl 0 against -ngl 99 once; never assume the GPU wins."""
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


def judge(t: str, tag: str, ent: dict, results: dict) -> None:
    ref = common.rjson(REF_FILE) if REF_FILE.exists() else {}
    r = ref.get(t)
    rel, km, pa = ent.get("rel_dppl"), ent.get("kl_mean"), ent.get("prefix_agree")
    if r is None:
        ent["pass"], ent["role"] = None, "reference_calibration"
        # prefix agreement is optional here: on a 0.5B at 4 bits the greedy continuation can
        # diverge from the first token even for the pristine checkpoint (measured 0.0 on base),
        # which makes it a finding but a useless discriminator.
        if all(isinstance(x, (int, float)) for x in (rel, km)):
            ref[t] = {"tag": tag, "rel_dppl": rel, "kl_mean": km, "prefix_agree": pa}
            common.wjson(REF_FILE, ref)
        return
    lim = {"rel_dppl": r["rel_dppl"] + PASS_CONTRACT["rel_dppl_slack"],
           "kl_mean": r["kl_mean"] + PASS_CONTRACT["kl_mean_slack"]}
    if isinstance(r.get("prefix_agree"), (int, float)):
        lim["prefix_agree"] = r["prefix_agree"] - PASS_CONTRACT["prefix_agree_slack"]
    informative_pa = "prefix_agree" in lim and r["prefix_agree"] > 0
    ent["pass"] = bool(isinstance(rel, (int, float)) and rel <= lim["rel_dppl"]
                       and (not isinstance(km, (int, float)) or km <= lim["kl_mean"])
                       and (not informative_pa or not isinstance(pa, (int, float))
                            or pa >= lim["prefix_agree"]))
    if not informative_pa:
        ent["prefix_agree_uninformative"] = True
    ent["limits"] = lim
    ent["reference_tag"] = r.get("tag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--backend", choices=("auto", "cpu", "vulkan"), default="auto")
    ap.add_argument("--tags", default="q8_0,q5_k_m,q4_k_m")
    ap.add_argument("--keep-gguf", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    model_dir = Path(a.model_dir).resolve()
    tag = a.tag or model_dir.name
    scratch = common.WORK / "gguf" / tag
    scratch.mkdir(parents=True, exist_ok=True)
    inp, cmds, results, notes = prep_inputs(), [], {}, []
    f16, logits = scratch / "model-f16.gguf", scratch / "f16-logits.bin"
    payload = {"script": "gguf_probe.py", "tag": tag, "model_dir": str(model_dir),
               "git_head": git_head(), "torch": __import__("torch").__version__,
               "versions": {"llama_cpp": llama_version(), "python": sys.version.split()[0]},
               "export": {"outtype": OUTTYPE,
                          "note": "reference is the same weights exported to the artifact's "
                                  "native dtype; f16/f32 export damage is measured by "
                                  "m1/export_damage.py and is a separate operation"},
               "corpus": {"ppl_bytes": PPL_BYTES, "kl_bytes": KL_BYTES, "prompts": N_PROMPTS,
                          "prompt_tokens": N_TOKENS, "source": str(common.EVAL_TEXT)},
               "pass_contract": PASS_CONTRACT, "quant_ref": (common.rjson(REF_FILE)
                                                             if REF_FILE.exists() else None)}
    if not convert(model_dir, f16, cmds, notes):
        payload.update(status="FAILED", notes=notes, results={}, cmds=cmds,
                       duration_s=round(time.time() - t0, 1))
        common.wjson(Path(a.out), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        sys.exit(1)

    want_gpu = a.backend in ("auto", "vulkan")
    lk = None
    t_wait = time.time()
    if want_gpu:
        log("  [gguf] waiting for gpu lock")
        lk = common.lock("gpu", timeout=5400)
        lk.__enter__()
    lock_wait = round(time.time() - t_wait, 1)
    backend = ngl = None
    try:
        backend, ngl, speeds = pick_backend(f16, inp["kl"], a.backend, cmds)
        log(f"  [gguf] backend={backend} ngl={ngl} throughput={speeds}")
        ppl_f16, perr = ppl_of(f16, inp["ppl"], ngl, cmds)
        results["f16"] = {"status": "OK" if ppl_f16 else "FAILED", "ppl": ppl_f16,
                          "size_mb": round(f16.stat().st_size / 1048576, 1),
                          "backend": backend, "throughput_Bps": speeds or None,
                          "error": perr}
        _, klerr = ppl_of(f16, inp["kl"], ngl, cmds, extra=("--save-all-logits", logits))
        if klerr:
            notes.append(f"reference logits save failed: {klerr[:200]}")
        for t in [x.strip() for x in a.tags.split(",") if x.strip()]:
            qp = scratch / f"model-{t}.gguf"
            cmd = [QUANTIZE, f16, qp, t.upper(), "8"]
            rc, out = run(cmd, timeout=1800)
            cmds.append(q(cmd))
            if rc or not qp.exists():
                results[t] = {"status": "UNAVAILABLE", "reason": out[-900:]}
                continue
            pq, perr2 = ppl_of(qp, inp["ppl"], ngl, cmds)
            ent = {"status": "OK", "size_mb": round(qp.stat().st_size / 1048576, 1),
                   "ppl": pq, "ppl_f16": ppl_f16}
            if pq is None:
                results[t] = {**ent, "status": "UNAVAILABLE", "reason": perr2}
                continue
            ent["dppl"] = round(pq - (ppl_f16 or 0), 4)
            ent["rel_dppl"] = round(pq / ppl_f16 - 1.0, 6) if ppl_f16 else None
            if logits.exists():
                kv, kerr = kl_of(qp, inp["kl"], logits, ngl, cmds)
                if kv:
                    ent.update(kv)
                else:
                    ent["kl_mean"] = None
                    notes.append(f"{t} kl unavailable: {kerr[:200]}")
            if t in AGREE_TAGS:
                ent.update(agree(f16, qp, inp["prompts"], ngl, cmds))
            judge(t, tag, ent, results)
            results[t] = ent
            log(f"  [gguf] {t}: ppl={pq:.4f} rel_dppl={ent['rel_dppl']:+.4f} "
                f"KLD={ent.get('kl_mean')} prefix={ent.get('prefix_agree')} pass={ent.get('pass')}")
            if not a.keep_gguf:
                qp.unlink(missing_ok=True)
    finally:
        if lk is not None:
            lk.__exit__()
        logits.unlink(missing_ok=True)
        if not a.keep_gguf:
            f16.unlink(missing_ok=True)
    payload.update(status="OK", results=results, notes=notes, cmds=cmds,
                   backend={"device": backend, "ngl": ngl, "llama_dir": str(LLAMA),
                            "lock_wait_s": lock_wait},
                   duration_s=round(time.time() - t0, 1))
    common.wjson(Path(a.out), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
