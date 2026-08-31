#!/usr/bin/env python3
"""Real llama.cpp GGUF quantization probe for Qwen2-family HF directories."""
from __future__ import annotations
import argparse, gc, json, math, os, re, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
import common

PASS_CONTRACT = {"rel_dppl_max": 0.02, "kl_mean_max": 0.01, "tokagree_min": 0.85,
                 "kl_unavailable_passes": True}
TAGS = ("q8_0", "q6_k", "q5_k_m", "q4_k_m", "iq4_xs")
QUANT_TYPES = {t: t.upper() for t in TAGS}
M1 = Path(__file__).resolve().parent
REPO = M1.parent
WORK = M1 / "work" / "gguf"
EVAL = M1 / "data" / "eval_wikitext.txt"
CONVERTER = Path("/home/admin/tools/llama.cpp-cuda-src/convert_hf_to_gguf.py")
LLAMA = Path("/home/admin/tools/llama.cpp-vulkan/llama-b9851")
QUANTIZE, PERPLEXITY, COMPLETION = (LLAMA / x for x in ("llama-quantize", "llama-perplexity", "llama-completion"))
SEED = 0
CORPUS_BYTES = 32768
PPL_CHUNKS = 4
BACKEND_NGL = "0"
BACKEND_NAME = "cpu"
BACKEND_SPEED = None


def qstr(cmd):
    import shlex
    return shlex.join(str(x) for x in cmd)


def run(cmd, env=None, timeout=1800):
    p = subprocess.run([str(x) for x in cmd], text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, env=env, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def git_head():
    p = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
                       capture_output=True, check=False)
    return p.stdout.strip() if p.returncode == 0 else "UNAVAILABLE: " + p.stderr.strip()


def tag_for(d):
    try:
        d.relative_to(WORK.parent)
        return d.name
    except ValueError:
        return "base"


def converter_env():
    env = os.environ.copy()
    try:
        import sentencepiece  # noqa: F401
        return env, None
    except ImportError:
        site = Path("/home/admin/laps/benchmarks/swebench/.venv/lib/python3.12/site-packages")
        if (site / "sentencepiece").exists():
            env["PYTHONPATH"] = str(site) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            return env, f"sentencepiece from {site}"
        return env, "sentencepiece unavailable"


def inputs(corpus_bytes):
    WORK.mkdir(parents=True, exist_ok=True)
    raw = EVAL.read_bytes()
    corpus = WORK / f"eval_{corpus_bytes // 1024}k.txt"
    if not corpus.exists() or corpus.read_bytes() != raw[:corpus_bytes]:
        corpus.write_bytes(raw[:corpus_bytes])
    prompts = WORK / "prompts.txt"
    if not prompts.exists():
        text = raw.decode("utf-8", errors="replace").replace("\r", " ").replace("\n", " ")
        rows = []
        for i in range(32):
            pos = (len(text) - 128) * i // 31 if len(text) > 128 else 0
            rows.append(text[pos:pos + 128].replace("\x00", " "))
        prompts.write_text("\n".join(rows) + "\n")
    return corpus, prompts, {"path": str(corpus), "byte_start": 0,
        "byte_end_exclusive": min(corpus_bytes, len(raw)), "source": str(EVAL),
        "prompts_path": str(prompts), "prompts_count": 32,
        "prompt_generation": "32 fixed 128-character prefixes at evenly spaced positions"}


def parse_ppl(s):
    vals = re.findall(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", s)
    return float(vals[-1]) if vals else None


def parse_kl(s):
    m = re.search(r"Mean\s+KLD:\s*([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else None


def ppl(path, corpus, cmds, chunks=None, logits=None):
    cmd = [PERPLEXITY, "-m", path, "-f", corpus, "-c", "512", "--temp", "0", "--seed", SEED,
           "-ngl", BACKEND_NGL, "--chunks", PPL_CHUNKS if chunks is None else chunks]
    if logits:
        cmd += ["--save-all-logits", logits]
    cmds.append(qstr(cmd))
    rc, out, err = run(cmd, timeout=1800)
    s = out + "\n" + err
    if rc:
        return None, f"llama-perplexity rc={rc}: {err[-1200:]}"
    val = parse_ppl(s)
    return val, "" if val is not None else "PPL not found: " + s[-1200:]


def completion(path, prompt, cmds):
    cmd = [COMPLETION, "-m", path, "-p", prompt, "-n", "32", "--temp", "0", "--seed", SEED,
           "-ngl", BACKEND_NGL, "--no-display-prompt", "-no-cnv", "--simple-io"]
    cmds.append(qstr(cmd))
    rc, out, err = run(cmd, timeout=300)
    return (out, "") if rc == 0 else (None, f"llama-completion rc={rc}: {err[-800:]}")


def agreement(f16, quant, prompts, cmds):
    rows = prompts.read_text().splitlines()
    same = 0
    for prompt in rows:
        a, e = completion(f16, prompt, cmds)
        if e: return None, e
        b, e = completion(quant, prompt, cmds)
        if e: return None, e
        same += a == b
    return same / len(rows), ""


def choose_backend(f16, corpus, cmds, requested):
    global BACKEND_NGL, BACKEND_NAME, BACKEND_SPEED
    if requested == "cpu":
        return
    free = None
    if torch.cuda.is_available():
        try: free = int(torch.cuda.mem_get_info()[0])
        except Exception: pass
    if requested == "auto" and free is not None and free < 2_500_000_000:
        return
    bench = WORK / "backend_8k.txt"
    bench.write_bytes(EVAL.read_bytes()[:8192])
    speeds = {}
    for name, ngl in (("cpu", "0"), ("vulkan", "99")):
        old = BACKEND_NGL; BACKEND_NGL = ngl
        t = time.monotonic()
        val, reason = ppl(f16, bench, cmds, chunks=1)
        elapsed = time.monotonic() - t
        if val is not None and elapsed > 0:
            # llama-perplexity's 8KiB benchmark is comparable across backends.
            speeds[name] = {"ppl": val, "elapsed_s": elapsed, "tokens_per_s": 8192 / elapsed}
        elif requested == "vulkan" and name == "vulkan":
            BACKEND_NGL = old
            return
    BACKEND_NGL = old
    if requested == "vulkan" and "vulkan" in speeds:
        BACKEND_NAME = "vulkan"; BACKEND_NGL = "99"
    elif requested == "auto" and speeds:
        BACKEND_NAME = max(speeds, key=lambda n: speeds[n]["tokens_per_s"])
        BACKEND_NGL = "99" if BACKEND_NAME == "vulkan" else "0"
    BACKEND_SPEED = speeds


def convert(model_dir, f16, cmds, notes):
    env, note = converter_env()
    cmd = [sys.executable, CONVERTER, "--outfile", f16, "--outtype", "f16", model_dir]
    cmds.append(qstr(cmd) + (f" [PYTHONPATH={env['PYTHONPATH']}]" if "PYTHONPATH" in env else ""))
    rc, _, err = run(cmd, env=env)
    if rc: return False, f"converter rc={rc}: {err[-1200:]}"
    if note: notes.append(note)
    cfg = json.loads((model_dir / "config.json").read_text())
    if cfg.get("tie_word_embeddings") is False:
        try:
            sys.path.insert(0, "/home/admin/tools/llama.cpp-cuda-src/gguf-py")
            from gguf import GGUFReader
            names = {str(t.name) for t in GGUFReader(str(f16)).tensors}
            notes.append("untied output.weight survived conversion" if "output.weight" in names else
                         "FINDING: tie_word_embeddings=false but output.weight absent; converter re-tied or dropped lm_head")
        except Exception as e: notes.append(f"untied output.weight check unavailable: {e}")
    return True, ""


def quant_one(tag, f16, scratch, corpus, prompts, fp, logits, cmds):
    qpath = scratch / f"model-{tag}.gguf"
    cmd = [QUANTIZE, f16, qpath, QUANT_TYPES[tag], 8]
    cmds.append(qstr(cmd)); rc, _, err = run(cmd)
    if rc or not qpath.exists(): return {"status": "UNAVAILABLE", "reason": f"llama-quantize rc={rc}: {err[-1200:]}"}
    qp, reason = ppl(qpath, corpus, cmds)
    if qp is None:
        qpath.unlink(missing_ok=True); return {"status": "UNAVAILABLE", "reason": reason}
    kl = "UNAVAILABLE"; kl_reason = ""
    if logits:
        klcmd = [PERPLEXITY, "-m", qpath, "-f", corpus, "-c", 512, "--chunks", 1, "--temp", 0,
                 "--seed", SEED, "-ngl", BACKEND_NGL, "--kl-divergence", "--kl-divergence-base", logits]
        cmds.append(qstr(klcmd)); krc, ko, ke = run(klcmd, timeout=1800)
        kv = parse_kl(ko + "\n" + ke) if krc == 0 else None
        if kv is not None: kl = {"kl_mean": kv}
        else: kl_reason = f"KL unavailable: rc={krc}; {ke[-800:]}"
    tok, reason = agreement(f16, qpath, prompts, cmds)
    if tok is None:
        qpath.unlink(missing_ok=True); return {"status": "UNAVAILABLE", "reason": reason}
    dppl = qp - fp; rel = dppl / fp if fp else math.inf
    kv = kl.get("kl_mean") if isinstance(kl, dict) else None
    passed = rel <= PASS_CONTRACT["rel_dppl_max"] and (kl == "UNAVAILABLE" or kv <= PASS_CONTRACT["kl_mean_max"]) and tok >= PASS_CONTRACT["tokagree_min"]
    result = {"status": "OK", "ppl_q": qp, "dppl": dppl, "rel_dppl": rel, "kl": kl,
              "tokagree": tok, "size_mb": qpath.stat().st_size / 1048576, "passes": passed}
    if kl_reason: result["kl_reason"] = kl_reason
    qpath.unlink(missing_ok=True); gc.collect()
    return result


def main():
    global CORPUS_BYTES
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tags", default=",".join(TAGS))
    ap.add_argument("--backend", choices=("auto", "cpu", "vulkan"), default="auto")
    ap.add_argument("--corpus-bytes", type=int, default=32768)
    args = ap.parse_args(); CORPUS_BYTES = args.corpus_bytes
    start = time.monotonic(); model_dir = args.model_dir.expanduser().resolve(); tag = tag_for(model_dir)
    scratch = WORK / tag; scratch.mkdir(parents=True, exist_ok=True)
    corpus, prompts, corpus_info = inputs(CORPUS_BYTES)
    cmds, notes, errors, results = [], [], [], {}
    f16 = scratch / "model-f16.gguf"
    if not f16.exists():
        ok, reason = convert(model_dir, f16, cmds, notes)
        if not ok: errors.append(reason)
    if f16.exists():
        choose_backend(f16, corpus, cmds, args.backend)
        fp, reason = ppl(f16, corpus, cmds)
        if fp is None: errors.append(reason)
        else:
            results["f16"] = {"status": "OK", "ppl_f16": fp, "size_mb": f16.stat().st_size / 1048576}
            logits = WORK / "f16_kl_logits.bin"
            _, kreason = ppl(f16, corpus, cmds, chunks=1, logits=logits)
            if logits.exists():
                notes.append("KL reference logits generated from fixed corpus")
            else:
                logits = None; notes.append("kl UNAVAILABLE: " + (kreason or "reference logits missing"))
            for qt in (x.strip().lower() for x in args.tags.split(",") if x.strip()):
                results[qt] = quant_one(qt, f16, scratch, corpus, prompts, fp, logits, cmds) if qt in QUANT_TYPES else {"status": "UNAVAILABLE", "reason": "unknown quantization tag"}
            if logits: logits.unlink(missing_ok=True)
        f16.unlink(missing_ok=True); gc.collect()
    else:
        for qt in args.tags.split(","): results[qt.strip()] = {"status": "UNAVAILABLE", "reason": "f16 conversion unavailable"}
    (WORK / "backend_8k.txt").unlink(missing_ok=True)
    try:
        if not any(scratch.iterdir()): scratch.rmdir()
    except OSError: pass
    obj = {"script": "gguf_probe.py", "model_dir": str(model_dir), "tag": tag, "git_head": git_head(),
           "torch": torch.__version__, "versions": {"python": sys.version.split()[0], "torch": torch.__version__},
           "results": results, "duration_s": time.monotonic() - start, "pass_contract": PASS_CONTRACT,
           "corpus": corpus_info, "backend": {"device": BACKEND_NAME, "ngl": BACKEND_NGL,
           "throughput_benchmark": BACKEND_SPEED, "quantize": str(QUANTIZE), "perplexity": str(PERPLEXITY), "completion": str(COMPLETION)},
           "cmds": cmds, "notes": notes}
    if errors: obj["error"] = errors
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    print(json.dumps(obj, indent=2, sort_keys=True))

if __name__ == "__main__": main()
