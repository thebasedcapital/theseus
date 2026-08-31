#!/usr/bin/env python3
"""K-10 second-corpus replication for native-bf16 llama.cpp damage.

The second corpus is a deterministic byte slice disjoint from M1's [0, 32768)
PPL / [0, 8192) KLD inputs.  Each artifact is converted to native bf16 GGUF,
then its own bf16 GGUF is used as the sole quantization reference.  No value in
m1/work/quant_ref.json is read or compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M1 = ROOT / "m1"
OUT = Path(__file__).resolve().parent
SOURCE = M1 / "data" / "eval_wikitext.txt"
BASE = Path(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"))
PREP = M1 / "work" / "prep_base_exact"
PREP_SCRIPT = M1 / "make_variants.py"
PYTHON = Path(os.environ.get("THESEUS_PY", sys.executable))
LLAMA = Path("/home/admin/tools/llama.cpp-vulkan/llama-b9851")
CONVERTER = Path("/home/admin/tools/llama.cpp-cuda-src/convert_hf_to_gguf.py")
EXTRA_SITE = Path("/home/admin/laps/benchmarks/swebench/.venv/lib/python3.12/site-packages")
PPL_BYTES = 32768
KL_BYTES = 8192
OFFSET = 65536
SEED = 0
SEQLEN = 512
TAGS = ("q8_0", "q4_k_m")
PASS_CONTRACT = {
    "mode": "within-artifact-reference-relative",
    "rel_dppl_slack": 0.010,
    "kl_mean_slack": 0.005,
    "reference_first": True,
    "note": "M1 amended reference-relative contract; each artifact calibrated independently on this slice",
}
KL_RE = {
    "kl_mean": r"Mean\s+KLD:\s*([0-9.eE+-]+)",
    "kl_median": r"Median\s+KLD:\s*([0-9.eE+-]+)",
    "kl_p90": r"90\.0%\s+KLD:\s*([0-9.eE+-]+)",
    "kl_p95": r"95\.0%\s+KLD:\s*([0-9.eE+-]+)",
    "kl_p99": r"99\.0%\s+KLD:\s*([0-9.eE+-]+)",
    "kl_p999": r"99\.9%\s+KLD:\s*([0-9.eE+-]+)",
    "kl_max": r"Maximum KLD:\s*([0-9.eE+-]+)",
}


def sha256_16(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]
NGL = os.environ.get("TSX_NGL", "0")



def corpus_manifest() -> dict:
    data = SOURCE.read_bytes()
    end = OFFSET + PPL_BYTES
    if end > len(data):
        raise RuntimeError(f"second slice ends at {end}, source has {len(data)} bytes")
    chunk = data[OFFSET:end]
    return {
        "source": str(SOURCE.resolve()),
        "source_size_bytes": len(data),
        "offset_bytes": OFFSET,
        "end_exclusive_bytes": end,
        "length_bytes": len(chunk),
        "sha256_16": sha256_16(chunk),
        "ppl_range": [OFFSET, end],
        "kl_range": [OFFSET, OFFSET + KL_BYTES],
        "old_m1_ranges": [[0, PPL_BYTES], [0, KL_BYTES]],
        "non_overlapping": OFFSET >= PPL_BYTES,
    }


def selfcheck() -> dict:
    m = corpus_manifest()
    data = SOURCE.read_bytes()
    again = data[m["offset_bytes"]:m["end_exclusive_bytes"]]
    source = Path(__file__).read_text()
    measure = source[source.index("def measure_artifact"):source.index("def main")]
    checks = {
        "source_exists": SOURCE.is_file(),
        "range_in_bounds": m["end_exclusive_bytes"] <= len(data),
        "non_overlap_old_ppl": m["offset_bytes"] >= PPL_BYTES,
        "non_overlap_old_kl": m["offset_bytes"] >= KL_BYTES,
        "length_exact": m["length_bytes"] == PPL_BYTES,
        "hash_stable": m["sha256_16"] == sha256_16(again),
        "reference_first_contract": PASS_CONTRACT["reference_first"] is True,
        "reference_first_order": source.index("ref_ppl, ref_meta") < source.index("for tag in TAGS"),
        "no_cross_corpus_comparison": "quant_ref.json" not in measure and "REF_FILE" not in measure,
    }
    if not all(checks.values()):
        raise SystemExit(json.dumps({"status": "FAIL", "checks": checks}, indent=2))
    return {"status": "PASS", "checks": checks, "corpus": m,
            "contract": PASS_CONTRACT}

def command_runner(commands: list):
    def run(cmd: list[str], *, env: dict | None = None, timeout: int = 3600) -> tuple[int, str]:
        started = time.time()
        e = dict(os.environ, **(env or {}))
        rec = {"argv": [str(x) for x in cmd], "cwd": str(ROOT), "started_unix": started}
        try:
            p = subprocess.run([str(x) for x in cmd], cwd=ROOT, env=e, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            out = p.stdout or ""
            rec.update({"returncode": p.returncode, "duration_s": round(time.time() - started, 3),
                        "output_tail": out[-1200:]})
            commands.append(rec)
            return p.returncode, out
        except Exception as exc:
            rec.update({"returncode": None, "duration_s": round(time.time() - started, 3),
                        "error": repr(exc)})
            commands.append(rec)
            return -1, repr(exc)
    return run

def ppl(model: Path, corpus: Path, run, *, logits: Path | None = None) -> tuple[float | None, dict]:
    cmd = [LLAMA / "llama-perplexity", "-m", model, "-f", corpus, "-c", str(SEQLEN),
           "--temp", "0", "--seed", str(SEED), "-ngl", NGL, "--chunks", "4"]
    if logits is not None:
        cmd += ["--save-all-logits", logits]
    rc, out = run(cmd)
    vals = re.findall(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", out)
    return (float(vals[-1]) if vals else None), {"returncode": rc, "output_tail": out[-1200:]}

def kld(model: Path, corpus: Path, logits: Path, run) -> dict:
    cmd = [LLAMA / "llama-perplexity", "-m", model, "-f", corpus, "-c", str(SEQLEN),
           "--temp", "0", "--seed", str(SEED), "-ngl", NGL, "--chunks", "1",
           "--kl-divergence", "--kl-divergence-base", logits]
    rc, out = run(cmd)
    vals = {k: (float(re.search(p, out).group(1)) if re.search(p, out) else None)
            for k, p in KL_RE.items()}
    vals.update({"returncode": rc, "output_tail": out[-1200:]})
    return vals





def convert(model: Path, target: Path, run) -> None:
    env = {"PYTHONPATH": str(EXTRA_SITE) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    rc, out = run([PYTHON, CONVERTER, "--outfile", target, "--outtype", "bf16", model], env=env,
                  timeout=1800)
    if rc or not target.exists():
        raise RuntimeError(f"bf16 conversion failed rc={rc}: {out[-800:]}")


def second_equivalence(model_a: Path, model_b: Path, manifest: dict) -> dict:
    """Compare fp32 forwards on tokens from the second slice, CPU-only and reference-first."""
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(M1))
    import common

    tok = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    text = SOURCE.read_bytes()[manifest["offset_bytes"]:manifest["end_exclusive_bytes"]].decode("utf-8", "replace")
    ids = tok(text, add_special_tokens=False, return_tensors=None)["input_ids"][:4096]
    usable = len(ids) - (len(ids) % SEQLEN)
    if usable < SEQLEN:
        raise RuntimeError(f"second corpus tokenized to only {len(ids)} tokens")
    x = torch.tensor(ids[:usable], dtype=torch.long).view(-1, SEQLEN)

    def logits_for(path: Path) -> torch.Tensor:
        model = common.load_model(path, dtype=torch.float32, device="cpu")
        out = []
        with torch.no_grad():
            for batch in x:
                out.append(model(input_ids=batch.unsqueeze(0)).logits.float().cpu())
        del model
        common.release("cpu")
        return torch.cat(out, 0)

    started = time.time()
    la = logits_for(model_a)
    lb = logits_for(model_b)
    diff = la - lb
    lp = torch.log_softmax(la, -1)
    lq = torch.log_softmax(lb, -1)
    npos = la.shape[0] * la.shape[1]
    kl = (lp.exp() * (lp - lq)).sum().item() / max(1, npos)
    agree = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
    def ppx(lg):
        loss, n = 0.0, 0
        for i in range(lg.shape[0]):
            loss += torch.nn.functional.cross_entropy(lg[i, :-1], x[i, 1:], reduction="sum").item()
            n += x.shape[1] - 1
        return math.exp(loss / n)
    pa, pb = ppx(la), ppx(lb)
    out = {
        "status": "EQUIVALENT" if kl <= 2e-3 and agree >= .995 and abs(pb / pa - 1) <= 2e-3 else "NOT_EQUIVALENT",
        "device": "cpu", "dtype": "fp32 forward over bf16 artifacts",
        "ntokens": usable, "seqlen": SEQLEN, "positions": npos,
        "ppl_base": pa, "ppl_prepared": pb, "rel_ppl": abs(pb / pa - 1),
        "kl_mean_nats": kl, "top1_agree": agree,
        "max_dlogit": diff.abs().max().item(), "duration_s": round(time.time() - started, 3),
        "gate": {"kl_mean_nats_max": 2e-3, "top1_agree_min": .995, "rel_ppl_max": 2e-3},
        "corpus_sha256_16": manifest["sha256_16"],
    }
    del la, lb, diff, lp, lq
    return out


def measure_artifact(name: str, model: Path, corpus_ppl: Path, corpus_kl: Path, run, scratch: Path) -> dict:
    f16 = scratch / f"{name}-bf16.gguf"
    logits = scratch / f"{name}-bf16-logits.bin"
    convert(model, f16, run)
    ref_ppl, ref_meta = ppl(f16, corpus_ppl, run)
    # Save logits on the KLD slice separately; never compare logits across corpus slices.
    _, ref_kl_meta = ppl(f16, corpus_kl, run, logits=logits)
    out = {"artifact": name,
           "reference": {"dtype": "bf16", "ppl": ref_ppl, **ref_meta,
                         "kl_reference": ref_kl_meta},
           "quantizations": {}}
    if ref_ppl is None or not logits.exists():
        out["status"] = "UNAVAILABLE"
        out["blocker"] = "bf16 reference PPL or logits unavailable"
        return out
    for tag in TAGS:
        qpath = scratch / f"{name}-{tag}.gguf"
        qcmd = [LLAMA / "llama-quantize", f16, qpath, tag.upper(), "8"]
        rc, text = run(qcmd, timeout=1800)
        ent = {"status": "UNAVAILABLE", "returncode": rc, "output_tail": text[-1200:]}
        if rc == 0 and qpath.exists():
            qp, meta = ppl(qpath, corpus_ppl, run)
            ent.update({"ppl": qp, **meta})
            if qp is not None:
                ent["dppl"] = qp - ref_ppl
                ent["rel_dppl"] = qp / ref_ppl - 1.0
                ent.update(kld(qpath, corpus_kl, logits, run))
                ent["status"] = "OK" if ent.get("kl_mean") is not None else "UNAVAILABLE"
                if ent["status"] == "UNAVAILABLE":
                    ent["blocker"] = "KLD parser or llama.cpp KLD output unavailable"
        out["quantizations"][tag] = ent
        qpath.unlink(missing_ok=True)
    out["status"] = "OK" if all(v.get("status") == "OK" for v in out["quantizations"].values()) else "UNAVAILABLE"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--out", default=str(OUT / "results.json"))
    ap.add_argument("--commands", default=str(OUT / "commands.json"))
    a = ap.parse_args()
    if a.selfcheck:
        print(json.dumps(selfcheck(), indent=2, sort_keys=True))
        return
    if not a.run:
        ap.error("choose --selfcheck or --run")
    started = time.time()
    manifest = corpus_manifest()
    OUT.mkdir(parents=True, exist_ok=True)
    corpus_ppl = OUT / "corpus_second_ppl.txt"
    corpus_kl = OUT / "corpus_second_kl.txt"
    raw = SOURCE.read_bytes()[OFFSET:OFFSET + PPL_BYTES]
    corpus_ppl.write_bytes(raw)
    corpus_kl.write_bytes(raw[:KL_BYTES])
    commands = []
    run = command_runner(commands)
    prep_created = False
    scratch = OUT / ".scratch"
    scratch.mkdir(exist_ok=True)
    result = {
        "status": "UNAVAILABLE", "verdict": "unavailable", "corpus": manifest,
        "contract": PASS_CONTRACT, "artifacts": {}, "equivalence": None,
        "reference_first": True, "cross_corpus_comparison": False,
        "llama_cpp": str(LLAMA), "llama_commit": "b9851", "timings": {},
    }
    try:
        if not PREP.exists():
            prep_created = True
            rc, text = run([PYTHON, PREP_SCRIPT, "--only", "prep_base_exact"],
                           env={"TSX_CPU": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                           timeout=1800)
            if rc or not PREP.exists():
                raise RuntimeError(f"prep_base_exact rebuild failed rc={rc}: {text[-1000:]}")
        result["equivalence"] = second_equivalence(BASE, PREP, manifest)
        if result["equivalence"]["status"] == "NOT_EQUIVALENT":
            result["verdict"] = "refutation"
            result["status"] = "OK"
            result["blocker"] = "fp32 second-corpus equivalence gate failed; surgery not compared"
        else:
            for name, model in (("base", BASE), ("prep_base_exact", PREP)):
                result["artifacts"][name] = measure_artifact(name, model, corpus_ppl, corpus_kl,
                                                               run, scratch)
            b = result["artifacts"]["base"]["quantizations"].get("q4_k_m", {})
            p = result["artifacts"]["prep_base_exact"]["quantizations"].get("q4_k_m", {})
            if isinstance(b.get("rel_dppl"), (int, float)) and isinstance(p.get("rel_dppl"), (int, float)):
                result["q4_relative_delta_prepared_minus_base"] = p["rel_dppl"] - b["rel_dppl"]
                result["verdict"] = "pass" if p["rel_dppl"] < b["rel_dppl"] else "neutral"
                result["status"] = "OK"
            else:
                result["verdict"] = "unavailable"
                result["blocker"] = "Q4 relative ΔPPL unavailable for base or prepared"
    except Exception as exc:  # noqa: BLE001
        result["verdict"] = "unavailable"
        result["blocker"] = repr(exc)
    finally:
        if prep_created:
            shutil.rmtree(PREP, ignore_errors=True)
        for p in scratch.glob("*"):
            p.unlink(missing_ok=True)
        scratch.rmdir()
        # Corpus files are persistent, small evidence of the exact input slice.
        result["timings"]["total_s"] = round(time.time() - started, 3)
        result["commands_file"] = str(Path(a.commands).resolve())
        Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        Path(a.commands).write_text(json.dumps({"commands": commands, "timings": result["timings"]},
                                               indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
