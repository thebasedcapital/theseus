#!/usr/bin/env python3
"""K-8 natural-history pair: ordinary LoRA and linear merge in opposite orders.

The script is deliberately fail-closed.  It imports the existing true-LoRA modules and
merge implementation; it never changes those probes and removes every temporary checkpoint
before returning.  A real run is expensive, so all operation choices are frozen in CONTRACT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
M1 = ROOT / "m1"
M3 = ROOT / "m3"
M3.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(M1))
import common  # noqa: E402
import adapt_probe  # noqa: E402

# Frozen before execution.  The adaptation seed is distinct from the specialist seed (2718)
# and the future-reserve seed (9091).
CONTRACT = {
    "history_pair": {"A": ["adapt.lora.rule", "merge.linear", "quantize.q4_k_m"],
                     "B": ["merge.linear", "adapt.lora.rule", "quantize.q4_k_m"]},
    "base": str(common.REF_MODEL),
    "specialist": str(common.WORK / "specialist"),
    "history_seed": 4242,
    "future_seed": 9091,
    "specialist_seed": 2718,
    "adapt": {"rank": 16, "alpha": 32, "steps": 80, "batch_size": 2,
              "grad_accum": 1, "seq_len": 128, "lr": 3e-4, "train_examples": 512,
              "heldout_examples": 128, "targets": list(adapt_probe.TARGETS),
              "rule": "reverse 10-digit identifier: rev: ID -> reversed(ID)"},
    "merge": {"kind": "linear", "alpha": 0.30, "formula": "candidate + alpha*(specialist-candidate)"},
    "quantizer": {"commit": "llama.cpp b9851", "type": "Q4_K_M", "threads": 8,
                  "export_dtype": "bf16", "source_dtype": "bf16"},
    "corpus": {"source": str(common.EVAL_TEXT), "ppl_bytes": 32768, "kl_bytes": 8192,
               "tokens": 2048, "sequence_length": 512, "seed": 0},
    "present_match": {"relative_ppl_max": 0.005, "mean_kl_max": 2e-3, "top1_min": 0.995,
                       "static_tolerances": [0.05, 0.1, 0.2, 0.4, 0.8]},
    "null": {"min_shuffles": 200, "seed": 7, "alpha": 0.05},
    "future_divergence": {"capture_abs": 0.05, "q4_rel_dppl_abs": 0.01},
}

LLAMA = Path("/home/admin/tools/llama.cpp-vulkan/llama-b9851")
CONVERTER = Path("/home/admin/tools/llama.cpp-cuda-src/convert_hf_to_gguf.py")
INSPECT = ROOT / "inspect/target/release/theseus-inspect"
SCAN = ROOT / "scan/target/release/theseus-scan"
PY_EXTRA = Path("/home/admin/laps/benchmarks/swebench/.venv/lib/python3.12/site-packages")


def git_head():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frozen_contract():
    return json.loads(json.dumps(CONTRACT, sort_keys=True))


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rule_examples(n: int, seed: int, offset: int = 0):
    rng = random.Random(seed)
    rows = []
    for _ in range(offset + n):
        ident = "".join(rng.choice("0123456789") for _ in range(10))
        rows.append(ident)
    return rows[offset:offset + n]


def make_data(tok, ids):
    rows, labels, masks = [], [], []
    for ident in ids:
        p = tok(f"rev: {ident} -> ", add_special_tokens=False)["input_ids"]
        t = tok(ident[::-1], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        x, y = (p + t)[:CONTRACT["adapt"]["seq_len"]], ([-100] * len(p) + t)[:CONTRACT["adapt"]["seq_len"]]
        m = [1] * len(x)
        pad = tok.eos_token_id
        rows.append(x + [pad] * (CONTRACT["adapt"]["seq_len"] - len(x)))
        labels.append(y + [-100] * (CONTRACT["adapt"]["seq_len"] - len(y)))
        masks.append(m + [0] * (CONTRACT["adapt"]["seq_len"] - len(m)))
    return tuple(torch.tensor(x, dtype=torch.long) for x in (rows, labels, masks))


def train_lora_state(start_sd: dict, tok, seed: int, device: str):
    """Apply the existing true-LoRA module to a supplied state and return merged bf16 state."""
    set_seed(seed)
    model = common.state_to_model(start_sd, common.REF_MODEL, dtype=torch.bfloat16, device=device)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    # Existing implementation: true LoRA wrappers and target selection, imported read-only.
    adapt_probe.replace_targets(model)
    params = [p for p in model.parameters() if p.requires_grad]
    ids = rule_examples(CONTRACT["adapt"]["train_examples"] + CONTRACT["adapt"]["heldout_examples"], seed)
    x, y, mask = make_data(tok, ids[:CONTRACT["adapt"]["train_examples"]])
    held = make_data(tok, ids[CONTRACT["adapt"]["train_examples"]:])
    task_before = adapt_probe.task_loss(model, held, device)
    model.train()
    t0 = time.perf_counter()
    for step in range(CONTRACT["adapt"]["steps"]):
        i = (step * CONTRACT["adapt"]["batch_size"]) % len(x)
        xb, yb, mb = (z[i:i + CONTRACT["adapt"]["batch_size"]].to(device) for z in (x, y, mask))
        out = model(input_ids=xb, attention_mask=mb)
        loss = F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.size(-1)),
                               yb[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    task_after = adapt_probe.task_loss(model, held, device)
    out_sd = {k: v.detach().cpu().clone() for k, v in start_sd.items()}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, adapt_probe.LoRALinear):
                delta = (mod.b.float().cpu() @ mod.a.float().cpu()) * mod.scale
                out_sd[name + ".weight"] = (out_sd[name + ".weight"].float() + delta).to(torch.bfloat16)
    del opt, model
    common.release(device)
    return out_sd, {"seed": seed, "steps": CONTRACT["adapt"]["steps"], "runtime_s": elapsed,
                    "base_frozen": True, "targets": list(CONTRACT["adapt"]["targets"]),
                    "task_loss_before": task_before, "task_loss_after": task_after,
                    "capture": (task_before - task_after) / task_before}


def save_sd(sd: dict, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    common.save_state(sd, dst, common.REF_MODEL)
    return dst


def construct(order: str, work: Path, device: str, commands=None):
    tok = common.load_tokenizer(common.REF_MODEL)
    base = common.load_state(common.REF_MODEL)
    specialist = common.load_state(Path(CONTRACT["specialist"]))
    if order == "A":
        adapted, ar = train_lora_state(base, tok, CONTRACT["history_seed"], device)
        merged = common.merge_sd(adapted, specialist, CONTRACT["merge"]["alpha"], ties=False)
        stages = ["adapt.lora.rule", "merge.linear"]
    elif order == "B":
        merged = common.merge_sd(base, specialist, CONTRACT["merge"]["alpha"], ties=False)
        adapted, ar = train_lora_state(merged, tok, CONTRACT["history_seed"], device)
        merged = adapted
        stages = ["merge.linear", "adapt.lora.rule"]
    else:
        raise ValueError(f"unknown order {order}")
    bf16_dir = save_sd(merged, work / f"{order}.bf16")
    q = quantize(bf16_dir, work, commands if commands is not None else [])
    if q.get("status") != "OK":
        raise RuntimeError(q.get("reason", "Q4_K_M history operation failed"))
    del base, specialist, merged, adapted
    common.release(device)
    return bf16_dir, Path(q["q4"]), {
        "order": order, "stages": stages, "adapt": ar,
        "history_final_op": "quantize.q4_k_m", "dtype": "bf16",
        "merge_alpha": CONTRACT["merge"]["alpha"], "q4_bytes": q["q4_bytes"]}


def inspect(path: Path, out: Path):
    binary = SCAN if path.suffix.lower() == ".gguf" else INSPECT
    if not binary.exists():
        return {"status": "UNAVAILABLE", "reason": f"scanner missing: {binary}"}
    argv = ([str(binary), "inspect", str(path), "--json", str(out)] if binary == SCAN
            else [str(binary), str(path / "model.safetensors"), "--json", str(out)])
    p = subprocess.run(argv, capture_output=True, text=True, check=False)
    if p.returncode or not out.exists():
        return {"status": "UNAVAILABLE", "reason": (p.stderr + p.stdout)[-1000:]}
    return json.loads(out.read_text())


def feature_map(scan):
    fam = scan.get("per_family", scan.get("families", {}))
    if isinstance(fam, list):
        fam = {x.get("family"): x for x in fam}
    total = scan.get("total", {})
    # Match analysis/pairs.py semantics: worst family, fallback total.
    keys = ("q4_block_mse", "dyn_range_log10", "row_energy_imbalance", "frac_below_f16_normal")
    out = {}
    for k in keys:
        vals = [float(v[k]) for v in fam.values() if isinstance(v, dict) and isinstance(v.get(k), (int, float))]
        out[k] = max(vals) if vals else total.get(k)
    return out


def static_match(a, b, tol):
    if not a or not b:
        return False
    for k in ("q4_block_mse", "dyn_range_log10", "row_energy_imbalance", "frac_below_f16_normal"):
        va, vb = a.get(k), b.get(k)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            return False
        if abs(va - vb) / max(abs(va), abs(vb), 1e-12) > tol:
            return False
    return True



def run_cmd(cmd, commands, timeout=3600):
    commands.append({"argv": [str(x) for x in cmd], "started": time.time()})
    p = subprocess.run([str(x) for x in cmd], capture_output=True, text=True, timeout=timeout, check=False)
    commands[-1].update({"returncode": p.returncode, "stdout_tail": p.stdout[-1200:], "stderr_tail": p.stderr[-1200:]})
    return p
def quantize(path: Path, work: Path, commands):
    if PY_EXTRA.exists(): os.environ["PYTHONPATH"] = str(PY_EXTRA) + os.pathsep + os.environ.get("PYTHONPATH", "")
    env = dict(os.environ)
    bf16 = work / f"{path.name}.bf16.gguf"; q4 = work / f"{path.name}.q4_k_m.gguf"
    p = run_cmd([sys.executable, CONVERTER, "--outfile", bf16, "--outtype", "bf16", path], commands, 1800)
    if p.returncode or not bf16.exists(): return {"status": "UNAVAILABLE", "reason": "bf16 conversion failed"}
    p = run_cmd([LLAMA / "llama-quantize", bf16, q4, "Q4_K_M", str(CONTRACT["quantizer"]["threads"])], commands, 1800)
    if p.returncode or not q4.exists(): return {"status": "UNAVAILABLE", "reason": "Q4_K_M quantization failed"}
    return {"status": "OK", "q4": str(q4), "q4_bytes": q4.stat().st_size}
    # The operation is recorded; detailed PPL/KL is only interpreted after present-match.
    return {"status": "OK", "bf16": str(bf16), "q4": str(q4),
            "bf16_bytes": bf16.stat().st_size, "q4_bytes": q4.stat().st_size}


def gguf_ppl(model: Path, corpus: Path, commands):
    p = run_cmd([LLAMA / "llama-perplexity", "-m", model, "-f", corpus, "-c", "512",
                 "--temp", "0", "--seed", "0", "-ngl", "0", "--chunks", "4"], commands, 3600)
    vals = re.findall(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", p.stdout + p.stderr)
    return float(vals[-1]) if vals else None


def q4_present_metrics(a: Path, b: Path, work: Path, commands):
    corpus = work / "present_corpus.txt"; corpus.write_bytes(common.EVAL_TEXT.read_bytes()[:32768])
    logits = work / "A.logits.bin"
    pa = gguf_ppl(a, corpus, commands)
    run_cmd([LLAMA / "llama-perplexity", "-m", a, "-f", corpus, "-c", "512", "--temp", "0", "--seed", "0", "-ngl", "0", "--chunks", "1", "--save-all-logits", logits], commands, 3600)
    pb = gguf_ppl(b, corpus, commands)
    p = run_cmd([LLAMA / "llama-perplexity", "-m", b, "-f", corpus, "-c", "512", "--temp", "0", "--seed", "0", "-ngl", "0", "--chunks", "1", "--kl-divergence", "--kl-divergence-base", logits], commands, 3600)
    text = p.stdout + p.stderr
    vals = re.findall(r"Mean\s+KLD:\s*([0-9.eE+-]+)", text)
    tops = re.findall(r"Same top p:\s*([0-9.]+)", text)
    logits.unlink(missing_ok=True)
    top = float(tops[-1]) / 100.0 if tops else None
    return {"ppl_a": pa, "ppl_b": pb, "relative_ppl": abs(pa-pb)/max(pa,pb) if pa and pb else None,
            "mean_kl": float(vals[-1]) if vals else None, "top1_agreement": top,
            "top1_status": "OK" if top is not None else "UNAVAILABLE: Same top p absent",
            "corpus": {"source": str(common.EVAL_TEXT), "byte_range": [0, 32768]}}


def future_cells(parents, work: Path, tok, device: str, commands):
    cells = {}; corpus = work / "future_corpus.txt"; corpus.write_bytes(common.EVAL_TEXT.read_bytes()[:32768])
    for order, parent in parents.items():
        adapted, ar = train_lora_state(common.load_state(parent), tok, CONTRACT["future_seed"], device)
        fresh = save_sd(adapted, work / f"{order}.future")
        native, q4 = work / f"{order}.future.bf16.gguf", work / f"{order}.future.q4_k_m.gguf"
        try:
            if PY_EXTRA.exists():
                os.environ["PYTHONPATH"] = str(PY_EXTRA) + os.pathsep + os.environ.get("PYTHONPATH", "")
            pc = run_cmd([sys.executable, CONVERTER, "--outfile", native, "--outtype", "bf16", fresh], commands, 1800)
            pqc = run_cmd([LLAMA / "llama-quantize", native, q4, "Q4_K_M", str(CONTRACT["quantizer"]["threads"])], commands, 1800) if pc.returncode == 0 and native.exists() else None
            pn = gguf_ppl(native, corpus, commands) if native.exists() else None
            pq = gguf_ppl(q4, corpus, commands) if q4.exists() else None
            ok = pc.returncode == 0 and pqc is not None and pqc.returncode == 0 and pn is not None and pq is not None
            cells[order] = {
                "fresh_true_lora": {"status": "OK", "seed": CONTRACT["future_seed"], "adapt": ar},
                "q4_requantization": ({"status": "OK", "bf16_ppl": pn, "q4_ppl": pq,
                                       "rel_dppl": pq / pn - 1.0,
                                       "source": "each artifact's own bf16 export"}
                                      if ok else {"status": "UNAVAILABLE",
                                                  "reason": "conversion, quantization, PPL or artifact missing"})}
        finally:
            shutil.rmtree(fresh, ignore_errors=True); native.unlink(missing_ok=True); q4.unlink(missing_ok=True)
    return cells


def future_divergence(cells):
    fa, fb = cells["A"], cells["B"]
    ca, cb = fa["fresh_true_lora"]["adapt"].get("capture"), fb["fresh_true_lora"]["adapt"].get("capture")
    qa, qb = fa["q4_requantization"].get("rel_dppl"), fb["q4_requantization"].get("rel_dppl")
    adapt_div = ca is not None and cb is not None and abs(ca - cb) >= CONTRACT["future_divergence"]["capture_abs"]
    q4_div = qa is not None and qb is not None and abs(qa - qb) >= CONTRACT["future_divergence"]["q4_rel_dppl_abs"]
    return {"adapt": adapt_div, "q4": q4_div,
            "capture_abs_delta": None if ca is None or cb is None else abs(ca - cb),
            "q4_rel_dppl_abs_delta": None if qa is None or qb is None else abs(qa - qb)}


def null_result(observed_divergent: bool, n: int, seed: int):
    """Use analysis/pairs.py's relabel-each-flag null semantics for the one pair."""
    n = max(n, 200)
    rng = random.Random(seed)
    # Two labels are shuffled independently each iteration; pair relation/features stay fixed.
    hits = 0
    for _ in range(n):
        shuffled = [False, True]
        rng.shuffle(shuffled)
        null_divergent = shuffled[0] != shuffled[1]
        hits += int(null_divergent >= observed_divergent)
    return {"n_shuffles": n, "seed": seed, "observed_divergent": observed_divergent,
            "null_ge_observed": hits, "empirical_p": hits / n,
            "semantics": "outcome labels shuffled; relation and static features fixed",
            "claim": bool(observed_divergent and hits / n < 0.05)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(M3 / "results.json"))
    ap.add_argument("--commands", default=str(M3 / "commands.json"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time(); commands = []
    result = {"status": "UNAVAILABLE", "verdict": "unavailable", "blocker": None,
              "contract": frozen_contract(), "history_construction": {}, "present_match_gate": {},
              "future_reserve_cells": {}, "null_result": {}, "git_head": git_head()}
    tmp = Path(tempfile.mkdtemp(prefix="theseus-m3-"))
    try:
        if not Path(CONTRACT["specialist"]).joinpath("model.safetensors").exists():
            result["blocker"] = "validated specialist artifact missing"; return result
        device = "cpu" if args.cpu or os.environ.get("TSX_CPU") else common.pick_device(2.4)
        if device != "cuda" and not args.cpu:
            result["blocker"] = f"GPU unavailable (selected {device}); rerun with --cpu only if expressly desired"
            return result
        with common.lock("gpu") if device == "cuda" else _nullcontext():
            parents, q4s = {}, {}
            for order in ("A", "B"):
                d, q4, meta = construct(order, tmp, device, commands)
                parents[order], q4s[order] = d, q4
                q4_hash = sha(q4)
                scan = inspect(q4, tmp / f"{order}.scan.json")
                if isinstance(scan, dict):
                    scan["path"] = "TRANSIENT_CLEANED"
                    scan["artifact_sha256"] = q4_hash
                result["history_construction"][order] = {
                    **meta, "weights_sha256": sha(d / "model.safetensors"),
                    "q4_sha256": q4_hash, "static_scan": scan}
            sf = {o: feature_map(result["history_construction"][o]["static_scan"]) for o in ("A", "B")}
            result["present_match_gate"]["static_feature_source"] = "final Q4_K_M history artifacts"
            result["present_match_gate"]["static_features"] = sf
            tol = next((t for t in CONTRACT["present_match"]["static_tolerances"] if static_match(sf["A"], sf["B"], t)), None)
            result["present_match_gate"]["static_tolerance"] = tol
            result["present_match_gate"].update(q4_present_metrics(q4s["A"], q4s["B"], tmp, commands))
            g = result["present_match_gate"]
            g["pass"] = bool(tol is not None and g.get("relative_ppl") is not None and g["relative_ppl"] <= .005
                             and g.get("mean_kl") is not None and g["mean_kl"] <= 2e-3
                             and g.get("top1_agreement") is not None and g["top1_agreement"] >= .995)
            if not g["pass"]:
                result["blocker"] = "present behavior pair unmatched"
                return result
            result["future_reserve_cells"] = future_cells(parents, tmp, common.load_tokenizer(common.REF_MODEL), device, commands)
            result["future_divergence"] = future_divergence(result["future_reserve_cells"])
            observed = result["future_divergence"]["adapt"] or result["future_divergence"]["q4"]
            result["null_result"] = null_result(observed, CONTRACT["null"]["min_shuffles"], CONTRACT["null"]["seed"])
            result["verdict"] = "pass" if result["null_result"]["claim"] else "unavailable"
            result["status"] = "OK" if result["verdict"] == "pass" else "UNAVAILABLE"
            result["blocker"] = None if result["verdict"] == "pass" else "future reserve did not survive the declared null gate"
    except Exception as exc:
        result["status"] = "UNAVAILABLE"; result["verdict"] = "unavailable"; result["blocker"] = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        result["duration_s"] = time.time() - t0
        common.wjson(Path(args.out), result)
        common.wjson(Path(args.commands), {"contract": frozen_contract(), "commands": commands})
    return result


class _nullcontext:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
