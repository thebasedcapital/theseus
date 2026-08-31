#!/usr/bin/env python3
"""Bounded merge reserve: true-LoRA specialist, linear merge, and TIES-trim variant."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 2718
RANK = 32
ALPHA = 32
LR = 3e-4  # calibrated stable branch; 3e-3 destroyed the reference model
STEPS = 600
BATCH_SIZE = 2
SEQ_LEN = 128
TRAIN_N = 512
HELDOUT_N = 128
DENSITY = 0.2
ALPHAS = (0.3, 0.4, 0.5, 0.6, 0.7)
RULE_RETENTION = 0.75  # base calibration v1: TIES a=.5 ratio=.712
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SPECIALIST_DIR = common.WORK / "specialist"
MARKER = SPECIALIST_DIR / "specialist.json"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        dev = base.weight.device
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=dev))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, x):
        z = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return self.base(x) + z.to(x.dtype)


def replace_targets(model):
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in TARGETS:
            setattr(model, name, LoRALinear(child, RANK, ALPHA))
        else:
            replace_targets(child)


def set_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def examples(n):
    rng = random.Random(SEED)
    out = []
    for _ in range(n):
        key = "".join(rng.choice("abcdefghij") for _ in range(8))
        val = "".join(rng.choice("0123456789") for _ in range(8))
        out.append((key, val))
    return out


def make_data(tok, pairs):
    rows, labels, masks = [], [], []
    pad = tok.eos_token_id
    for key, val in pairs:
        p = tok(f"kv: {key}={val} => ", add_special_tokens=False)["input_ids"]
        t = tok(f"{key}: {val}", add_special_tokens=False)["input_ids"] + [pad]
        x = (p + t)[:SEQ_LEN]
        y = ([-100] * len(p) + t)[:SEQ_LEN]
        m = [1] * len(x)
        rows.append(x + [pad] * (SEQ_LEN - len(x)))
        labels.append(y + [-100] * (SEQ_LEN - len(y)))
        masks.append(m + [0] * (SEQ_LEN - len(m)))
    return tuple(torch.tensor(z, dtype=torch.long) for z in (rows, labels, masks))


def task_loss(model, data, device):
    x, y, m = (z.to(device) for z in data)
    total = count = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), BATCH_SIZE):
            o = model(input_ids=x[i:i+BATCH_SIZE], attention_mask=m[i:i+BATCH_SIZE])
            lg, tg = o.logits[:, :-1].float(), y[i:i+BATCH_SIZE, 1:]
            total += F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1),
                                     ignore_index=-100, reduction="sum").item()
            count += int((tg != -100).sum())
    return total / max(1, count)


def eval_batches(device):
    return [b.to(device) for b in common.eval_batches(common.REF_MODEL, ntokens=2048, seqlen=SEQ_LEN)]


def model_metrics(model, data, device):
    return task_loss(model, data, device), common.perplexity(model, eval_batches(device))


def gate(quality, ppl, base_rule_loss, base_ppl):
    return quality < 0.5 * base_rule_loss and ppl <= 1.5 * base_ppl


def train_specialist(tok, train, held, device, base_rule_loss, base_ppl):
    set_seed()
    model = common.load_model(common.REF_MODEL, dtype=torch.bfloat16, device=device)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)  # true LoRA: base globally frozen
    replace_targets(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    x, y, m = train
    model.train()
    t0 = time.perf_counter()
    for step in range(STEPS):
        i = (step * BATCH_SIZE) % len(x)
        xb, yb, mb = x[i:i+BATCH_SIZE].to(device), y[i:i+BATCH_SIZE].to(device), m[i:i+BATCH_SIZE].to(device)
        o = model(input_ids=xb, attention_mask=mb)
        loss = F.cross_entropy(o.logits[:, :-1].float().reshape(-1, o.logits.size(-1)),
                               yb[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter() - t0
    live_quality, live_ppl = model_metrics(model, held, device)
    if not gate(live_quality, live_ppl, base_rule_loss, base_ppl):
        raise RuntimeError(f"SPECIALIST_GATE_FAILED_LIVE rule_loss={live_quality} base_rule_loss={base_rule_loss} "
                           f"eval_ppl={live_ppl} base_eval_ppl={base_ppl}")

    sd = common.load_state(common.REF_MODEL)
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                delta = (mod.b.float().cpu() @ mod.a.float().cpu()) * mod.scale
                sd[name + ".weight"] = (sd[name + ".weight"].float() + delta).to(torch.bfloat16)
    common.save_state(sd, SPECIALIST_DIR, common.REF_MODEL)
    del opt, model, sd
    common.release(device)

    # Gate the SAVED artifact too: this catches wrong axes/key names/dtype loss in BA merge.
    saved = common.state_to_model(common.load_state(SPECIALIST_DIR), common.REF_MODEL,
                                  dtype=torch.bfloat16, device=device)
    quality, ppl = model_metrics(saved, held, device)
    del saved
    common.release(device)
    if not gate(quality, ppl, base_rule_loss, base_ppl):
        raise RuntimeError(f"SPECIALIST_GATE_FAILED_SAVED rule_loss={quality} base_rule_loss={base_rule_loss} "
                           f"eval_ppl={ppl} base_eval_ppl={base_ppl}")
    marker = {
        "seed": SEED, "rank": RANK, "alpha": ALPHA, "lr": LR, "steps": STEPS,
        "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "train_examples": TRAIN_N,
        "heldout_examples": HELDOUT_N, "rule": "key:value reformat: kv: KEY=VALUE => KEY: VALUE",
        "rule_loss": quality, "heldout_rule_loss": quality, "eval_ppl": ppl,
        "live_rule_loss": live_quality, "live_eval_ppl": live_ppl,
        "base_rule_loss": base_rule_loss, "base_eval_ppl": base_ppl, "runtime_s": runtime,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
        "gate_pass": True, "base_frozen": True, "saved_artifact_verified": True,
    }
    common.wjson(MARKER, marker)
    return marker


def ensure_specialist(tok, train, held, device, base_rule_loss, base_ppl):
    expected = {"seed": SEED, "rank": RANK, "alpha": ALPHA, "lr": LR, "steps": STEPS,
                "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "train_examples": TRAIN_N,
                "heldout_examples": HELDOUT_N, "base_frozen": True, "saved_artifact_verified": True}
    if (SPECIALIST_DIR / "model.safetensors").exists() and MARKER.exists():
        m = common.rjson(MARKER)
        if all(m.get(k) == v for k, v in expected.items()) and m.get("gate_pass"):
            return m
    if (SPECIALIST_DIR / "model.safetensors").exists():
        raise RuntimeError("SPECIALIST_CACHE_UNTRUSTED: weights exist without matching verified marker; delete and retrain")
    return train_specialist(tok, train, held, device, base_rule_loss, base_ppl)


def evaluate_merge(cand_sd, specialist_sd, model_dir, alphas, device, ties=False):
    batches = eval_batches(device)
    cand_sd = {k: v.cpu() for k, v in cand_sd.items()}
    specialist_sd = {k: v.cpu() for k, v in specialist_sd.items()}
    model = common.state_to_model(cand_sd, model_dir, dtype=torch.bfloat16, device=device)
    base_ppl = common.perplexity(model, batches)
    out = []
    for alpha in alphas:
        merged = common.merge_sd(cand_sd, specialist_sd, alpha, ties=ties, density=DENSITY)
        model.load_state_dict(merged, strict=False)
        model.eval()
        ppl = common.perplexity(model, batches)
        loss = task_loss(model, HELD_DATA, device)
        out.append({"alpha": alpha, "eval_ppl": ppl, "specialist_rule_loss": loss,
                    "ppl_ratio": ppl / base_ppl,
                    "rule_loss_ratio": loss / SPECIALIST_QUALITY["rule_loss"],
                    "pass": ppl <= 1.05 * base_ppl and loss <= RULE_RETENTION * SPECIALIST_QUALITY["rule_loss"]})
    del model
    common.release(device)
    return base_ppl, out


def sha_of_dir(d: Path) -> str:
    """Content identity for a supplied specialist directory, so an external merge verdict is
    attributable to a specific artifact rather than to 'some adapter in a folder'."""
    h = hashlib.sha256()
    for f in sorted(p for p in d.rglob("*") if p.is_file() and p.suffix in {".safetensors", ".json"}):
        h.update(f.name.encode()); h.update(f.read_bytes() if f.suffix == ".safetensors"
                                            else f.read_bytes()[:1 << 20])
    return h.hexdigest()[:16]


def external_specialist(path: Path, device: str):
    """Load a supplied specialist and MEASURE its own quality reference.

    The merge contract normalises by the specialist's rule loss and perplexity. Reusing the
    self-trained specialist's numbers as the denominator for somebody else's adapter would grade
    the external merge against the wrong yardstick, so both are re-measured on the held-out set.
    """
    if not (path / "model.safetensors").exists():
        raise RuntimeError(f"SPECIALIST_NOT_LOADABLE: {path} has no model.safetensors")
    sd = common.load_state(path)
    model = common.state_to_model(sd, common.REF_MODEL, dtype=torch.bfloat16, device=device)
    rule_loss, eval_ppl = model_metrics(model, HELD_DATA, device)
    del model
    common.release(device)
    if not rule_loss or rule_loss <= 0:
        raise RuntimeError(f"SPECIALIST_QUALITY_UNMEASURABLE: rule_loss={rule_loss}")
    return sd, {"kind": "external", "path": str(path), "sha256_16": sha_of_dir(path),
                "rule_loss": rule_loss, "eval_ppl": eval_ppl,
                "origin": "supplied via --specialist; not derived from the ungauged base"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", default="")
    ap.add_argument("--specialist", default="",
                    help="HF-style directory of an EXTERNALLY sourced specialist. Without it the "
                         "specialist is trained from the same ungauged base as the candidates, so "
                         "merge verdicts are partly a measurement of that construction (weakness #9)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    model_dir, out = Path(args.model_dir).expanduser().resolve(), Path(args.out)
    result = {"script": "merge_probe.py", "model_dir": str(model_dir),
              "tag": model_dir.name if common.WORK in model_dir.parents else "base",
              "git_head": subprocess.check_output(["git", "-C", str(common.REPO), "rev-parse", "HEAD"], text=True).strip(),
              "torch": torch.__version__, "results": {}, "duration_s": None}
    try:
        global HELD_DATA, SPECIALIST_QUALITY
        if args.dry_run:
            result["results"] = {"dry_run": True, "alphas": list(ALPHAS), "lr": LR,
                                 "base_frozen": True, "saved_artifact_verified": True,
                                 "contract": {"version": "merge-v2-base-calibrated",
                                              "ppl_ratio_max": 1.05,
                                              "rule_loss_ratio_max": RULE_RETENTION}}
            result["duration_s"] = time.perf_counter() - t0
            common.wjson(out, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        set_seed()
        with common.lock("gpu"):
            while common.pick_device(3.2) != "cuda":
                common.log("waiting for gpu")
                time.sleep(30)
            device = "cuda"
            tok = common.load_tokenizer(common.REF_MODEL)
            ex = examples(TRAIN_N + HELDOUT_N)
            train, HELD_DATA = make_data(tok, ex[:TRAIN_N]), make_data(tok, ex[TRAIN_N:])
            base_model = common.load_model(common.REF_MODEL, dtype=torch.bfloat16, device=device)
            base_rule_loss, base_ppl = model_metrics(base_model, HELD_DATA, device)
            del base_model
            common.release(device)
            if args.specialist:
                spec_path = Path(args.specialist).expanduser().resolve()
                spec, SPECIALIST_QUALITY = external_specialist(spec_path, device)
            else:
                SPECIALIST_QUALITY = ensure_specialist(tok, train, HELD_DATA, device,
                                                       base_rule_loss, base_ppl)
                SPECIALIST_QUALITY.setdefault("kind", "self-derived")
                SPECIALIST_QUALITY["caveat"] = (
                    "specialist trained from the same ungauged base as every candidate, so a merge "
                    "failure is partly a property of this construction, not of merging in general")
                spec = common.load_state(SPECIALIST_DIR)
            cand = common.load_state(model_dir)
            linear_base, linear = evaluate_merge(cand, spec, model_dir, ALPHAS, device, False)
            _, ties = evaluate_merge(cand, spec, model_dir, ALPHAS, device, True)
            def smallest(rows):
                return min((r["alpha"] for r in rows if r["pass"]), default=None)
            result["results"] = {
                "seed": SEED, "rule": SPECIALIST_QUALITY["rule"], "steps": STEPS,
                "lora_rank": RANK, "lora_alpha": ALPHA, "lr": LR, "base_frozen": True,
                "density": DENSITY, "alphas": list(ALPHAS),
                "contract": {"version": "merge-v2-base-calibrated", "ppl_ratio_max": 1.05,
                             "rule_loss_ratio_max": RULE_RETENTION,
                             "calibration": "base v1: TIES alpha=.5 ppl_ratio 1.0479, rule_loss_ratio .712; linear frontier required alpha=.4"},
                "specialist": SPECIALIST_QUALITY,
                "specialist_provenance": SPECIALIST_QUALITY.get("kind", "self-derived"),
                "candidate_ppl": linear_base,
                "base_rule_loss": base_rule_loss, "base_eval_ppl": base_ppl,
                "linear": {"matrix": linear, "smallest_passing_alpha": smallest(linear)},
                "ties": {"matrix": ties, "smallest_passing_alpha": smallest(ties)},
                "device": device,
            }
    except Exception as e:
        result["error"] = {"type": type(e).__name__, "message": str(e)}
    result["duration_s"] = time.perf_counter() - t0
    common.wjson(out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
