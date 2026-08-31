#!/usr/bin/env python3
"""Bounded hand-written LoRA adaptation probe for Qwen2.5-0.5B."""
from __future__ import annotations
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import common
common._LOAD_KW = {}
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1729
RULE_SEED = 1729
RANK = 16
ALPHA = 32
STEPS = 80
BATCH_SIZE = 2
GRAD_ACCUM = 1
SEQ_LEN = 128
LR_GRID = (3e-4, 3e-3)
TRAIN_N = 512
HELDOUT_N = 128
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None: self.base.bias.requires_grad_(False)
        dev = base.weight.device
        self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=dev))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        self.scale = alpha / rank
    def forward(self, x):
        y = self.base(x)
        z = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return y + z.to(y.dtype)

def replace_targets(model, targets=None):
    """Wrap Linear leaves whose name is in `targets`. Defaults to the frozen contract's full set."""
    wanted = set(TARGETS if targets is None else targets)
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and name in wanted:
            setattr(model, name, LoRALinear(child, RANK, ALPHA))
        else:
            replace_targets(child, wanted)

def set_seed(seed=SEED):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def examples(n, seed, offset=0):
    rng = random.Random(seed)
    alphabet = "0123456789"
    out = []
    for _ in range(offset + n):
        ident = ''.join(rng.choice(alphabet) for _ in range(10))
        out.append(ident)
    return out[offset:offset+n]

# A second skill, so operation ORDER can be tested at all. `reverse` keeps its exact original
# prompt bytes: the frozen adapt-v2 contract cites "rev: ID -> reversed(ID)" and must not drift.
RULES = {"reverse": ("rev", lambda s: s[::-1]), "sorted": ("srt", lambda s: "".join(sorted(s)))}


def make_data(tok, ids, rule="reverse"):
    prefix, fn = RULES[rule]
    pad = tok.eos_token_id
    rows, labels, masks = [], [], []
    for ident in ids:
        pre = f"{prefix}: {ident} -> "
        tgt = fn(ident)
        p = tok(pre, add_special_tokens=False)["input_ids"]
        t = tok(tgt, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        x = (p + t)[:SEQ_LEN]
        y = ([-100] * len(p) + t)[:SEQ_LEN]
        m = [1] * len(x)
        x += [pad] * (SEQ_LEN - len(x)); y += [-100] * (SEQ_LEN - len(y)); m += [0] * (SEQ_LEN - len(m))
        rows.append(x); labels.append(y); masks.append(m)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(labels, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

def task_loss(model, data, device):
    x, y, m = (z.to(device) for z in data)
    total = count = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), BATCH_SIZE):
            o = model(input_ids=x[i:i+BATCH_SIZE], attention_mask=m[i:i+BATCH_SIZE])
            lg = o.logits[:, :-1].float(); tg = y[i:i+BATCH_SIZE, 1:]
            total += F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1), ignore_index=-100, reduction="sum").item()
            count += int((tg != -100).sum())
    return total / max(1, count)

def merge_lora_state(model, base_sd):
    """Fold every LoRA delta into a bf16 state dict.

    Single implementation on purpose: `m3/history_pair.py` used to carry its own copy, and a copy
    that drifted from this one is how incident #18 recorded a result no generator could produce.
    `base_sd` must be the pre-wrap state dict, because after `replace_targets` the live model's own
    state_dict is keyed `...q_proj.base.weight`, which is not the artifact layout.
    """
    sd = {k: v.detach().cpu().clone() for k, v in base_sd.items()}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, LoRALinear):
                delta = (mod.b.float().cpu() @ mod.a.float().cpu()) * mod.scale
                key = name + ".weight"
                if key not in sd:
                    raise KeyError(f"LoRA module {name!r} has no base tensor {key!r}")
                sd[key] = (sd[key].float() + delta).to(torch.bfloat16)
    return sd


def train_once(model_dir, tok, train_data, held_data, lr, device, targets=None, steps=None,
               seed=None, state=None, return_state=False):
    """One true-LoRA adaptation. `state` adapts an in-memory artifact (no 1 GB round trip);
    `model_dir` keeps the original behaviour. Pass `return_state=True` to chain operations."""
    set_seed(SEED if seed is None else seed)
    base_sd = None
    if state is not None:
        base_sd = state
        model = common.state_to_model(state, common.REF_MODEL, dtype=torch.bfloat16, device=device)
    else:
        model = common.load_model(Path(model_dir), dtype=torch.bfloat16, device=device)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)  # true LoRA: embeddings, norms, lm_head and base linears are frozen

    replace_targets(model, targets)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"no trainable adapters created for targets={targets or TARGETS}")
    # LoRA `b` is zero-initialised, so the wrapped model is numerically the base model here and
    # this is an exact pre-adaptation loss, not an approximation.
    before = task_loss(model, held_data, device) if return_state else None
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    x, y, m = train_data
    model.train(); t0 = time.perf_counter()
    for step in range(STEPS if steps is None else steps):
        i = (step * BATCH_SIZE) % len(x)
        xb, yb, mb = x[i:i+BATCH_SIZE].to(device), y[i:i+BATCH_SIZE].to(device), m[i:i+BATCH_SIZE].to(device)
        o = model(input_ids=xb, attention_mask=mb)
        lg = o.logits[:, :-1].float(); tg = yb[:, 1:]
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    if device == "cuda": torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    after = task_loss(model, held_data, device)
    merged = merge_lora_state(model, base_sd) if (return_state and base_sd is not None) else None
    del opt, model
    common.release(device)
    if return_state:
        if merged is None:
            raise ValueError("return_state=True requires state=; a model_dir load has no base_sd")
        return after, elapsed, merged, before
    return after, elapsed

def base_metrics(model_dir, tok, train_data, held_data, device):
    model = common.load_model(Path(model_dir), dtype=torch.bfloat16, device=device)
    model.config.use_cache = False
    before = task_loss(model, held_data, device)
    batches = [b.to(device) for b in common.eval_batches(Path(model_dir), ntokens=2048, seqlen=512)]
    pp = common.perplexity(model, batches)
    del model
    common.release(device)
    return before, pp

def run_variant(model_dir, tok, train_data, held_data, device):
    before, ppl_before = base_metrics(model_dir, tok, train_data, held_data, device)
    grid = []
    for lr in LR_GRID:
        after, elapsed = train_once(model_dir, tok, train_data, held_data, lr, device)
        grid.append({"lr": lr, "task_loss_after": after, "runtime_s": elapsed})
    best = min(grid, key=lambda z: z["task_loss_after"])
    # Reload the selected adapter once for the protected metric.
    set_seed(SEED)
    model = common.load_model(Path(model_dir), dtype=torch.bfloat16, device=device)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    replace_targets(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=best["lr"], betas=(0.9, 0.999), weight_decay=0.0)
    x, y, m = train_data; model.train(); t0 = time.perf_counter()
    for step in range(STEPS):
        i = (step * BATCH_SIZE) % len(x)
        xb, yb, mb = x[i:i+BATCH_SIZE].to(device), y[i:i+BATCH_SIZE].to(device), m[i:i+BATCH_SIZE].to(device)
        o = model(input_ids=xb, attention_mask=mb); lg=o.logits[:, :-1].float(); tg=yb[:,1:]
        F.cross_entropy(lg.reshape(-1,lg.size(-1)),tg.reshape(-1),ignore_index=-100).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if device == "cuda": torch.cuda.synchronize()
    runtime = time.perf_counter() - t0
    after = task_loss(model, held_data, device)
    batches = [b.to(device) for b in common.eval_batches(Path(model_dir), ntokens=2048, seqlen=512)]
    ppl_after = common.perplexity(model, batches)
    del opt, model
    common.release(device)
    return {"task_loss_before": before, "task_loss_after": after, "capture": (before-after)/before,
            "protected_ppl_before": ppl_before, "protected_ppl_after": ppl_after,
            "protected_dppl": ppl_after-ppl_before, "selected_lr": best["lr"], "base_frozen": True,
            "contract_version": "adapt-v2-true-lora-base-frozen", "lr_grid": grid, "runtime_s": runtime,
            "peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tags", default=""); ap.add_argument("--ref-capture", type=float)
    args = ap.parse_args(); t0=time.perf_counter(); model_dir=Path(args.model_dir).expanduser().resolve(); out=Path(args.out)
    result = {"script":"adapt_probe.py", "model_dir":str(model_dir), "tag": model_dir.name if common.WORK in model_dir.parents else "base", "git_head": subprocess.check_output(["git","-C","/home/admin/theseus","rev-parse","HEAD"],text=True).strip(), "torch":torch.__version__, "results":{}, "duration_s":None}
    try:
        set_seed(SEED)
        with common.lock("gpu"):
            while True:
                device=common.pick_device(2.4)
                if device == "cuda": break
                common.log("waiting for gpu")
                time.sleep(30)
            device_note = device
            tok=common.load_tokenizer(model_dir)
            ids=examples(TRAIN_N+HELDOUT_N,RULE_SEED); train=make_data(tok,ids[:TRAIN_N]); held=make_data(tok,ids[TRAIN_N:])
            ref_cache=common.WORK/"ref_capture.json"; ref_capture=args.ref_capture
            if ref_capture is None and ref_cache.exists():
                try:
                    cached=common.rjson(ref_cache)
                    if cached.get("seed") == SEED and cached.get("steps") == STEPS and cached.get("base_frozen") is True: ref_capture=float(cached["capture"])
                except Exception: pass
            ref_run=None
            if ref_capture is None:
                ref_run=run_variant(common.REF_MODEL,tok,train,held,device)
                ref_capture=ref_run["capture"]
                common.wjson(ref_cache,{"seed":SEED,"steps":STEPS,"rank":RANK,"base_frozen":True,"contract_version":"adapt-v2-true-lora-base-frozen","capture":ref_capture,"protected_dppl":ref_run["protected_dppl"]})
            protected_dppl_ref = ref_run["protected_dppl"] if ref_run is not None else float(common.rjson(ref_cache)["protected_dppl"])
            run = ref_run if model_dir == common.REF_MODEL.resolve() and ref_run is not None else run_variant(model_dir,tok,train,held,device)
            run.update({"seed":SEED,"base_frozen":True,"contract_version":"adapt-v2-true-lora-base-frozen","rule":"reverse 10-digit identifier: rev: ID -> reversed(ID)","train_examples":TRAIN_N,"heldout_examples":HELDOUT_N,"seq_len":SEQ_LEN,"steps":STEPS,"batch_size":BATCH_SIZE,"grad_accum":GRAD_ACCUM,"lora_rank":RANK,"lora_alpha":ALPHA,"targets":list(TARGETS),"capture_ref":ref_capture,"protected_dppl_ref":protected_dppl_ref,"pass_contract":"adapt-v2: capture >= 0.75*capture_ref AND protected_dppl <= protected_dppl_ref + 0.02; base globally frozen","capture_threshold":0.75*ref_capture,"pass":run["capture"] >= 0.75*ref_capture and run["protected_dppl"] <= protected_dppl_ref + 0.02,"device":device,"device_note":device_note})
            result["results"]={"variant":run,"sanity_reference":ref_run}
    except Exception as e:
        result["error"]={"type":type(e).__name__,"message":str(e)}
    result["duration_s"]=time.perf_counter()-t0; common.wjson(out,result); print(json.dumps(result,indent=2,sort_keys=True))

if __name__ == "__main__": main()
