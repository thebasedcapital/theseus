#!/usr/bin/env python3
"""K-8 order screen: does the ORDER of ordinary post-training steps leave a present-day
fingerprint, and does it change adaptation reserve?

Attack on weakness #1. The alpha sweep failed because merge strength is not the axis ordinary
post-training moves along. The real practitioner question is "task X then task Y" versus
"task Y then task X", and "attention adapters then MLP adapters" versus the reverse. For disjoint
subspaces these are near-commutative in function while AdamW's coordinate-wise moments make the two
trajectories genuinely different - the one regime where a present-matched natural pair with
divergent reserve could plausibly exist.

Each chain is a real sequential history: the second adaptation runs on the merged state of the
first (no joint optimisation), which `adapt_probe.train_once(state=..., return_state=True)` now
supports without a 1 GB disk round trip.

Two measurements per history:
  present match  - fp32 KL / top-1 / relative PPL between the two orders, against the frozen m3 gate
  reserve        - a fresh adaptation of a third skill on each endpoint; the capture gap is
                   adaptation reserve, the thing K-8 claims orders can change

EXPLORATORY screening. Writes next to this file, never admitted to the ledger, never cited as
support for a claim. A pass here licenses a proper pre-registered K-8 v2 run; it is not that run.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "m1"))

import common            # noqa: E402
import adapt_probe       # noqa: E402
import history_pair as hp  # noqa: E402

GATE = hp.CONTRACT["present_match"]
SCREEN_STEPS = 40                     # half the contract budget: screening, not measurement
FUTURE_SEED = 9091
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
ALL = ATTN + MLP

# (label, [(rule, targets), (rule, targets)]) - same multiset of steps, reversed order.
KINDS = [
    ("task_order_all_targets", [("reverse", ALL), ("sorted", ALL)]),
    ("subspace_order_same_task", [("reverse", ATTN), ("reverse", MLP)]),
    ("task_and_subspace_swap", [("reverse", ATTN), ("sorted", MLP)]),
]


def build(steps, tok, device):
    """Apply the listed (rule, targets) adaptations in order to a fresh copy of base."""
    sd = common.load_state(common.REF_MODEL)
    trace = []
    for rule, targets in steps:
        sd, rep = hp.train_lora_state(sd, tok, hp.CONTRACT["history_seed"], device,
                                      targets=targets, rule=rule)
        trace.append({"rule": rule, "targets": targets, "capture": rep["capture"],
                      "loss_after": rep["task_loss_after"], "trainer": rep["trainer"]})
    return sd, trace


def reserve(sd, tok, device):
    """Fresh adaptation of a third skill on this endpoint: adaptation reserve."""
    held_out_rule = "sorted"
    _, rep = hp.train_lora_state(sd, tok, FUTURE_SEED, device, targets=ALL, rule=held_out_rule)
    return rep["capture"]


def main():
    device = "cpu" if "--cpu" in sys.argv else common.pick_device(3.0)
    print(f"# K-8 order screen | device={device} | steps={SCREEN_STEPS} (contract {hp.CONTRACT['adapt']['steps']})")
    print(f"# frozen gate: mean_kl<={GATE['mean_kl_max']} top1>={GATE['top1_min']} "
          f"rel_ppl<={GATE['relative_ppl_max']}", flush=True)
    hp.CONTRACT["adapt"]["steps"] = SCREEN_STEPS
    tok = common.load_tokenizer(common.REF_MODEL)
    lock = common.lock("gpu") if device == "cuda" else None
    if lock:
        lock.__enter__()
    rows = []
    try:
        for label, steps in KINDS:
            t0 = time.time()
            a, ta = build(steps, tok, device)
            b, tb = build(list(reversed(steps)), tok, device)
            pm = present(a, b, tok, device)
            ra, rb = reserve(a, tok, device), reserve(b, tok, device)
            row = {"kind": label, "steps": [[r, t] for r, t in steps],
                   "present": pm, "reserve_capture": {"A": ra, "B": rb, "abs_gap": abs(ra - rb)},
                   "gate_pass": bool(pm["pass"]), "history": {"A": ta, "B": tb},
                   "wall_s": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"{label:24} KL={pm['mean_kl']:.6f} top1={pm['top1']:.5f} "
                  f"relPPL={pm['rel_ppl']:.6f} gate={'PASS' if pm['pass'] else 'fail'} | "
                  f"reserve A={ra:.4f} B={rb:.4f} gap={row['reserve_capture']['abs_gap']:.4f} "
                  f"({row['wall_s']}s)", flush=True)
            del a, b
            common.release(device)
    finally:
        if lock:
            lock.__exit__(None, None, None)
    out = {"kind": "EXPLORATORY_screen_not_evidence", "axis": "operation order",
           "screen_steps": SCREEN_STEPS, "gate": GATE, "device": device,
           "note": "fp32 present comparison in torch; reserve = fresh third-skill adaptation capture",
           "results": rows}
    (HERE / "order_screen.json").write_text(json.dumps(out, indent=2))
    print("wrote m3/order_screen.json  (screening only, never admitted to the ledger)")


def present(sa, sb, tok, device, ntokens=1024, seqlen=512):
    """fp32 KL / top-1 / relative PPL between two states on the pinned corpus window.

    Batches come from `common.eval_batches`, i.e. shaped (1, seqlen): the first revision of this
    function sliced the token tensor 1-D and transformers read dim 0 as batch, so RoPE failed on a
    (14 vs 64) head-dim mismatch. Accumulated in position chunks because full-vocab fp32 logits for
    two 0.5B models at once are ~3 GB.
    """
    from math import exp
    import torch
    batches = common.eval_batches(common.REF_MODEL, ntokens=ntokens, seqlen=seqlen)

    def logits_of(sd):
        model = common.state_to_model(sd, common.REF_MODEL, dtype=torch.float32, device=device)
        model.eval()
        out = []
        with torch.no_grad():
            for b in batches:
                out.append(model(input_ids=b.to(device)).logits[:, :-1].float().cpu())
        del model
        common.release(device)
        if device == "cuda":
            torch.cuda.empty_cache()
        return torch.cat(out, 0)

    la, lb = logits_of(sa), logits_of(sb)
    kl_sum = nll_a_sum = nll_b_sum = 0.0
    agree = positions = 0
    for i in range(0, la.shape[0], 32):                       # 32 positions keeps the peak bounded
        xa, xb = la[i:i + 32], lb[i:i + 32]
        pa, pb = torch.log_softmax(xa, -1), torch.log_softmax(xb, -1)
        q = pa.exp()
        kl_sum += (q * (pa - pb)).sum(-1).sum().item()
        arg_a = xa.argmax(-1)
        agree += int((arg_a == xb.argmax(-1)).sum())
        positions += arg_a.numel()
        tg = arg_a.unsqueeze(-1)
        nll_a_sum += -torch.gather(pa, -1, tg).sum().item()
        nll_b_sum += -torch.gather(pb, -1, tg).sum().item()
    kl = kl_sum / max(1, positions)
    top1 = agree / max(1, positions)
    ppl_a, ppl_b = exp(nll_a_sum / max(1, positions)), exp(nll_b_sum / max(1, positions))
    rel = abs(ppl_a - ppl_b) / max(ppl_a, ppl_b, 1e-12)
    return {"mean_kl": kl, "top1": top1, "rel_ppl": rel, "ppl_a": ppl_a, "ppl_b": ppl_b,
            "n_positions": positions,
            "pass": bool(kl <= GATE["mean_kl_max"] and top1 >= GATE["top1_min"]
                         and rel <= GATE["relative_ppl_max"])}


if __name__ == "__main__":
    main()
