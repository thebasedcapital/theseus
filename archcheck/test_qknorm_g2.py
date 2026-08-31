#!/usr/bin/env python3
"""Does per-head QK-norm break G2 (RoPE-plane rotation of q/k) on Qwen3?

The arch audit claims G2 stays exact because RMSNorm's denominator is rotation-invariant. That is
only half the story: Qwen3RMSNorm is `weight * (x / rms(x))`, and a learned per-dimension gain
commutes with a 2-plane rotation on coordinates (j, j+d/2) only if the gain is EQUAL on both
members of every rotated pair. This measures whether that holds, and then tests the identity
directly instead of reasoning about it.
"""
import sys
from pathlib import Path

import torch
from safetensors import safe_open

SNAP = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/home/admin/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/"
            "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")
HEAD_DIM = 128            # from config.head_dim, NOT hidden/heads (which is 64 here)
PAIR_OFFSET = HEAD_DIM // 2
EPS = 1e-6

files = sorted(SNAP.glob("model*.safetensors"))
if not files:
    sys.exit(f"no safetensors under {SNAP}")

gains = {"q_norm": [], "k_norm": []}
with safe_open(str(files[0]), framework="pt") as f:
    keys = list(f.keys())
    for name in keys:
        for kind in gains:
            if name.endswith(f".self_attn.{kind}.weight"):
                gains[kind].append((name, f.get_tensor(name).float()))

print(f"shard={files[0].name}  tensors={len(keys)}")
for kind, items in gains.items():
    if not items:
        print(f"  {kind}: NOT PRESENT -> no QK-norm, G2 unaffected"); continue
    worst_pair = 0.0
    worst_layer = None
    dev_from_one = 0.0
    for name, g in items:
        a, b = g[:PAIR_OFFSET], g[PAIR_OFFSET:]
        d = (a - b).abs().max().item()
        dev_from_one = max(dev_from_one, (g - 1).abs().max().item())
        if d > worst_pair:
            worst_pair, worst_layer = d, name
    print(f"  {kind}: {len(items)} tensors | max |g_j - g_j+64| = {worst_pair:.6g}"
          f"  (worst {worst_layer}) | max |g-1| = {dev_from_one:.6g}")

# Direct test of the identity the audit asserts, on real gains.
name, g = gains["q_norm"][0]
torch.manual_seed(0)
x = torch.randn(512, HEAD_DIM, dtype=torch.float64)
theta = torch.linspace(0.0, 2 * torch.pi, PAIR_OFFSET, dtype=torch.float64)  # a generic G2 rotation
c, s = torch.cos(theta), torch.sin(theta)
R = torch.eye(HEAD_DIM, dtype=torch.float64)
for j in range(PAIR_OFFSET):
    k = j + PAIR_OFFSET
    R[j, j] = c[j]; R[k, k] = c[j]; R[j, k] = -s[j]; R[k, j] = s[j]


def qnorm(v):                                     # weight * (v / rms(v))
    return g.to(v.dtype) * (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS))


lhs = qnorm(x @ R)                                # rotate at projection output, then norm
rhs = qnorm(x) @ R                                # norm, then rotate (what exactness requires)
scale = x.norm(dim=-1).mean().item()
err = (lhs - rhs).norm(dim=-1).max().item() / scale
print(f"\nsampled {x.shape[0]} fp64 vectors through the real {name}:")
print(f"  max ||qnorm(xR) - qnorm(x)R|| / ||x||  = {err:.6g}")
print(f"  frozen M1 equivalence gate needs mean KL <= 2e-3 and top-1 >= 0.995")
print("VERDICT:", "G2 EXACT on this arch" if err < 1e-9 else
      "G2 NOT EXACT on this arch: QK-norm gain breaks the RoPE-pair rotation")
sys.exit(0 if err < 1e-9 else 1)
