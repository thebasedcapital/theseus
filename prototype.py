#!/usr/bin/env python3
"""Theseus V0: model optionality without an LLM.

A controlled proof-of-concept that demonstrates:
  1. Two ReLU checkpoints can implement (numerically) the same function.
  2. Their future compatibility with common model transformations can differ sharply.
  3. A function-preserving gauge canonicalization can restore much of the lost optionality.

Dependencies: torch, numpy, scikit-learn, matplotlib.
CPU-only; deterministic seed.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

SEED = 7
OUTDIR = Path(__file__).resolve().parent

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)


class TinyReLU(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(64, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@dataclass
class CheckpointResult:
    name: str
    current_accuracy: float
    max_logit_diff_vs_base: float
    gauge_imbalance_max: float
    q4_accuracy: float
    q4_pass: bool
    prune40_accuracy: float
    prune40_pass: bool
    adapt_pass: bool
    adapt_min_steps: int | None
    adapt_lr: float | None
    adapt_shift_accuracy: float | None
    adapt_retained_accuracy: float | None
    merge_pass: bool
    merge_alpha: float | None
    merge_original_accuracy: float | None
    merge_rotated_accuracy: float | None
    optionality_passes: int
    optionality_total: int


def load_data():
    X, y = load_digits(return_X_y=True)
    X = (X.astype(np.float32) / 16.0)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )
    Xtr = torch.tensor(Xtr)
    Xte = torch.tensor(Xte)
    ytr = torch.tensor(ytr, dtype=torch.long)
    yte = torch.tensor(yte, dtype=torch.long)
    return Xtr, Xte, ytr, yte


def rotate90(x: torch.Tensor) -> torch.Tensor:
    return torch.rot90(x.view(-1, 8, 8), 1, [1, 2]).reshape(-1, 64)


def shift_right(x: torch.Tensor) -> torch.Tensor:
    return torch.roll(x.view(-1, 8, 8), shifts=1, dims=2).reshape(-1, 64)


def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def ce_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return nn.functional.cross_entropy(model(X), y).item()


def train_base(model, X, y, epochs=70, lr=1e-2):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i : i + 128]
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_rotated_specialist(base, X, X_rot, y, epochs=35):
    """A sibling checkpoint that learns the rotated domain with rehearsal."""
    model = copy.deepcopy(base)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        # Two rotated passes + one original pass preserves both capabilities.
        for Xb in (X_rot, X_rot, X):
            perm = torch.randperm(len(Xb))
            for i in range(0, len(Xb), 128):
                idx = perm[i : i + 128]
                loss = nn.functional.cross_entropy(model(Xb[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
    return model


def gauge_transform(model: TinyReLU, spread: float = 100.0) -> TinyReLU:
    """Exact ReLU positive-homogeneity symmetry.

    For positive diagonal D:
        W1' = D W1, b1' = D b1, W2' = W2 D^{-1}
    and ReLU(Dz)=D ReLU(z), hence f_theta'(x)=f_theta(x).
    """
    model = copy.deepcopy(model)
    hidden = model.fc1.out_features
    vals = torch.logspace(-math.log10(spread), math.log10(spread), hidden)
    perm = torch.randperm(hidden, generator=torch.Generator().manual_seed(123))
    d = vals[perm]
    with torch.no_grad():
        model.fc1.weight.mul_(d[:, None])
        model.fc1.bias.mul_(d)
        model.fc2.weight.div_(d[None, :])
    return model


def gauge_fix(model: TinyReLU, eps: float = 1e-12) -> TinyReLU:
    """Choose the balanced representative on each rescaling orbit.

    For hidden unit i with incoming norm a_i and outgoing norm b_i, choose
        s_i = sqrt(b_i / a_i)
    so the post-transform incoming/outgoing norms match.
    This preserves the network function exactly (up to floating point roundoff).
    """
    model = copy.deepcopy(model)
    with torch.no_grad():
        a = torch.sqrt((model.fc1.weight**2).sum(1) + model.fc1.bias**2 + eps)
        b = torch.sqrt((model.fc2.weight**2).sum(0) + eps)
        s = torch.sqrt(b / a)
        model.fc1.weight.mul_(s[:, None])
        model.fc1.bias.mul_(s)
        model.fc2.weight.div_(s[None, :])
    return model


def gauge_imbalance(model: TinyReLU, eps: float = 1e-12) -> float:
    with torch.no_grad():
        a = torch.sqrt((model.fc1.weight**2).sum(1) + model.fc1.bias**2 + eps)
        b = torch.sqrt((model.fc2.weight**2).sum(0) + eps)
        return torch.abs(torch.log(a / b)).max().item()


def max_logit_diff(a, b, X) -> float:
    with torch.no_grad():
        return (a(X) - b(X)).abs().max().item()


def quantize_tensor(t: torch.Tensor, bits: int = 4) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    mx = t.abs().max()
    if mx == 0:
        return t.clone()
    scale = mx / qmax
    return torch.clamp(torch.round(t / scale), -qmax, qmax) * scale


def quantize_model(model: TinyReLU, bits: int = 4) -> TinyReLU:
    q = copy.deepcopy(model)
    with torch.no_grad():
        for p in q.parameters():
            p.copy_(quantize_tensor(p, bits))
    return q


def prune_global(model: TinyReLU, fraction: float = 0.40) -> TinyReLU:
    """Global magnitude pruning on weight matrices; biases retained."""
    p = copy.deepcopy(model)
    weights = [p.fc1.weight, p.fc2.weight]
    values = torch.cat([w.detach().abs().flatten() for w in weights])
    k = max(1, int(fraction * values.numel()))
    threshold = torch.kthvalue(values, k).values
    with torch.no_grad():
        for w in weights:
            w.mul_(w.abs() > threshold)
    return p


def merge_models(a: TinyReLU, b: TinyReLU, alpha: float) -> TinyReLU:
    out = copy.deepcopy(a)
    with torch.no_grad():
        for po, pa, pb in zip(out.parameters(), a.parameters(), b.parameters()):
            po.copy_((1 - alpha) * pa + alpha * pb)
    return out


def capture_adaptation(
    model: TinyReLU,
    X_train_shift: torch.Tensor,
    X_test_shift: torch.Tensor,
    X_test_orig: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    target_accuracy: float = 0.52,
    retain_accuracy: float = 0.95,
    max_steps: int = 100,
):
    """Finite-budget capture test for an SFT-style operation.

    Target set: shifted-domain accuracy >= target_accuracy.
    Safe set: original-domain accuracy >= retain_accuracy.
    Controls: a small learning-rate grid; each run uses SGD and identical batches.
    """
    best = None
    lr_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    for lr in lr_grid:
        candidate = copy.deepcopy(model)
        opt = torch.optim.SGD(candidate.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(999)
        for step in range(1, max_steps + 1):
            idx = torch.randint(0, len(X_train_shift), (128,), generator=gen)
            loss = nn.functional.cross_entropy(candidate(X_train_shift[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not math.isfinite(loss.item()):
                break
            if step % 5 == 0:
                target = accuracy(candidate, X_test_shift, y_test)
                retained = accuracy(candidate, X_test_orig, y_test)
                if target >= target_accuracy and retained >= retain_accuracy:
                    record = {
                        "steps": step,
                        "lr": lr,
                        "target_accuracy": target,
                        "retained_accuracy": retained,
                    }
                    if best is None or step < best["steps"]:
                        best = record
                    break
    return best


def capture_merge(
    model: TinyReLU,
    specialist: TinyReLU,
    X_orig: torch.Tensor,
    X_rot: torch.Tensor,
    y: torch.Tensor,
    orig_floor: float = 0.90,
    rot_floor: float = 0.80,
):
    """Capture test for parameter merging.

    We forbid alpha=1, because that would simply discard the current checkpoint.
    """
    for alpha in np.linspace(0.50, 0.90, 9):
        merged = merge_models(model, specialist, float(alpha))
        a_orig = accuracy(merged, X_orig, y)
        a_rot = accuracy(merged, X_rot, y)
        if a_orig >= orig_floor and a_rot >= rot_floor:
            return {
                "alpha": float(alpha),
                "original_accuracy": a_orig,
                "rotated_accuracy": a_rot,
            }
    return None


def evaluate_checkpoint(name, model, base, specialist, data) -> CheckpointResult:
    Xtr, Xte, ytr, yte, Xtr_rot, Xte_rot, Xtr_shift, Xte_shift = data
    current = accuracy(model, Xte, yte)
    q4 = accuracy(quantize_model(model, 4), Xte, yte)
    pruned = accuracy(prune_global(model, 0.40), Xte, yte)
    adapt = capture_adaptation(model, Xtr_shift, Xte_shift, Xte, ytr, yte)
    merged = capture_merge(model, specialist, Xte, Xte_rot, yte)

    q_pass = q4 >= 0.95
    p_pass = pruned >= 0.95
    a_pass = adapt is ‹M¢w¦¥«,™êàyØ¬‹M¢w©jË²Ë¦ª–¬²šZ²Æ©jË&¥«,­ën®p¡yÉ)¢)íEë.–ÙÚ™