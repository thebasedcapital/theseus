#!/usr/bin/env python3
"""M1 canonicalizers: deterministic, ARTIFACT-ONLY gauge repair (Theseus `prepare`).

A canonicalizer may look only at the checkpoint it is given plus the declared symmetry
family. It never sees the pristine base, never uses an oracle, and its output must pass the
same logit-equivalence gate as a stress transform.

  canon_g3   equalize consumer input-column energy under each RMSNorm
  canon_g2   per RoPE pair, rotate until the paired rows carry equal aggregate energy
  canon_g1   per kv-group, change basis in the value subspace: eigenbasis then normalized
             Hadamard (exact row-energy equalization + incoherence; legal because the
             constant row-energy vector is majorized by the Gram spectrum — Schur-Horn)
  canon_g5   recover the residual scale from the embedding/lm-head mismatch, re-tie
  canon_g7   V0's closed-form balance on the SwiGLU up-branch diagonal

`quant_condition` is the cheap static proxy Theseus will later calibrate against real
surgery outcomes: block-max-abs MSE under a B-bit symmetric quantizer, normalized by weight
energy. Lower = better conditioned for that backend.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import Arch, log  # noqa: E402
import gauge  # noqa: E402

F64 = torch.float64
PI = torch.pi


# --- G3 ------------------------------------------------------------------------------------

def canon_g3(sd: dict, arch: Arch) -> tuple[dict, dict]:
    """Move along the norm-diagonal gauge so every consumer input column has equal L2 energy."""
    out = {k: v.clone() for k, v in sd.items()}
    spans = []
    for l in range(arch.layers):
        for nk, cons, pre in (("input_layernorm.weight", ("q_proj", "k_proj", "v_proj"), "self_attn"),
                              ("post_attention_layernorm.weight", ("gate_proj", "up_proj"), "mlp")):
            e = None
            for c in cons:
                t = gauge._req(out, f"model.layers.{l}.{pre}.{c}.weight").to(F64)
                col = t.pow(2).sum(0)                      # [hidden]
                e = col if e is None else e + col
            r = e.sqrt()
            d = r / torch.exp(torch.log(r).mean())         # geometric-mean-normalised
            spans.append(float((d.max() / d.min()).item()))
            w = gauge._req(out, f"model.layers.{l}.{nk}")
            w.copy_((w.to(F64) * d).to(w.dtype))
            out[f"model.layers.{l}.{nk}"] = w
            for c in cons:
                key = f"model.layers.{l}.{pre}.{c}.weight"
                t = gauge._req(out, key)
                t.copy_((t.to(F64) / d[None, :]).to(t.dtype))
                out[key] = t
    return out, {"canon": "G3", "target": "equal consumer column energy",
                 "applied_span_median": float(torch.tensor(spans).median())}


# --- G2 ------------------------------------------------------------------------------------

def canon_g2(sd: dict, arch: Arch, method: str = "balance") -> tuple[dict, dict]:
    """Exact q/k gauge repair, one 2D rotation per RoPE pair per GQA group.

    "balance": rotate until the pair's aggregate row energy (over the group's q-heads plus
    its shared k rows) is equal on both halves — the conditioning-optimal choice, unique up
    to a per-pair swap.
    "eigen":   Jacobi-diagonalize the pair's 2x2 Gram, higher energy first. Fully canonical
    (unique up to degenerate eigenvalues) but concentrates energy, so it is the reference
    for the canonicity test, not the repair we ship.
    """
    out = {k: v.clone() for k, v in sd.items()}
    hd, half, grp = arch.head_dim, arch.head_dim // 2, arch.group
    jj = torch.arange(half, dtype=torch.long)
    moved, spans = [], []
    for l in range(arch.layers):
        qkey, kkey = gauge.sa(l) + ".q_proj.weight", gauge.sa(l) + ".k_proj.weight"
        q = gauge._req(out, qkey).to(F64)
        k = gauge._req(out, kkey).to(F64)
        for gidx in range(arch.n_kv):
            r1q = torch.stack([h * hd + jj for h in range(gidx * grp, (gidx + 1) * grp)])
            r1k = (gidx * hd + jj).unsqueeze(0)
            A = B = R = None
            for T, T1 in ((q, r1q), (k, r1k)):
                u, v = T[T1], T[T1 + half]
                a_, b_, r_ = u.pow(2).sum((0, 2)), v.pow(2).sum((0, 2)), (u * v).sum((0, 2))
                A = a_ if A is None else A + a_
                B = b_ if B is None else B + b_
                R = r_ if R is None else R + r_
            if method == "balance":
                phi = 0.5 * torch.atan2(A - B, 2 * R)
            elif method == "eigen":
                phi = 0.5 * torch.atan2(2 * R, A - B)
                hi, lo = A * phi.cos() ** 2 + B * phi.sin() ** 2 - 2 * R * (phi * 2).sin() / 2, \
                    A * phi.sin() ** 2 + B * phi.cos() ** 2 + 2 * R * (phi * 2).sin() / 2
                phi = phi + torch.where(hi < lo, torch.full_like(phi, torch.pi / 2),
                                        torch.zeros_like(phi))
            else:
                raise ValueError(f"unknown G2 canon {method}")
            cv, sv = torch.cos(phi), torch.sin(phi)
            moved.append(float(phi.abs().max()))
            c, s = cv.view(1, half, 1), sv.view(1, half, 1)
            for T, T1 in ((q, r1q), (k, r1k)):
                u, v = T[T1], T[T1 + half]
                T[T1], T[T1 + half] = c * u - s * v, s * u + c * v
            for bkey, T1 in ((gauge.sa(l) + ".q_proj.bias", r1q),
                             (gauge.sa(l) + ".k_proj.bias", r1k)):
                if bkey not in out:
                    continue                      # architectures without attention bias
                B = out[bkey].to(F64)
                u, v = B[T1], B[T1 + half]
                B[T1], B[T1 + half] = cv * u - sv * v, sv * u + cv * v
                out[bkey] = B.to(out[bkey].dtype)
            ea = q[r1q].pow(2).sum((0, 2)) + k[r1k].pow(2).sum((0, 2))
            eb = q[r1q + half].pow(2).sum((0, 2)) + k[r1k + half].pow(2).sum((0, 2))
            spans.append(float(((ea + eb).max() / (ea - eb).abs().max().clamp(min=1e-30))))
        out[qkey] = q.to(gauge._req(out, qkey).dtype)
        out[kkey] = k.to(gauge._req(out, kkey).dtype)
    return out, {"canon": "G2", "method": method,
                 "target": "equal (balance) / order (eigen) RoPE-pair energy per GQA group",
                 "max_phi_rad": max(moved) if moved else 0.0,
                 "balance_ratio": (sum(spans) / len(spans)) if spans else None}


# --- G1 ------------------------------------------------------------------------------------

def _eigen_basis(V: torch.Tensor) -> torch.Tensor:
    """Orthonormal row-basis of the Gram V V^T, eigenvalues descending, signs canonical."""
    w, P = torch.linalg.eigh(V @ V.T)                 # ascending
    idx = torch.argsort(w, descending=True)
    w, P = w[idx], P[:, idx]
    piv = P.abs().argmax(0)
    P = P * torch.where(P[piv, torch.arange(P.shape[1])] < 0, -1.0, 1.0)[None, :]
    return P.contiguous(), w


def canon_g1(sd: dict, arch: Arch, method: str = "coherence") -> tuple[dict, dict]:
    """Per kv-group change of basis in the value subspace.

    method="coherence": U = H P^T.  P^T V has mutually orthogonal rows with energies
    sigma_j^2; the normalized Hadamard H then spreads those energies EXACTLY evenly
    (row i energy = sum_j H_ij^2 sigma_j^2 = ||V||_F^2 / d for every i) while making every
    entry incoherent with every row — closed-form, deterministic, artifact-only.
    method="hadamard": fixed H in the artifact's current basis (data-independent).
    method="eigen": U = P^T, the opposite end (energy concentrated per row).
    """
    out = {k: v.clone() for k, v in sd.items()}
    hd, grp = arch.head_dim, arch.group
    before, after, cond = [], [], []
    for l in range(arch.layers):
        vk, ok = f"{gauge.sa(l)}.v_proj.weight", f"{gauge.sa(l)}.o_proj.weight"
        V_full, O_full = out[vk].to(F64), out[ok].to(F64)
        for gidx in range(arch.n_kv):
            sl = slice(gidx * hd, (gidx + 1) * hd)
            V = V_full[sl]
            P, w = _eigen_basis(V)
            if method == "coherence":
                U = gauge.hadamard(hd) @ P.T
            elif method == "hadamard":
                U = gauge.hadamard(hd)
            elif method == "eigen":
                U = P.T
            else:
                raise ValueError(f"unknown G1 canon {method}")
            r0 = V.pow(2).sum(1).sqrt()
            g = U @ V
            r1 = g.pow(2).sum(1).sqrt()
            before.append(float((r0.max() / r0.min().clamp(min=1e-30)).item()))
            after.append(float((r1.max() / r1.min().clamp(min=1e-30)).item()))
            cond.append(float(((U @ V).pow(2).sum(1).mean() * hd) / (V.pow(2).sum())))
            V_full[sl] = g
            bkey = f"{gauge.sa(l)}.v_proj.bias"
            if bkey in out:                        # v bias lives in the same subspace
                B = out[bkey].to(F64)
                B[sl] = (U @ B[sl]).to(out[bkey].dtype)
                out[bkey] = B
            for h in range(gidx * grp, (gidx + 1) * grp):
                cs = slice(h * hd, (h + 1) * hd)
                O_full[:, cs] = O_full[:, cs] @ U.T
        out[vk] = V_full.to(sd[vk].dtype)
        out[ok] = O_full.to(sd[ok].dtype)
    return out, {"canon": "G1", "method": method,
                 "row_energy_span_before": float(torch.tensor(before).median()),
                 "row_energy_span_after": float(torch.tensor(after).median()),
                 "balance_ratio_after": float(torch.tensor(cond).mean())}


# --- G7 ------------------------------------------------------------------------------------

def canon_g7(sd: dict, arch: Arch) -> tuple[dict, dict]:
    """Balance the SwiGLU up-branch diagonal: c_j = sqrt(B_j / A_j) where A_j = ||up_j: || and
    B_j = ||down_:,j|| are NORMS (not energies — V0 §7 states it the same way), so both sides
    meet at sqrt(A_j B_j) regardless of which point of the orbit was handed in.

    This is math.md section 7's V0 gauge-fix, unchanged, applied to the one MLP direction that
    survives a gated linear unit. Deterministic and artifact-only."""
    out = {k: v.clone() for k, v in sd.items()}
    spans = []
    for l in range(arch.layers):
        ukey = f"{gauge.mlp(l)}.up_proj.weight"
        dkey = f"{gauge.mlp(l)}.down_proj.weight"
        ubkey = f"{gauge.mlp(l)}.up_proj.bias"
        u = out[ukey].to(F64)
        d = out[dkey].to(F64)
        A = u.pow(2).sum(1).clamp(min=0).sqrt()          # row norms of up_proj
        if ubkey in out:                                 # bias is part of the same neuron
            A = (A.pow(2) + out[ubkey].to(F64).pow(2)).sqrt()
        B = d.pow(2).sum(0).clamp(min=0).sqrt()          # column norms of down_proj
        c = (B / A.clamp(min=1e-30)).sqrt()
        c = c / torch.exp(torch.log(c).mean())          # keep the global scale neutral
        spans.append(float((c.max() / c.min()).item()))
        out[ukey] = (c[:, None] * u).to(out[ukey].dtype)
        if ubkey in out:
            ub = out[ubkey]
            out[ubkey] = (c * ub.to(F64)).to(ub.dtype)
        out[dkey] = (d / c[None, :]).to(out[dkey].dtype)
    return out, {"canon": "G7", "target": "equalize up/down per-neuron energy (V0 gauge-fix)",
                 "applied_span_median": float(torch.tensor(spans).median())}


# --- G5 ------------------------------------------------------------------------------------

def canon_g5(sd: dict, arch: Arch) -> tuple[dict, dict]:
    """Gauge slice: the tied point. lm_head is the intrinsic scale witness.

    The G5 orbit through an artifact is {(c*E, H, c*A)}: A = the stream-writing tensors
    (o_proj, down_proj); embeddings scale with c; every RMSNorm is untouched because its
    output is 0-homogeneous. At most one point satisfies embed == lm_head, so the tie
    condition fixes the gauge intrinsically — no knowledge of the base is used. Getting
    there is the whole-artifact move, not an embed-only edit (embed-only would change the
    function)."""
    if "lm_head.weight" not in sd:
        return {k: v.clone() for k, v in sd.items()}, {"canon": "G5", "detected": False,
                                                       "reason": "tied artifact: no witness"}
    e = sd["model.embed_tokens.weight"].to(F64)
    h = sd["lm_head.weight"].to(F64)
    c = float(((e * h).sum() / (h * h).sum()).item())       # least squares: embed ~ c * head
    rho = float(((e - c * h).norm() / e.norm().clamp(min=1e-30)).item())
    if rho > 0.05:      # embed is not a scalar multiple of the head: not a G5 orbit point
        return {k: v.clone() for k, v in sd.items()}, {"canon": "G5", "detected": False,
                                                       "reason": "not on G5 orbit",
                                                       "tie_residual": rho, "c_lsq": c}
    out = {k: v.clone() for k, v in sd.items()}
    out["model.embed_tokens.weight"] = (e / c).to(sd["model.embed_tokens.weight"].dtype)
    for l in range(arch.layers):
        for key in (f"{gauge.sa(l)}.o_proj.weight", f"{gauge.mlp(l)}.down_proj.weight"):
            t = out[key]
            out[key] = (t.to(F64) / c).to(t.dtype)
    out.pop("lm_head.weight", None)                          # re-tied artifact
    return out, {"canon": "G5", "detected": True, "c_recovered": c}


# --- combined --------------------------------------------------------------------------------


def run(sd: dict, arch: Arch, fams, g1_method: str = "coherence",
        g2_method: str = "balance") -> tuple[dict, dict]:
    out, man = sd, []
    for f in fams:
        if f == "G5":
            out, m = canon_g5(out, arch)
        elif f == "G3":
            out, m = canon_g3(out, arch)
        elif f == "G2":
            out, m = canon_g2(out, arch, g2_method)
        elif f == "G7":
            out, m = canon_g7(out, arch)
        elif f == "G1":
            out, m = canon_g1(out, arch, g1_method)
        else:
            raise ValueError(f"unknown canon family {f}")
        man.append(m)
    return out, {"canonicalize": man}


# --- static conditioning proxy ----------------------------------------------------------------

def quant_condition(sd: dict, bits: int = 4, block: int = 32,
                     only: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj")) -> dict:
    """Predicted relative MSE of block-max-abs symmetric quantization, per tensor family.

    proxy = sum_blocks(amax^2 * n) / (12 * (2^(bits-1)-1)^2 * sum w^2)
    """
    s = 2 ** (bits - 1) - 1
    acc: dict[str, list[float]] = {}
    for k, t in sd.items():
        if not k.endswith(".weight") or "layers" not in k:
            continue
        tag = k.split(".")[-2]
        if tag not in only:
            continue
        w = t.to(F64)
        if w.ndim != 2 or w.shape[1] % block:
            continue
        x = w.reshape(-1, block)
        amax = x.abs().amax(1)
        num = float((amax.pow(2).sum() * block) / (12.0 * s * s))
        den = float(w.pow(2).sum())
        acc.setdefault(tag, []).append(num / den if den > 0 else float("inf"))
    return {t: sum(v) / len(v) for t, v in sorted(acc.items())}


if __name__ == "__main__":
    import json
    arch = common.read_arch(common.REF_MODEL)
    sd = common.load_state(common.REF_MODEL)
    base_c = quant_condition(sd)
    log(f"base conditioning {json.dumps(base_c)}")
    spec = sys.argv[1] if len(sys.argv) > 1 else "G3:random:1+G1:haar:1"
    s2, m = gauge.apply_spec(sd, arch, spec)
    c2, cm = canon_all(s2, arch)
    print(json.dumps({"spec": spec, "cond_base": base_c, "cond_stressed": quant_condition(s2),
                      "cond_repaired": quant_condition(c2), "manifest": cm}, indent=2))
