#!/usr/bin/env python3
"""M1 gauge library: EXACT, architecture-valid function-preserving transforms for Qwen2-class decoders.

Every family below is provably function-preserving in exact arithmetic (fp64 here, bf16
storage afterwards) and is verified numerically by m1/verify_equiv.py before any surgery is
trusted. Families that are not valid for an architecture must fail closed.

Attention biases matter enormously here: Qwen2.5-0.5B carries q/k biases with absmax 79 / 130
(a long-context YaRN artifact), so anything that rotates q/k or v rows MUST rotate the matching
bias entries or it silently changes the function. m1/test_gauge_math.py keeps a permanent
sensitivity control that fails the suite if that detection ever stops working.

Families (derivations in ../M1_NOTES.md):

  G1 vo_orth      per-GQA-group value/output change of basis with any U in O(d_head):
                  v rows and v bias of group g <- U·(·); o columns of every q-head in g <- ·Uᵀ.
                  RoPE never touches V/O, so U is unrestricted. No ReLU-MLP analogue exists.

  G2 rope_pairs   per-pair 2D rotation of the paired rows/bias entries (j, j+d/2) of q (per
                  head) and k (per group). The score term is q_jᵀ R(θ_j(t−s)) k_j and 2D
                  rotations commute, so a shared rotation is invisible. Distinct frequencies
                  make the 2-planes non-isomorphic, so by Schur's lemma this is the WHOLE
                  commutant of the RoPE family: G2 is the maximal exact q/k gauge. GQA forces
                  all q-heads of a group to share the group's angles (k is shared).

  G3 norm_diag    RMSNorm scale absorption: w <- w⊙d, consumer INPUT columns <- W/d for every
                  consumer of that norm (q,k,v | gate,up), d > 0. Biases sit on the output
                  side and are correctly untouched.

  G4 head_perm    query-head permutation inside a kv group + kv-group permutation, weights and
                  biases. Control family: exact and quantization-neutral (whole rows move).

  G5 res_scale    global residual-stream scaling. RMSNorm output is 0-homogeneous, so only
                  embed, o_proj and down_proj move; exact only as eps -> 0, unless the config
                  constant is moved eps -> eps/c² (declared, optional). The head must stay put,
                  which the embedding tie forbids in one tensor → structure-changing.
                  Quantization-neutral per tensor; damages fp16 runtimes and merges instead.

  G6 neuron_perm  SwiGLU hidden-unit permutation (gate/up rows + biases, down columns).

  G7 up_diag      SwiGLU up-branch diagonal: row j of up_proj (and its bias) times c_j,
                  column j of down_proj divided by c_j, any nonzero c_j. V0's ReLU-style
                  scaling gauge DOES survive a GLU — the gated branch cannot scale
                  (silu(c g) != c silu(g)) but its multiplicative partner can. The exact MLP
                  group is therefore permutation x (R minus {0})^intermediate, not permutation alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import Arch, log  # noqa: E402

F64 = torch.float64
FAM = ("G1", "G2", "G3", "G4", "G5", "G6", "G7")


# --- linear algebra helpers ---------------------------------------------------------------

def gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def haar(d: int, seed: int) -> torch.Tensor:
    """Deterministic Haar-distributed orthogonal matrix (QR of a Gaussian)."""
    A = torch.randn(d, d, generator=gen(seed), dtype=F64)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R))


def hadamard(d: int) -> torch.Tensor:
    """Normalised Sylvester Hadamard; d must be a power of two."""
    if d & (d - 1):
        raise ValueError(f"Hadamard needs a power-of-2 dimension, got {d}")
    H = torch.ones(1, 1, dtype=F64)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / (d ** 0.5)


def per_pair_angles(n_pairs: int, seed: int) -> torch.Tensor:
    return torch.rand(n_pairs, generator=gen(seed), dtype=F64) * (2 * torch.pi)


def row_gather(order, hd: int) -> torch.Tensor:
    """Row indices realizing a block permutation of unit size hd."""
    ar = torch.arange(hd)
    return torch.cat([int(h) * hd + ar for h in order]).to(torch.long)


# --- tensor access helpers ----------------------------------------------------------------

def sa(l: int) -> str:
    return f"model.layers.{l}.self_attn"


def mlp(l: int) -> str:
    return f"model.layers.{l}.mlp"


def _req(sd: dict, k: str) -> torch.Tensor:
    if k not in sd:
        raise SystemExit(f"gauge: expected tensor {k} — fail closed (unknown artifact layout)")
    return sd[k]


def _rot_rows(T: torch.Tensor, idx: torch.Tensor, c: torch.Tensor, s: torch.Tensor):
    """Rotate row pairs (idx, idx+half) of T by per-pair angles c,s.

    T is a 2-D weight [rows, in] or a 1-D bias [rows]; idx is [r, half] and the partner of a
    pair is idx + idx.shape[-1], which is the rotate_half layout (j, j + head_dim/2) inside
    every head block.
    """
    half = idx.shape[-1]
    u, v = T[idx], T[idx + half]
    shape = [1] * u.dim()
    shape[idx.dim() - 1] = -1
    cc, ss = c.reshape(shape), s.reshape(shape)
    return cc * u - ss * v, ss * u + cc * v


def _apply_rot(T: torch.Tensor, idx: torch.Tensor, c: torch.Tensor, s: torch.Tensor):
    a, b = _rot_rows(T, idx, c, s)
    T[idx], T[idx + idx.shape[-1]] = a, b


# --- G1: value/output change of basis -----------------------------------------------------

def g1_vo_orth(sd: dict, arch: Arch, mode: str = "haar", seed: int = 0,
               layers: list[int] | None = None, u_for=None) -> tuple[dict, dict]:
    """U per (layer, kv-group). mode: haar | hadamard | svd (energy-concentrating)."""
    sd = {k: v.clone() for k, v in sd.items()}
    hd, grp = arch.head_dim, arch.group
    used, conc = [], []
    for l in (layers or list(range(arch.layers))):
        bkey = f"{sa(l)}.v_proj.bias"
        for gidx in range(arch.n_kv):
            vkey, okey = f"{sa(l)}.v_proj.weight", f"{sa(l)}.o_proj.weight"
            sl = slice(gidx * hd, (gidx + 1) * hd)
            v, o = _req(sd, vkey), _req(sd, okey)
            vb = v[sl].to(F64)
            if u_for is not None:
                U = u_for(l, gidx, vb)
            elif mode == "haar":
                U = haar(hd, seed * 1000 + l * 32 + gidx)
            elif mode == "hadamard":
                U = hadamard(hd)
            elif mode == "svd":
                P, S, _ = torch.linalg.svd(vb, full_matrices=False)
                U = P.T                                # rows -> descending singular dirs
                conc.append(float((S[: hd // 2].pow(2).sum() / S.pow(2).sum()).item()))
            else:
                raise ValueError(f"unknown G1 mode {mode}")
            v[sl] = (U @ vb).to(v.dtype)
            if bkey in sd:                             # v bias lives in the same subspace
                b = sd[bkey]
                b[sl] = (U @ b[sl].to(F64)).to(b.dtype)
            for h in range(gidx * grp, (gidx + 1) * grp):
                cs = slice(h * hd, (h + 1) * hd)
                o[:, cs] = (o[:, cs].to(F64) @ U.T).to(o.dtype)
            used.append((l, gidx))
    return sd, {"family": "G1", "mode": mode, "seed": seed, "groups": len(used),
                "v_bias_transformed": bkey in sd,
                "svd_top_half_energy": (sum(conc) / len(conc)) if conc else None}


# --- G2: RoPE-pair rotations of q/k -------------------------------------------------------

def g2_rope_pairs(sd: dict, arch: Arch, mode: str = "random", seed: int = 0,
                  angles: torch.Tensor | None = None) -> tuple[dict, dict]:
    """Rotate rows and biases of every RoPE pair (j, j+hd/2) of q (per head) and k (per group)."""
    sd = {k: v.clone() for k, v in sd.items()}
    hd, grp = arch.head_dim, arch.group
    if hd % 2:
        raise SystemExit("G2 needs an even head_dim (RoPE pairing)")
    half = hd // 2
    jj = torch.arange(half, dtype=torch.long)
    for l in range(arch.layers):
        for gidx in range(arch.n_kv):
            if angles is None:
                if mode == "random":
                    ph = per_pair_angles(half, seed * 100003 + l * 977 + gidx)
                elif mode == "quarter":
                    ph = torch.full((half,), torch.pi / 4, dtype=F64)
                else:
                    raise ValueError(f"unknown G2 mode {mode}")
            else:
                ph = angles.to(F64)
            c, s = torch.cos(ph), torch.sin(ph)
            r1q = torch.stack([h * hd + jj for h in range(gidx * grp, (gidx + 1) * grp)])
            r1k = (gidx * hd + jj).unsqueeze(0)
            for prefix, idx in ((f"{sa(l)}.q_proj", r1q), (f"{sa(l)}.k_proj", r1k)):
                for suffix in (".weight", ".bias"):
                    key = prefix + suffix
                    if key not in sd:
                        continue
                    T = _req(sd, key).to(F64)
                    _apply_rot(T, idx, c, s)
                    sd[key] = T.to(_req(sd, key).dtype)
    return sd, {"family": "G2", "mode": mode, "seed": seed,
                "bias_transformed": f"{sa(0)}.q_proj.bias" in sd}


# --- G3: RMSNorm scale absorption ---------------------------------------------------------

def g3_norm_diag(sd: dict, arch: Arch, mode: str = "random", seed: int = 0,
                 decades: float = 3.0, d_attn: torch.Tensor | None = None,
                 d_mlp: torch.Tensor | None = None) -> tuple[dict, dict]:
    """w <- w·d ; consumer INPUT columns <- W/d. Random log-uniform d stresses 32-blocks."""
    sd = {k: v.clone() for k, v in sd.items()}
    g = gen(seed)
    lg10 = torch.log(torch.tensor(10.0, dtype=F64))

    def mk() -> torch.Tensor:
        if mode == "random":
            u = torch.rand(arch.hidden, generator=g, dtype=F64)
            return torch.exp((2 * u - 1) * decades * lg10)
        if mode == "pow2":
            # exact in bf16: multiplying a bf16 value by 2^k is lossless (k stays well inside
            # the exponent range), so the gauge itself costs no representation noise and any
            # downstream difference is purely the change of coordinates. +-decades of MAGNITUDE
            # maps to k = +-(decades * log2 10) rounded, so this spans the same range as "random".
            kmax = int(round(decades * 3.321928094887362))
            k = torch.randint(-kmax, kmax + 1, (arch.hidden,), generator=g, dtype=torch.int32)
            return torch.pow(torch.tensor(2.0, dtype=F64), k.to(F64))
        if mode == "smooth":                 # block-constant: a super-block scale absorbs it
            u = torch.rand(arch.hidden // 32, generator=g, dtype=F64)
            return torch.exp((2 * u - 1) * decades * lg10).repeat_interleave(32)
        raise ValueError(f"unknown G3 mode {mode}")

    spans = []
    for l in range(arch.layers):
        for nk, pre, cons in (("input_layernorm.weight", "self_attn", ("q_proj", "k_proj", "v_proj")),
                              ("post_attention_layernorm.weight", "mlp", ("gate_proj", "up_proj"))):
            tag = "attn" if pre == "self_attn" else "mlp"
            wkey = f"model.layers.{l}.{nk}"
            w = _req(sd, wkey)
            d = (d_attn if (tag == "attn" and d_attn is not None) else
                 d_mlp if (tag == "mlp" and d_mlp is not None) else mk()).to(F64)
            spans.append(float((d.max() / d.min()).item()))
            w.copy_((w.to(F64) * d).to(w.dtype))
            for cname in cons:
                key = f"{sa(l) if pre == 'self_attn' else mlp(l)}.{cname}.weight"
                t = _req(sd, key)
                t.copy_((t.to(F64) / d[None, :]).to(t.dtype))
    return sd, {"family": "G3", "mode": mode, "seed": seed, "decades": decades,
                "max_d_span": max(spans) if spans else None}


# --- G4: head / group permutations (control) ----------------------------------------------

def g4_head_perm(sd: dict, arch: Arch, seed: int = 0, within: bool = True,
                 groups: bool = True) -> tuple[dict, dict]:
    sd = {k: v.clone() for k, v in sd.items()}
    hd = arch.head_dim
    gq = torch.randperm(arch.n_kv, generator=gen(seed)) if groups else torch.arange(arch.n_kv)
    pw = torch.randperm(arch.group, generator=gen(seed + 1)) if within else torch.arange(arch.group)
    head_map = [int(gq[new_g]) * arch.group + int(pw[new_i])
                for new_g in range(arch.n_kv) for new_i in range(arch.group)]
    q_idx, kv_idx = row_gather(head_map, hd), row_gather([int(g) for g in gq], hd)
    for l in range(arch.layers):
        for key, idx in ((f"{sa(l)}.q_proj.weight", q_idx), (f"{sa(l)}.q_proj.bias", q_idx),
                         (f"{sa(l)}.k_proj.weight", kv_idx), (f"{sa(l)}.k_proj.bias", kv_idx),
                         (f"{sa(l)}.v_proj.weight", kv_idx), (f"{sa(l)}.v_proj.bias", kv_idx)):
            if key not in sd:
                continue
            t = _req(sd, key)
            sd[key] = t.to(F64)[idx].to(t.dtype)
        okey = f"{sa(l)}.o_proj.weight"
        o = _req(sd, okey)
        sd[okey] = o.to(F64)[:, q_idx].to(o.dtype)
    return sd, {"family": "G4", "seed": seed, "within": within, "groups": groups,
                "bias_transformed": f"{sa(0)}.q_proj.bias" in sd}


# --- G5: global residual scaling (unties lm_head) -----------------------------------------

def g5_res_scale(sd: dict, arch: Arch, c: float = 8.0,
                 rms_norm_eps: float | None = None) -> tuple[dict, dict]:
    """Residual stream z -> c·z.

    RMSNorm output is 0-homogeneous only in the limit eps -> 0:
        rms(c z) = c z / sqrt(c² m + eps'),  which equals  z / sqrt(m + eps)  iff eps' = c² eps,
    so with a nonzero eps the naive move drifts by O(eps). Rewriting the config constant
    eps -> c²·eps makes it EXACT — an artifact-level metadata edit carrying a weight
    symmetry, which is exactly math.md §1's state = (theta, a). Pass the model's current eps
    in `rms_norm_eps` to request the patch; leave it None to measure the floor instead.

    Only embed, o_proj and down_proj move (normalized branches are invariant). The head must
    stay put and it is tied to embed, so the tie has to be broken to express this gauge at all.
    """
    if c <= 0:
        raise ValueError("c must be positive")
    sd = {k: v.clone() for k, v in sd.items()}
    if "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
    e = _req(sd, "model.embed_tokens.weight")
    e.copy_((e.to(F64) * c).to(e.dtype))
    for l in range(arch.layers):
        for key in (f"{sa(l)}.o_proj.weight", f"{mlp(l)}.down_proj.weight"):
            t = _req(sd, key)
            t.copy_((t.to(F64) * c).to(t.dtype))
    sd["model.embed_tokens.weight"] = e
    patch = {"rms_norm_eps": float(rms_norm_eps) * c * c} if rms_norm_eps else {}
    return sd, {"family": "G5", "c": c, "untied": True, "eps_patch": patch or None,
                "eps_source": rms_norm_eps}


# --- G6: SwiGLU neuron permutation --------------------------------------------------------

def g6_neuron_perm(sd: dict, arch: Arch, seed: int = 0) -> tuple[dict, dict]:
    sd = {k: v.clone() for k, v in sd.items()}
    p = torch.randperm(arch.intermediate, generator=gen(seed))
    for l in range(arch.layers):
        for name in ("gate_proj", "up_proj"):
            wkey, bkey = f"{mlp(l)}.{name}.weight", f"{mlp(l)}.{name}.bias"
            t = _req(sd, wkey)
            sd[wkey] = t.to(F64)[p].to(t.dtype)
            if bkey in sd:
                b = _req(sd, bkey)
                sd[bkey] = b.to(F64)[p].to(b.dtype)
        dkey = f"{mlp(l)}.down_proj.weight"
        d = _req(sd, dkey)
        sd[dkey] = d.to(F64)[:, p].to(d.dtype)
    return sd, {"family": "G6", "seed": seed}


# --- G7: SwiGLU up-branch diagonal (V0's mechanism, reborn) -------------------------------

def g7_up_diag(sd: dict, arch: Arch, mode: str = "random", seed: int = 0,
               decades: float = 3.0) -> tuple[dict, dict]:
    """out = Σ_j silu(g_j)·u_j · d_j  is invariant under  u_j ← c_j u_j ,  d_j ← d_j / c_j
    (row j of gate/up weights and column j of down_proj), for ANY nonzero c_j — the gated
    branch may not move, because silu(c g) ≠ c silu(g), but the multiplicative partner can.

    This is the direct transformer analogue of V0's ReLU scaling gauge (math.md §1), and it
    survives SwiGLU precisely because GLU factorizes the nonlinearity away from one branch.
    """
    sd = {k: v.clone() for k, v in sd.items()}
    if mode not in ("random", "few"):
        raise ValueError(f"unknown G7 mode {mode}")
    lg10 = torch.log(torch.tensor(10.0, dtype=F64))
    spans = []
    for l in range(arch.layers):
        u = _req(sd, f"{mlp(l)}.up_proj.weight")
        d = _req(sd, f"{mlp(l)}.down_proj.weight")
        g = gen(seed * 7919 + l)
        c = torch.exp((2 * torch.rand(arch.intermediate, generator=g, dtype=F64) - 1) * decades * lg10)
        if mode == "few":                      # concentrate the damage on 0.1% of neurons
            keep = torch.zeros(arch.intermediate, dtype=F64) + 1.0
            idx = torch.randperm(arch.intermediate, generator=g)[: max(1, arch.intermediate // 1000)]
            keep[idx] = c[idx]
            c = keep
        spans.append(float((c.max() / c.min()).item()))
        ub = f"{mlp(l)}.up_proj.bias"
        u.copy_((c[:, None] * u.to(F64)).to(u.dtype))
        if ub in sd:                           # bias moves with its own row
            b = sd[ub]
            b.copy_((c * b.to(F64)).to(b.dtype))
        d.copy_((d.to(F64) / c[None, :]).to(d.dtype))
    return sd, {"family": "G7", "mode": mode, "seed": seed, "decades": decades,
                "max_c_span": max(spans) if spans else None}


# --- spec dispatch ------------------------------------------------------------------------

def apply_spec(sd: dict, arch: Arch, spec: str,
               cfg_eps: float | None = None) -> tuple[dict, dict]:
    """spec grammar: FAMILY[:mode[:seed]]  e.g. "G1:haar:7", "G3:random:1", "G5:c8".
    Compose with "+": "G3:random:1+G1:haar:1+G2:random:1".
    G5 takes an optional eps patch: "G5:c8:1" keeps the seed slot for the model eps only
    through g5_res_scale(rms_norm_eps=...), which make_variants passes explicitly."""
    out, manifests, cfg_patch = sd, [], {}
    for part in spec.split("+"):
        bits = part.split(":")
        fam, mode = bits[0], (bits[1] if len(bits) > 1 else "")
        seed = int(bits[2]) if len(bits) > 2 and bits[2].lstrip("-").isdigit() else 0
        if fam == "G1":
            out, m = g1_vo_orth(out, arch, mode or "haar", seed)
        elif fam == "G2":
            out, m = g2_rope_pairs(out, arch, mode or "random", seed)
        elif fam == "G3":
            out, m = g3_norm_diag(out, arch, mode or "random", seed)
        elif fam == "G4":
            out, m = g4_head_perm(out, arch, seed)
        elif fam == "G5":
            if not mode.startswith("c"):
                raise ValueError("G5 needs c<factor>, e.g. G5:c8")
            want_eps = len(bits) > 2 and bits[2] == "eps"
            out, m = g5_res_scale(out, arch, float(mode[1:]),
                                  rms_norm_eps=cfg_eps if want_eps else None)
            if m.get("eps_patch"):
                cfg_patch.update(m["eps_patch"])
        elif fam == "G6":
            out, m = g6_neuron_perm(out, arch, seed)
        elif fam == "G7":
            out, m = g7_up_diag(out, arch, mode or "random", seed)
        else:
            raise ValueError(f"unknown family {fam} in {spec}")
        manifests.append(m)
    return out, {"spec": spec, "transforms": manifests, "config_patch": cfg_patch or None}


if __name__ == "__main__":
    import json
    arch = common.read_arch(common.REF_MODEL)
    sd = common.load_state(common.REF_MODEL)
    spec = sys.argv[1] if len(sys.argv) > 1 else "G2:random:1"
    s2, man = apply_spec(sd, arch, spec)
    log(json.dumps(man, indent=2, default=str))
    log(f"keys added: {sorted(k for k in s2 if k not in sd)}")
