#!/usr/bin/env python3
"""M1 gauge math tests on a tiny Qwen2-shaped model, in fp32, no artifacts on disk.

Properties checked per family:
  1. EXACTNESS     the stress transform leaves logits unchanged (max abs diff ~ 1e-6)
  2. REPAIR        the artifact-only canonicalizer leaves the function unchanged, from
                   either side of the orbit
  3. CANONICITY    J(canon(base)) == J(canon(stressed)): the canonicalizer lands on the same
                   measurable conditioning regardless of which orbit point it is handed, so
                   `prepare` selects a representative rather than undoing a stress it was
                   never told about. (Bitwise orbit representatives are NOT asserted: sign
                   and per-pair-swap ambiguities inside an orthogonal family are genuine
                   residual gauge freedom, not a bug. G5 has an intrinsic witness — the tie —
                   so its bitwise representative IS asserted.)

Run: /home/admin/counterpoint/.venv/bin/python m1/test_gauge_math.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import canonicalize as canon  # noqa: E402
import gauge  # noqa: E402

CFG = dict(model_type="qwen2", vocab_size=97, hidden_size=32, intermediate_size=64,
           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
           max_position_embeddings=256, rope_theta=1_000_000.0, rms_norm_eps=1e-6)
TOL_EXACT, TOL_J, TOL_BITWISE_G5 = 2e-5, 0.02, 1e-5


def make():
    """Untied tiny Qwen2 whose lm_head starts equal to the embedding, i.e. a materialized
    tied artifact — that is what makes the G5 tie witness testable."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(3)
    m = Qwen2ForCausalLM(Qwen2Config(**{**CFG, "tie_word_embeddings": False})).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.mul_(1 + torch.randn(()) * 0.3)      # HF init is too tame to stress anything
            if n.endswith(".bias"):
                # HF zero-initializes biases; Qwen2.5-0.5B ships q/k biases with absmax
                # 79/130, and a transform that forgets them is only detectable if biases
                # are actually nonzero. This line is what makes the bias tests mean it.
                p.normal_(0, 0.5)
        m.lm_head.weight.copy_(m.model.embed_tokens.weight)   # tied point, AFTER the noise
    assert float(m.model.layers[0].self_attn.q_proj.bias.abs().max()) > 0.05, "test model lost its biases"
    return m


def sd_of(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def logits(m, x):
    with torch.no_grad():
        return m(input_ids=x).logits


def reload(m, sd):
    """Load a state dict into a fresh copy of m, filling a dropped lm_head from embed."""
    from transformers import Qwen2ForCausalLM
    m2 = Qwen2ForCausalLM(m.config).float().eval()
    sd = dict(sd)
    if "lm_head.weight" in m2.state_dict() and "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
    missing, unexpected = m2.load_state_dict(sd, strict=False)
    bad = [k for k in list(missing) + list(unexpected)
           if "rotary" not in k and "inv_freq" not in k]
    assert not bad, (missing, unexpected)
    return m2


CASES = [
    # name          transform                                                  canon families
    ("G1:haar",  lambda sd, a: gauge.g1_vo_orth(sd, a, "haar", 1),            ("G1",)),
    ("G1:svd",   lambda sd, a: gauge.g1_vo_orth(sd, a, "svd", 1),             ("G1",)),
    ("G1:had",   lambda sd, a: gauge.g1_vo_orth(sd, a, "hadamard", 0),        ("G1",)),
    ("G2:rand",  lambda sd, a: gauge.g2_rope_pairs(sd, a, "random", 1),       ("G2",)),
    ("G2:qtr",   lambda sd, a: gauge.g2_rope_pairs(sd, a, "quarter", 0),      ("G2",)),
    ("G3:rand",  lambda sd, a: gauge.g3_norm_diag(sd, a, "random", 1, 2.),    ("G3",)),
    ("G3:smth",  lambda sd, a: gauge.g3_norm_diag(sd, a, "smooth", 1, 2.),    ("G3",)),
    ("G4:perm",  lambda sd, a: gauge.g4_head_perm(sd, a, 1),                  None),
    ("G6:perm",  lambda sd, a: gauge.g6_neuron_perm(sd, a, 1),                None),
    ("G5:c4",    lambda sd, a: gauge.g5_res_scale(sd, a, 4.0),                ("G5",)),
    ("G7:rand",  lambda sd, a: gauge.g7_up_diag(sd, a, "random", 1, 2.),       ("G7",)),
    ("G7:few",   lambda sd, a: gauge.g7_up_diag(sd, a, "few", 1, 3.),          ("G7",)),
    # multi-family: sequential per-family projections do NOT commute (G3 and G7 both move
    # up_proj magnitudes), so the landing point is family-order dependent. Exactness is still
    # asserted; the J residual is reported as post-prepare gauge debt, and is a real finding.
    ("bad_all",  lambda sd, a: gauge.apply_spec(sd, a,
        "G3:random:1+G1:haar:1+G2:random:1+G7:random:1"),
     ("G3", "G2", "G7", "G1")),
    ("bad_all_r",  lambda sd, a: gauge.apply_spec(sd, a,
        "G7:random:1+G3:random:1+G1:haar:1"),
     ("G7", "G3", "G1")),
]

TOL_EXACT, TOL_BITWISE_G5 = 2e-5, 1e-5
TOL_J_DEFAULT, TOL_J_G1 = 0.02, 0.12   # G1 keeps residual sign/permutation freedom inside
                                      # the balanced set, so its objective match is a band


def jdiff(ja: dict, jb: dict) -> float:
    """Worst relative difference of the static conditioning proxy across tensor families."""
    return max(abs(ja[t] / jb[t] - 1.0) for t in ja if jb.get(t)) if ja and jb else 0.0


def g5_eps_probe(c: float = 4.0):
    """The residual-scale gauge is exact only up to RMSNorm's epsilon: n(z)=z/sqrt(mean z^2
    +eps) is 0-homogeneous in the limit eps->0. Shrinking eps must shrink the drift."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    x2 = torch.randint(0, CFG["vocab_size"], (2, 48))
    out = {}
    for eps in (1e-6, 1e-12):
        torch.manual_seed(3)
        mm = Qwen2ForCausalLM(Qwen2Config(**{**CFG, "rms_norm_eps": eps,
                                             "tie_word_embeddings": False})).float().eval()
        with torch.no_grad():
            for p in mm.parameters():
                p.mul_(1 + torch.randn(()) * 0.3)
            mm.lm_head.weight.copy_(mm.model.embed_tokens.weight)
        sd = sd_of(mm)
        a = common.Arch.from_config(mm.config.to_dict())
        base = logits(mm, x2)
        stressed = logits(reload(mm, gauge.g5_res_scale(sd, a, c)[0]), x2)
        out[eps] = float((stressed - base).abs().max() / base.abs().max())
    return out

def main():
    m = make()
    torch.manual_seed(0)
    x = torch.randint(0, CFG["vocab_size"], (2, 48))
    a = common.Arch.from_config(m.config.to_dict())
    base_sd = sd_of(m)
    base_lg = logits(m, x)
    base_j = canon.quant_condition(base_sd)
    fails = []
    print(f"tiny qwen2: hidden={a.hidden} q/kv={a.n_q}/{a.n_kv} hd={a.head_dim} "
          f"group={a.group} inter={a.intermediate} layers={a.layers}")
    print(f"base J = {', '.join(f'{k}={v:.3f}' for k, v in base_j.items())}")
    for name, fn, fams in CASES:
        got = fn(base_sd, a)
        sd2 = got[0] if isinstance(got, tuple) else got
        d = float((logits(reload(m, sd2), x) - base_lg).abs().max())
        row = {"exact": d}
        if fams:
            c_base, _ = canon.run(base_sd, a, fams)
            c_str, _ = canon.run(sd2, a, fams)
            lg_b, lg_s = logits(reload(m, c_base), x), logits(reload(m, c_str), x)
            row["canon_exact"] = float((lg_b - base_lg).abs().max())
            row["canon_exact2"] = float((lg_s - base_lg).abs().max())
            row["bitwise"] = float((lg_s - lg_b).abs().max())
            row["j"] = jdiff(canon.quant_condition(c_base), canon.quant_condition(c_str))
        # G5 is exact only up to RMSNorm's epsilon floor (see g5_eps_probe); everything
        # else is an exact symmetry of the architecture and must land at fp32 round-off.
        tol = 3e-3 if name.startswith("G5") else TOL_EXACT
        ok = d <= tol
        if fams:
            ok = ok and row["canon_exact"] <= tol and row["canon_exact2"] <= tol
            if name.startswith("bad_all"):
                print(f"{name:9s} (post-prepare gauge debt, not asserted: families do not "
                      f"commute) J_residual={row['j']:.3f}")
            else:
                ok = ok and row["j"] <= (TOL_J_G1 if "G1" in fams else TOL_J_DEFAULT)
            if fams == ("G5",):
                ok = ok and row["bitwise"] <= TOL_BITWISE_G5
        detail = " ".join(f"{k}={v:.2e}" if k != "j" else f"J={v:.2e}" for k, v in row.items())
        print(f"{name:9s} {detail:78s} {'OK' if ok else 'FAIL'}")
        if not ok:
            fails.append(name)
    eps = g5_eps_probe()
    print(f"G5 drift / logit-scale by RMSNorm eps: 1e-6 -> {eps[1e-6]:.2e}   "
          f"1e-12 -> {eps[1e-12]:.2e}")
    if eps[1e-12] > TOL_EXACT or eps[1e-12] >= eps[1e-6]:
        print("  FAIL: residual-scale gauge drift is not the RMSNorm epsilon floor")
        fails.append("G5:eps-floor")

    # Sensitivity control: the same G2 rotation applied to WEIGHTS ONLY — the exact bug this
    # suite was written to catch (Qwen2.5-0.5B carries q/k biases with absmax 79/130) — MUST
    # fail. If it ever stops failing, the suite is blind and the numbers above mean nothing.
    def weights_only_g2(sd, a):
        sd = {k: v.clone() for k, v in sd.items()}
        for l in range(a.layers):
            for pfx in ("q_proj", "k_proj"):
                key = f"{gauge.sa(l)}.{pfx}.weight"
                t = sd[key].to(torch.float64)
                for g in range(a.n_kv):
                    ph = gauge.per_pair_angles(a.head_dim // 2, 17 + 7 * l + g)
                    jj = torch.arange(a.head_dim // 2)
                    idx = (torch.stack([h * a.head_dim + jj
                                        for h in range(g * a.group, (g + 1) * a.group)])
                           if pfx == "q_proj" else (g * a.head_dim + jj).unsqueeze(0))
                    t[idx], t[idx + a.head_dim // 2] = gauge._rot_rows(
                        t, idx, torch.cos(ph), torch.sin(ph))
                sd[key] = t.to(sd[key].dtype)
        return sd
    sab = float((logits(reload(m, weights_only_g2(base_sd, a)), x) - base_lg).abs().max())
    print(f"sensitivity control (G2 weights-only, biases forgotten): max|dlogit|={sab:.2e} "
          f"= {sab / float(base_lg.abs().max()):.1%} of logit scale")
    if sab < 100 * TOL_EXACT:
        print("  FAIL: suite cannot detect a forgotten attention bias")
        fails.append("sensitivity")
    # G5's declared config edit (eps -> eps/c^2) must close the drift completely.
    from transformers import Qwen2Config, Qwen2ForCausalLM
    c5, eps5 = 4.0, 1e-6
    sd5 = gauge.g5_res_scale(base_sd, a, c5, rms_norm_eps=eps5)[0]
    m_eps = Qwen2ForCausalLM(Qwen2Config(**{**CFG, "rms_norm_eps": eps5 * c5 ** 2,
                                            "tie_word_embeddings": False})).float().eval()
    sd5 = dict(sd5)
    m_eps.load_state_dict(sd5, strict=False)
    m_eps.eval()
    d5 = float((logits(m_eps, x) - base_lg).abs().max())
    print(f"G5 with eps->c^2*eps (exactness claim): max|dlogit|={d5:.2e}")
    if d5 > TOL_EXACT:
        print("  FAIL: eps-patched residual scaling is not exact")
        fails.append("G5:eps-fix")
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
