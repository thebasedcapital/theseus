#!/usr/bin/env python3
"""archcheck/probe.py — cross-architecture static-scanner classification for Theseus.

Owner: ArchAudit (shared worktree /home/admin/theseus). Read-only: loads config.json and the
safetensors HEADER (u64 length + JSON; it never seeks past the header, so no tensor bytes are
read). Classification is evidence-based: architecture facts + GGUF-import transform knowledge
are sourced from llama.cpp conversion/*.py, the block layout notes come from ggml-common.h and
the runtime conventions come from upstream transformers source. Every non-obvious claim in the
output carries a "file:line" anchor; anything the source or bytes cannot support is printed
UNAVAILABLE with a reason (fail closed; null never means False, I8).

Usage:  archcheck/probe.py <hf_dir_or_config.json>
Exit:   0 = arch known and every gauge family claimed EXACT for the artifact is supported
        1 = FAIL CLOSED: unknown arch, or a required family/feature is UNAVAILABLE here
        2 = usage, 3 = unreadable config
Never loads the model, never writes anything, stdlib only.
"""

import json
import os
import re
import struct
import sys

FAMILIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

###############################################################################
# Static per-architecture facts.  Every row is grounded in the two source trees
# on this box:
#   L = /home/admin/tools/llama.cpp-cuda-src   (conversion/*.py, ggml/src/ggml-common.h)
#   T = /home/admin/counterpoint/.venv/lib/python3.12/site-packages/transformers/models/<arch>/
# Gauge semantics (M1_NOTES.md §2 / m1/gauge.py): G1 value-subspace basis change
# (needs GQA: U constant per kv-group when kv<q), G2 RoPE-plane rotations (needs the
# rotate_half pairing AND distinct per-pair frequencies for the claimed maximality; pair
# rotations stay exact even with duplicated frequencies), G3 RMSNorm scale absorption
# (needs a scale-invariant pre-norm: rms/z, NOT LayerNorm), G5 stream-scale (tie must be
# breakable), G7 SwiGLU up-branch diagonal (needs a GLU gate/up split).
###############################################################################

# arch id -> facts:
#   attn  "full" | "moe-full" (dense attention + MoE/MLA MLP) | "ssm" | other
#   mlp   "glu" | "post-ln-linear" | "ssm"
#   norm  "rms-w" (true scale) | "rms-1pw" (stored as 1+w offset) | "ln~" (LayerNorm)
#   tie   True | False | "either"
#   rope  "rotate_half" | "none" | "mrope" | "partial"

ARCHS = {
    "llama": {
        "hf": ("LlamaForCausalLM",),
        "mt": ("llama",),
        "attn": "full", "mlp": "glu", "norm": "rms-w", "tie": "either", "rope": "rotate_half",
        "bias_fact": "no attention projection bias (nn.Linear bias defaults False)",
        "transforms": [
            ("q/k_proj ROW REPERMUTED at import: HF keeps rotate_half pairs (j, j+d/2) apart, "
             "GGUF interleaves them — LlamaModel.permute reshape(n_head,2,..).swapaxes(1,2)",
             "conversion/llama.py:163-168 layout def, :244-249 apply, undo_permute=True :33"),
            ("MoE (Mixtral/VLlama) experts named w1/w2/w3 merged to "
             "layers.{}.feed_forward.experts.{w1,w2,w3}.weight",
             "conversion/llama.py:256-295"),
        ],
        "danger": "RoPE-pair row ORDER differs between the HF artifact (pairs d/2 apart) and "
                  "the GGUF (pairs adjacent); all five static aggregates are row-permutation-"
                  "invariant so the census survives, but any per-row feature read off GGUF bytes "
                  "(or a G2 pairing assertion on GGUF) is invalid unless it restores the HF frame.",
    },
    "mistral": {
        "hf": ("MistralForCausalLM",),
        "mt": ("mistral",),
        "attn": "full", "mlp": "glu", "norm": "rms-w", "tie": "either", "rope": "rotate_half",
        "bias_fact": "no attention projection bias (mistral/modeling_mistral.py:134-137)",
        "transforms": [
            ("HF-format Mistral routes through LlamaModel -> q/k rows REPERMUTED like llama",
             "conversion/llama.py:17-25 register includes MistralForCausalLM; :244-249"),
            ("Native --mistral-format path: undo_permute=False (community format already in "
             "ggml order)", "conversion/mistral.py:33-35"),
        ],
        "danger": "sliding_window (default 4096, mistral/configuration_mistral.py:80) is a "
                  "RUNTIME mask: it does not break G1/G2/G3/G5/G7 exactness but invalidates any "
                  "long-context equivalence claim that assumes full attention.",
    },
    "gemma": {
        "hf": ("GemmaForCausalLM",),
        "mt": ("gemma",),
        "attn": "full", "mlp": "glu", "norm": "rms-1pw", "tie": True, "rope": "rotate_half",
        "bias_fact": "no attention projection bias",
        "transforms": [
            ("norm.weight stored as OFFSET w, runtime multiplies by (1+w); import ADDS 1.0",
             "T gemma/modeling_gemma.py:68,77; conversion/gemma.py:62-66"),
            ("lm_head.weight skipped on import (tied to embed)",
             "conversion/gemma.py:33-42, :126-134"),
        ],
        "danger": "1+w storage: the stored norm weight is an OFFSET near 0, not a scale. The "
                  "inspector meters only rank-2 tensors (inspect/src/main.rs:553-557) so norms "
                  "do NOT corrupt today's dyn_range/census; but ANY probe that aggregates norm "
                  "tensors (or compares gemma dyn_range to qwen's without adding 1) silently "
                  "misstates dyn_range and the weight census. G3 is unaffected (the +1 is "
                  "outside the consumer columns it absorbs).",
    },
    "gemma2": {
        "hf": ("Gemma2ForCausalLM",),
        "mt": ("gemma2",),
        "attn": "full", "mlp": "glu", "norm": "rms-1pw", "tie": True, "rope": "rotate_half",
        "bias_fact": "attention_bias configurable (gemma2/modeling_gemma2.py:233-244)",
        "transforms": [
            ("norm.weight stored as OFFSET w, runtime multiplies by (1+w); import ADDS 1.0",
             "T gemma2/modeling_gemma2.py:53,62; conversion/gemma.py:112-116"),
            ("lm_head.weight skipped on import", "conversion/gemma.py:101-108, :126-134"),
        ],
        "danger": "Same 1+w norm caveat as gemma. Plus: head_dim 256, sliding_window 4096 "
                  "alternating global/local, attn_logit_softcapping 50 and final_logit_"
                  "softcapping 30 (gemma2/configuration_gemma2.py:72-91). Softcapping is a "
                  "monotone postprocessor of unnormalized scores and does NOT break any gauge; "
                  "the sliding window is runtime-only (mistral caveat).",
    },
    "qwen2": {
        "hf": ("Qwen2ForCausalLM",),
        "mt": ("qwen2",),
        "attn": "full", "mlp": "glu", "norm": "rms-w", "tie": "either", "rope": "rotate_half",
        "bias_fact": "q/k/v_proj bias=True hard-coded (qwen2/modeling_qwen2.py:189-191), "
                     "o/MLP bias=False",
        "transforms": [
            ("NO weight transform on import (identity map — the M1 reference path)",
             "conversion/qwen.py:65-69 (Qwen2Model.modify_tensors)"),
        ],
        "danger": "Reference architecture; nothing extra. RoPE pairs (j, j+d/2) stay apart in "
                  "both artifact and GGUF.",
    },
    "qwen3": {
        "hf": ("Qwen3ForCausalLM",),
        "mt": ("qwen3",),
        "attn": "full", "mlp": "glu", "norm": "rms-w", "tie": "either", "rope": "rotate_half",
        "bias_fact": "attention_bias configurable — False in Qwen3-0.6B/1.7B "
                     "(qwen3/modeling_qwen3.py:225-234)",
        "transforms": [
            ("NO weight transform on import",
             "conversion/qwen.py:154-252 (Qwen3Model extends Qwen2Model)"),
        ],
        "danger": "QK-NORM BREAKS G2. Per-head q_norm/k_norm run AFTER projection, BEFORE RoPE "
                  "(qwen3/modeling_qwen3.py:237-238, :252-257) as weight * (x / rms(x)). The "
                  "denominator is rotation-invariant, but the per-dimension gain is not: a "
                  "projection-output 2-plane rotation on (j, j+d/2) commutes with diag(g) only if "
                  "g_j == g_{j+d/2} for every pair, and trained gains violate that enormously "
                  "(Qwen3-0.6B-Base layer 0: max |g_j - g_{j+64}| = 95.3 on k_norm). Measured "
                  "through the real weights: max ||qnorm(xR) - qnorm(x)R|| / ||x|| = 1.23, i.e. the "
                  "'exact' rotation moves q by 123% of its own scale. G2 is therefore UNAVAILABLE "
                  "here. G1/G3/G5/G7 are unaffected: there is no norm on v/o, and G3/G7 act on "
                  "input columns and the MLP. Reproduce with "
                  "archcheck/test_qknorm_g2.py <snapshot-dir>.",
    },
    "phi3": {
        "hf": ("Phi3ForCausalLM",),
        "mt": ("phi3",),
        "attn": "full", "mlp": "glu", "norm": "rms-w", "tie": False, "rope": "partial",
        "bias_fact": "no attention bias",
        "transforms": [
            ("No weight transform; rope_dimension_count derived from partial_rotary_factor "
             "(0.4 in real phi-3) — only rot_pct of the head dim is rotated; long/short rope "
             "factor tensors emitted (longrope)",
             "conversion/phi.py:145-165; T phi3/configuration_phi3.py:98,120"),
        ],
        "danger": "PARTIAL rotary: G2 pair rotations on the rotated pairs are still exact, but "
                  "the 'G2 is the maximal q/k gauge' claim only holds for the rotated band; the "
                  "unrotated tail has a strictly larger commutant. Any G2 canonicity or "
                  "'distinct frequencies' assertion must be restricted to the rotated band. "
                  "sliding_window default 2047: runtime-only caveat.",
    },
    "deepseek_v2_v3": {
        "hf": ("DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"),
        "mt": ("deepseek_v2", "deepseek_v3"),
        "attn": "moe-full", "mlp": "glu", "norm": "rms-w", "tie": "either", "rope": "rotate_half",
        "bias_fact": "no attention projection bias",
        "transforms": [
            ("MLA: kv_b_proj split into k_b_proj/v_b_proj and k_b_proj TRANSPOSED on import",
             "conversion/deepseek.py:405-433"),
            ("MoE experts merged to 3D layers.{}.mlp.experts.{gate,up,down_proj}.weight from "
             "per-expert 2D mlp.experts.{e}.{gate,up,down_proj}.weight",
             "conversion/deepseek.py:391-408"),
        ],
        "danger": "MLA hides k/v inside shared low-rank projections (no independent per-row value "
                  "subspace), so G1 is UNAVAILABLE even though q/k/v/o-named tensors exist; G2 "
                  "applies only to the q_a/q_b RoPE portion. Expert tensors named "
                  "gate/up/down_proj COLLIDE with dense-family substring keying (ERR, below).",
    },
    "qwen_moe": {
        "hf": ("Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"),
        "mt": ("qwen2_moe", "qwen3_moe"),
        "attn": "moe-full", "mlp": "glu", "norm": "rms-w", "tie": "either", "rope": "rotate_half",
        "bias_fact": "none",
        "transforms": [
            ("Per-expert 2D mlp.experts.{e}.{gate,up,down}_proj.weight merged to 3D on import",
             "conversion/qwen.py:100-151 (Qwen2Moe/Qwen3MoeModel)"),
        ],
        "danger": "Expert tensors named gate/up/down_proj COLLIDE with dense-family substring "
                  "keying. transformers 5.16 also has a FILE-LEVEL fused layout "
                  "(gate_up_proj: [n_experts, 2*ff, hidden], down_proj: [n_experts, hidden, ff]; "
                  "qwen2_moe/modeling_qwen2_moe.py:287-288) — the header probe detects which "
                  "regime is on disk.",
    },
    "mixtral": {
        "hf": ("MixtralForCausalLM",),
        "mt": ("mixtral",),
        "attn": "moe-full", "mlp": "glu", "norm": "rms-w", "tie": "either", "rope": "rotate_half",
        "bias_fact": "none",
        "transforms": [
            ("Experts on disk are w1/w2/w3 (NOT gate/up/down) and DO NOT match the 7 family "
             "patterns — the inspector silently DROPS them from the census",
             "conversion/llama.py:256-295 (block_sparse_moe.experts.{e}.{w1,w2,w3}.weight)"),
        ],
        "danger": "w1/w2/w3 expert weights never enter any family aggregate AND are not counted "
                  "in weights/total (family_of returns None -> inspect/src/main.rs:537-547 "
                  "`continue`): the census undercounts ~half the params with no UNAVAILABLE flag. "
                  "Dense-attention q/k rows DO hit the llama permute path (same register, "
                  "undo_permute=True).",
    },
    "gpt2": {
        "hf": ("GPT2LMHeadModel", "GPT2Model"),
        "mt": ("gpt2",),
        "attn": "full", "mlp": "post-ln-linear", "norm": "ln~", "tie": True, "rope": "none",
        "pe": "absolute",
        "bias_fact": "c_attn/c_proj fused qkv naming; no q_proj/... keys at all",
        "transforms": [("(converter: gpt2.py) q/k/v stored fused as c_attn — no per-family split",
                        "conversion/gpt2.py")],
        "danger": "None of the 7 families exist by name (verified on the cached gpt2 artifacts); "
                  "every gauge is UNAVAILABLE.",
    },
    "mamba": {
        "hf": ("MambaForCausalLM", "MambaLMHeadModel", "Mamba2ForCausalLM",
               "FalconMambaForCausalLM"),
        "mt": ("mamba", "mamba2"),
        "attn": "ssm", "mlp": "ssm", "norm": "rms-w", "tie": False, "rope": "none",
        "pe": "none",
        "bias_fact": "n/a",
        "transforms": [
            ("A_log -> A = -exp(A_log); conv1d squeeze; mamba2 dt_bias -> dt_proj.bias",
             "conversion/mamba.py:83-90, :174-184, :194-199"),
        ],
        "danger": "SSM block has NO q/k/v/o/gate/up/down — zero of the 7 families exist; all "
                  "gauges UNAVAILABLE.",
    },
    "hybrid": {
        "hf": ("Qwen3NextForCausalLM", "JambaForCausalLM", "DeepseekVLHybridForCausalLM"),
        "mt": ("qwen3_next", "jamba"),
        "attn": "moe-full", "mlp": "glu", "norm": "rms-w", "tie": "either", "rope": "rotate_half",
        "bias_fact": "varies",
        "transforms": [
            ("Qwen3Next: in_proj_qkvz re-split/re-order, A_log=-exp, norm.+1 on linear-attn "
             "lognorm layers", "conversion/qwen.py:295-337"),
            ("Jamba: A_log -> A", "conversion/jamba.py:106-112"),
        ],
        "danger": "Hybrids mix an SSM branch with attention on a per-layer schedule "
                  "(full_attention_interval / moe_layer_freq): per-family features exist only on "
                  "attention/GLU layers and the census cannot assume layer-uniform families; "
                  "UNAVAILABLE until per-layer-type mapping is declared.",
    },
}

HF2ARCH: dict[str, str] = {}
MT2ARCH: dict[str, str] = {}
for _aid, _f in ARCHS.items():
    for _h in _f["hf"]:
        HF2ARCH[_h] = _aid
    for _m in _f["mt"]:
        MT2ARCH[_m] = _aid


def fail(msg: str, code: int) -> int:
    print(msg)
    return code


def load_config(target: str) -> tuple[str, dict, str]:
    """Return (config_path, config, hf_dir_or_None)."""
    p = target
    if os.path.isdir(p):
        p = os.path.join(p, "config.json")
    if not os.path.isfile(p):
        raise SystemExit(fail(f"probe: no config at {p!r}", 3))
    with open(p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return p, cfg, os.path.dirname(p) or "."


def read_safetensors_header(model_file: str) -> list[tuple[str, str, list[int]]] | None:
    """Return [(name, dtype, shape)] from the header ONLY; never touch tensor payload.
    None if the file is missing or unreadable (unavailable evidence — not a fact)."""
    try:
        with open(model_file, "rb") as f:
            ln = struct.unpack("<Q", f.read(8))[0]
            if ln == 0 or ln > (1 << 28):
                return None
            hdr = json.loads(f.read(ln))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    out = []
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        try:
            out.append((k, v["dtype"], list(v["shape"])))
        except (TypeError, KeyError):
            continue
    return out or None


def header_class_stats(entries, arch: str) -> dict:
    """Runtime-observable facts from the header only."""
    if entries is None:
        return {}
    names = [n for n, _, _ in entries]
    fam_keys = {f: [n for n in names if f in n] for f in FAMILIES}
    return {
        "family_key_hits": {f: len(v) for f, v in fam_keys.items()},
        "no_qwen_families": all(not v for v in fam_keys.values()),
        "qk_norm_tensors": sorted({n for n in names
                                   if n.endswith(("q_norm.weight", "k_norm.weight"))}),
        "norm_tensors": sorted({n for n in names
                                if n.endswith(("layernorm.weight", "norm.weight", ".ln.weight"))}),
        "rank3_plus": sorted({n for n, _, s in entries if len(s) > 2}),
        "has_bias": any(n.endswith(".bias") for n in names),
    }


def header_keying_issues(entries, families, arch):
    """Header-driven MoE and fused-layout audit against the current boundary-aware scanners."""
    if entries is None:
        return [("WARN", "?", "no model.safetensors header found; byte-level checks unavailable")]
    issues = []
    names = [n for n, _, _ in entries]
    expert_marker = re.compile(r"(^|\.)experts\.|block_sparse_moe\.experts|\.experts\.")
    per_expert = [n for n in names if expert_marker.search(n)
                  and re.search(r"\.experts\.\d+\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$", n)]
    if per_expert:
        issues.append(("INFO", "experts",
                       "boundary-aware scanners separate recognized 2-D expert tensors into "
                       "expert_gate/expert_up/expert_down families"))
    fused = [n for n in names if expert_marker.search(n) and
             ("gate_up_proj" in n or ("down_proj" in n and len(next(s for nn, _, s in entries if nn == n)) > 2))]
    for n in fused[:6]:
        shape = next(s for nn, _, s in entries if nn == n)
        issues.append(("ERR", "experts",
                       f"fused expert tensor {n!r} rank {len(shape)} is explicit UNAVAILABLE; "
                       "the row scanner cannot deaggregate it"))
    if any(re.search(r"\.experts\.\d+\.(w1|w2|w3)\.weight$", n) for n in names):
        issues.append(("INFO", "experts", "Mixtral w1/w2/w3 map to expert gate/down/up families"))
    seen_r = set()
    for n, _, s in entries:
        if len(s) != 2:
            for f in families:
                if f in n:
                    key = ("bias" if n.endswith(".bias") else "rank", f)
                    if key in seen_r:
                        continue
                    seen_r.add(key)
                    msg = (f"rank-1 {f} bias skipped as expected" if n.endswith(".bias") else
                           f"non-2D tensor {n!r} is explicit UNAVAILABLE")
                    issues.append(("WARN", f, msg))
                    break
    return issues


def norm_storage_note(arch: str) -> str:
    a = ARCHS[arch]
    if a["norm"] == "rms-1pw":
        return ("norms STORED AS 1+w OFFSET (runtime multiplies by 1+w) — any dyn_range/census "
                "that reads the stored values as scales is wrong on this family by construction")
    if a["norm"] == "ln~":
        return ("LayerNorm (mean-subtracting): G3's scale-absorption does not hold "
                "(bias inside the norm)")
    return "rms norm stored as true scale; dyn_range/census directly comparable to Qwen2 convention"


def main() -> int:
    if len(sys.argv) != 2:
        return fail("usage: archcheck/probe.py <hf_dir_or_config.json>", 2)
    target = sys.argv[1]
    cfg_path, cfg, hf_dir = load_config(target)

    arch_names = cfg.get("architectures") or []
    model_type = cfg.get("model_type") or (arch_names[0].lower() if arch_names else "")
    arch_id = None
    for cand in arch_names:
        if cand in HF2ARCH:
            arch_id = HF2ARCH[cand]
            break
    if arch_id is None:
        arch_id = MT2ARCH.get(model_type)
    if arch_id is None:
        return fail(
            f"probe: UNAVAILABLE: unknown architecture {arch_names or model_type!r} (fail "
            "closed; ROADMAP policy: never guess transformations) — config " + cfg_path, 1)

    entries = read_safetensors_header(os.path.join(hf_dir, "model.safetensors")) if hf_dir else None
    stats = header_class_stats(entries, arch_id)
    info = ARCHS[arch_id]
    issues = header_keying_issues(entries, FAMILIES, arch_id)

    q_heads = cfg.get("num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads", q_heads)
    hidden_act = cfg.get("hidden_act", "silu")
    tfm = cfg.get("transformers_version", "?")
    rope_theta = cfg.get("rope_theta")

    fam_hits = stats.get("family_key_hits") or {}
    separable_attn = any(fam_hits.get(f, 0) > 0 for f in ("q_proj", "k_proj", "v_proj", "o_proj"))

    # ---- gauge exactness decisions (config-derived + arch facts + header evidence) ----
    gauges = {}
    if info["attn"] in ("full", "moe-full"):
        groups = ("kv-group (GQA, U constant per group)" if kv_heads and q_heads and kv_heads < q_heads
                  else "per-head (MHA, kv==q)")
        g1, g1r = "EXACT", (f"attention present; groups={groups} — value subspace read only by "
                            "V/O, RoPE never touches it")
        if "deepseek" in arch_id:
            g1, g1r = "UNAVAILABLE", ("MLA fuses k/v into shared low-rank projections — no "
                                      "independent V/O value subspace (deepseek_v2 MLA)")
        elif not separable_attn:
            if entries is None:
                g1, g1r = "UNAVAILABLE", ("no model.safetensors header found next to config — "
                                          "separable family tensors cannot be certified from "
                                          "config alone (fail closed; bytes are the evidence)")
            else:
                g1, g1r = "UNAVAILABLE", (f"no separable q/k/v/o family tensors on disk "
                                          f"(header hits {fam_hits}); value subspace is fused "
                                          f"(e.g. GPT-2 c_attn)")
        gauges["G1"] = (g1, g1r)

        rope = info["rope"]
        if rope == "partial":
            pf = cfg.get("partial_rotary_factor") or 1.0
            g2 = "PARTIAL" if pf < 1.0 else "EXACT"
            g2r = (f"RoPE rotate_half pairing present; partial_rotary_factor={pf} -> rotated band "
                   f"{int(round(pf * 100))}%: pair rotations exact there but the 'maximal "
                   "commutant' claim stops at the band edge (unrotated tail duplicates "
                   "frequency 0)")
        elif rope == "mrope":
            g2, g2r = "ADAPTER-REQUIRED", ("mrope_section/multimodal rope: frequency bands are "
                                           "concatenated per section, not the (j,j+d/2) split — "
                                           "raw G2 pair indices are wrong")
        elif rope == "rotate_half":
            if stats.get("qk_norm_tensors"):
                g2, g2r = "UNAVAILABLE", (
                    "per-head QK-norm sits between the projection and RoPE: Qwen3RMSNorm is "
                    "g * (x / rms(x)), and a 2-plane rotation on (j,j+d/2) commutes with diag(g) "
                    "only when g_j == g_{j+d/2} for every pair. Trained gains are nowhere near "
                    "pair-constant, so the pair rotation is NOT exact. Measured on real weights by "
                    "archcheck/test_qknorm_g2.py; promote to EXACT only if that test reports a "
                    "pair-constant gain.")
            else:
                g2, g2r = "EXACT", (f"RoPE rotate_half pairs (j,j+d/2), theta={rope_theta}; distinct "
                                    "per-pair frequencies -> G2 is the maximal q/k gauge")
        else:
            g2, g2r = "UNAVAILABLE", f"no RoPE ({rope!r})"
        gauges["G2"] = (g2, g2r)
    else:
        gauges["G1"] = ("UNAVAILABLE", f"no attention-based self-attention ({info['attn']})")
        gauges["G2"] = ("UNAVAILABLE", f"no RoPE rotary attention ({info['attn']})")

    norm = info["norm"]
    if norm in ("rms-w", "rms-1pw"):
        g3 = "EXACT"
        g3r = ("RMSNorm pre-norm; rms(z) is 0-homogeneous so a positive diagonal on the "
               "consumers' input columns is exact (shared consumer set)"
               + (" — norm stored as 1+w; the diagonal lands on consumer columns, still exact"
                  if norm == "rms-1pw" else ""))
    else:
        g3, g3r = "UNAVAILABLE", ("LayerNorm (mean-subtracting): not scale-invariant, G3's "
                                  "positive-diagonal absorption is not exact")
    gauges["G3"] = (g3, g3r)

    tied = cfg.get("tie_word_embeddings", info["tie"] if isinstance(info["tie"], bool) else False)
    g5r = ("scale embed·c, Wo·c, Wd·c and lm_head/c^-1; "
           + ("tie_word_embeddings=True -> implementable only by breaking the tie "
              "(lm_head is tied to embed, M1 ships untied)"
              if tied else "checkpoint is untied -> direct"))
    g5st = "EXACT"
    if info.get("pe") == "absolute" and info["attn"] != "ssm":
        g5st, g5r = "PARTIAL", (
            "shipped M1 G5 moves only embed/Wo/Wd, which is exact only for RoPE-only "
            "position encoding; GPT-2 adds an additive learned position embedding "
            "(wpe: modeling_gpt2.py:493, :576-577), so embed·c without wpe·c is NOT an "
            "exact stream scale — scale wpe with embed (4-tensor gauge) to make G5 exact")
    elif info["attn"] == "ssm":
        g5r += "; SSM stack: no Wo/Wd writers, embed·c + lm_head/c^-1 suffice"
    gauges["G5"] = (g5st, g5r)

    if info["mlp"] == "glu":
        g7, g7r = "EXACT", (f"SwiGLU ({hidden_act!r}) gate/up split: scale up_proj row j by "
                            "c_j and down_proj column j by c_j^-1; the MLP output is unchanged")
        if "deepseek" in arch_id:
            g7r += " on dense/shared-expert GLU layers; routed experts still need a gauge adapter"
    else:
        g7, g7r = "UNAVAILABLE", (f"MLP is {info['mlp']!r} — no gate/up split "
                                  "(G7 requires a GLU variant)")
    gauges["G7"] = (g7, g7r)

    # ---- per-family static-feature trust (evidence-driven) ----
    err_fams = {f for s, f, _ in issues if s == "ERR"}
    fused_experts = "experts" in err_fams
    fam_trust = {}
    for f in FAMILIES:
        reasons = []
        if entries is None:
            reasons.append("no model.safetensors header next to config; byte evidence unavailable")
        if stats.get("no_qwen_families") and arch_id != "gpt2":
            reasons.append("no family key matched any safetensors name")
        if arch_id == "mamba":
            reasons.append("SSM block, no attention/GLU families")
        if arch_id == "gpt2":
            reasons.append("GPT-2 uses c_attn/c_proj/c_fc naming")
        if f in err_fams:
            reasons.append("header evidence marks this family unavailable")
        if f in ("gate_proj", "up_proj", "down_proj") and fused_experts:
            reasons.append("fused expert stack unavailable; dense family remains separate")
        if f in ("gate_proj", "up_proj", "down_proj") and arch_id in ("deepseek_v2_v3", "qwen_moe", "mixtral", "hybrid"):
            reasons.append("scanner separates expert families, but the gauge/canonicalizer adapter is unavailable")
        if arch_id == "hybrid":
            reasons.append("hybrid layer schedule not declared")
        fam_trust[f] = "UNAVAILABLE: " + "; ".join(reasons) if reasons else "EXACT"

    # -------- report --------
    print(f"archcheck/probe.py  target={target}")
    print(f"config            = {cfg_path}  (transformers_version={tfm})")
    print(f"architectures     = {arch_names}  model_type={model_type}")
    print(f"classified as     = {arch_id}")
    print(f"heads             = q={q_heads} kv={kv_heads} hidden={cfg.get('hidden_size')} "
          f"layers={cfg.get('num_hidden_layers')} rope_theta={rope_theta}")
    print()
    print("GAUGE FAMILIES (what Theseus would claim EXACT / PARTIAL / UNAVAILABLE):")
    for g in ("G1", "G2", "G3", "G5", "G7"):
        st, why = gauges[g]
        print(f"  {g:4} {st:<15} {why}")
    print()
    print("PER-FAMILY STATIC FEATURES (inspector aggregates on the artifact's own bytes):")
    for f in FAMILIES:
        print(f"  {f:12} {fam_trust[f]}")
    print()
    print("ARCH-LEVEL IMPORT/STORAGE FACTS (from source, file:line):")
    for t, anchor in info["transforms"]:
        print(f"  - {t}")
        print(f"      [{anchor}]")
    print(f"  norm storage:  {norm_storage_note(arch_id)}")
    print(f"  attn proj bias: {info['bias_fact']}  (has .bias tensors in header: {stats.get('has_bias')})")
    if info["danger"]:
        print(f"  danger:        {info['danger']}")
    if stats.get("qk_norm_tensors"):
        print(f"  qk_norm seen in header: {stats['qk_norm_tensors'][:4]}{' ...' if len(stats['qk_norm_tensors']) > 4 else ''}")
    print()
    print("HEADER EVIDENCE:")
    if entries is None:
        print("  UNAVAILABLE: no model.safetensors header readable next to the config — "
              "bytes-level collision checks skipped (still fail-closed on arch facts)")
    else:
        print(f"  {len(entries)} tensors; family key hits = {stats['family_key_hits']}")
        if stats["rank3_plus"]:
            print(f"  rank>2 tensors (inspector skips -> absent from census): {stats['rank3_plus'][:6]}")
    for sev, fam, msg in issues:
        print(f"  [{sev}] {fam}: {msg}")

    has_err = any(s == "ERR" for s, _, _ in issues)
    print()
    unavailable = any(v.startswith("UNAVAILABLE") for v in fam_trust.values()) or any(
        v.startswith("UNAVAILABLE") for v, _ in gauges.values())
    if has_err or unavailable:
        print("VERDICT: FAIL CLOSED (exit 1) — at least one gauge family or per-family feature "
              "is UNAVAILABLE / mis-keyed on this architecture; do not use raw numbers.")
        return 1
    if any(v.startswith("PARTIAL") for v, _ in gauges.values()):
        print("VERDICT: PASS WITH ADAPTER (exit 0) — gauge families exact but not maximal "
              "(partial rotary / sectioned rope); numbers safe only with the adapter applied.")
        return 0
    print("VERDICT: OK (exit 0) — all gauge families exact and per-family features trustworthy "
          "under the stated conventions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
