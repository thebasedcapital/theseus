#!/usr/bin/env python3
"""archcheck/mk_synthetic_fixtures.py — regenerate the header fixtures used to verify
`probe.py`'s fail-closed behaviour on architectures that are NOT cached on this box.

stdlib only. Writes, per arch, a `config.json` plus a `model.safetensors` **header only**
(8-byte u64 length + JSON; no tensor payload is ever written). `probe.py` reads only the
header, so these are fully sufficient to exercise classification and the MoE/fused/partial
paths. Usage:  mk_synthetic_fixtures.py [out_dir]   (default /tmp/archcheck_fixtures)

Fixtures and the verdict each must produce (run `probe.py <dir>` to check):
  dsv3     DeepSeek-V3 MLA + per-expert 2-D mlp.experts.<e>.{gate,up,down}_proj  -> ERR, exit 1
  mixtral  dense q/k/v/o + block_sparse_moe.experts.<e>.{w1,w2,w3}              -> silent drop, exit 1
  q3moe    Qwen3-MoE fused 3-D mlp.experts.gate_up_proj/down_proj               -> ERR, exit 1
  phi3     Phi-3 partial_rotary_factor=0.5                                       -> G2 PARTIAL, exit 0
  mamba    Mamba2 config only (no header)                                        -> exit 1 (fail closed)
  hybrid_only  Qwen3Next config only (no header)                                 -> exit 1 (fail closed)
  unknown   unmapped architecture                                                 -> exit 1 (UNAVAILABLE)
"""
import json
import os
import struct
import sys


def write_hdr(d: str, cfg: dict, tensors: dict) -> None:
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    hdr = {k: {"dtype": dt, "shape": s} for k, (dt, s) in tensors.items()}
    blob = json.dumps(hdr).encode()
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/archcheck_fixtures"

    # -- DeepSeek-V3 style: MLA dense attention + per-expert 2-D gate/up/down --
    dsbase = {
        "architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3",
        "hidden_size": 7168, "intermediate_size": 18432, "num_hidden_layers": 4,
        "num_attention_heads": 128, "num_key_value_heads": 128, "hidden_act": "silu",
        "q_lora_rank": 1536, "kv_lora_rank": 512, "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64, "v_head_dim": 128, "n_routed_experts": 256,
        "n_shared_experts": 1, "num_experts_per_tok": 8, "rope_theta": 10000,
        "tie_word_embeddings": False, "rms_norm_eps": 1e-6,
    }
    dst = {}
    for l in range(4):
        for f in ("q_proj", "k_proj", "v_proj", "o_proj"):
            dst[f"model.layers.{l}.self_attn.{f}.weight"] = ("F32", [7168, 7168])
        for e in (0, 1, 2, 3):
            for f in ("gate_proj", "up_proj", "down_proj"):
                shape = [7168, 18432] if f != "down_proj" else [18432, 7168]
                dst[f"model.layers.{l}.mlp.experts.{e}.{f}.weight"] = ("F32", shape)
    dst["model.norm.weight"] = ("F32", [7168])
    write_hdr(os.path.join(out, "dsv3"), {**dsbase, "first_k_dense_replace": 1}, dst)

    # -- Mixtral: dense q/k/v/o + block_sparse_moe w1/w2/w3 (no family match) --
    mix = {
        "architectures": ["MixtralForCausalLM"], "model_type": "mixtral",
        "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 4,
        "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_act": "silu",
        "num_local_experts": 8, "num_experts_per_tok": 2, "rope_theta": 1000000,
        "tie_word_embeddings": False, "rms_norm_eps": 1e-5,
    }
    mxt = {}
    for l in range(4):
        for f in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mxt[f"model.layers.{l}.self_attn.{f}.weight"] = ("F32", [4096, 4096])
        for f in ("gate_proj", "up_proj", "down_proj"):
            shape = [4096, 14336] if f != "down_proj" else [14336, 4096]
            mxt[f"model.layers.{l}.mlp.{f}.weight"] = ("F32", shape)
        for e in (0, 1, 2):
            for f in ("w1", "w2", "w3"):
                shape = [4096, 14336] if f in ("w1", "w3") else [14336, 4096]
                mxt[f"model.layers.{l}.block_sparse_moe.experts.{e}.{f}.weight"] = (
                    "F32", shape)
    write_hdr(os.path.join(out, "mixtral"), mix, mxt)

    # -- Qwen3-MoE fused 3-D regime (transformers 5.16 on-disk layout) --
    q3m = {
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "hidden_size": 1024, "moe_intermediate_size": 2048, "num_hidden_layers": 4,
        "num_attention_heads": 16, "num_key_value_heads": 8, "hidden_act": "silu",
        "num_experts": 16, "num_experts_per_tok": 4, "rope_theta": 1000000,
        "tie_word_embeddings": False, "rms_norm_eps": 1e-6,
    }
    q3t = {}
    for l in range(4):
        for f in ("q_proj", "k_proj", "v_proj", "o_proj"):
            q3t[f"model.layers.{l}.self_attn.{f}.weight"] = ("F32", [1024, 1024])
        q3t[f"model.layers.{l}.mlp.experts.gate_up_proj.weight"] = ("F32", [16, 4096, 1024])
        q3t[f"model.layers.{l}.mlp.experts.down_proj.weight"] = ("F32", [16, 1024, 2048])
    q3t["model.norm.weight"] = ("F32", [1024])
    write_hdr(os.path.join(out, "q3moe"), q3m, q3t)

    # -- Phi-3 partial rotary (header voice incomplete would be wrong here — it IS complete) --
    phi = {
        "architectures": ["Phi3ForCausalLM"], "model_type": "phi3",
        "hidden_size": 3072, "intermediate_size": 8192, "num_hidden_layers": 4,
        "num_attention_heads": 32, "num_key_value_heads": 32, "hidden_act": "silu",
        "partial_rotary_factor": 0.5, "rope_theta": 10000, "sliding_window": 2047,
        "tie_word_embeddings": False, "rms_norm_eps": 1e-5,
    }
    phit = {}
    for l in range(4):
        for f in ("q_proj", "k_proj", "v_proj", "o_proj"):
            phit[f"model.layers.{l}.self_attn.{f}.weight"] = ("F32", [3072, 3072])
        for f in ("gate_proj", "up_proj", "down_proj"):
            shape = [3072, 8192] if f != "down_proj" else [8192, 3072]
            phit[f"model.layers.{l}.mlp.{f}.weight"] = ("F32", shape)
    write_hdr(os.path.join(out, "phi3"), phi, phit)

    # -- Mamba2 and Qwen3Next: config-only (probe must fail closed on the missing header) --
    os.makedirs(os.path.join(out, "mamba"), exist_ok=True)
    json.dump({"architectures": ["Mamba2ForCausalLM"], "model_type": "mamba2",
               "hidden_size": 1024, "state_size": 128, "time_step_rank": 64,
               "conv_kernel": 4, "num_hidden_layers": 4, "expand": 2, "hidden_act": "silu",
               "tie_word_embeddings": False, "rms_norm_eps": 1e-5},
              open(os.path.join(out, "mamba", "config.json"), "w"))
    os.makedirs(os.path.join(out, "hybrid_only"), exist_ok=True)
    json.dump({"architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
               "hidden_size": 1024, "moe_intermediate_size": 2048, "num_hidden_layers": 4,
               "num_attention_heads": 16, "num_key_value_heads": 8, "hidden_act": "silu",
               "num_experts": 16, "num_experts_per_tok": 4, "rope_theta": 100000,
               "full_attention_interval": 4, "moe_layer_freq": 2,
               "tie_word_embeddings": False, "rms_norm_eps": 1e-6},
              open(os.path.join(out, "hybrid_only", "config.json"), "w"))
    os.makedirs(os.path.join(out, "unknown"), exist_ok=True)
    json.dump({"architectures": ["MegaCorpMysteryForCausalLM"], "model_type": "megacorp_mystery",
               "hidden_size": 1024, "num_hidden_layers": 2, "num_attention_heads": 8},
              open(os.path.join(out, "unknown", "config.json"), "w"))

    print(f"fixtures written under {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
