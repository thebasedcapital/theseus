# ARCHITECTURES — are Theseus's gauges and static features correct beyond Qwen2?

Owner: ArchAudit. Companion to `archcheck/probe.py` (the executable classification oracle below).
This is the cross-architecture correctness audit (goal 5 of the breadth push): every exactness
claim is either verified against the source trees on this box or marked `UNAVAILABLE` (fail
closed, invariant I8 — `null`/missing evidence is never a PASS).

Source anchors use two roots:

| root | path |
|---|---|
| L | `/home/admin/tools/llama.cpp-cuda-src`            (conversion/*.py, ggml/src/ggml-common.h) |
| T | `<site-packages>/transformers/models/<arch>/` (whatever environment holds transformers 5.16) |
| I | `/home/admin/theseus/inspect/src/main.rs`          (the static-feature inspector) |

Semantics used below: gauge families are the five *knottable* exact transforms from
`M1_NOTES.md` §2 / `m1/gauge.py` — **G1** value-subspace basis change (per kv-group under GQA),
**G2** RoPE-plane rotations (pairs `(j, j+d/2)`, distinct frequencies for the claimed maximality),
**G3** RMSNorm scale absorption (pre-norm with a shared consumer set), **G5** global stream scale
(tie situation matters), **G7** SwiGLU up-branch diagonal (needs a gate/up split). G4/G6 are
structural permutations (valid wherever the row-block families exist) and are not the risk surface.

the five static features the inspector reports per family — `q4_block_mse` (+`_pooled`),
`dyn_range_log10`, `row_energy_imbalance`, `amax_over_rms`, `frac_below_f16_normal`/`below_f16`,
`weights` — are computed from the artifact's own safetensors bytes, row-by-row (`Acc::feed_row`
I main.rs:314, driven per 2-D tensor row at :575-588). Which of them survive on a given
architecture is the census question this file answers.

---

## 0. MoE keying status

**The original silent-keying bug is fixed in both Rust scanners.** Classification now matches
exact dot-separated components:

- dense `q_proj` through `down_proj` remain the original seven families;
- 2-D `experts.<e>.gate_proj/up_proj/down_proj` map to
  `expert_gate/expert_up/expert_down`;
- Mixtral `w1/w2/w3` map to expert gate/down/up semantics;
- llama.cpp per-expert `ffn_gate.<e>/ffn_down.<e>/ffn_up.<e>` map to expert families;
- fused rank-3 HF tensors and GGUF `ffn_*_exps` stacks are explicit `UNAVAILABLE`, never merged
  into dense families or silently dropped;
- dense fused QKV remains `UNAVAILABLE`.

Trusted 2-D expert tensors enter expert-family and total statistics. Quant/export preflight uses
the worst trusted dense or expert family. Adaptation remains unavailable on MoE until an
operation-specific calibration exists.

Remaining limits: fused expert stacks are visible but unmetered; Gemma `1+w` norm storage still
needs an explicit convention if norms enter the census; hybrids need a layer schedule; GPT-2 and
Mamba have no compatible seven-family surface and fail closed.

---

## 1. The table (14 rows; >= 8 required)

Legend: `E` exact · `P` partial (exact for a band, not maximal) · `U` unavailable · `A` needs an
adapter · `n/a` not applicable. "import" = weight transform applied by `convert_hf_to_gguf.py`;
"norm" = RMSNorm storage convention (`1+w` = stored as an offset). `–` = fine, no corruption.

| arch (HF class) | attn/MLP | G1 | G2 | G3 | G5 | G7 | import transforms | norm | census verdict |
|---|---|---|---|---|---|---|---|---|---|
| **llama / llama-2** (LlamaForCausalLM) | full / SwiGLU | E | E | E | E* | E | q/k ROWS REPERMUTED (`undo_permute=True`) | scale | – (row-order note) |
| **llama-3.x** (LlamaForCausalLM) | full / SwiGLU | E | E | E | E* | E | same as llama (3.1/3.3 add longrope factors; kv=8 GQA in 8B) | scale | – |
| **mistral** (MistralForCausalLM) | full / SwiGLU | E | E | E | E | E | HF→LlamaModel: q/k REPERMUTED; native mistral-format: none | scale | – (sliding window runtime-only) |
| **gemma** (GemmaForCausalLM) | full / SwiGLU | E | E | E | E* | E | `norm.weight += 1`; lm_head skipped | **1+w offset** | norm probe must add 1 |
| **gemma-2** (Gemma2ForCausalLM) | full / SwiGLU | E | E | E | E* | E | `norm.weight += 1`; lm_head skipped | **1+w offset** | norm probe must add 1 |
| **qwen2** (Qwen2ForCausalLM) | full / SwiGLU | E | E | E | E* | E | **NONE (identity)** — the M1 reference | scale | – (the baseline) |
| **qwen3** (Qwen3ForCausalLM) | full / SwiGLU | E | **U** | E | E* | E | **NONE** | scale | **qk-norm breaks G2** (measured err 1.23·‖x‖⁻¹; probe fails closed) |
| **phi-3** (Phi3ForCausalLM) | full / SwiGLU | E | **P** | E | E | E | none (rope_dim = rot_pct of head dim; longrope factor tensors) | scale | G2 band caveat |
| **deepseek v2/v3** (DeepseekV2/3ForCausalLM) | MLA / MoE | **U** (MLA) | E† | E | E | E‡ | kv_b split+transposed; experts merged to 3D | scale | 2-D experts separated; fused stacks unavailable |
| **qwen2moe / qwen3moe** | full / MoE | E | **E / U** | E | E | E‡ | experts merged to 3D | scale | qwen3moe has per-head q/k norm (T qwen3_moe/modeling_qwen3_moe.py:147-148, :162-163) → G2 UNAVAILABLE, same as qwen3 |
| **mixtral** (MixtralForCausalLM) | full / MoE(w1w2w3) | E | E | E | E | **U‡** | experts w1/w2/w3 merged to 3D; q/k permuted | scale | w1/w2/w3 separated; merged stacks unavailable |
| **mamba / mamba2** | SSM / SSM | U | U | E(n/t) | E | U | A_log→A, conv1d squeeze, dt_bias rename | scale | no families at all |
| **qwen3next / jamba** (Qwen3NextForCausalLM, JambaForCausalLM) | hybrid / hybrid | U‡ | E† | E | E* | E‡ | in_proj_qkvz re-split, A_log→A, norm+1 (lognorm) | scale (+1+w lognorm) | **census wrong until layer-type map** |
| **gpt2** (GPT2LMHeadModel) | full / LN-MLP | U | U | **U** (LayerNorm) | **P** | U | none (c_attn fused) | LayerNorm | no qwen families by name |

\* G5 is exact for any incl. tied arch, but a **tied** checkpoint (gemma/gemma2/gemma3 tie
embed↔lm_head and skip lm_head on import; qwen2.5-0.5B / many llama-3.x tie by config) is only
expressible by breaking the tie (M1 ships untied). `E*` = exact, tie-break required.
† G2 on DeepSeek applies to the q_a/q_b RoPE portion (MLA keeps per-head rope on query/key
projections while value/key are shared low-rank — no G1). On hybrids (qwen3next/jamba) the
attention layers occupy a strict subset of the block schedule, so G1/G2 are exact only on those
layers and the family census is invalid until the layer-type map is declared — the row above's
per-layer entries are statements only where the map is known, and the whole-artifact probe
verdict fails closed.
‡ routed-expert `up_proj` scaling is exact on GLU layers but is meaningless until the MoE keying
adapter lands; on Mixtral the w1/w2/w3 tensors are dropped, so `gate/up/down` cannot be trusted
even though the gauges are defined.
**gpt2's G5 is `P` (PARTIAL), not `E`:** the shipped M1 G5 moves only `embed`/`Wo`/`Wd`, which is
an exact stream scale only for RoPE-only position encoding; GPT-2 has an additive learned `wpe`
added into `embed` at the input (T gpt2/modeling_gpt2.py:493, :576-577), so `embed·c` without
`wpe·c` is not exact until the gauge moves the 4-tensor set (see §2 G5).

---

## 2. What each exactness column depends on (evidence)

### G1 — value-subspace basis change
Requires a softmax-attention read of a per-head value subspace by `V`/`O`, with RoPE never
touching it, and — under GQA — a single `U` per kv-group (M1_NOTES.md §2). Every row above that
is not `U` meets this from the named `q_proj/k_proj/v_proj/o_proj` families.
- **deepseek v2/v3 are G1-UNAVAILABLE by design**: MLA replaces the per-head `k/v/o` with shared
  low-rank `kv_b_proj` and `q_b_proj` (the converter splits and transposes them on import,
  L conversion/deepseek.py:415-431; weight layout `kv_b = [n_kv,(qk_nope+v),n]` split along
  dim 1). There is no independent per-row value subspace to rotate.
- **gpt2 is G1-UNAVAILABLE**: no separable q/k/v/o tensors exist (`c_attn` fused QKV,
  T gpt2/modeling_gpt2.py), so there is nothing to gauge by family name.

### G2 — RoPE-plane rotations
Requires the `rotate_half` pairing convention (pairs `(j, j+d/2)`) and, for the *maximality*
claim, distinct per-pair frequencies.
- Pairing: qwen2 `rotate_half` splits `x[..., :d/2]` and `x[..., d/2:]` and recombines
  `(-x2, x1)` (T qwen2/modeling_qwen2.py:105-107, applied at :133-134); the same convention in
  gemma (T gemma/modeling_gemma.py:135), gemma2 (:120), qwen3 (:121), llama/mistral/phi3 (all
  inherit the same rope util). So the pair structure is identical everywhere RoPE is used.
- Distinct frequencies: `inv_freq = 1/base**(arange(0,dim,2)/dim)` (T qwen2/modeling_qwen2.py:86)
  — each pair gets a distinct angle; the G2 = maximal-commutant result (Schur) is then sound.
- **phi-3 is G2-PARTIAL**: real phi-3 configs set `partial_rotary_factor` (T phi3/configuration_phi3.py:98,120;
  L conversion/phi.py:153-154 consumes it into `rope_dims`, :164 writes `rope_dimension_count`). Pairs beyond the rotated band
  duplicate frequency 0 → the unrotated tail has a strictly larger commutant, so "G2 is maximal"
  and any G2 canonicity claim must be restricted to the rotated band. Individual pair rotations
  in the unrotated band remain exact (any equal q/k orthogonal transform commutes).
- **qwen3 has per-head-dim QK-NORM** `q_norm/k_norm` after projection, before RoPE
  (T qwen3/modeling_qwen3.py:237-238, applied :252-253). **G2 is NOT exact** — an earlier revision
  of this audit claimed "rotation-invariant per token, so G2 stays exact", which considered only the
  `rms(x)` denominator and ignored the learned per-dimension gain. `Qwen3RMSNorm(x) = g * (x/rms(x))`,
  and a projection-output 2-plane rotation on `(j, j+hd/2)` commutes with `diag(g)` only if
  `g_j == g_{j+hd/2}` for every pair. Measured on Qwen3-0.6B-Base: `max|g_j - g_{j+64}| = 95.3`
  (k_norm, layer 0), and sampling 512 fp64 vectors through the real weights gives
  `max‖qnorm(xR) - qnorm(x)R‖ / ‖x‖ = 1.229`. `archcheck/test_qknorm_g2.py` reproduces this;
  `archcheck/probe.py` reports G2 UNAVAILABLE and `m1/gauge.g2_rope_pairs` refuses to run.
  G1/G3/G5/G7 are unaffected (no norm on v/o; G3 and G7 act on input columns and the MLP).
- **mRoPE** (Qwen2/3-VL, multimodal) sections frequencies (`mrope_section`); raw G2 pair indices
  `(j, j+d/2)` are wrong there → `A` adapter required, never silently applied.
- import question: llama-family archs REPERMUTE q/k rows on import (see §3); the pair relations
  are re-tenanted but the aggregate static features (row × column stats) are invariant to row
  permutation, so no static number is corrupted — only any per-row/GGUF-side read needs the
  frame restored.

### G3 — RMSNorm scale absorption
Requires a scale-invariant (mean-free) pre-norm: `rms(z)=z/sqrt(mean z²+eps)`.
- qwen2/llama/mistral/gemma/gemma2/qwen3/phi3/deepseek/mixtral all use RMSNorm-style pre-norms:
  exact. Gemma stores the norm weight as offset and multiplies by `1+w` at runtime
  (T gemma/modeling_gemma.py:68,77; gemma2/modeling_gemma2.py:53,62) — G3's diagonal lands on
  consumer columns *after* the +1, so it is still exact; only any norm-family census must add 1.
- **gpt2 is G3-UNAVAILABLE**: LayerNorm subtracts the mean (bias inside the norm), so a positive
  diagonal on consumers is not a symmetry (T gpt2/modeling_gpt2.py).

### G5 — global stream scale
Exact whenever you may edit the writers (`embed`, `Wo`, `Wd`) and the token reader (`lm_head`):
same 0-homogeneity argument as G3. Tied checkpoints (gemma, gemma2, gemma3, qwen2.5-0.5B, many
llama-3.x small) require breaking the tie; M1 ships untied (M1_NOTES.md §2 G5).
This is **exact only if the position encoding scales with the stream**. Every arch in the table
except GPT-2 (and mamba, which has no position encoding) uses RoPE-only position encoding — the
`embed·c` move scales the whole layer-0 input. GPT-2 adds a learned `wpe` into `embed` at the
input (`hidden_states = inputs_embeds + position_embeds`, T gpt2/modeling_gpt2.py:493, :576-577),
so the shipped 3-tensor move is not an exact stream scale there: the exact variant must move the
4-tensor set `(embed, wpe, Wo, Wd)`. Mamba has no `Wo`/`Wd`; `embed·c + lm_head/c^-1` suffices.

### G7 — SwiGLU up-branch diagonal
Scale `up_proj` row $j$ by $c_j$ and `down_proj` column $j$ by $c_j^{-1}$. This leaves
`down_proj(silu(gate) ⊙ up)` unchanged for any elementwise-activation GLU.
- **gpt2/mamba are G7-UNAVAILABLE** (no gate/up split at all).
- Mixtral's routed experts (w1/w2/w3) are defined but the keys never reach a family (R1) —
  unusable until keying is fixed.

---

## 3. Which architectures TRANSFORM weights at import (evidence)

These are the per-arch `modify_tensors` bodies in the converter — the single most important
thing to get right before trusting any *GGUF-side* number (the static features themselves are
computed on the HF artifact, so they are only affected when a probe reads GGUF bytes instead).

| arch | import behaves | anchor |
|---|---|---|
| qwen2 / qwen3 (dense) | **identity — nothing transformed** (M1's verified path) | L conversion/qwen.py:65-69, :154-252 |
| llama, llama-2/3.x, HF-mistral, mixtral | **q/k_proj rows REPERMUTED** (`undo_permute=True`; pairs `(j,j+d/2)` d/2-apart become adjacent; also reorders `q_proj.bias`/`k_proj.bias`) | L conversion/llama.py:33 (flag), :163-168 (permute def), :244-249 (apply); :19-21 (register: LlamaForCausalLM/MistralForCausalLM/MixtralForCausalLM → LlamaModel) |
| mistral native `--mistral-format` | `undo_permute=False` — community format already in ggml order | L conversion/mistral.py:28-29 |
| llama-4 | `undo_permute=False` — no q/k reorder | L conversion/llama.py:362 |
| gemma / gemma-2 / gemma-3 | `norm.weight += 1` (offset→scale); `lm_head.weight` dropped (tied) | L conversion/gemma.py:62-66, :112-116; gemma3 `norm_shift` gemma.py:124 |
| deepseek v1 | q/k permute (own copy) | L conversion/deepseek.py:151-165 |
| deepseek v2/v3 | kv_b split+transpose (MLA), experts merged per-layer | L conversion/deepseek.py:386-411 (experts), :415-431 (kv_b) |
| qwen2/3-MoE | per-expert `mlp.experts.<e>.*proj` merged to 3D | L conversion/qwen.py:110-139 (per-expert merge), :98-109 (fused gate_up_proj) |
| mixtral | `block_sparse_moe.experts.<e>.{w1,w2,w3}` merged to 3D | L conversion/llama.py:250-276 (w1/w2/w3 merge), :340-344 (residual check) |
| plamo | `shuffle_attn_q_weight`/`shuffle_attn_output_weight` (GQA broadcast reorder) | L conversion/plamo.py:34-56 |
| phi-3 | none (rope_dim from partial factor) | L conversion/phi.py:145-165 |
| mamba / mamba2 | `A_log→A=−exp(·)`, conv1d squeeze, `dt_bias→dt_proj.bias` | L conversion/mamba.py:83-85 (A_log), :87-89 (conv1d squeeze), :174-175 (dt_bias), :194-196 (mamba2 A_log) |
| jamba (hybrid) | `A_log→A` (attention layers remain) | L conversion/jamba.py:106-112 |
| qwen3next (hybrid) | `in_proj_qkvz` re-split/re-order, `A_log`, norm+1 | L conversion/qwen.py:296-303 (A_log/dt_bias/conv1d/norm+1), :305-337 (in_proj_qkvz) |

Consequence: **no weight transform at all** is currently verified only for qwen2-class and
phi-3-class dense archs; everything else either reorders q/k rows, shifts norms, or reshapes
experts. The probe prints these per arch so a user sees the transform before trusting a GGUF-side
number.

### Quant block layout (what `q4_block_mse` actually models)
`q4_block_mse` = mean per-tensor ratio `Σ_blocks amax²·n/(12·7²·Σw²)` with BLOCK=32 contiguous
(I main.rs:16-18; the distinction from `_pooled` is the PIPELINE_FAILURES #11 convention trap).
GGML lays quantization blocks along the fastest (in_feature) dim of the transposed gguf tensor;
the 32-element unit matches:
- Q4_0: `block_q4_0` QK4_0=32, one half-scale + 4-bit quants (L ggml/src/ggml-common.h:184-189)
- Q8_0: `block_q8_0` QK8_0=32, int8 quants (:241-246)
- Q4_K: super-block QK_K=256 split as "8 blocks of 32 elements each" (:89, :313-328)
- Q5_K: 8×32 (comment :330-331, struct :333-346); Q3_K: 16×16 (:305-311); Q6_K: 16×16 (:352-358)
The proxy block granularity is architecture-independent. Recognized 2-D expert layouts are now
keyed and metered separately. Fused rank-3 MoE storage remains explicit `UNAVAILABLE` because the
row scanner cannot deaggregate the expert stack.

---

## 4. What would break M1's published results (Qwen2-only facts vs. transferable facts)

Published claims in `CLAIMS.md` rest on Qwen2.5-0.5B measurements. Which survive a new
architecture?

| Claim | Qwen2-only dependency | Breaks if you point it at… |
|---|---|---|
| K-1 (>= five exact gauges) | **Qwen2 is the rare arch with NO import transform**, so gauge↔quant reasoning never crossed a q/k permute or a `1+w` norm shift | llama/mistral/mixtral/deepseek (q/k permute; gauges exact in HF frame, numbers still exact through the identical import — safe if the pipeline never reads GGUF rows); gemma (norm offset — gauges fine, any norm census wrong); deepseek (G1 vanishes — MLA). The five-gauge *set* is not universal: Mamba has two; GPT-2 has none by name. |
| K-2 (export format is an op; census predicts f16 damage) | calibrated on dense Qwen | trusted 2-D MoE expert families are now included; fused stacks remain unavailable; Gemma norm-inclusive census still needs a storage-convention adapter. |
| K-4 (quantization reserve differs by gauge state) | Q8 uses fitted v3 at n=20; Q4 remains v2 provisional | base rates and thresholds still need per-architecture evidence. |
| K-6 (static preflight) | Q8 v3 is in-sample Qwen calibration | MoE adaptation is unavailable until separately calibrated; quant/export can use trusted expert families. |
| K-5 / K-10 (canonicalizer) | Qwen family keys and scale norms | MoE gauge transforms still require an architecture adapter even though scan keying is fixed. |
| G2 maximality | distinct per-pair frequencies | phi-3 (partial rotary) and mRoPE VLM archs: the "maximal commutant" and pair indices are wrong beyond the rotated band. |

Bottom line: **every result that is "the pipeline is exact end-to-end" transfers only to dense,
clean-keyed, scale-normed, rotate-half RoPE archs (qwen3, and modulo tie-breaking the
llama/mistral family).** Everything MoE, gemma-norm-inclusive, partial-rotary, or hybrid needs an
adapter before its numbers are trustworthy. Qwen2.5-0.5B's measured numbers themselves are not
endangered (they were measured on the artifact, under its own import path, with identity import
I main.rs / CLAIMS K-2 pipeline control `m1/check_gguf_layout.py`); the risk is claiming they
generalize.

---

## 5. Prioritized remaining adapters

1. **Fused MoE stack metering.** Add exact rank-3 `ffn_*_exps` / `gate_up_proj` handling without
   pretending a fused gate/up tensor is separable when its layout does not prove the split.
2. **Norm-storage convention field.** Stamp true-scale versus `1+w` offset for Gemma and hybrid
   lognorm layers.
3. **G2 pairing pin.** Carry partial rotary and mRoPE section metadata and restrict rotations to
   the validated band.
4. **q/k import row-order declaration.** Stamp llama/mistral/deepseek-v1/plamo reorder behavior.
5. **Hybrid layer schedule.** Aggregate attention and SSM blocks separately.

The old highest-priority keying bug is closed. Fused expert metering is now the first missing
adapter because the scanners expose it explicitly rather than returning a plausible wrong total.

---

## 6. How probe.py encodes this

`archcheck/probe.py <hf_dir_or_config.json>`:
- parses `config.json` + the safetensors **header only** (8-byte u64 length + JSON; never seeks
  into tensor payload) — no GPU, no torch, stdlib only;
- classifies into the table above (unknown arch ⇒ exit 1, UNAVAILABLE, never guesses);
- prints per-gauge status (EXACT/PARTIAL/UNAVAILABLE + reason), per-family static-feature trust,
  the import/storage facts with file:line, and the header evidence (family key hits, expert-name
  collisions → ERR/fail closed, w1/w2/w3 drop, rank>2 skips, qk-norm presence);
- exit 0 only when every EXACT claim holds on this artifact; otherwise exit 1 with reasons.

Verified on this box against the cached artifacts: `Qwen/Qwen2.5-0.5B` (exit 0, all EXACT),
`Qwen/Qwen3-0.6B-Base` (exit 0), `gpt2` and `gpt2-medium` (exit 1, fail closed on every gauge —
correct: GPT-2 has none of the seven families by name, no RoPE/GLU, LayerNorm). Synthetic
config+header fixtures confirm: DeepSeek-V3 per-expert `gate/up/down_proj` (ERR, fail closed),
Mixtral `w1/w2/w3` (silent-drop WARN, gate/up/down UNAVAILABLE), Qwen3-MoE fused 3-D
`gate_up_proj`/`down_proj` (ERR, fail closed), phi-3 `partial_rotary_factor` (exit 0, PASS WITH
ADAPTER — G2 PARTIAL), plus fail-closed-unknown (exit 1), Qwen3Next/Jamba hybrid config-only
(exit 1) and Mamba (exit 1). The `gpt2` G5 = PARTIAL (additive `wpe` breaks the 3-tensor move,
see §2 G5) is reproduced on the real artifact. Only Qwen-class dense configs plus GPT-2 are
cached on this box; every non-cached table row is validated against the transformers and
llama.cpp sources cited above rather than fabricated measurements.
