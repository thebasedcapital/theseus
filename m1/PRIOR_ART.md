# M1 prior art and novelty boundary

Produced by an external research pass over primary sources (papers, official repos), then
annotated with what M1 actually measured. Every item below is either verified against a
primary source or explicitly marked as ours. Research pass: 2026-08-30.

## 1. Positioning in one paragraph

That rotations which preserve full-precision behaviour change quantized quality is
**established**: SpinQuant reports a **13-point** spread in average zero-shot accuracy between
the best and worst *random* dense rotations of LLaMA-2-7B at W4A4 over 100 trials, and up to
That rotations which preserve full-precision behaviour change quantized quality is
**established**. Verified from the SpinQuant abstract (arXiv:2405.16406v4, ICLR 2025): the
authors "identify a collection of applicable rotation parameterizations that lead to
**identical outputs in full-precision** Transformer architectures while enhancing quantization
accuracy", and find "some random rotations lead to much better quantization than others, with
an **up to 13 points** difference in downstream zero-shot reasoning performance". QuIP / QuIP# /
QuaRot / QTIP / OmniQuant all exploit coordinate changes for the same reason. M1 therefore must
**not** be phrased as "rotations change quantization".

Scope limit worth stating out loud: SpinQuant's rotations are evaluated on LLaMA-2/3 (MHA, with
an independently headwise V/O rotation), at W4A4 with activations and KV cache rotated *in the
runtime*. It does not cover the shared-KV GQA coupling (our G1), the RoPE-compatible q/k
commutant (G2), QKV biases, RMSNorm/tie metadata (G3/G5), the SwiGLU up-branch diagonal (G7),
commodity GGUF K-quant artifacts, or any non-quantization operation. M1 extends the
**intervention and the endpoint**, not the core phenomenon. *(The finer details our research
pass reported — 100 random trials, random-Hadamard spread ≈ 6 points, §2.2/Fig. 4 — are
promoted after the research pass read §2.2 verbatim: 100 randomized trials, LLaMA-2-7B W4A4,
average zero-shot accuracy; best random rotation beats worst by 13 points, and random-Hadamard
final-performance variance is "as large as 6 points" (arXiv:2405.16406v4 §2.2). I independently
verified the abstract wording.)*
surgeries** — llama.cpp GGUF K-quants, bounded AdamW LoRA, coordinate-wise linear/TIES merges —
including operations that no quantization paper measures; that an **artifact-only** canonicalizer
(sees one checkpoint, never the original) recovers the lost reserve; and that these effects are
**operation-specific**, i.e. the same gauge can be harmless to one operation and fatal to
another. Working title for the phenomenon: *gauge dependence of downstream checkpoint surgery
under exact Transformer symmetries*.

## 2. Symmetry-family audit (what we got right, what the audit corrected)

| family | status | source / note |
|---|---|---|
| **G1** per-GQA-group value/output orthogonal basis change | VERIFIED (derivation ours; GQA constraint ours) | GQA shares one K/V per query group (Ainslie et al., arXiv:2305.13245 §2.2), so `U` must be constant over the group's query heads. Closest published instance: SpinQuant's headwise V/O rotation `R2` (§3.1), which does not state the shared-KV constraint. |
| **G2** RoPE-plane rotations = full commutant | VERIFIED as an elementary derivation; **no paper states the centralizer theorem** | RoPE construction: arXiv:2104.09864. Distinct frequencies ⇒ 2-planes are non-isomorphic ⇒ Schur gives `SO(2)^{d/2}`. Caveat we record: repeated or aliased frequencies enlarge the commutant to isotypic blocks. Qwen `rotate_half` pairs `(j, j+d/2)` (HF `modeling_qwen2.py`). |
| **G3** RMSNorm scale absorption | VERIFIED | RMSNorm rescale invariance: Zhang & Sennrich arXiv:1910.07467 Eq. 4–7; QuaRot absorbs RMSNorm scales into following weights as preprocessing (arXiv:2404.00456 §3.4). |
| **G5** global residual scaling | **REFUTED as exact** at fixed ε; **VERIFIED exact with the config edit ε → c²ε** | `RMSNorm(cz; ε) = γz/sqrt(mean z² + ε/c²)`, so exactness needs `ε' = c²ε`. This is an artifact-level metadata edit carrying a weight symmetry — math.md §1's `state = (θ, a)` made concrete. Measured here: drift 1.1e-3 of logit scale at ε=1e-6, **exactly 0.0** with the ε patch. Tied embeddings force breaking the tie to express it at all (Qwen2.5-0.5B card: tied). |
| **G6** neuron permutation | VERIFIED | Standard permutation symmetry (Git Re-Basin, arXiv:2209.04836 §2–3). |
| **G7** SwiGLU **up-branch diagonal** | **NEW for us — the audit refuted our original claim** | We first wrote "SwiGLU admits only permutation, no per-unit rescaling". Wrong: with SwiGLU = (Swish(xW) ⊙ xV)W₂ (Shazeer arXiv:2002.05202 Eq. 5–6), `V_j → c_j V_j`, `W₂[:,j] → W₂[:,j]/c_j` is exact for any nonzero `c_j`, because the *gated* branch is what cannot scale (`SiLU(cg) ≠ c SiLU(g)`). So the exact MLP group is permutation × (R\{0})^intermediate — **V0's ReLU-style gauge survives a GLU**, in the multiplicative partner. Implemented as `gauge.g7_up_diag`, repaired by `canon_g7` = V0 §7's balance `c_j = sqrt(B_j/A_j)` on norms. |
| attention **biases** | VERIFIED critical | Qwen2 builds q/k/v projections with bias (HF source); Qwen2.5-0.5B ships `|b_q|max = 79`, `|b_k|max = 130` (our measurement). Any row transform must transform the matching bias entries. No primary source found linking those magnitudes to YaRN — treat as our measurement. |

## 3. Quantization is coordinate-dependent (prior art we do not re-claim)

* **QuIP** arXiv:2307.13304 — left/right rotation of weights and proxy Hessian for incoherence.
* **QuIP#** arXiv:2402.04396 — randomized-Hadamard incoherence processing + E8 lattice codebooks.
* **QuaRot** arXiv:2404.00456 — fixed randomized Hadamards through residual/FFN/attention/KV,
  fused into the runtime; removes activation outliers while preserving FP output.
* **SpinQuant** arXiv:2405.16406 — *learns* the residual `R1` and V/O `R2` on the Stiefel
  manifold with FP outputs frozen; §2.2/Fig.4 = the 13-point random-rotation spread (W4A4).
* **QTIP** arXiv:2406.11235 §2.1 — Hadamard incoherence before trellis coding.
* **OmniQuant** arXiv:2308.13137 §3.3 — learned diagonal/shift equivalent transforms moving
  activation outliers into weights.

Difference in framing that survives scrutiny: all of these **change the runtime or the
objective of one quantizer**. M1 changes nothing at runtime, uses the checkpoint exactly as a
local user would hand it to `llama-quantize`, and measures *several* downstream operations,
plus recovery from the artifact alone.

## 4. Same-function-different-future adjacent work

* **Merge / lineage:** Git Re-Basin arXiv:2209.04836 aligns independently trained models before
  weight-space arithmetic; TIES arXiv:2306.01708 (sign disagreement / redundancy trimming);
  DARE arXiv:2311.03099 (drop-and-rescale deltas). Together these establish that parameter
  coordinates matter to merge arithmetic — not that explicit Transformer-gauge copies of one
  checkpoint differ, which is M1's controlled version.
* **Adam is not rotation-equivariant (the mechanism our LoRA probe leans on):** Zhang et al.,
  *Understanding Adam Requires Better Rotation Dependent Assumptions*, arXiv:2410.19964
  (NeurIPS 2025) — §2.1 defines rotation equivariance, Prop. 1 proves SGD+momentum has it and
  states Adam loses it to element-wise division, Fig. 2 shows divergent trajectories, and §3.1
  trains GPT-2 in rotated parameter spaces where global rotations degrade Adam. This covers
  orthogonal reparameterization of *trainable* parameters; nobody publishes the compound
  frozen-base-gauge + adapter case, so the allowed sentence is: "AdamW's coordinatewise moments
  are non-equivariant to general orthogonal reparameterization (arXiv:2410.19964); applied to the
  induced adapter coordinates of these gauges, non-equivalence is expected, and M1 measures it."
  Do **not** write that the literature proves M1's G1/G2 LoRA divergence.
* **LoRA is parametrization-sensitive:** **LoRA-RITE** arXiv:2410.20625 §3.1 Eq. 12 — for
  equivalent factors `A₂ = A₁R, B₂ = B₁R^{-T}`, equality of a preconditioned update for
  arbitrary gradients requires `X₁ᵀX₁ = R^{-T}X₂ᵀX₂Rᵀ`; an adversarial `R` makes the RHS
  non-diagonal, so **no nonzero diagonal preconditioner** (Adam/Adagrad/RMSProp) can satisfy it.
  Their invariance theorem covers factor-space `GL(r)` only, so it is evidence and a template,
  **not** a proof for M1's model-coordinate gauges. Honest wording: AdamW LoRA is *not expected*
  to be equivariant under G1/G2; a strict negative control would need an equivariant optimizer
  plus transformed adapter init **and** transformed moment states. We report LoRA results as
  empirical, never as a theorem.
* **Rank/scale:** rsLoRA arXiv:2312.03732 (`α/√r`); PiSSA arXiv:2404.02948 (spectral selection of
  the LoRA basis, and lower QLoRA error) — supports "the coordinate frame changes the adapter's future".

## 5. Refutation checks we ran or adopted

* *"llama.cpp K-quants are orthogonally invariant"* — **false**: Q4_K uses `QK_K=256`
  super-blocks of eight contiguous 32-element sub-blocks with 6-bit packed scales, Q6_K sixteen
  contiguous 16-element groups (`ggml/src/ggml-common.h`, `ggml-quants.c`); rounding is applied
  to fixed coordinate groups, so a rotation redistributes entries across them.
* *"the GGUF exporter absorbs RMSNorm into q/k/v, so G3 is invisible"* — **checked here**:
  llama.cpp's `Qwen2Model` (`conversion/qwen.py`) only rewrites names/vocab; norm weights map to
  separate `blk.N.attn_norm.weight` tensors with no folding, and there is no q/k re-layout for
  Qwen2 (contrast `conversion/plamo.py:shuffle_attn_q_weight`). So the rotated coordinates are
  exactly what the K-quant blocks see.
* *"your LoRA probe is a null test by equivariance"* — see §4 (LoRA-RITE): not expected to hold,
  and we do not claim it as a theorem.

## 6. Five sources a skeptic will cite against M1

QuaRot 2404.00456 · SpinQuant 2405.16406 · QuIP# 2402.04396 · OmniQuant 2308.13137 ·
Git Re-Basin 2209.04836 — plus LoRA-RITE 2410.20625 as the sharpest probe-specific objection.
