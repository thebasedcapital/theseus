# M1 notes — exact gauges of a real decoder-only Transformer

Companion to `math.md` (V0 formulation) and `ROADMAP.md`. This file records the derivation
and the frozen protocol for M1, so the numbers in `M1_RESULTS.md` can be audited without
re-reading the source.

## 0. Question and exit condition

> Can two function-equivalent real Transformers have different futures?

M1 exit condition (ROADMAP M1): function-equivalent real Transformer checkpoints show
materially different outcome under at least one **real** operation, with exact-transformation
verification and multi-seed replication.

## 1. Subject artifact and environment

| item | value |
|---|---|
| checkpoint | `Qwen/Qwen2.5-0.5B` (base, bf16, tied embeddings) |
| architecture | `Qwen2ForCausalLM`: 24 layers, hidden 896, 14 q-heads / 2 kv-heads (GQA group 7), `head_dim` 64, SwiGLU MLP `intermediate` 4864, RMSNorm pre-norm, RoPE `theta=1e6` (`rotate_half` convention), q/k/v biases, no qk-norm, `tie_word_embeddings=true` |
| params | 494.03 M |
| eval corpus | `m1/data/eval_wikitext.txt` — pinned 401,943-char slice (94,099 Qwen tokens) of WikiText-2 raw `test`, CC BY-SA 4.0, provenance in `m1/data/PROVENANCE.json`; verification uses the first 4,096 tokens, 8×512 |
| torch | 2.13.0+cu130, transformers 5.16.1, safetensors 0.8.0 (`/home/admin/counterpoint/.venv`) |
| surgery backend | llama.cpp `b9851` (Vulkan, Quadro RTX 4000) for real GGUF `Q8_0/Q6_K/Q5_K_M/Q4_K_M` + `f16`; hand-written LoRA and linear/TIES merges in torch |
| contention | one 8 GB GPU shared with the desktop and sibling agents → `common.pick_device()` + `common.lock("gpu")` serialize big loads; a probe that cannot get resources records `UNAVAILABLE`, never a fake PASS |

## 2. What is exactly symmetric here (and what V0 had that this does not)

Write one decoder block with pre-norm, GQA and SwiGLU:

```
n1 = rms(z) ⊙ w1                       rms(z) = z / sqrt(mean(z²) + eps)
q  = RoPE_t( n1 Wq^T + bq )            (per head, pairs (j, j + d/2) rotated by θ_j t)
k  = RoPE_t( n1 Wk^T + bk )            k is shared by the 7 q-heads of its kv-group
a  = concat_h softmax(q_h k_h^T/√d) V_h n1
z' = z + a Wo^T
z''= z' + ( silu(n2 ⊙ Wg^T) ⊙ (n2 Wu^T) ) Wd^T      n2 = rms(z') ⊙ w2
```

**G1 — value-subspace basis change (per kv-group).** `V` and `O` are the only places the
value coordinates are read, RoPE never touches them, so for any `U ∈ O(d_head)`:

```
V_g ← U V_g        Wo[h·d:(h+1)·d, :] ← Wo[h·d:(h+1)·d, :] Uᵀ   for every q-head h in group g
```

gives `softmax(·)·(U V_g n1) = U a` and then `Uᵀ` cancels it. `U` is *unrestricted*
(rotations, reflections, Hadamard) — this family has no ReLU-MLP analogue and is the
strongest thing V0 could not express. Under GQA, `U` must be constant across the seven
query heads of a group, because `V_g` is shared; a per-query-head `U` is **not** a symmetry.

**G2 — RoPE-plane rotations (the full q/k freedom).** The score term of pair `j` is
`q_jᵀ R(θ_j (t−s)) k_j`. Applying the same 2D rotation `R(φ_j)` to the paired rows
`(j, j+d/2)` of `Wq` (per head) and `Wk` (per group) gives
`qᵀ R(φ)ᵀ R(θ(t−s)) R(φ) k = qᵀ R(θ(t−s)) k`, since 2D rotations commute. Distinct
frequencies make the irreducible 2-planes non-isomorphic, so by Schur's lemma the commutant
of the RoPE family is *exactly* the product of these per-pair rotations: G2 is the maximal
exact q/k gauge, not one example of it. Biases are untouched (they are rotated with the
rows in the same plane? no — `b_q[j], b_q[j+d/2]` must be rotated by the same `R(φ_j)`;
implemented for q/k, verified by the gate).

**G3 — RMSNorm scale absorption.** `rms(·)` output is independent of any positive diagonal
applied *after* it, so for `d ∈ R_+^hidden`:

```
w1 ← w1 ⊙ d        Wq,Wk,Wv ← Wq / d[None,:]   (input columns)
w2 ← w2 ⊙ d'       Wg,Wu     ← Wg / d'[None,:]
```

is exact. One `d` serves all consumers of that norm — you cannot stress `Wq` alone.

**G5 — global residual scaling.** Because `rms(z)` is **0-homogeneous** (scale-invariant),
scaling the whole stream `z → c z` needs compensation only in the tensors that *write into*
the stream or *read as tokens*:

```
embed ← c·embed        Wo ← c·Wo        Wd ← c·Wd        (norms, q,k,v,gate,up: untouched)
```

`lm_head` must stay put, and it is tied to `embed` — so G5 is only expressible by breaking
the tie. Measured consequence: this gauge is **invisible to per-tensor max-abs
quantization** (a tensor's scale rides along with its amax) and visible to fp16 runtimes and
task-vector merges. That asymmetry is a result, not a defect.

**G4 / G6 — permutations (controls).** Query-head and kv-group permutation (`Wq` row blocks +
`Wo` column blocks; `Wk/Wv` row blocks), and SwiGLU neuron permutation (`Wg/Wu` rows + `Wd`
columns). Exact. `G4` is also quantization-neutral because whole rows move, and llama.cpp
blocks are contiguous along the input dimension — a useful control that proves the harness is
not flagging *any* byte change as damage.

**What died with ReLU.** V0's gauge was `W1 ← D W1, W2 ← W2 D⁻¹` for `D > 0`, which exists
because `σ(Dz) = Dσ(z)`. SwiGLU kills it twice over: `silu` is neither positively homogeneous
nor odd, so neither per-unit positive rescaling nor per-unit sign flip survives. The MLP of a
Qwen2 decoder therefore has *no* diagonal gauge — its only exact freedom is permutation, and
the gauges that remain live in the norms (G3), the value subspace (G1), the RoPE planes (G2),
and the stream scale (G5).

## 3. Stresses: deliberately not optimized

`make_variants.VARIANTS` names every artifact. Stress draws are simple declared parameters
— a Haar-random orthogonal (QR of a fixed-seed Gaussian), uniform random angles, log-uniform
random scales, random permutations. **No stress is chosen by minimizing or maximizing any
objective**, so a damage result cannot be an artifact of adversarial construction. The
adversarial end of an orbit is included separately and labelled (`g1_svd`: the value block
rotated into its eigenbasis, which concentrates energy into the first rows).

Each stressed artifact has a paired `_rep` sibling: same stress, then the artifact-only
canonicalizer. Repair is never the inverse of the known stress: the canonicalizer sees one
checkpoint and one declared family, and it must pass the same equivalence gate as the stress.

## 4. Canonicalizers and how far "canonical" goes

| family | rule (artifact-only) | canonicity |
|---|---|---|
| G3 | equalize consumer input-column energy: `d_j ∝ ‖W[:,j]‖` (geometric-mean normalised), summed over `q,k,v` and separately over `gate,up` | canonical up to one global scalar per norm, which is invisible to block-max-abs quantizers |
| G2 | one rotation per RoPE pair that makes the aggregate paired energy equal (`φ = ½ atan2(A−B, 2R)`) | canonical up to a per-pair swap; the `eigen` variant (Jacobi diagonalization, descending) is fully canonical and is kept as the reference |
| G1 | `U = H Pᵀ`: eigenbasis of the value Gram then normalized Hadamard. `Pᵀ V` has orthogonal rows with energies `σ_j²`; Hadamard spreads them **exactly** evenly (`Σ_j H_ij² σ_j² = ‖V‖_F²/d` for every `i`) and makes entries incoherent | canonical up to the residual `± ` sign and ordering freedom inside the balanced set, so the *measured objective* is asserted to match (band 12%) while bitwise equality is not claimed |
| G5 | least-squares `c` from `embed ≈ c·lm_head`, whole-artifact inverse move, re-tie | bitwise canonical when the tie is the only witness; guarded — refuses to act when `embed` is not a scalar multiple of the head (a genuinely untied checkpoint is not on the G5 orbit) |

Two honest limits found while building this, both encoded in `m1/test_gauge_math.py`:

* **The residual-scale gauge has an exactness floor at RMSNorm's `eps`.** On the tiny probe
  model, G5 drift is `1.75e-3` of the logit scale at `eps=1e-6` and **exactly 0** at
  `eps=1e-12`. A real checkpoint with `mean(z²) ≫ eps` is unaffected, but the bound is
  property-of-the-model: how far you may travel along G5 before the function moves depends
  on activation magnitudes.
* **Canonicity is an objective, not a bit pattern.** Within an orthogonal family there is
  leftover discrete freedom (signs, pair swaps) that changes stored bytes and mildly changes
  block amaxes. Theseus should therefore report *conditioning* and *reserve*, never "this
  checkpoint is now canonical".

## 5. Frozen verification gate (`verify_equiv.GATE`, set before any variant was measured)

| metric | threshold | why |
|---|---|---|
| `kl_mean_nats` (mean `KL(P_base ‖ P_var)`, 4,096 positions) | ≤ 2e-3 | the distribution must not move |
| `top1_agree` (teacher-forced argmax) | ≥ 0.995 | greedy behaviour must be unchanged |
| `rel_ppl` on the pinned corpus | ≤ 2e-3 | quality read, independent of the comparison |
| `max_dlogit` | ≤ 0.5 | gross-error tripwire only |

Control: base vs base gives `max_dlogit = 0`, `kl = 0`, `top1 = 1.0`, `ppl = 17.7102` — the
harness floor is exact, so any nonzero drift in a variant is the artifact's, not the tooling's.

**The gate is not being relaxed.** On the real checkpoint the `G3` stresses trip `max_dlogit`
(1.12 stressed / 1.91 repaired) while KL (8.8e-5 / 1.9e-4 nats), teacher-forced top-1
(0.9961 / 0.9954) and PPL (0.1%) all pass. Diagnosis in `m1/control_precision_floor.py`:
gauges are computed in fp64 but *stored* in the artifact's bf16, so every touched entry
re-rounds, and `G3` touches five tensor families across 24 layers twice (stress + repair).
Consequence, and it is a result rather than an excuse: **a checkpoint's own precision caps how
far you may travel along a gauge orbit for free** — that is artifact-level state in the
`state = (θ, a)` sense of math.md §1. The clean handling is to keep the frozen thresholds
untouched, report the tripwire as a separate column, and run the primary `G3` evidence with
**bf16-exact power-of-two scales** (`g3_pow2`: multiplying a bf16 value by `2^k` is lossless),
so the storage-noise argument cannot be made at all.
Equivalence is measured in **fp32 forwards over the bf16 artifacts** (what the artifact
really is) and never with two models resident at once.

## 6. Operations and contracts

Real tools, real weights; Theseus measures existing operators rather than reimplementing them.

| op | implementation | pass contract (frozen in the probe) |
|---|---|---|
| `q8_0, q6_k, q5_k_m, q4_k_m`, `f16` | `convert_hf_to_gguf.py` → `llama-quantize` → `llama-perplexity` / `llama-completion` | damage of quantizing *this* artifact against *its own* f16 GGUF: relative ΔPPL and greedy continuation agreement on 32 fixed prompts |
| bounded LoRA | hand-written r=16 adapters on all 7 projections, fixed steps/batch/lr-grid, seed-fixed data order | task capture ≥ 0.75 × the base-measured capture under the identical budget, and protected-corpus ΔPPL ≤ 0.05 |
| linear / TIES merge | candidate + `α`·(specialist − candidate), density-trimmed variant | exists `α` with merged PPL ≤ 1.05 × candidate PPL **and** merged specialist-rule loss ≤ 0.7 × specialist loss |

## 7. Threats to validity

* **Prior art overlap.** Rotating weights to make them quantization-friendly is exactly the
  QuaRot/QuIP# incoherence trick. The difference in framing is load-bearing: those methods
  change the *runtime* (rotations folded into activations, online Hadamard in the kernel) to
  improve one quantizer. Here nothing changes at runtime; the artifact alone is moved along an
  exact symmetry and the *downstream operation* outcome is what changes — including merges and
  adapters, which no quantization paper measures. See the prior-art brief in `M1_RESULTS.md`.
* **Merge demo is constructed.** We deliberately merge a gauged candidate with a specialist
  derived from the ungauge base. Real independently-trained finetunes do drift in basis (the
  git-re-matching / permutation-symmetry literature), but our drift is a designed worst case,
  and the result must be read as "gauge mismatch destroys merging", not "merges usually fail".
* **`prepare` could be seen as undoing a known stress.** Guarded by `_rep` artifacts being
  produced by the same canonicalizer that is applied to the pristine base (`prep_base`) and by
  the canonicity tests in `test_gauge_math.py`.
* **Single model, single scale.** 0.5B base-only. M3's natural-history pairs are the real
  test of whether this matters in people's actual checkpoints.

## 8. Files

```
m1/common.py             artifacts, pinned corpus, fp32 verification metrics, resource lock
m1/gauge.py              G1..G6 exact transforms + spec parser
m1/canonicalize.py       artifact-only repairs + quant_condition() static proxy
m1/make_variants.py      VARIANTS registry (stress / stress+repair pairs, 3 seeds)
m1/verify_equiv.py       frozen equivalence gate, JSON out
m1/test_gauge_math.py    property tests on a tiny Qwen2 (exactness / repair / canonicity)
m1/gguf_probe.py         llama.cpp quantization operations (sibling agent)
m1/adapt_probe.py        bounded LoRA capture (sibling agent)
m1/merge_probe.py        linear + TIES merge capture (sibling agent)
m1/run_m1.py             phase-2 driver: gate → ops → reserve vector → free disk
m1/data/                 pinned eval corpus + provenance
m1/work/                 scratch (gitignored): variants, gguf, probes, results
```
