# Gauge dependence of downstream checkpoint surgery under exact Transformer symmetries

**Theseus — technical report.** Version 1.0, 2026-08-31.
Code, evidence and failure log: <https://github.com/thebasedcapital/theseus> (MIT).
Status: single-repository report, no peer review. Every number below is read from a persisted
artifact in that repo; each headline claim carries the experiment that would refute it.

---

## 1. Abstract

A checkpoint is usually treated as a function plus some bytes that implement it. I show that for
real Transformers this identification is unsafe: two artifacts can compute the *same* function, to
the last bit, and still have sharply different futures under the operations local users
perform. On Qwen2.5-0.5B, a rescaling drawn from the RMSNorm scale-absorption symmetry, restricted
to powers of two so that bf16 storage is exact, yields a different file whose logits are
bit-identical to the original (`max|Δlogit| = 0.00e+00`, mean KL `0`, top-1 `1.00000`, perplexity
`12.1351` vs base `12.1399`). That checkpoint is then destroyed by 8-bit quantization (KL
divergence `0.00094 → 10.69` nats) and loses almost all of its adaptation capacity under a
three-seed rank-16 LoRA probe (mean capture `0.9705 → 0.0989`), while merges that the base survives
both fail. An artifact-only repair that never sees the original restores both reserves, and in one
case reproduces the pristine checkpoint's file byte for byte.

I therefore treat the *reserve* of a checkpoint, defined per operation family, as a first-class
measurable quantity, and argue it cannot be summarized by a scalar. The same function-preserving
move that improves 4-bit quantization damage on two disjoint corpora (`-0.185` and `-0.478`
percentage points of relative damage) makes both merge operators fail. The repo ships the static
detector (~2 s, no model load), the append-only evidence ledger behind these numbers, and a log of
20 pipeline incidents. One surfaced while writing this report: a recorded natural-history attempt
proved not reproducible from its own generator, and §11 withdraws it.

---

## 2. The claim, stated falsifiably

Find checkpoints `θ_A`, `θ_B` with

> `f_{θ_A} ≈ f_{θ_B}` on all current evaluation, while `R(θ_A) ≠ R(θ_B)` for the vector `R` of
> future-operation reserves,

where `θ_B` is generated from `θ_A` by an *exact* symmetry of the architecture, so the divergence
cannot be attributed to current task quality. The strongest version is `≈` replaced by equality in
floating-point output, which §5 achieves.

---

## 3. Definitions

A checkpoint is a state `s = (θ, a)`, where `a` carries artifact-level facts: quantization scales,
`rms_norm_eps`, embedding-tie metadata, adapter structure. Concrete operators act on `s`, not on the
equivalence class `[θ]`.

Each surgery family `o` (quantize, prune, LoRA-adapt, merge, distill, edit, unlearn) is a controlled
transition `s_{t+1} = F_o(s_t, u_t, ξ_t)` with control `u_t` (bit-width, learning rate, merge
coefficient, rank), subject to a **safe set** `K = {s : U_j(s) ≥ c_j ∀j}` protecting existing
capability and a **target set** `T_o = {s : G_o(s) ≥ τ_o}`. The operation-specific capture basin is

```
C_o^H = { s0 ∈ K : ∃ u_{0:H-1} with s_t ∈ K ∀t < H and s_H ∈ T_o } .
```

Writing `J*_o(s)` for the minimum control cost of reaching `T_o` while staying in `K`, the reserve
under budget `B_o` is `R_o(s; B_o) = max(0, 1 − J*_o(s)/B_o)`, and `R(s) = [R_1(s), …, R_m(s)]`. A
deployment-specific scalar `Ω(s) = Σ_o p_o R_o(s)` is admissible only with the weights `p_o`
recorded alongside it. This is viability theory applied to post-training; the ingredients are prior
art (§12), and the object claimed here is the heterogeneous, operation-specific surface with both
artifact and canonical views.

Gauge dependence enters through positive homogeneity. For a ReLU unit, `σ(Dz) = Dσ(z)` gives
`W₁' = DW₁`, `W₂' = W₂D⁻¹` with unchanged function; magnitude pruning, per-tensor quantization,
SGD and raw weight averaging are *not* invariant under the resulting group. The measurable gap
between acting on the artifact and acting on a canonical representative is avoidable lifecycle debt

```
D_o^gauge(θ) = R_o^canon([θ]) − R_o^artifact(θ) .
```

---

## 4. Symmetry audit

Every family was checked against a primary source before use; the audit corrected two of my own
claims. Full derivation in `m1/PRIOR_ART.md` and `M1_NOTES.md`.

| family | transformation | status |
|---|---|---|
| **G1** | orthogonal basis change shared across a GQA group's value/output heads | derivation mine; the shared-KV constraint follows from GQA (arXiv:2305.13245); closest published instance is SpinQuant's headwise `R2`, which does not state it |
| **G2** | rotations inside the RoPE 2-planes of q/k | the full commutant `SO(2)^{d/2}` (distinct frequencies ⇒ non-isomorphic 2-planes ⇒ Schur); no paper states this theorem as far as I could find |
| **G3** | RMSNorm scale absorption into the consumer | known invariance (arXiv:1910.07467), already used as preprocessing by QuaRot (arXiv:2404.00456 §3.4) |
| **G5** | global residual scaling | **refuted as exact** at fixed `ε`; exact only with the artifact edit `ε → c²ε`. A metadata change carrying a weight symmetry, i.e. `s = (θ, a)` made concrete |
| **G6** | neuron permutation | standard (arXiv:2209.04836) |
| **G7** | SwiGLU **up-branch** diagonal, `V_j → c_jV_j`, `W₂[:,j] → W₂[:,j]/c_j` | **new to me**: my prior-art pass refuted my original assertion that a GLU admits no per-unit rescaling. The *gated* branch cannot scale (`SiLU(cg) ≠ c·SiLU(g)`); the multiplicative partner can |
| bias | attention q/k/v biases | critical: Qwen2.5-0.5B ships `max|b_q| = 79`, `max|b_k| = 130`, so any row transform must transform the matching bias entries |

**Important negative space.** That rotations preserving full-precision behaviour change
quantized quality is *established*: SpinQuant reports up to **13 points** of downstream spread
between random dense rotations at W4A4 (arXiv:2405.16406 §2.2), and QuIP/QuIP#/QuaRot/QTIP/
OmniQuant all exploit coordinate changes deliberately. I do not claim that. What I measure is
different in intervention and endpoint: nothing changes at runtime, the checkpoint is handed to
commodity `llama-quantize` exactly as a local user would, several *non-quantization* operations are
measured on the same artifact, and recovery is attempted from the artifact alone.

---

## 5. The collapse

Subject `Qwen/Qwen2.5-0.5B` (bf16, tied embeddings); surgery executed by llama.cpp `b9851` K-quants,
hand-written AdamW LoRA, and task-vector merges. Equivalence gate frozen before measurement
(`m1/verify_equiv.py`): mean KL ≤ `2e-3` nats **and** teacher-forced top-1 ≥ `0.995` **and** relative
PPL change ≤ `2e-3`, over 4,096 tokens of a pinned WikiText-2 slice. Harness floor is exactly zero
(base vs base).

`g3_pow2` rescales the RMSNorm/consumer pairs by `2^k`, `k ∈ [-10,10]`, on q, k, v, gate and up
across all 24 layers, and is bit-exact because bf16 × `2^k` is lossless:

| artifact | `max\|Δlogit\|` | mean KL | top-1 | Q8_0 KLD | LoRA capture (3 seeds) | linear merge | TIES |
|---|---:|---:|---:|---:|---:|---|---|
| `base` | 0 | 0 | 1.00000 | 0.00094 | **0.9705** | pass | pass |
| `g3_pow2` | **0.00e+00** | **0.00e+00** | **1.00000** | **10.69** | **0.0989** | fail | fail |
| `g3_pow2_rep` | 0.00e+00 | 0.00e+00 | 1.00000 | 0.00106 | **0.9753** | fail | fail |
| `g7_rand` | 3.15e-1 | 1.95e-5 | 0.99683 | — | **0.1931** | fail | fail |
| `g7_rand_rep` | 4.8e-01 | — | — | 0.00088 | **0.9359** | fail | fail |

Same logits, same perplexity, reserve collapsed. Both halves of that sentence were re-derived on 2026-08-31 after incident #18 showed a recorded cell here could be unreproducible from its own generator: `make_variants` rebuilt `g3_pow2` to the identical sha256 (`0c106a426af05dc8`), equivalence came back `max|Δlogit| 0.0 / KL 0.0 / top-1 1.0`, and the 8-bit collapse measured KLD `10.690189` against the recorded `10.690304` with perplexity `633431.0375` reproduced exactly. Full log in `analysis/data/reverification.json`. `g7_rand` repeats the result under a weaker
equivalence condition (exact in fp32 arithmetic, not bit-identical under bf16 compute), and both
effects clear a deliberately conservative bar: a gap must exceed three times the largest
within-variant SD across the panel. Of the nine gauged artifacts in that panel only G3 and G7 clear
it; the smaller G1/G2/permutation differences do not survive optimizer variance and are **not** claimed.

Storage precision bounds the free part of an orbit. A continuous (non-lattice) version of the same
gauge drifts by `max|Δlogit| = 5.40e-01` purely from bf16 re-rounding, isolating rounding as the
mechanism by re-running one transform at fp64 (error `9.39e-05`) and storing it back to bf16
(`2.69`).

Export is a separately metered operation, and it nearly produced a false result. The stressed
artifact measured perplexity `177` after an f16 GGUF export before any quantizer ran; the same
weights exported to their native bf16 give `12.1351` (base `12.1399`). Cause: `9.87 %` of the
stressed weights fall below f16's normal range (base: `0.28 %`), minimum `1.8e-11`. An identity
control, pristine bytes written through my own writer and re-exported, reproduces `12.1399` exactly.

---

## 6. Recovery, including one exact round trip

The repair sees a single artifact and no base. For G5 it recovers the global scale from the tie
witness (`embed ≈ c·lm_head`, least squares) and applies the whole-file inverse on the power-of-two
lattice. The result is not close to the original, it is the original:

```
$ cd m1 && python make_variants.py --only g5_c8     # rebuild the stressed artifact (7 s)
$ python m1/verify_g5_recovery.py                    # repair it, compare, hash (17 s)
key sets identical: True  (n=290)
tensors differing: 0/290
repaired sha: 88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342
pristine sha: 88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342
BYTE-FOR-BYTE: CONFIRMED      repair meta: {"canon": "G5", "detected": true, "c_recovered": 8.0}
max|Δlogit| 0   KL 0   top-1 1.00000
```

The scale `c = 8.0` is recovered from the embedding/LM-head tie alone; the repair is never given
the pristine file. `verify_g5_recovery.py` re-derives this end to end, so the claim no longer rests
on a log line from a run whose artifact is gitignored.

Adaptation recovery is consistent across families: `g3_pow2` `0.0989 → 0.9753` and `g7_rand`
`0.1931 → 0.9359` mean capture, with all repaired artifacts passing the fp32 equivalence gate and
G3 staying bit-identical under bf16 compute.

The repair has a precision cost, reported against myself. The full continuous canonicalizer run on
the *pristine* checkpoint fails its own equivalence gate (`max|Δlogit| 0.96`, top-1 `0.99487` under
the `0.995` bar) because the value-subspace Hadamard rewrites every v/o entry and the continuous
multipliers are not bf16-representable. `m1/rescue.py` therefore refuses to ship that path; only the
lattice-exact `{G5, G3, G7}` repair is offered as a mutation.

---

## 7. Pre-registered prediction, and where it is incomplete

Before any surgery number existed I committed a purely static quantity to disk
(`m1/predict.py` → `PREDICTIONS.json`): block-max-abs 4-bit conditioning `J` over the 32-element
blocks a K-quant actually rounds, and its debt `J(variant) − J(base)`.

It predicted damage for `g3_pow2` (debt `+0.01831`) and neutrality for the value/output rotation
`g1_haar` (debt `−0.00009`), and the latter is the interesting case because it cuts against the
fashionable story: the QuaRot-style gain from rotating the value space comes from activation and
KV-cache outliers, which a **weight-only** 4-bit GGUF quantizer never touches. Measured KLD for
`g1_haar` is `0.03200` = `1.00×` base, i.e. distributionally indistinguishable.

It is incomplete as a success story and I will not smooth it. On the *second* damage statistic the
same artifact fails: relative ΔPPL `+3.97 %`, past the contract limit, roughly `1.8` pp worse than
base. Two damage measures disagree on one checkpoint. That is recorded as a disagreement
(`CLAIMS.md` K-4), with its own refuter: a rerun on a longer corpus where ΔPPL falls back inside
base-plus-slack would mean the effect is corpus variance rather than gauge state. The practical
conclusion is the one in §8: a reserve collapsed to one number hides exactly this.

---

## 8. Reserve is a vector, not a score

`prep_base_exact` applies the lattice repair to the untouched base; it never sees a stressed
ancestor. Its fp32 outputs are exactly equal to base on a second, disjoint corpus slice, and:

| corpus | base Q4 relative ΔPPL | prepared | prepared − base |
|---|---:|---:|---:|
| original `[0,32768)` | +2.195 % | **+2.010 %** | **−0.185 pp** |
| disjoint `[65536,98304)` | +2.608 % | **+2.131 %** | **−0.478 pp** |

The same artifact **fails both merge operators**. Its single-seed LoRA row passes. One
function-preserving coordinate change simultaneously improves one future operation and destroys
another, so no scalar health ordering preserves the outcomes. The schema rejects scalar `health`
fields; reserves are stored as typed maps and any scalarization must name its weights.

---

## 9. A cheap static detector

`inspect/` is a zero-dependency Rust binary that parses the safetensors container itself and streams
bf16/f16/f32 weights without loading a model: `357,826,560` weights metered in about two seconds.
It reports per family `q4_block_mse`, `dyn_range_log10`, `row_energy_imbalance` and
`frac_below_f16_normal`, the last being the export gate. Against the measured outcomes:

| artifact | total J | dyn range | frac below f16 normal | flags | measured outcome |
|---|---:|---:|---:|---:|---|
| `base` | 0.01123 | 8.83 | 0.00282 | 0 | capture 0.9705; Q4 KLD 0.0319 |
| `g3_pow2` | 0.02955 | 14.6 | 0.0987 | **15** | capture **0.0989**; Q8 KLD **10.69** |
| `g3_pow2_rep` | 0.01148 | 8.76 | 0.00281 | **0** | capture **0.9753** |
| `bad_all` (4 families) | 0.03315 | 18.81 | 0.15486 | 20 | adaptation + both merges fail |
| `bad_all_exact` | 0.01159 | 9.99 | 0.00279 | **0** | adaptation + both merges **still fail** |
| `prep_base_exact` | 0.01144 | 9.01 | 0.00280 | **0** | Q4 improves; both merges fail |

Every verdict carries its basis in the reason column, so `v3 n=20, fitted in-sample` can be told
apart from `v2 provisional constant, not fitted`, and the Q4 row reads
`no contract: fit refused (precision 0.278 < 0.3125)`. Flags are specific: `g3_pow2` raises nothing on `o_proj`/`down_proj`, the two families a norm-diagonal
gauge does not touch. Two implementations are cross-validated: the Rust `q4_block_mse` and the
Python/torch fp64 version agree per family to `≤ 4.4e-09`. Getting that agreement took a real bug,
pooled ratio-of-sums versus mean-of-per-tensor-ratios, which disagreed by up to `5.7 %`; both
conventions are now emitted under separate names.

**Everyday value.** `theseus-scan preflight` rates each operation from the artifact's bytes in about
a second and flags the f16 export trap before somebody downloads ~400 MB of noise. The gate is
opt-in: `--fail-on-risk` exits 1 only for a **judged** `AT_RISK`. `UNAVAILABLE` rows (merge, AWQ-LoRA)
and `UNKNOWN` rows (Q4, whose fitted cut was refused at precision 0.278) deliberately never trip
it - the tool refuses to convert "I have no threshold" into either a pass or a failure. That is I8
enforced in the exit code, not just in the schema.

---

## 10. Threshold calibration, including a refused fit

Nine additional equivalence-gated artifacts, measured on CPU with native-bf16 export, bring Q8 to
`n = 20` and produce an immutable threshold contract `v3`: predict Q8 risk when
`q4_block_mse > 0.01282348`, giving **recall 1.0, precision 0.40, specificity 0.833**. The shipped
detectors use it. Earlier `v2` verdicts were invalidated through ledger edges rather than rewritten.

Q4 also reached `n = 20`. Its recall-preserving cut has precision `0.278`, below the required
`0.3125`, so **no Q4 threshold is emitted** and the operation is reported `UNAVAILABLE`. Export and
adaptation remain below `n = 20`. `v3` is in-sample calibration on one checkpoint family, not an
out-of-sample predictor, and the refuter is explicit: any measured Q8 false negative, or
out-of-sample recall below `0.95`, kills it.

Architecture handling is deliberately conservative. The audit probe fails closed on Mamba/Hybrid
layers, on fused QKV, and on rank-3 fused expert stacks (explicit `UNAVAILABLE`, never a silently
missing family), and passes with an adapter where rotary or sections are partial. Trusted 2-D MoE
experts are keyed into separate `expert_gate`/`expert_up`/`expert_down` families by boundary-aware
matching. Raw numbers from an unsupported architecture are refused rather than guessed.

---

## 11. What is not shown

Stated in the repo itself (`m1/passport.py` `UNCLAIMED`, `CLAIMS.md`):

- **Natural histories have never been tested.** An earlier draft of this report described a built
  `adapt → merge → Q4_K_M` versus `merge → adapt → Q4_K_M` pair that allegedly matched perplexity
  within `0.054 %` while failing the distributional gate at mean KL `0.032311` / top-1 `0.88235`.
  **That account is withdrawn**: the committed harness cannot produce it. `train_lora_state` called
  `opt.step()` with no optimizer ever constructed, `CONTRACT.adapt.lr` was declared but never
  consumed, and the stored `adapt` records lack the `capture` / `task_loss_*` fields that function
  always returns. Its self-check passed only because it replaced that function with a lambda
  (incident #18). The JSON is kept as a failed-attempt record and cited as nothing. K-8 is open.
- **The obvious fix for that failure does not work.** Re-running the repaired harness as an
  exploratory screen (`m3/screens/`, not evidence), sweeping merge alpha from `0.30` down to `0.02`
  leaves mean KL at `0.029-0.048`, i.e. 15-24x outside the `2e-3` gate, and non-monotone in alpha:
  at `alpha = 0.02` the merge is nearly a no-op and the orders are still distributionally distinct.
  Weakening the operations cannot produce a present-matched natural pair, because the two orders
  adapt from different coordinates. The screen also voids a detail the withdrawn record had made
  look elegant - a pair inside the perplexity tolerance while outside the distributional one. At
  every screened alpha **both** fail. What the screen does keep is narrow but reproducible: the two
  orders are indistinguishable in the features that drive quantization reserve (gaps `<= 2e-3`) and
  differ most in `row_energy_imbalance` (`0.017-0.094`), the feature M1 ties to adaptation capture.
- **Caveat travelling with every figure in the two bullets above.** That sweep ran `m3`'s then-private
  copy of the training loop, which inherited AdamW's default `weight_decay=1e-2` rather than the
  contract's `0.0`. The duplicate has since been deleted so `m3` delegates to
  `m1/adapt_probe.train_once`, and `m3/screens/README.md` records the divergence. These values are
  superseded until the sweep is re-run under the consolidated trainer.
- **One model, one scale, base (non-instruct) weights.** Two corpora now support the Q4 result; no
  second architecture does.
- **Merge tests are constructed** against a specialist derived from the ungauged base.
- **Lattice repair is not a universal cure.** `bad_all_exact` clears every static flag and still
  fails adaptation and both merges.
- **Thresholds are in-sample** (see §10).
- **The first adaptation/merge panel was invalid, and it stays on record.** It trained the whole
  model at a LoRA learning rate, because embeddings, norms and the head were still trainable.
  Twenty-one cells are archived and count toward nothing; the replacement contract is
  `adapt-v2-true-lora-base-frozen`.

---

## 12. Prior art that constrains the novelty claim

Ingredients, not claims: viability kernels and capture basins (Aubin, Frankowska, Saint-Pierre);
quantitative controllability through minimum energy; Optimization Readiness for future trainability
(arXiv:2605.09044); quantization designed for downstream adapter correctability (ProjQ,
arXiv:2606.00494); positive-homogeneous gauge redundancy and gauge fixing (arXiv:2602.14729);
functionally identical gauges changing training dynamics (arXiv:2608.06766); rescaling-invariant path
metrics (Gonon et al., PMLR 2025); sequential bounded plasticity decay (CellFill,
arXiv:2608.20873). Coordinate changes for quantization: QuIP `2307.13304`, QuIP# `2402.04396`,
QuaRot `2404.00456`, SpinQuant `2405.16406`, QTIP `2406.11235`, OmniQuant `2308.13137`. Merge
arithmetic and coordinates: Git Re-Basin `2209.04836`, TIES `2306.01708`, DARE `2311.03099`.

Two objections deserve their strongest form rather than a strawman. Adam is not rotation-equivariant
(arXiv:2410.19964, Prop. 1: SGD with momentum is, Adam loses it to element-wise division), which
*predicts* the LoRA divergence but does not cover the compound case of a frozen gauged base plus an
adapter; I report LoRA results as empirical, never as a theorem. And LoRA-RITE (arXiv:2410.20625
§3.1 Eq. 12) shows that for equivalent factors `A₂ = A₁R`, `B₂ = B₁R^{-T}`, equality of a
preconditioned update for arbitrary gradients requires `X₁ᵀX₁ = R^{-T}X₂ᵀX₂Rᵀ`, which an adversarial
`R` defeats for *any* nonzero diagonal preconditioner; that theorem is about factor-space `GL(r)`,
so it is evidence and a template, not a proof for model-coordinate gauges.

The candidate new object is the heterogeneous, operation-specific lifecycle surface across learning,
alignment, merging, editing, unlearning, pruning, quantization and distillation, with both
artifact-level and symmetry-canonical views.

---

## 13. Reproduction

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # torch ≥2.13, transformers ≥5.16
cd m1 && python prep_data.py                        # pinned corpus, sha256 f58687fa…8af8
python make_variants.py                             # gauges, incl. g3_pow2
python verify_equiv.py                              # equivalence gate, frozen thresholds
cargo run --release --manifest-path inspect/Cargo.toml -- inspect <file> --json
cargo test --manifest-path scan/Cargo.toml          # 13 tests; inspect crate: 9
python m1/verify_g5_recovery.py                     # re-derive the byte-for-byte result (§6)
python -m ledger.cli verify                         # cell provenance: commit + generator + shape
python analysis/reserve.py                          # quantitative reserve vectors, no GPU needed
python archcheck/test_qknorm_g2.py <snapshot-dir>   # G2 exactness test for QK-norm architectures
python m1/test_gauge_math.py                        # gauge algebra property tests
python -m unittest discover -s analysis             # 37 tests; ledger suite: 21
```

Quantization rows require a llama.cpp build (`b9851` here); adaptation and merge rows are pure
torch. Deterministic seeds: `1729`, `23`, `44` for the three-seed adaptation panels; `42` for the
lattice gauge. The pinned corpus is `m1/data/eval_wikitext.txt`, 401,943 chars / 94,099 Qwen tokens,
derived from WikiText-2 raw `test` (CC BY-SA 4.0) and regenerated rather than redistributed.
`m1/work/` outputs and Rust `target/` are gitignored.

---

## 14. Evidence governance

The reason the negative results are in this document is structural, not stylistic. `ledger/` is an
append-only store: `182` admission-clean cells, each binding `(artifact, operation, environment)` to
a verdict, re-derivable and re-checkable, plus `20` incident records linked to the invariant that now
prevents recurrence — including #18, which voids the natural-history attempt described in §11.
`python -m ledger.cli verify` audits every persisted cell against three claims it makes about itself:
the recorded commit resolves, the named generator existed in that commit's tree, and the cell carries
the fields its own writer cannot omit. A `contract_version`-tagged adaptation record missing `capture`
is the #18 signature and fails the run; pre-contract cells are reported as warnings rather than
silently compared, which is how §11's superseded alpha figures stay labelled as superseded.
`CLAIMS.md` holds the claim register with an explicit refuter per claim, and states
`UNSUPPORTED → PRELIMINARY → CONTROLLED → CONFIRMED` from the obligations table; prose never upgrades
a state. Generated files (`M1_TABLE.md`, passports, reports) are renderings of the ledger, never
hand-written, so a session that loses its working tree loses nothing. Invariants I1–I10 in
`SYSTEM.md` encode the rules that the incidents taught.

---

## 15. Open questions

1. Do ordinary post-training histories leave hidden lifecycle state that current evaluation cannot
   see? Requires two present-matched pairs surviving a registered shuffled null (§11). I have not
   produced them.
2. Is there a canonicalization that is safe for *all* measured operations? The evidence so far says
   no, and `prepare` therefore needs an operation target.
3. Can reserves be predicted from static features out of sample? In-sample Q8 is recall `1.0` /
   precision `0.40`; Q4 refuses to fit.
4. Which symmetry groups matter for which operation, at scale and on MoE and multimodal stacks?

Cite as: *Theseus: gauge dependence of downstream checkpoint surgery under exact Transformer
symmetries*, technical report, `github.com/thebasedcapital/theseus`, 2026.
