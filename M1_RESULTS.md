# M1 — can two function-equivalent real Transformers have different futures?

Subject: `Qwen/Qwen2.5-0.5B` (base, bf16, tied), surgery executed by real tools (llama.cpp
`b9851` GGUF K-quants, hand-written AdamW LoRA, task-vector merges). Nothing here is
model-judged or simulated; every number comes from a persisted JSON under `m1/work/` and the
scripts that produced it are in `m1/`. Derivations: `M1_NOTES.md`. Novelty boundary:
`m1/PRIOR_ART.md`. Table generator: `m1/report.py`; static analysis: `m1/analyze.py`.

## 0. Headline: checkpoints with provably *bit-identical* behaviour, different bytes

The exponent-lattice gauges (§1, bottom) remove every numerical caveat from the equivalence half
of M1. `G3:pow2` rescales the RMSNorm/consumer pair by `2^k`, `k ∈ [-10,10]`, per input column of
q, k, v, gate, up in all 24 layers — 3 orders of magnitude of coordinate change, and because
bf16 × `2^k` is exact, the resulting artifact is a *different file that computes the identical
function to the last bit*:

| artifact | sha256 (file digest) | max\|Δlogit\| | mean KL | top-1 | PPL |
|---|---|---:|---:|---:|---:|
| `base` | `88c142557820ccad…` | 0 (self) | 0 | 1.00000 | 17.7102 |
| `g3_pow2` (stressed) | `0c106a426af05dc8…` | **0.00e+00** | **0.00e+00** | **1.00000** | 17.7102 |
| `g3_pow2_rep` (repaired) | `eee3345d64d604b1…` | **0.00e+00** | **0.00e+00** | **1.00000** | 17.7102 |
| `g5_c8_rep` (repaired) | `88c142557820ccad…` | 0.00e+00 | 0 | 1.00000 | 17.7102 |

`g5_c8_rep`'s digest is the *pristine checkpoint's own blob hash*: the artifact-only canonicalizer
regenerated the original file byte for byte from a tie-broken, embedding-scaled descendant it had
never seen (see §1). And the `ε → c²ε` config edit is what makes residual scaling bitwise-exact:
without it, `g5_c8` drifts by `max|Δlogit| 1.67e-01` / `KL 3.89e-06` purely because
`rms(cz) ≠ c·rms(z)` at finite ε; with it, `g5_c8_eps` is exactly zero.

So the setup M1 needs is on the table: **identical logits, identical perplexity, different bytes**,
and the only remaining question is whether real surgery treats them differently. That is what
§5's panel answers.

## 1. The exact gauges, verified on a real checkpoint

Gate (frozen before measurement, `verify_equiv.GATE`): fp32 forwards over the stored bf16
artifacts, first 4,096 tokens of the pinned WikiText-2 slice.
Pass needs mean `KL(P_base‖P_var) ≤ 2e-3` nats **and** teacher-forced top-1 agreement `≥ 0.995`
**and** relative PPL change `≤ 2e-3`; `max|Δlogit| ≤ 0.5` is a gross-error tripwire only.
Control: base vs base = `0, 0, 1.00000, ppl 17.7102` — the harness floor is exactly zero.

| checkpoint | gauge | max\|Δlogit\| | mean KL (nats) | top-1 | PPL | verdict |
|---|---|---:|---:|---:|---:|---|
| `base` | — | 0 | 0 | 1.00000 | 17.7102 | reference |
| `g1_haar` | per-GQA-group value/output O(64) basis change | 1.76e-1 | 1.11e-5 | 0.99902 | 17.7082 | EQUIVALENT |
| `g1_haar_rep` | + artifact-only repair | 3.05e-1 | 2.09e-5 | 0.99780 | 17.7087 | EQUIVALENT |
| `g1_svd` | adversarial: value block into eigenbasis | 2.08e-1 | 2.25e-5 | 0.99585 | 17.7083 | EQUIVALENT |
| `g2_rand` | RoPE-plane q/k rotations (full q/k commutant) | 2.23e-1 | 3.72e-5 | 0.99756 | 17.7144 | EQUIVALENT |
| `g3_smooth` | norm-diagonal, block-constant | 4.20e-1 | 5.97e-5 | 0.99609 | 17.7063 | EQUIVALENT |
| `g7_rand` | SwiGLU up-branch diagonal, ±3 decades | 3.15e-1 | 1.95e-5 | 0.99683 | 17.7083 | EQUIVALENT |
| `g7_few` | same, extreme on 0.1 % of neurons | 8.11e-3 | 1.07e-8 | 1.00000 | 17.7101 | EQUIVALENT |
| `g4_perm` | **control** head/group permutation | 1.26e-4 | 1.09e-9 | 1.00000 | 17.7102 | EQUIVALENT |
| `g6_perm` | **control** SwiGLU neuron permutation | 1.64e-4 | ≈0 (−5e-9) | 1.00000 | 17.7106 | EQUIVALENT |
| `g3_rand` | norm-diagonal, log-uniform per column | **1.12** | 8.76e-5 | 0.99609 | 17.6928 | tripwire tripped |
| `g3_rand_rep` | + repair | **1.91** | 1.86e-4 | 0.99536 | 17.6792 | tripwire tripped |
| `g3_smooth_rep` | + repair | **1.50** | 1.55e-4 | 0.99512 | 17.7059 | tripwire tripped |

So: a random orthogonal change of basis inside every GQA value subspace, the whole RoPE-compatible
q/k freedom, the SwiGLU up-branch diagonal, and the RMSNorm scale absorption are all *the same
model* — mean KL between `1e-5` and `2e-4` nats over 4,096 positions, teacher-forced greedy
agreement 99.5–99.9 %, perplexity unchanged in the third decimal, and PPL moved less than the
checkpoint's own corpus-order noise. The two controls are exact to `1.6e-4`, which also proves the
harness is not calling any byte change "different".

### Why three rows say "tripwire tripped" instead of EQUIVALENT, and what I did about it

`g3_rand` fails only `max|Δlogit|` (1.12; repaired 1.91) while passing all three distributional
criteria. `m1/control_precision_floor.py` settles it: for the same transform computed in fp64 and
kept in fp32 (no bf16 re-storage) the error is **9.39e-05**; stored back into bf16 it becomes
**2.69** on that token subset (8.6 % of the logit scale). Gauges re-round every entry they touch,
and `G3` touches q, k, v, gate, up and both norms in all 24 layers.

That is a genuine property of the artifact, not an excuse: **a checkpoint's storage precision caps
how far you may move along a gauge orbit for free.** Two consequences, both implemented:

* the frozen thresholds were **not** relaxed;
* `G3` stress and repair now run on the bf16 **exponent lattice** (`gauge.g3_norm_diag(mode="pow2")`,
  `canonicalize.canon_g3(snap_pow2=True)`): multiplying a bf16 value by `2^k` is lossless, so the
  whole stress→repair pair stays exactly representable, and the argument disappears. Measured
  static effect of that lattice repair: `J` 0.01123 (base) → 0.02955 (stressed, 2.6×) → 0.01148
  (repaired).

Two bug classes found and fixed here, both worth stating because they were silent:
Qwen2 carries **q/k/v attention biases** (`|b_q| = 79`, `|b_k| = 130` on this checkpoint) and my
first G1/G2 rotated only weights — `m1/test_gauge_math.py` now carries a permanent sensitivity
control that fails the suite unless a weights-only rotation is detected (measured separation
5.2e-3 vs 1.2e-7, ≈ 43 000×); and a `G5` repair that re-ties the head must also stop claiming
`tie_word_embeddings = false`, otherwise `transformers` silently initialises a fresh head
(produced `ppl = 1.1e15` before the fix).

## 2. Property tests backing the algebra

`m1/test_gauge_math.py` (tiny Qwen2, fp32, exit 0, 13 cases): exactness `6e-8…1.8e-7` for
G1/G2/G3/G4/G6/G7 and the composed multi-family stress; repair exact from both ends of the orbit;
and **canonicity as an objective** — `J(canon(base))` vs `J(canon(stressed))` agree to `1e-8` for
G2/G3/G7 and within a documented 12 % band for G1, whose balanced set keeps residual
sign/permutation freedom. G5's drift is proved to be the RMSNorm ε floor: `1.4e-3` of logit scale
at `ε = 1e-6`, **exactly 0.0** at `ε = 1e-12`, and **exactly 0.0** at `ε = 1e-6` when the config is
moved `ε → c²ε` — an artifact-level metadata edit carrying a weight symmetry, which is the
`state = (θ, a)` of `math.md §1` made concrete.

### The canonicalizer has a precision cost of its own — reported against myself

`prep_base` is the artifact-only canonicalizer run on the **pristine** checkpoint (the `theseus
prepare` code path). It is **not** equivalent to base under the frozen gate: `max|Δlogit| 0.96`,
teacher-forced top-1 agreement `0.99487` (below the 0.995 bar), mean KL `9.8e-05`, PPL 17.7070.
Cause: the value-subspace Hadamard (`canon_g1`) rewrites every v/o entry, and the continuous
`canon_g3`/`canon_g7` multipliers are not representable in bf16 — so "repair" costs about as much
precision as re-exporting the checkpoint at its native dtype.

Consequence for the tool design, and it is a real requirement rather than a footnote: `prepare`
must either emit higher-precision artifacts, restrict itself to lattice-exact families (the
`pow2` modes), or refuse when the induced drift exceeds the user's declared tolerance. Theseus
judging checkpoints must be judged the same way.

### The G5 repair returned the original file, byte for byte

`canon_g5` gets one artifact: a checkpoint whose embedding was scaled by 8 and whose head was
detached from it. It never sees the base. Recovering the scale from the tie witness (`c` by least
squares, `embed ≈ c·lm_head`) and applying the whole-artifact inverse move produced

```text
sha256 m1/work/g5_c8_rep/model.safetensors        = 88c142557820ccad55bb59756bfcfcf891de9cc6
sha256 ~/.cache/huggingface/hub/.../88c1425578…    = 88c142557820ccad55bb59756bfcfcf891de9cc6
tensors differing from the pristine checkpoint     = 0 / 290
equivalence metrics                                 max|Δlogit| 0, KL 0, top-1 1.00000
```

The same digest for `g5_c8_eps_rep`, and `g5_c8` / `g5_c8_eps` share a digest with each other
(49/290 tensors differing from base, `+ lm_head.weight`) — confirming the `ε → c²ε` variant changes
*only* the config, never a weight. For a stress built from a power of two, bf16 storage is exactly
representable, so the repair is not "close to base", it **is** base. That is the gauge-fixing
property `math.md §7` wanted: a canonical representative per orbit, verified on a 0.5 B checkpoint.

## 3. Pre-registered prediction (written before any surgery number existed)

`m1/predict.py` snapshots a purely static quantity — block-max-abs 4-bit conditioning `J`
(`quant_condition`, 32-element blocks, the unit llama.cpp K-quants actually round) — and the
"debt" `J(variant) − J(base)`. Committed as `m1/work/PREDICTIONS.json` /
`PREDICTIONS_new.json` before the probe queue ran:

| predicted | variants |
|---|---|
| **Q4 damage expected** (debt > 1e-3) | `g3_rand` **+0.01852**, `g7_rand` **+0.00371** |
| repair returns it to neutral | `g3_rand_rep` +0.00021, `g7_rand_rep` +0.00000, `g3_smooth_rep` +0.00021 |
| **quantization-neutral expected** | `g1_haar` −0.00009, `g1_svd` +0.00012, `g2_rand` +0.00001, `g4_perm`, `g6_perm`, `g7_few`, controls |

Note what this predicts *against* the popular story: rotating the value/output subspace (G1) is
expected to be **harmless to a weight-only K-quant**, because QuaRot-style gains come from
activation and KV-cache outliers, which a 4-bit weight-only GGUF quantizer never touches. If the
measured panel agrees, the interesting claim is not "rotations change quantization" (SpinQuant
already showed that, up to 13 points) but **which gauges matter to which operation**.

## 4. Surgery calibration on the untouched checkpoint

Reference-relative contracts, fixed on base *before* any variant was measured:

* **llama.cpp Q4_K_M on pristine base** (f16 GGUF of the same weights is the reference;
  32 KiB corpus slice for PPL, 8 KiB for KL, Vulkan):
  `ppl 12.1399 → 12.4159` (`rel_dppl +2.27 %`), mean KLD `0.0319` nats,
  p99.9 `0.3998`, max `0.4043`, artifact `379 MB` vs `948 MB`, 339 s per sweep.
  Consequence: an absolute cap like `rel_dppl ≤ 0.02` or `kl_mean ≤ 0.01` fails the *pristine*
  checkpoint, so the quantization contract is `≤ base + 0.010 rel_dppl` and `≤ base + 0.005 KLD`
  with greedy prefix retention `≥ base − 0.10`. Exact 32-token equality was also abandoned as a
  statistic: base Q4 scores `0.00` on it because one divergent token zero-codes a whole prompt —
  the probe now reports mean shared greedy prefix length.
* **bounded LoRA r16 on pristine base** (deterministic synthetic rule: reverse a 10-digit
  identifier; 512 train / 128 held-out, 80 steps, batch 2 × seq 128, lr chosen from {3e-4, 3e-3}):
  task loss `2.7228 → 0.0733`, capture `0.9731`, selected lr `3e-4`, and protected WikiText PPL
  `16.9918 → 19.9516` (**+2.96**). Again the absolute cap was wrong: honest LoRA costs general-PPL
  collateral at this budget, so the contract is reference-relative
  (`capture ≥ 0.75 × capture_ref` and `protected_dppl ≤ dppl_ref + 0.02`).

## 5. Status of the surgery panel

Phase 1 (the equivalence gate above) is complete for 18 checkpoints. Phase 2 — the same three
real operations run on base and on each stressed/repaired pair, with three replication seeds for
the combined stress — is executing under `m1/queue.sh`, which refuses to probe any checkpoint that
did not pass the gate. Its output lands in `M1_TABLE.md` (per-op damage + pass/fail per
checkpoint), `M1_ANALYSIS.md` (rank correlation of static `J` against measured damage, i.e. the
M6 seed), and `m1/work/m1_optionality.svg`. I will report the panel numbers as they land; nothing
in this file depends on them.

## 6. What is already established, independent of the panel

1. A real 0.5 B decoder-only Transformer has at least five *exactly* function-preserving
   reparameterizations that ordinary local-ML tooling does not quotient out: GQA value-subspace
   basis change, the full RoPE-compatible q/k commutant, RMSNorm scale absorption, SwiGLU
   up-branch diagonal, residual-stream scaling (the last one requiring the embedding tie to break,
   and exact only with a `ε → c²ε` config edit).
2. Some of them are invisible to bf16 storage arithmetic and some cost representation noise —
   measurable, and the lattice trick removes that cost.
3. A cheap artifact-only statistic (32-block conditioning) already separates the gauges that
   should matter to a 4-bit weight quantizer from those that should not, *before* running the
   surgery — the pre-registration in §3.
4. V0's ReLU scaling gauge survives into a GLU transformer (G7) in the multiplicative branch — a
   correction our own prior-art pass caught before it went into a claim.
