# M1 — can two function-equivalent real Transformers have different futures?

> Belief state with obligations and refuters lives in [`CLAIMS.md`](CLAIMS.md) (K-1…K-10). This
> file is the evidence narrative for milestone 1. Numbers here are cited by the claim register.

> **RESULT INVALIDATION CLOSED (2026-08-31):** The first adaptation/merge panel was full-model
> training at LoRA learning rates because embeddings, norms and the LM head remained trainable.
> Twenty-one cells are archived under `m1/work/invalidated/full_model_training/` and count toward
> no claim. The corrected probes freeze the whole base before adapters; the replacement panel and
> three-seed adaptation replication are complete. Equivalence, export, static and quantization
> measurements were unaffected.

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

The two repairs of the *same* lattice stress isolate the rounding explanation on the real model:

| artifact | repair rule | max\|Δlogit\| | mean KL | top-1 |
|---|---|---:|---:|---:|
| `g3_pow2_rep` | column-energy balance snapped to `2^k` | **0.00e+00** | **0.00e+00** | **1.00000** |
| `g3_pow2_rep_raw` | same balance, continuous multiplier | 5.40e-01 | 5.29e-05 | 0.99707 |

Same stress, same objective, same data — only the representability of the gauge parameter differs,
and the drift appears exactly where the theory says it must.

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

Same story for the combined stress: `bad_all` (G3 pow2 + G1 Haar + G2 RoPE + G7, four families at
once) certifies cleanly — `max|Δlogit| 0.333`, KL `6.7e-05`, top-1 `0.99585`, PPL 17.7095 — while
`bad_all_rep`, that artifact run through the *full* canonicalizer, fails the distributional gate
(`top-1 0.99170`, `max|Δlogit| 1.11`, KL `2.2e-04`). The stress is not the problem; the repair is.

Consequence for the tool design, and it is a real requirement rather than a footnote: `prepare`
must either emit higher-precision artifacts, restrict itself to lattice-exact families (the
`pow2` modes), or refuse when the induced drift exceeds the user's declared tolerance. Theseus
judging checkpoints must be judged the same way. All three are implemented and measured:
`bad_all_exact` / `prep_base_exact` run the repair over `{G5, G3, G7}` with `snap_pow2=True`, which
rewrites 175/290 tensors by powers of two only — different bytes, and the gate decides whether the
behaviour is still bit-identical.

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
- **bounded true-LoRA r16 on pristine base** (reverse a 10-digit identifier; 512 train / 128
  held-out, 80 steps, batch 2 × seq 128, lr chosen from {3e-4, 3e-3}): seed 1729 captures
  **0.9860** of the available task-loss reduction and moves protected WikiText PPL by **+1.224**.
  Across seeds 1729/23/44, mean capture is **0.9705**, range 0.9509–0.9860, SD 0.0146. The
  reference-relative pass contract is `capture ≥ 0.75 × capture_ref` and
  `protected_dppl ≤ dppl_ref + 0.02`.

## 4b. Importer audit: the GGUF sees the coordinates we wrote

Everything above would be meaningless if `convert_hf_to_gguf.py` re-expressed the weights before
quantizing. Audited against the actual artifacts (`m1/check_gguf_layout.py`, layout JSON in
`m1/work/gguf/g1_haar.layout.json`) and against the importer source:

* the Qwen2 GGUF carries 290 tensors with the expected per-block names, q/k/v biases at
  896/128/128, `qwen2.rope.freq_base` in metadata, and `attn_norm`/`ffn_norm` as **separate F32
  tensors** — so no RMSNorm absorption happens on export and the `G3` column scaling really is
  present when the K-quant blocks are formed;
* **no q/k row permutation or interleaving**: `conversion/qwen.py::Qwen2Model.modify_tensors` only
  re-prefixes names, whereas `llama.py` has an explicit `permute` on q/k weights and biases and
  `plamo.py` a separate `shuffle_attn_q_weight` — neither applies to Qwen2. Therefore the RoPE
  pairs my `G2` gauge rotates are the same pairs the quantizer sees, and the `G2` row needs no
  caveat.

## 5. Corrected true-LoRA adaptation: replicated gauge effects

The corrected probe globally freezes embeddings, norms, LM head and all base Linear weights before
installing rank-16 adapters. Every artifact uses the same data order, 80 steps, two-learning-rate
grid and three optimizer seeds. The registered effect gate is deliberately conservative: a gap
must exceed three times the largest within-variant SD in the panel.

| checkpoint | mean capture | range | SD | gap vs base | 3σ result |
|---|---:|---:|---:|---:|---|
| `base` | 0.9705 | 0.9509–0.9860 | 0.0146 | — | reference |
| `g1_haar` | 0.9141 | 0.8876–0.9336 | 0.0194 | −5.6 pp | below 3σ |
| `g1_haar_rep` | 0.8839 | 0.7300–0.9628 | 0.1088 | −8.7 pp | below 3σ |
| `g2_rand` | 0.9058 | 0.8794–0.9551 | 0.0349 | −6.5 pp | below 3σ |
| **`g3_pow2`** | **0.0989** | 0.0400–0.1988 | 0.0710 | **−87.2 pp** | **effect** |
| `g3_pow2_rep` | **0.9753** | 0.9639–0.9876 | 0.0097 | +0.5 pp | restored |
| `g4_perm` | 0.8827 | 0.7788–0.9839 | 0.0837 | −8.8 pp | below 3σ |
| `g5_c8` | 0.9813 | 0.9722–0.9860 | 0.0065 | +1.1 pp | below 3σ |
| **`g7_rand`** | **0.1931** | 0.1537–0.2226 | 0.0290 | **−77.7 pp** | **effect** |
| `g7_rand_rep` | **0.9359** | 0.8853–0.9806 | 0.0391 | −3.5 pp | restored |

Only G3 and G7 clear the 3σ bar. The smaller G1/G2/permutation differences do not survive
optimizer variance and are not claims. `g3_pow2` is the strongest case: its stored artifact is
bit-identical in logits and PPL to base in both fp32 and bf16 compute, yet its mean adaptation
capture falls from 0.9705 to 0.0989. Artifact-only lattice repair returns it to 0.9753.

G7 tells the same story with a weaker equivalence condition. It is exact in fp32 arithmetic but
not bit-identical under bf16 compute; mean capture falls to 0.1931 and repair returns it to 0.9359.
The evidence therefore supports adaptation reserve as a property of artifact coordinates plus
runtime arithmetic, not of the realized fp32 function alone.

One negative result matters: `bad_all_exact` clears every static flag but still fails the
single-seed adaptation contract. Lattice repair is not a universal cure; it repairs G3/G7
adaptation in the tested orbits, not every composed gauge state.
## 5b. Quantization rows: measured under a corrected export

The first quantization pass was invalidated by a confound I found mid-run and fixed
(`PIPELINE_FAILURES.md` #10): the probe exported every artifact as **f16 GGUF**, which is what
every public quantize script does, but for a gauge that pushes weights below f16's normal range
the export itself is destructive, so "quantization damage" was mostly export damage. Measured on
one artifact, three exports of the *same* weights:

| export of `g3_pow2` | ppl |
|---|---:|
| bf16 (the artifact's native dtype) | **12.1351** (base bf16: 12.1399 → 0.04 %) |
| f16 | 177.3286 |
| f32 | 177.3922 |
| f16 with f32 KV cache | 177.3286 (so not the cache) |

and an **identity round-trip control** proves my artifact rewriting is faithful: base weights
written through `save_state` export to f16 GGUF at ppl 12.1399, exactly the pristine checkpoint's
number. The probe therefore re-exports at the artifact's native dtype (`TSX_OUTTYPE`, default
`bf16`) and `m1/export_damage.py` measures the export step as its own operation. Quantization rows
are being regenerated under that definition; the `f16`-source numbers are kept only as evidence for
this subsection.

## 5c. Quantization: same result, same repair, measured through a native-dtype export

All rows below: artifact → bf16 GGUF (its native dtype; the bf16 reference ppl is shown, base
12.1351) → real `llama-quantize` → `llama-perplexity --kl-divergence` against that same artifact's
bf16 model. Debt is the pre-registered static number from §3 (`+0.01831` for `g3_pow2` was computed
from the artifact bytes at 00:00, before its first surgery result existed at 00:18+).

| checkpoint | static debt | bf16 ppl | Q8_0 KLD | Q5_K_M KLD | Q4_K_M KLD | Q4 rel ΔPPL | Q4 pass |
|---|---:|---:|---:|---:|---:|---:|---|
| `base` | 0 | 12.1351 | 0.00094 | 0.01672 | 0.03191 | +2.20 % | reference |
| `g1_haar` | −0.00009 | 12.1246 | 0.00091 | 0.01754 | 0.03200 (1.00×) | **+3.97 %** | **FAIL (ΔPPL)** |
| `g1_haar_rep` | −0.00010 | 12.1277 | 0.00093 | 0.01745 | 0.03041 (0.95×) | +3.15 % | pass |
| `g2_rand` | +0.00001 | 12.1318 | 0.00102 | 0.01723 | 0.03256 (1.02×) | +2.97 % | pass |
| **`g3_pow2`** | **+0.01831** | **12.1351** | **10.69030** | — | — (KL undefined) | **+2,619,002 %** | **FAIL** |
| **`g3_pow2_rep`** | +0.00025 | 12.1351 | 0.00106 | 0.02053 | 0.03501 (1.10×) | **+1.84 %** | pass |

Readings that matter:

* `g3_pow2` — bit-identical logits, bit-identical bf16 export perplexity (12.1351 = base to four
  decimals) — is destroyed by **8-bit** quantization (KLD 10.7 nats) and by 4-bit (+2.6 M % ΔPPL,
  KL undefined because the distributions no longer overlap). Its repair, which only equalizes
  consumer column energy on the exponent lattice, gives 1.10× base KLD and the *best* ΔPPL in the
  table. Same function, destroyed reserve, restored reserve.
* `g1_haar` is the honest complication: its **distributional** damage is indistinguishable from
  base (KLD 1.00×, q8/q5 identical to three digits) while its relative ΔPPL is 1.8 pp worse and
  crosses the contract limit. Two damage statistics disagree, so the reserve vector must stay a
  vector — collapsing it to one score is exactly the mistake `ROADMAP.md` warns against ("a single
  mysterious health score").
| `g4_perm` (control) | 0 | 12.1 | 0.03194 (**1.00×**) | +2.29 % | pass |
| `g7_rand` | +0.00371 | — | (KL undefined: distributions no longer overlap) | — | **FAIL** |
| `g7_rand_rep` | +0.00000 | 12.1 | 0.03137 (**0.98×**) | +2.14 % | pass |

* Nine additional CPU/native-bf16 probes bring Q8 and Q4 to n=20. Q8 emits contract v3 at
  `q4_block_mse > 0.01282348` with recall 1.0, precision 0.40 and specificity 0.833. Q4's
  recall-preserving fit is refused because precision 0.278 is below the required 0.3125.
  Q8 v3 is an in-sample calibrated preflight rule; it is not an out-of-sample model.
* G3/G7 repair restores adaptation across three seeds and restores quantization where measured,
  but `bad_all_exact` shows that clearing static flags does not guarantee every operation passes.

### Pre-registered prediction ledger

| checkpoint | frozen debt prediction | measured Q4 | verdict |
|---|---|---|---|
| `base` | neutral | reference | held |
| `g2_rand` | neutral | 1.02× base KLD | held |
| `g1_haar` | neutral | 1.00× KLD but ΔPPL over limit | false negative on contract verdict |
| `g3_pow2` | damage | Q8 KLD 10.69, Q4 ΔPPL +2.6M % | held |
| `g3_pow2_rep` | neutral | 1.10× KLD, ΔPPL better than base | held |

## 5d. Merge reserve and the vector result

The corrected specialist is true LoRA, validated after saving, and calibrated so the untouched
base passes linear merge at α=0.3 and TIES-trim at α=0.4. The merge contract requires
PPL ratio ≤1.05 and retained specialist rule quality ≤0.75 of the specialist loss.

| checkpoint | linear | TIES-trim | note |
|---|---|---|---|
| `base` | pass α=0.3 | pass α=0.4 | reference |
| `g1_haar`, `g1_haar_rep`, `g2_rand`, `g3_pow2`, `g3_pow2_rep` | fail | fail | same present function, incompatible coordinates |
| `g4_perm` | fail | fail | whole-head permutation is not a merge control |
| `g5_c8` | **pass α=0.4** | fail | merge algorithm matters |
| `g7_rand`, `g7_rand_rep` | fail | fail | stress and repair both incompatible with this specialist |
| `bad_all`, `bad_all_exact` | fail | fail | clearing static flags does not repair merge reserve |
| `prep_base_exact` | fail | fail | pristine prepare improves Q4 but hurts merge |

G5 initially raised `KeyError: lm_head.weight` because its untied artifact stores an explicit head
while the tied specialist omits it. `common.merge_sd` now materializes the tied head from the
embedding on either side and rejects every other key mismatch; the rerun above is the corrected
cell.

`prep_base_exact` proves the vector claim cleanly. It improves Q4 relative ΔPPL from +2.195 % to
+2.010 % on the original slice and from +2.608 % to +2.131 % on disjoint bytes `[65536,98304)`.
The second-slice fp32 outputs are exactly equal (KL 0, top-1 1.0, max Δlogit 0). Yet both merge
operators fail. No scalar checkpoint-health ordering can preserve those operation-specific outcomes.

## 5e. First natural-history attempt: pair construction failed the present-match gate

The harness executed two ordinary histories through final Q4 artifacts:

- A: true-LoRA adaptation → linear merge → Q4_K_M;
- B: linear merge → the same adaptation → Q4_K_M.

The final artifacts match static features at tolerance 0.05 and PPL within 0.054 %, but mean KL
is 0.032311 and teacher-forced top-1 agreement is 0.88235. They are not the same current model.
The harness therefore stopped before fresh adaptation, re-quantization reserve and the shuffled
null. K-8 remains unsupported; `m3/results.json` is a failed-attempt cell, not positive evidence.

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

## Appendix — the diagnostic is cross-validated by two independent implementations

`inspect/` is a zero-dependency Rust binary that parses the safetensors container itself
(8-byte header length, JSON header, tensor data after it) and streams bf16/f16/f32 weights
without ever loading a model: 357,826,560 weights metered in **2.0 s**. Its `q4_block_mse` is an
independent implementation of the same 32-block max-abs statistic that
`canonicalize.quant_condition` computes in torch fp64, and on the pristine checkpoint and two
gauged artifacts the two agree per tensor family to `<= 4.4e-09`:

| family | python J | rust J | g3_pow2_rep_raw | g7_rand_rep |
|---|---|---|---|---|
| q_proj | 0.01132 | 0.01132 | 0.01079 | 0.01132 |
| k_proj | 0.01176 | 0.01176 | 0.01219 | 0.01176 |
| v_proj | 0.01282 | 0.01282 | 0.01428 | 0.01282 |
| o_proj | 0.01018 | 0.01018 | 0.01018 | 0.01018 |
| gate_proj | 0.01076 | 0.01076 | 0.01081 | 0.01076 |
| up_proj | 0.01069 | 0.01069 | 0.01077 | 0.01069 |
| down_proj | 0.01110 | 0.01110 | 0.01110 | 0.01111 |

Getting this right mattered: the first version pooled block statistics across tensors
(ratio-of-sums) while the registered predictions used the mean of per-tensor ratios, and the two
disagreed by up to 5.7 % on q/k/v. Both conventions are now emitted explicitly
(`q4_block_mse` = mean of per-tensor ratios, as registered; `q4_block_mse_pooled` = ratio of
sums), so a number in this project always carries its definition.

The inspector also reports the two features that the LoRA collapse in §5 should depend on and the
quantization statistic does not: `dyn_range_log10` (how much exponent the family spans — the f16
export killer) and `row_energy_imbalance` (how unequal the optimizer's per-coordinate geometry is
— the AdamW-capture killer), plus `frac_below_f16_normal` which is exactly the
`--fail-above` preflight gate for the export path.

### Inspector verdicts: the static scan against the measured outcomes

`theseus-inspect` run on each artifact (2-11 s, no model load, no GPU). Thresholds are the
provisional ones in `inspect/src/main.rs`, calibrated on the base/gauged contrast and labelled as
provisional in the binary's own output:

| artifact | total J | dyn range | frac below f16 normal | flags raised | measured surgery outcome |
|---|---:|---:|---:|---:|---|
| `base` | 0.01123 | 8.83 | 0.00282 | 0 | capture mean 0.9705; Q4 KLD 0.0319 |
| `g3_pow2` | 0.02955 | 14.6 | 0.0987 | 15 | capture mean **0.0989**; Q8 KLD **10.69**; f16 export 177 ppl |
| `g3_pow2_rep` | 0.01148 | 8.76 | 0.00281 | **0** | capture mean **0.9753**; Q4 KLD 0.0350 |
| `bad_all` | 0.03315 | **18.81** | **0.15486** | 20 | adaptation and both merges fail |
| `bad_all_exact` | 0.01159 | 9.99 | 0.00279 | **0** | adaptation and both merges still fail |
| `prep_base_exact` | 0.01144 | 9.01 | 0.00280 | **0** | Q4 improves; adaptation passes once; both merges fail |

Two things to take from this. The flags are **specific**: `g3_pow2` raises nothing on
`o_proj`/`down_proj`, which are exactly the two families the norm-diagonal gauge does not touch,
and `bad_all` adds a `down_proj` quant flag, which is where the SwiGLU up-branch diagonal lands.
And the repaired artifacts raise **no flags at all**, at total J within 2 % of the pristine
checkpoint — which is the `prepare` claim stated in the tool's own vocabulary, produced by code
that never saw the original.
