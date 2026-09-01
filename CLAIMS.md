# CLAIMS — live register

State vocabulary: `UNSUPPORTED → PRELIMINARY → CONTROLLED → CONFIRMED`, or `REFUTED`
(`SYSTEM.md` §1, `SCHEMA.md` §3). Pointers are today evidence locations; `theseus import-m1`
re-points them at ledger cell ids without changing a number. Nothing here is upgraded in prose:
if the obligations table says `needs 2`, the state says CONTROLLED, not CONFIRMED.

---

## K-1 — Qwen2.5-0.5B admits at least five exact gauges that ordinary tooling does not quotient out

**State: CONTROLLED.**

| obligation | evidence |
|---|---|
| equivalence, fp32 | `m1/work/equiv/*.json` — 18 checkpoints, mean KL ≤ 2.0e-4 nats, top-1 ≥ 0.9951 over 4,096 positions; lattice gauges exactly 0 |
| equivalence, bf16 compute | `m1/work/compute_dtype_check.json` — `g3_pow2` exactly 0.00e+00; `g7_rand` top-1 0.98096 (**see refuter**) |
| controls | null-gauge and permutation: `m1/test_gauge_math.py` (13 properties, exit 0), `g4_perm` dlogit 1.26e-04 |
| replication | 3 stress seeds (`bad_all`, `_s2`, `_s3`) |
| second architecture | Qwen3-0.6B-Base, 4 measured equivalence cells in `m1/work-qwen3/equiv/`: g3_pow2 **bit-identical** (max\|Δlogit\| 0.0, KL 0.0, top-1 1.00000), g1_haar KL 6.3e-06, g5_c8 KL 1.1e-07, g7_rand EQUIVALENT but tripwire-flagged at 0.669 (vs 0.315 on Qwen2); G2 **refused** on 56 QK-norm tensors |
| algebra | `M1_NOTES.md` §2; `inspect/` unit tests 1-2 pin the dtype arithmetic |

**Refuter:** any gauge family whose equivalence cell shows KL > 2e-3 or top-1 < 0.995 in *both*
compute dtypes. Partially fired for `g7_rand` under bf16, and again on Qwen3 where its max\|Δlogit\| is 0.669 against the 0.5 tripwire compute → the claim is stated as
"exact in fp32 arithmetic, gauge-dependent under bf16", which is weaker and true.

---

## K-2 — Export format is an operation, and it can destroy a function-equivalent checkpoint

**State: CONTROLLED** (n = 1 artifact, but with the identity control in place).

`g3_pow2` (logits bit-identical to base in fp32 and bf16 compute, ppl 16.9471/16.9889 unchanged):

| export of the same weights | perplexity |
|---|---:|
| bf16 (native dtype) | 12.1351 (base bf16: 12.1399) |
| f16 | 177.3286 |
| f32 | 177.3922 |
| f16 with f32 KV cache | 177.3286 |

Supporting census: 9.87 % of `g3_pow2`'s weights fall below f16's normal range (base: 0.28 %),
minimum 1.8e-11 — `m1/work/export_damage.json`, `inspect` per-family numbers.
Pipeline-trust control: base bytes written through my own `save_state` re-export to f16 at
ppl **12.1399**, identical to the pristine checkpoint, so the divergence is the gauge × runtime,
not the writer (`m1/check_gguf_layout.py` + `m1/work/gguf/g1_haar.layout.json` confirm no q/k
re-layout and no norm absorption on the Qwen2 import path).

**Refuter:** an export-dtype sweep on a second artifact where f16 stays within 1 % of bf16 while
the static census still shows > 5 % subnormal weights — which would mean my mechanism story
(underflow) is wrong even though the numbers are right.

**Practical consequence for a user:** `theseus-scan preflight` rates each operation from the
artifact's own bytes in about a second. Reporting is the default and exits 0; the gate is opt-in
via `--fail-on-risk`, which exits 1 only when a **judged** row says AT_RISK - a row that is
`UNAVAILABLE` (merge, awlora) or `UNKNOWN` (Q4, whose fit was refused) never trips it, because
declining to claim is not the same as claiming badly. `theseus-inspect --fail-above FRAC` is a
separate, numeric gate on one statistic. Reproducing the g3_pow2 row needs the artifact, which is
gitignored: rebuild it with `python m1/make_variants.py --only g3_pow2` after `m1/prep_data.py`.

---

## K-3 — Function-equivalent checkpoints have materially different adaptation reserve

**State: CONTROLLED.** The corrected probe freezes every base parameter before installing
rank-16 adapters. Three optimizer seeds per artifact establish the effect:

| checkpoint | mean capture | range | SD | gap vs base |
|---|---:|---:|---:|---:|
| `base` | 0.9705 | 0.9509–0.9860 | 0.0146 | — |
| **`g3_pow2`** | **0.0989** | 0.0400–0.1988 | 0.0710 | **−87.2 pp** |
| `g3_pow2_rep` | 0.9753 | 0.9639–0.9876 | 0.0097 | +0.5 pp |
| **`g7_rand`** | **0.1931** | 0.1537–0.2226 | 0.0290 | **−77.7 pp** |
| `g7_rand_rep` | 0.9359 | 0.8853–0.9806 | 0.0391 | −3.5 pp |

**Cross-architecture replication (Qwen3-0.6B-Base, same contract, same 3 seeds, all at one commit).**
base `0.9613 ± 0.0080`, `g3_pow2` `0.2511 ± 0.0211` (gap **−0.710**), `g3_pow2_rep` `0.9376 ± 0.0744`
(gap −0.024). Largest within-variant sd `0.0744` puts the 3σ bar at `0.223`, which `g3_pow2` clears by
over 3x: **K-3 holds on a second architecture**. Cells: `m1/work-qwen3/seed_replicate.json`.

Kept attached rather than smoothed: the seed drives LoRA **init only** - `seed_replicate.py:72-73`
pins the data via `RULE_SEED` - and an extra Qwen3 init gave base capture **0.676** against the
panel's 0.951-0.971. Qwen3 is markedly more init-sensitive than Qwen2 (0.951-0.986 across 3 seeds),
so three seeds are thin on this model and its base mean is provisional.

The registered gate is conservative: a gap must exceed three times the largest within-variant SD
across the panel. Only G3 and G7 pass it. G1, G2 and the head-permutation control do not.
`m1/work/seed_replicate.json` carries the contract and all per-seed grids.

**Refuter:** G3 and G7 fall below the 3σ bar on a fresh three-seed panel.

## K-4 — …and different quantization reserve at the same bit-width

**State: CONTROLLED** (same corpus, same toolchain, damage measured against each artifact's own
bf16 reference; base calibration cell present).

| checkpoint | Q8_0 KLD | Q4_K_M KLD (× base) | Q4 rel ΔPPL | verdict |
|---|---:|---:|---:|---|
| `base` | 0.00094 | 0.03191 (1.00×) | +2.20 % | reference |
| `g3_pow2` | **10.690** | undefined (no overlap) | +2.62e6 % | fail |
| `g3_pow2_rep` | 0.00106 | 0.03501 (1.10×) | +1.84 % | pass |
| `bad_all` (4 families) | unavailable | unavailable | unavailable | adaptation + both merges fail |
| `bad_all_exact` | 0.00102 | 0.03095 (0.97×) | +3.27 % | Q4, adaptation and both merges fail |
| `g7_rand_rep` | 0.00088 | 0.03137 (0.98×) | +2.14 % | pass |
| `g1_haar` | 0.00091 | 0.03200 (1.00×) | **+3.97 %** | **fail on ΔPPL, neutral on KL** |
| `g4_perm` | 0.00095 | 0.03194 (1.00×) | +2.29 % | pass |
| `g5_c8` | 0.00089 | 0.03252 (1.02×) | — | pass (predicted neutral ✓) |

**Cross-architecture replication (2026-08-31).** On Qwen3-0.6B-Base the same `g3_pow2` gauge is again bit-identical (max|Δlogit| 0.0, KL 0.0, top-1 1.00000) and quantization again destroys it: bf16-export ppl 12.004 on both artifacts, then Q8_0 ppl 1.20e9 and Q4_K_M KLD 18.83 versus base 0.001491 / 0.091089. The static cause transfers nearly numerically (J 0.01020→0.02886 vs Qwen2 0.01123→0.02955; dyn range 8.66→14.54 vs 8.83→14.6; frac below f16 normal 0.0021→0.0860 vs 0.00282→0.0987; flags 0→21 vs 0→15). Cells: `m1/work-qwen3/*.gguf.json`, `*.static.json`. Adaptation and merge reserve have NOT been measured there.

`g1_haar` is the interesting one: distributionally indistinguishable, perplexity-wise over the
limit. Recorded as a disagreement between two damage statistics rather than smoothed into one.

**Refuter:** a rerun of `g1_haar` under a longer corpus where its ΔPPL falls inside base+slack,
which would mean the effect is corpus sampling variance, not gauge state.

---

## K-5 — An artifact-only lattice canonicalizer restores G3/G7 adaptation reserve

**State: CONTROLLED.** The repair sees only the stressed artifact. It applies the exact
power-of-two G5/G3/G7 path, then the same three-seed true-LoRA probe:

- `g3_pow2`: 0.0989 mean capture → `g3_pow2_rep`: **0.9753**;
- `g7_rand`: 0.1931 → `g7_rand_rep`: **0.9359**;
- all four repaired/stressed artifacts pass the fp32 equivalence gate; G3 is bit-identical in
  bf16 compute too.

This is operation-specific. `bad_all_exact` clears every static flag but still fails adaptation
and both merge operators. The full non-lattice canonicalizer is diagnostic-only: `m1/rescue.py`
refuses to ship it even when a finite token probe happens to return `EQUIVALENT`.

On Qwen3-0.6B-Base the repair also replicates: `g3_pow2` 0.2511 → `g3_pow2_rep` 0.9376, landing within 0.024 of that panel's base. G7 has not been measured there.

**Refuter:** either repaired G3/G7 mean remains more than 3σ below base.

## K-6 — A fitted static threshold predicts Q8 risk; Q4 remains unsupported

**State: PARTIAL.** Nine new equivalence-gated artifacts were measured with CPU llama.cpp,
native-bf16 export and the original corpus contract. Q8 reaches n=20 and emits immutable contract
v3 at `q4_block_mse > 0.01282348`: recall 1.0, precision 0.40, specificity 0.833. Eight v2 verdicts
are invalidated through edges, not rewritten. The Rust scanners now use this threshold for Q8.

Q4 also reaches n=20 but does not pass the information gate: its recall-preserving cut has
precision 0.278, below the required 0.3125. Export and adaptation remain below n=20. Contract v3
is in-sample calibration, not an out-of-sample predictor.

**Refuter:** any measured Q8 false negative, or out-of-sample recall below 0.95.

## K-7 — Reserve is a vector; no scalar summarizes it

**State: CONTROLLED.** `prep_base_exact` is the clean counterexample to a scalar health ordering.
It is equivalent to base and improves Q4 relative ΔPPL from +2.195 % to +2.010 %, yet it fails
both calibrated merge operators. Its single-seed true-LoRA row passes. One coordinate change can
improve one future operation and reduce another, so the schema rejects scalar `health` fields.

## K-10 — Lattice prepare improves Q4 on a pristine checkpoint across two corpora, but hurts merge

**State: CONTROLLED.** `prep_base_exact` applies the exact lattice path to untouched
Qwen2.5-0.5B. It never sees a stressed ancestor.

| corpus | base Q4 rel ΔPPL | prepared Q4 rel ΔPPL | prepared minus base |
|---|---:|---:|---:|
| original `[0,32768)` | +2.195 % | **+2.010 %** | **−0.185 pp** |
| disjoint `[65536,98304)` | +2.608 % | **+2.131 %** | **−0.478 pp** |

The second slice has sha256 prefix `c2cc1b4175c60879`. Base and prepared fp32 outputs are exactly
equal on it: KL 0, top-1 1.0, max Δlogit 0. Q8 also improves slightly. The same prepared artifact
still fails both calibrated merge operators, so `prepare` needs an operation target.

**Refuter:** a third corpus or architecture where prepared Q4 no longer beats base, or a prepared
merge cell that passes under the same contract.

## K-8 — Natural histories, not constructed gauges, produce divergent reserves

**State: UNSUPPORTED, and the earlier attempt record is quarantined as non-evidence.**

`m3/results.json` describes an ordered-history pair (`adapt → merge → Q4` versus
`merge → adapt → Q4`, identical budgets and seeds) that matched static features at tolerance 0.05
and PPL within 0.054 % but failed the registered present-match gate: mean KL 0.032311, teacher-forced
top-1 agreement 0.88235. **Those numbers cannot be cited.** The committed generator cannot produce
them: `m3/history_pair.py::train_lora_state` called `opt.step()` with no optimizer ever constructed,
`CONTRACT.adapt.lr` was declared but never consumed, and the recorded `adapt` dicts lack the
`capture` / `task_loss_before` / `task_loss_after` keys that function always returns. No commit of
that file could run (incident #18). `m3/selfcheck.py` reported PASS the whole time because it replaced
`train_lora_state` with a lambda, so the "future-path" check certified a path that had never executed.

The file is kept deliberately, as a failed-attempt record, not deleted. K-8 has still never been
tested: it needs a re-run under the fixed harness before any claim about natural histories, positive
or negative, is made.

The first re-run under the repaired harness is an exploratory screen (`m3/screens/`, never evidence,
never ledger-admitted) and it closes the obvious route: sweeping merge alpha over 0.30 → 0.02 leaves
mean KL at 0.029-0.048, i.e. **15-24× outside** the `2e-3` gate, and non-monotone in alpha. Weakening
the operations does not make the two orders agree, because they adapt from different coordinates.

Qualification that has to stay attached to those figures: that sweep executed `m3`'s then-private
copy of the training loop, which used AdamW's default `weight_decay=1e-2` instead of the contract's
`0.0`. The duplicate has since been deleted in favour of delegating to `m1/adapt_probe.train_once`.
The plateau is unlikely to be a weight-decay artefact, but the numbers are superseded until re-run.

Equally important, a flattering detail from the withdrawn record does **not** survive re-running: the
claim that a pair sat inside the perplexity tolerance while failing the distributional one was
specific to the void numbers. At every screened alpha, **both** statistics fail (`rel ΔPPL` 0.0069 to
0.0343 against a 0.005 limit). That observation is therefore void too and is not carried forward.

A second construction axis has since been screened through the consolidated trainer
(`m3/order_screen.json`): the **order** of real sequential adaptations, same multiset of steps per
arm. Every arm fails the gate by 21-53x on KL, so order is not invisible to current evaluation at
this scale - while it moves reserve hard, one arm ending at capture **-0.0547** against the other's
`0.9540`. Two natural routes to a present-matched pair are therefore measured closed: weakening
merge strength, and swapping operation order. "Hidden from ordinary evaluation" is now the specific
part of this claim with no support, and the honest reading is that history changes the future
emphatically at 0.5B but changes the present too.

**Refuter:** two present-matched history pairs surviving the registered shuffled null would support
it; a present-matched pair whose reserves are equal would refute it. Neither has been earned.

## K-9 — Merge compatibility is gauge-dependent

**State: CONTROLLED.** The corrected specialist is true LoRA, saved-artifact verified and
calibrated so base passes linear merge at α=0.3 and TIES-trim at α=0.4. Eleven gauged/prepared
representatives fail both operators; G5 is the exception, passing linear at α=0.4 while failing
TIES-trim. The result depends on coordinates and on the merge algorithm.

The tied-head storage bug found on G5 is fixed in `common.merge_sd`: tied checkpoints materialize
`lm_head.weight` from the embedding before key comparison; every other key mismatch fails closed.

**Attempted on Qwen3-0.6B-Base and refused, not fabricated.** Running `merge_probe` there fails
its own specialist quality gate: rule learning passes easily (`rule_loss` 0.0717 against a 0.962
ceiling) but collateral perplexity on the rule holdout is **45.46 where the gate allows 42.44**
(1.5 × base 28.2955, measured on the rule task rather than on WikiText). The probe raised `SPECIALIST_GATE_FAILED_LIVE` and wrote error cells for all
three candidates (`m1/work-qwen3/*.merge.json`) rather than emitting merge verdicts from a
specialist that wrecked the model. This claim is therefore still single-architecture, and the
blocker was two-stage. Stage one was real: the 600-step / lr 3e-4 budget breached the collateral term. A calibration sweep (`m1/calibrate_specialist.py`, 7 configs, ALL passing at the softer budget) fixed it - keeping Qwen2's rank 32 and alpha/rank = 1 since rank changes merge arithmetic, and only softening the optimisation to 150 steps at lr 1e-4 gave rule loss 0.0194 with rule-holdout ppl 29.17, inside the 42.44 ceiling. Stage two is a contract defect, logged as incident #21: with a specialist that good, the absolute term `rule_loss_ratio <= 0.75` demands merged loss below 0.0146, better than the specialist itself, so the **pristine base cannot pass** (1.227 at a=0.5). No K-9 verdict can be issued there. What IS measured, and striking, is the perplexity axis where the contract is reference-relative: linear merge ppl_ratio base **1.00**, `g3_pow2` **696,800**, `g3_pow2_rep` **1.057** (`m1/work-qwen3/*.merge.json`) - the gauge destroys mergeability on Qwen3 too and the repair restores it. Mechanism replicated, verdict withheld until the retention frontier is calibrated per architecture.

**Refuter:** the base reference fails on rerun, or two independent gauged representatives pass
both operators under the same calibrated contract.
