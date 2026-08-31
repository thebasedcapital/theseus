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
| algebra | `M1_NOTES.md` §2; `inspect/` unit tests 1-2 pin the dtype arithmetic |

**Refuter:** any gauge family whose equivalence cell shows KL > 2e-3 or top-1 < 0.995 in *both*
compute dtypes. Partially fired for `g7_rand` under bf16 compute → the claim is stated as
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

**Practical consequence for a user:** `theseus preflight` answers `export.gguf.f16: AT_RISK` for
this artifact in 2 s and exit 1, before anyone downloads 400 MB of noise.

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

**Refuter:** either repaired G3/G7 mean remains more than 3σ below base.

## K-6 — The current provisional static thresholds predict operation risk

**State: REFUTED.** The static features remain useful measurements, but the current n=2 thresholds
are not a predictor. On the frozen measured slice, Q4 has TP=1, FN=2 at n=10. A preflight tool
cannot tolerate those false negatives. `analysis/thresholds.py` prints candidate cuts but refuses
to emit a replacement until each flag has at least 20 labelled cells and precision reaches 1.25×
the failure base rate.

The honest surviving statement is narrower: G3/G7 stress raises the expected conditioning and
their lattice repairs remove it. That does not license the current thresholds for new artifacts.

## K-7 — Reserve is a vector; no scalar summarizes it

**State: CONTROLLED.** `prep_base_exact` is the clean counterexample to a scalar health ordering.
It is equivalent to base and improves Q4 relative ΔPPL from +2.195 % to +2.010 %, yet it fails
both calibrated merge operators. Its single-seed true-LoRA row passes. One coordinate change can
improve one future operation and reduce another, so the schema rejects scalar `health` fields.

## K-10 — Lattice prepare helps Q4 on a pristine checkpoint but hurts merge reserve

**State: CONTROLLED.** `prep_base_exact` applies the exact lattice path to untouched
Qwen2.5-0.5B. It never sees a stressed ancestor.

| artifact | equivalence | Q4 rel ΔPPL | true-LoRA capture | linear merge | TIES-trim |
|---|---|---:|---:|---|---|
| `base` | reference | +2.195 % | 0.9860 | pass at α=0.3 | pass at α=0.4 |
| `prep_base_exact` | EQUIVALENT | **+2.010 %** | 0.9900 | fail | fail |

The Q4 gain is real but small and measured on one corpus. The merge loss is large enough to cross
both calibrated contracts. `prepare` therefore needs an operation target; “make the model
healthier” is not a valid command.

**Refuter:** a repeated Q4 corpus where the prepared artifact no longer beats base, or a prepared
merge cell that passes under the same contract.

## K-8 — Natural histories, not constructed gauges, produce divergent reserves

**State: UNSUPPORTED. Highest-value open claim.** Two matched pairs at 0.5B:
`sft→merge→q4` vs `merge→sft→q4`, matched on L0 features and current behaviour, compared on
adaptation and re-quantization reserve. The ledger already stores `ancestry` edges, so this is a
selection query, not a new mechanism. Until this lands, M1 remains "a symmetry curiosity with
excellent evidence".

## K-9 — Merge compatibility is gauge-dependent

**State: CONTROLLED.** The corrected specialist is true LoRA, saved-artifact verified and
calibrated so base passes linear merge at α=0.3 and TIES-trim at α=0.4. Eleven gauged/prepared
representatives fail both operators; G5 is the exception, passing linear at α=0.4 while failing
TIES-trim. The result depends on coordinates and on the merge algorithm.

The tied-head storage bug found on G5 is fixed in `common.merge_sd`: tied checkpoints materialize
`lm_head.weight` from the embedding before key comparison; every other key mismatch fails closed.

**Refuter:** the base reference fails on rerun, or two independent gauged representatives pass
both operators under the same calibrated contract.
