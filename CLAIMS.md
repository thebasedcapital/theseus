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

**State: PRELIMINARY** (needs replication of the *probe*, not just the stress).

Bounded LoRA r16, 80 steps, batch 2 × seq 128, identical data order and seed:

| checkpoint | capture | Δ vs base | protected ΔPPL |
|---|---:|---:|---:|
| `base` | 0.9731 | — | +2.96 |
| `g3_pow2` | **0.1559** | **−81.7 pp** | +4,316,255 |
| `g7_rand` | **0.0600** | **−91.3 pp** | +55,656,363 |
| `g2_rand` | 0.9520 | −2.11 pp | +2.54 |
| `g1_haar` | 0.9648 | −0.83 pp | +2.73 |
| `g4_perm` | 0.9088 | −6.43 pp | — |

The catastrophic rows (0.16, 0.06 against 0.97) need no error bars. The 0.8–2.1 pp rows do:
`m1/seed_replicate.py` (queued) supplies them, and PLAN §3 ranks that work ahead of anything that
merely widens the matrix.

**Refuter:** |gap| < 3·sd across seeds for the small gaps (would demote G1/G2 adaptation effects to
noise), or a base re-measurement whose spread swallows them.

---

## K-4 — …and different quantization reserve at the same bit-width

**State: CONTROLLED** (same corpus, same toolchain, damage measured against each artifact's own
bf16 reference; base calibration cell present).

| checkpoint | Q8_0 KLD | Q4_K_M KLD (× base) | Q4 rel ΔPPL | verdict |
|---|---:|---:|---:|---|
| `base` | 0.00094 | 0.03191 (1.00×) | +2.20 % | reference |
| `g3_pow2` | **10.690** | undefined (no overlap) | +2.62e6 % | fail |
| `g3_pow2_rep` | 0.00106 | 0.03501 (1.10×) | +1.84 % | pass |
| `bad_all` (4 families) | pending | pending | pending | capture 0.0906 |
| `bad_all_exact` | 0.00102 | 0.03095 (0.97×) | +3.27 % | pass, capture 0.9482 |
| `g7_rand_rep` | 0.00088 | 0.03137 (0.98×) | +2.14 % | pass |
| `g1_haar` | 0.00091 | 0.03200 (1.00×) | **+3.97 %** | **fail on ΔPPL, neutral on KL** |
| `g4_perm` | 0.00095 | 0.03194 (1.00×) | +2.29 % | pass |
| `g5_c8` | 0.00089 | 0.03252 (1.02×) | — | pass (predicted neutral ✓) |

`g1_haar` is the interesting one: distributionally indistinguishable, perplexity-wise over the
limit. Recorded as a disagreement between two damage statistics rather than smoothed into one.

**Refuter:** a rerun of `g1_haar` under a longer corpus where its ΔPPL falls inside base+slack,
which would mean the effect is corpus sampling variance, not gauge state.

---

## K-5 — An artifact-only canonicalizer restores the reserve without seeing the original

**State: CONTROLLED for G3/G1; PARTIAL for G7 (the partial is a result).**

* `g5_c8_rep` reproduced the pristine file **byte-for-byte** (`sha256` equal to the HF blob hash,
  0/290 tensors differing) — a true section of that orbit.
* `g3_pow2_rep`: capture 0.9829 (base 0.9731), Q4 KLD 1.10× base, static debt +0.018314 → +0.000251.
* `g7_rand_rep`: quantization fully restored (0.98× base KLD) but capture only to 0.8407 vs 0.9731.
* `bad_all_exact`, `prep_base_exact`: 0 inspector flags, total J within 2 % of pristine.

**Counter-obligation (kept in your face):** the *full* canonicalizer, which includes the
value-subspace Hadamard, **fails the equivalence gate by itself** (`prep_base` top-1 0.99487,
`bad_all_rep` 0.99170) because a wide-mixing rotation re-rounds every entry it touches in bf16.
`prepare` must therefore emit higher precision, restrict itself to lattice-exact families, or
refuse past a declared drift budget. This is a documented defect of my tool, not of the idea.

**Refuter:** a repaired artifact that beats base on one axis while its static debt stays > 1e-3 —
would mean the "debt" statistic is not what the repair is actually doing.

---

## K-6 — Static L0 features predict which operations are at risk

**State: PRELIMINARY (7/7 directional; n too small; refuter armed).** Predictions were frozen in
`m1/work/PREDICTIONS.json` / `PREDICTIONS_new.json` / `debts_lattice.json` before the surgery cells
they are graded against existed, and `m1/analyze.py` scores them (currently `held: 7, broken: 0`).
Thresholds in `inspect/src/main.rs` are labelled provisional with their n in the source and in every
invocation's output.

**Refuter:** any artifact where flags say OK and a measured cell fails (false negative is the only
direction that hurts a preflight tool), or ≥ 20 labelled cells with Spearman ρ < 0.3.

## K-7 — Reserve is a vector; no scalar summarizes it

**State: CONTROLLED.** `g7_rand_rep` is quantization-pristine (0.98×) and adaptation-deficient
(−13 pp). `g1_haar` is KL-neutral and ΔPPL-failing. `g4_perm` is quantization-inert and costs
6.4 pp of capture — which also forced the wording fix that "control" must always name its
operation. The schema has no scalar field for health, and `render` rejects one.

## K-10 — `prepare` improves reserve on a checkpoint nobody stressed  ← strongest practical claim

**State: CONTROLLED** (equivalence verified in fp32; replication 1; controls: null-gauge, flags).

Running the lattice-only canonicalizer (`{G5, G3, G7}` with `snap_pow2`, bf16-lossless) on the
**pristine** Qwen2.5-0.5B, compared against the pristine checkpoint measured through the identical
bf16 export path:

| | equivalence vs base | LoRA capture | Q8_0 KLD | Q4_K_M KLD | Q4 rel ΔPPL | flags |
|---|---|---:|---:|---:|---:|---:|
| `base` | reference | 0.9731 | 0.000940 | 0.031914 | +2.195 % | 0 |
| `prep_base` (full canonicalizer) | **top-1 0.99487 → NOT equivalent** | — | — | — | — | 0 |
| **`prep_base_exact`** (lattice-only) | **EQUIVALENT** | **0.9924** | 0.001017 | 0.031624 | **+2.010 %** | 0 |

Same dtype, same corpus, same tools, verified-equivalent model: **+1.9 pp adaptation capture** and
**8 % less 4-bit perplexity damage**, bought by changing nothing but the coordinates. That is the
product — not a diagnosis, a treatment — and it comes from the family of transforms that is
representable in the format people actually ship.

The combined-stress contrast keeps it honest: `bad_all` (4 families) certifies equivalent with
capture **0.0906**; `bad_all_exact` recovers to **0.9482** with Q4 KLD 0.97× base and ΔPPL +3.27 % —
recovered, with a 2.5 pp residue. Repair quality is a continuum and the register says where each
artifact sits on it.

**Refuter:** a second architecture family where lattice-prepare does not reduce Q4 ΔPPL; or the
bf16-compute equivalence cell for `prep_base_exact` (not yet run — cheap, and it is the honest gap).

## K-8 — Natural histories, not constructed gauges, produce divergent reserves

**State: UNSUPPORTED. Highest-value open claim.** Two matched pairs at 0.5B:
`sft→merge→q4` vs `merge→sft→q4`, matched on L0 features and current behaviour, compared on
adaptation and re-quantization reserve. The ledger already stores `ancestry` edges, so this is a
selection query, not a new mechanism. Until this lands, M1 remains "a symmetry curiosity with
excellent evidence".

## K-9 — Merge compatibility is gauge-dependent

**State: BLOCKED, honestly.** Cells are running; the probe's first specialist was broken (ppl 40,694,
rule loss 9.22) and its fail-closed gate correctly refused to emit a matrix rather than produce
garbage that would have looked like a result. The inspector answers `merge.linear: UNAVAILABLE`
for any artifact, because coordinates cannot be compared without a second checkpoint — that blank
is load-bearing.
