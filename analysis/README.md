# analysis — base rates, threshold fitting, matched pairs

Read-only statistics layer over the Theseus scan/measure pipeline. Every module is
`stdlib + numpy` only, owns its own data under `analysis/data/`, and treats missing
evidence as `unavailable`, never as `False` (SYSTEM.md I8).

Modules (this slice owns all of them):

| module | job |
|---|---|
| `loader.py` | reads Inspector schema v1 JSON scans, harvest `manifest.jsonl`/`edges.jsonl`, ledger `cell` records, and an explicit `labels.jsonl`. Missing paths print `no data yet: <path>` and yield empty frames — never an exception, never invented rows. |
| `risk_flags.py` | the single definition of a risk flag: feature, threshold, aggregate rule, damage metric, and the one stated catastrophic-divergence multiple (100x the operation's reference measurement). |
| `baserates.py` | per-flag prevalence (per family, per **architecture**, per artifact) with hand-implemented Wilson 95% intervals; measured confusion matrices (sensitivity/specificity + intervals); catastrophic-divergence rates. Zero measured outcomes => `outcomes: unavailable (reason)`. |
| `thresholds.py` | sweeps >=5 candidate cuts per flag with precision/recall; chooses the largest cut with recall >=0.95, ties on precision; emits a NEW contract version with its invalidation set only when the gate passes (>=20 labelled cells and precision >=1.25x the fail base rate). History is never rewritten. |
| `pairs.py` | matched-lineage-pair finder for claim K-8: >=3 tolerance sensitivity points (0.05…0.8) and a mandatory >=200-shuffle outcome-permutation null with empirical p per tolerance. No null, no claim. |
| `fixtures.py` | synthetic labelled data with known ground truth (planted one-family effect + pure-noise variants). |
| `test_analysis.py` | `python -m unittest discover -s analysis` — asserts effect recovery, null silence, no-data behaviour, Wilson closed-form endpoints. |

## Run in the no-data state

Every CLI must be callable with no inputs and exit 0 reporting the absence:

```sh
python analysis/baserates.py  --root /tmp/empty_dir     # prevalence n=0, outcomes: unavailable
python analysis/thresholds.py --root /tmp/empty_dir     # contract unchanged: nothing emitted
python analysis/pairs.py      --root /tmp/empty_dir     # pairs: unavailable (no lineage edges)
```

Missing scan dirs, harvest slices, ledger dirs and label files each print
`no data yet: <path>` and produce empty frames.

## Run against real data

`analysis/data/evidence/` is the frozen evidence slice (20 scanned artifacts, 34 measured
pass/fail labels); `analysis/data/harvest/` is a snapshot of the live harvest cache
(`harvest/cache/`, 390 manifest entries, 352 lineage edges). Point `root` at the freeze:

```sh
python analysis/baserates.py  --root analysis/data/evidence
python analysis/thresholds.py --root analysis/data/evidence      # gates on n; see below
python analysis/pairs.py --root analysis/data --scans analysis/data/evidence/scans.jsonl --labels analysis/data/evidence/labels.jsonl --null 200
```

Contract output lands in `analysis/data/contracts/` as `contract-<N>.json`, never an edit of
an older version; each file carries the `invalidates` list of prior verdicts it flips.

## Trap 1 — the f16-export confound (K-2 / incident #10)

An `export.gguf.f16` of a wide-dynamic-range artifact collapses (g3_pow2: bf16 ppl 12.14,
f16 ppl 177). Any "quantization damage" measured from an **f16-exported** source is really
*export* damage, not the quantizer's. Consequences enforced here:

- `export.f16` is its own operation/flag, with its own reference (`ppl_ratio_f16_over_bf16`,
  pristine base = 1.0004). It is never folded into quant damage.
- A damage number is only compared to a **same-operation** reference cell
  (SCHEMA I3: same dtype/conditions), and `risk_flags.is_catastrophic` refuses a ratio whose
  reference is below `REF_ABS_MIN` (the ratio is *undefined*, never False).
- The published `frac_below_f16_normal` census (9.87% of g3_pow2's weights below 2^-14 vs
  0.28% base) is the *static* predictor for this flag; the measured ratio is the outcome.
  Both appear in `baserates.py` (prevalence vs confusion) and must not be confused with each
  other.

## Trap 2 — the aggregation-convention trap (PIPELINE_FAILURES #11)

`q4_block_mse` is the **mean of per-tensor ratios**; `q4_block_mse_pooled` is the **ratio of
sums** of the same blocks. They differ by up to 5.7%, and M1 was briefly fooled by exactly
that. Rules kept here:

- The two are separate fields, loaded separately, never averaged together.
- `risk_flags` thresholds cut on `q4_block_mse` only; `_pooled` is never mixed into a
  threshold or a prevalence denominator.
- `baserates.py` treats a missing feature as *no evidence* for that row (I8): a scan that
  only recorded `pooled` cannot be e.g. called quant-safe through the per-tensor cut.
- SCHEMA requires `features.total.convention`; a scan doc without it is still loaded (the
  per-family fields are the working set) but convention is surfaced in `loader._scan_json_rows`.

## Thresholds gate (PLAN §5 / K-6 refuter)

No learned predictor before the ledger has >=20 labelled cells; a flag only fires a new
contract when its chosen cut both catches >=95% of true fails and reaches precision
>=1.25x the flag's own fail base rate — otherwise the flag is a rumor, not a predictor, and
`contract unchanged` is the honest output. `n` per flag is printed so the reader can see how
far each flag is from the gate.

## Current fitted status

The reproducible freeze now contains 26 scanned artifacts and 55 measured labels. CPU-only,
native-bf16 augmentation adds nine equivalence-gated artifacts with matching scan JSONs.

- `quant.q8_0`: n=20, contract v3 emitted at `q4_block_mse > 0.01282348`; recall 1.0,
  precision 0.40, specificity 0.833. Eight prior verdicts are invalidated, never rewritten.
- `quant.q4_k_m`: n=20, fit refused. Best recall-preserving cut has precision 0.278, below the
  required 0.3125.
- export and adaptation remain below n=20.

Identical evidence reuses contract v3 byte-for-byte. It does not emit duplicate v4 files.
