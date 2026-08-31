# Exploratory screens (NOT evidence)

Files here are screening output. They are never admitted to the ledger and never cited as support
for a claim, because they were produced while searching for a configuration rather than under a
pre-registered contract. Same standing as a pre-registration's predicted rows: informative about
where to look, worthless as a result.

## 2026-08-31 — merge-alpha sweep for K-8

Runs: `2026-08-31_alpha_screen_run1.json` (4 configs, no scanner built),
`2026-08-31_alpha_screen_run2.json` (3 configs, scanner present; reproduces run 1 exactly).

Purpose: after incident #18 (the K-8 harness could not execute, so `m3/results.json` was withdrawn),
find whether weakening the merge makes an order-swapped natural pair pass the frozen present-match
gate (`mean_kl <= 2e-3`, `top1 >= 0.995`, `relative_ppl <= 5e-3`) while leaving reserve differences
worth probing.

This is the **first genuine execution** of `adapt -> merge -> Q4_K_M` vs `merge -> adapt -> Q4_K_M`
with a working optimizer. ~97 s per configuration.

| merge α | mean KL | top-1 | rel ΔPPL | gate | B adapt capture |
|---:|---:|---:|---:|---|---:|
| 0.30 | 0.047975 | 0.88235 | 0.0343 | fail | 0.9364 |
| 0.15 | 0.037317 | 0.88235 | 0.00853 | fail | 0.9513 |
| 0.06 | 0.029418 | 0.90196 | 0.01375 | fail | 0.9544 |
| 0.02 | 0.031228 | 0.89804 | 0.00692 | fail | 0.9787 |

Findings:

1. **Weakening the merge does not close the distributional gap.** KL plateaus around `0.029-0.048`,
   i.e. 15-24x outside the `2e-3` gate, and is *not* monotone in alpha. At `alpha = 0.02` the merge is
   nearly a no-op and the pair is still distributionally distinct. The divergence therefore comes from
   adapting from different starting coordinates (plus Q4 rounding of two different bf16 parents), not
   from merge strength. The "just make the operations weaker" route to a present-matched natural pair
   is dead.
2. **`alpha = 0.30` reproduces nothing from the withdrawn `m3/results.json`**, which reported
   KL `0.032311` and rel ΔPPL `0.00054` at exactly that alpha. Independent confirmation for incident
   #18 that the old JSON did not come from this code.
3. A-side capture is constant (`0.929842`) across the sweep as it should be, because A adapts before
   merging; B-side capture rises as alpha shrinks, because B then adapts from a near-pristine base.
4. Run 1 recorded `NA` reserve-feature gaps because `scan/target/release/theseus-scan` had not been
   built in-tree. Run 2 rebuilt it and reproduced run 1's gate numbers **exactly** (KL `0.047975`,
   `0.029418`, `0.031228`), so the fixed harness is deterministic, not just executable.

Run 2 static-feature gaps between the two orders (relative, same ladder the contract uses):

| merge α | `q4_block_mse` | `dyn_range_log10` | `row_energy_imbalance` | `frac_below_f16_normal` |
|---:|---:|---:|---:|---:|
| 0.30 | 0.0001 | 0.0019 | **0.0750** | 0.0066 |
| 0.06 | 0.0000 | 0.0000 | **0.0942** | 0.0019 |
| 0.02 | 0.0001 | 0.0000 | **0.0171** | 0.0043 |

5. The two orders are almost indistinguishable in the features that drive **quantization** reserve
   (`q4_block_mse` and `dyn_range_log10` gaps at or below `2e-3`) while the **`row_energy_imbalance`**
   gap - the feature M1 identified as the adaptation killer - is 10-90x larger and tracks the
   capture difference between the orders. If K-8 has a real effect at this scale, this is where it
   should show up: divergent adaptation reserve at effectively identical quantization reserve.
   Screening cannot establish that; only a gated re-run can.

Implication for K-8: the gate itself, not the pair, is the obstacle. Either "same present" has to be
redefined as *indistinguishable to ordinary evaluation* (with distributional drift reported rather
than gated), or pair construction needs a mechanism that equalizes the endpoint by construction
instead of shrinking the operations. Decided as an open question, not silently.

## Caveat on the alpha sweep, found while fixing weakness #1

The sweep above ran through `m3/history_pair.py`'s **private copy** of the LoRA training loop. That
copy called `AdamW(params, lr=...)`, inheriting AdamW's default `weight_decay=1e-2`, whereas the
contract and the real probe (`m1/adapt_probe.py`) use `weight_decay=0.0`. So the numbers above were
produced by an adaptation operation that is not the one the contract specifies.

The duplicate is now gone: `m3.train_lora_state` delegates to `adapt_probe.train_once`, which owns
the single implementation. Directionally the plateau is unlikely to be a weight-decay artefact
(KL barely moves across a 15x range of alpha), but the specific values are superseded and the sweep
should be re-run under the consolidated trainer before it is quoted anywhere.
