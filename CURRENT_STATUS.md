# Current Status

## Research status

**V0 controlled smoke test: PASS.**

A small ReLU classifier was trained to 98.22% test accuracy. An exact positive-homogeneity gauge transformation produced a checkpoint with the same 98.22% accuracy and maximum held-out logit difference of approximately `5.72e-06`.

Despite equivalent current behavior, the transformed artifact failed every V0 future-operation contract:

| Checkpoint | Q4 | 40% prune | finite-budget adaptation | constrained merge | Optionality |
|---|---:|---:|---:|---:|---:|
| base | pass | pass | pass | pass | 4/4 |
| same-function / bad-gauge | fail | fail | fail | fail | 0/4 |
| gauge-fixed | pass | pass | pass | pass | 4/4 |

A deterministic function-preserving gauge-fix restored all four operation contracts.

## What this demonstrates

The bytes/parameterization of a checkpoint can contain lifecycle-relevant state that is invisible to current task behavior. Model quality and model future-transformability are therefore distinct observables.

## What it does not demonstrate yet

- The effect has not yet been reproduced on a modern Transformer.
- The stressed bad-gauge state is deliberately constructed; natural mixed post-training histories have not yet been shown to produce the same separation.
- The V0 operation contracts are toy contracts, not final benchmark definitions.
- The current optionality score is binary and coarse.
- There is no production checkpoint parser/CLI yet.

See `ROADMAP.md` for the next gates.
