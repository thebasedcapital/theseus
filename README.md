# Theseus: model optionality / checkpoint lifecycle diagnostics

> Two checkpoints can compute the same function and have very different futures. Theseus
> measures the difference, per operation, and then repairs it without changing the function.

This repo is organised as a tower, and each layer has exactly one job:

```
SYSTEM.md    how the thing is built and WHY each rule exists (invariants I1-I10)
PLAN.md      claims + evidence obligations + budget + ordering   -> what is owed
SCHEMA.md    the record contract (artifact / cell / claim / incident) -> what is stored
RUNBOOK.md   the agent-facing verbs, session procedure, triage  -> how to drive it
CLAIMS.md    the live claim register with refuters               -> what is believed
ROADMAP.md   the scientific wish-list (operations, architectures, Track B surface)
M1_*         evidence for milestone 1: NOTES (derivations), RESULTS (measurements),
             TABLE/ANALYSIS (generated), m1/PRIOR_ART.md (novelty boundary)
inspect/     the zero-dependency Rust static diagnostic + preflight (its own crate, its own tests)
prototype.py V0: the deterministic ReLU smoke test
```

The rule that keeps the tower honest: **generated files are generated** (`views/`, `M1_TABLE.md`,
`M1_ANALYSIS.md`, passports, figures) and **narrative files are only written by hand at the top two
layers** (`SYSTEM.md`, `RUNBOOK.md`). Everything below that is a rendering of the ledger, so a
session that loses its working tree loses nothing.

## M1 in one screen (real transformer, real surgery)

Qwen2.5-0.5B, exact architecture-valid gauges verified to logit equivalence, surgery executed by
`llama.cpp b9851` and hand-written AdamW LoRA / task-vector merges:

| checkpoint | max\|Δlogit\| vs base | Q4_K_M KLD | true-LoRA mean capture | merge |
|---|---:|---:|---:|---|
| `base` | 0 | 0.0319 | 0.9705 | linear + TIES pass |
| `g3_pow2` | **0.00e+00** | undefined; Q8 KLD **10.69** | **0.0989** | both fail |
| `g3_pow2_rep` | 0.00e+00 | 0.0350 | **0.9753** | both fail |
| `g7_rand` | 3.1e-01 | undefined | **0.1931** | both fail |
| `g7_rand_rep` | 4.8e-01 | 0.0314 | **0.9359** | both fail |
| `g5_c8` | 1.7e-01 | 0.0325 | 0.9813 | linear passes, TIES fails |
| `prep_base_exact` | EQUIVALENT | 0.0316 | 0.9900 (one seed) | both fail |

For G3, logits agree to the last bit in fp32 and bf16 compute, yet mean adaptation capture falls
from 0.9705 to 0.0989 and Q8 KLD jumps from 0.00094 to 10.69. Artifact-only lattice repair returns
adaptation to 0.9753 and Q4 close to base. G7 repeats the adaptation result under fp32-equivalent,
bf16-sensitive arithmetic. Both effects survive a conservative three-seed, 3σ gate.

There is no single reserve score. `prep_base_exact` improves Q4 relative ΔPPL from +2.195 % to
+2.010 % while causing both merge operators to fail. And the provisional static threshold is
refuted by Q4 false negatives, so preflight prints its n and refuses to learn a replacement below
20 labelled cells.

`theseus-inspect preflight` prints the provisional operation matrix and the calibration count.
It localizes G3/G7 stress correctly, but Q4 has measured false negatives, so the current thresholds
are diagnostic hints rather than a validated predictor. UNAVAILABLE remains a first-class verdict.

`inspect/` is a zero-dependency Rust implementation of the static half: it parses the safetensors
container itself and prints per-family 4-bit conditioning, dynamic range, row-energy imbalance and
f16-export risk in ~2 s over 357 M weights, with flags that localize to exactly the tensor families
each gauge touches — 15 flags on `g3_pow2`, 0 on every repaired artifact.

## V0 in one screen

Two checkpoints can have the same current behavior but very different futures.

The script trains a small ReLU classifier on `sklearn` digits, then constructs three checkpoints:

1. **base** - the trained model.
2. **same-function / bad-gauge** - an exact positive-homogeneity rescaling of hidden neurons. It computes the same function as the base model up to floating-point error.
3. **gauge-fixed** - the bad-gauge model canonicalized by balancing each hidden unit's incoming and outgoing norms. This also preserves the function.

Each checkpoint then faces four real operations:

- per-tensor symmetric **INT4 quantization**;
- **40% global magnitude pruning**;
- a finite-budget **SGD adaptation capture test** on a shifted input domain while retaining original accuracy;
- a constrained **parameter merge capture test** with a sibling specialist.

No LLM, API, judge model, prompt, or agent participates in the experiment.

## Smoke-test result

| checkpoint | current acc. | Q4 | prune 40% | SFT capture | merge capture | optionality |
|---|---:|---:|---:|---:|---:|---:|
| base | 98.22% | 97.56% ✅ | 98.22% ✅ | ✅ | ✅ | **4/4** |
| same-function / bad-gauge | 98.22% | 9.56% ❌ | 78.67% ❌ | ❌ | ❌ | **0/4** |
| gauge-fixed | 98.22% | 98.00% ✅ | 98.22% ✅ | ✅ | ✅ | **4/4** |

The max logit difference between the base and bad-gauge checkpoint is only about `5.7e-06`.

So the current predictor is effectively unchanged, while its compatibility with future model surgery collapses.

Then a function-preserving canonicalization restores the future-operation tests.

## Why this matters

This establishes a minimal but strong result:

> **Current model quality does not identify the state variables needed to predict future model surgery.**

The model lifecycle therefore lives in a richer state space than benchmark scores or even the realized input-output function alone.

A useful benchmark can expose two related quantities:

- **artifact optionality** - what can safely be done to the checkpoint exactly as stored;
- **canonical optionality** - what can safely be done after a function-preserving symmetry canonicalization.

Their gap is avoidable lifecycle debt.

See [`math.md`](math.md) for the viability / reachability formulation.

## Run

```bash
python prototype.py
```

Expected runtime is CPU-friendly. The script asserts the core smoke result and writes:

```text
results.json
results.csv
smoke_output.txt
optionality.png
```

## Operation contracts in V0

The thresholds are intentionally explicit and easy to audit:

- **Q4 pass**: original test accuracy remains at least `0.95` after per-tensor symmetric INT4 quantization.
- **Prune pass**: original test accuracy remains at least `0.95` after 40% global magnitude pruning.
- **SFT capture**: within 100 SGD steps and a fixed LR grid, shifted-domain accuracy reaches at least `0.52` while original-domain accuracy remains at least `0.95`.
- **Merge capture**: using a merge coefficient between `0.50` and `0.90`, original accuracy remains at least `0.90` and rotated-domain accuracy reaches at least `0.80`. `alpha=1` is forbidden because that would simply discard the checkpoint being tested.

These are toy contracts, not proposed final benchmark thresholds. Their job is to make the V0 hypothesis falsifiable.

## Mathematical direction

For operation family `o`, define a constrained state transition

```text
s_(t+1) = F_o(s_t, u_t, ξ_t)
```

with safe set `K` and target set `T_o`.

The operation-specific capture basin is the set of checkpoints from which some allowed control sequence reaches `T_o` without leaving `K`.

The future-operation vector is then the collection of capture margins across surgery families.

The V0 result also shows that a benchmark should not blindly use Euclidean parameter-space geometry. Gauge symmetries can move a checkpoint far in weight coordinates without changing its function. Symmetry-aware path metrics or deterministic canonicalization are better candidates for intrinsic reserve measures.

## Important prior-art boundaries from the search

We should not claim novelty for the individual ingredients:

- viability/capture basins come from classical control theory;
- automated post-training search already exists;
- future trainability / Optimization Readiness exists (`arXiv:2605.09044`);
- quantization-aware downstream plasticity exists in ProjQ (`arXiv:2606.00494`);
- ReLU gauge redundancy and gauge fixing are active topics (`arXiv:2602.14729`);
- same-predictor gauges can change learning dynamics (`arXiv:2608.06766`);
- symmetry-aware pruning/path metrics already exist;
- CellFill derives a finite plasticity lifetime under its bounded in-cell update rule (`arXiv:2608.20873`).

The still-interesting object is the **heterogeneous lifecycle surface**: whether one checkpoint remains learnable, alignable, mergeable, editable, unlearnable, prunable, quantizable, distillable, and repairable after mixed histories of those operations.
