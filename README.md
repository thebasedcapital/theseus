# Theseus 🧬 — model optionality / checkpoint lifecycle diagnostics

> Two checkpoints can compute the same function and have very different futures. Theseus
> measures the difference, per operation, and then repairs it without changing the function.

This repo is organised as:

| path | what |
|---|---|
| `prototype.py`, `results.json`, `math.md`, `optionality.svg` | **V0** — the deterministic, no-LLM ReLU smoke test (4 operations, 4/4 → 0/4 → 4/4) |
| `ROADMAP.md` | the plan: Track A scientific validity, Track B usable local-ML tool, milestones M1…M6 |
| `M1_NOTES.md` | **M1** derivation of the exact symmetry group of a Qwen2 decoder + the frozen protocol |
| `M1_RESULTS.md`, `M1_TABLE.md`, `M1_ANALYSIS.md` | **M1** measured results: the reserve table and the static-proxy analysis |
| `m1/` | M1 code: gauges, canonicalizers, equivalence gate, llama.cpp/LoRA/merge probes, report |
| `m1/PRIOR_ART.md` | the novelty boundary against QuaRot / SpinQuant / QuIP# / Git Re-Basin / LoRA-RITE |

## M1 in one screen (real transformer, real surgery)

Qwen2.5-0.5B, exact architecture-valid gauges verified to logit equivalence, surgery executed by
`llama.cpp b9851` and hand-written AdamW LoRA / task-vector merges:

| checkpoint | max\|Δlogit\| vs base | Q4_K_M KLD (× base) | LoRA r16 capture | verdict |
|---|---:|---:|---:|---|
| `base` | 0 | 0.0319 (1.00×) | 0.973 | pass |
| `g3_pow2` — RMSNorm diagonal, `2^k` scales | **0.00e+00** | KLD undefined; **Q8_0 alone: 10.69** | **0.156** | **fail** |
| `g3_pow2_rep` — artifact-only repair | 0.00e+00 | 0.0350 (1.10×) | 0.983 | pass |
| `g7_rand` — SwiGLU up-branch diagonal | 3.1e-01 | KL undefined | **0.060** | fail |
| `g7_rand_rep` | 4.8e-01 | 0.0314 (0.98×) | 0.841 | pass |
| `g1_haar` — value-subspace O(64) per GQA group | 1.8e-01 | 0.0320 (1.00×) | 0.965 | pass (ΔPPL over limit) |
| `g4_perm` — permutation control | 1.3e-04 | 0.0319 (1.00×) | — | pass |

Same function — for `g3_pow2` the logits agree to the last bit and pre-adaptation perplexity is
identical to four decimals — and an 80-step LoRA that base survives turns it into a 4.3-million
perplexity model that learns 16 % of the task. A canonicalizer that never sees the original
checkpoint gives it back. Two corollaries worth stealing:

* **export dtype is an operation.** The same artifact exports to bf16 GGUF at ppl 12.1351 (base:
  12.1399) and to f16/f32 GGUF at 177 — before any quantizer runs. `theseus preflight` must meter
  the export, not just the quant.
* **there is no single reserve score.** The repaired `g7` artifact is quantization-pristine
  (0.98× base KLD) and still 13 points short of base on adaptation capture. Different operations
  read different features of the same bytes.

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
