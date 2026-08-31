# Theseus Roadmap — scientific scope

> **Sequencing and process are superseded by [`PLAN.md`](PLAN.md)** (claims, evidence obligations,
> budgets, ordering) and the invariants in [`SYSTEM.md`](SYSTEM.md). What stays authoritative here
> is the *content*: which operations, which architectures, which formulations, what not to build.
> Where the two disagree about what to do next, `PLAN.md` wins; where they disagree about what
> would be interesting, this file wins.

Theseus is a no-LLM model-lifecycle diagnostics layer.

The core question is not only **"what can this checkpoint do now?"** but:

> **"What can still safely be done to this checkpoint next?"**

V0 proves that current behavior can be insufficient to answer that question: function-equivalent ReLU checkpoints can have radically different compatibility with quantization, pruning, finite-budget adaptation, and parameter merging.

The remaining work splits into two tracks that must advance together.

---

## 0. Current checkpoint: V0 smoke test ✅ / M1 phase 1 ✅ (see CLAIMS.md K-1…K-7)

Already implemented:

- deterministic CPU-only ReLU experiment;
- exact function-preserving gauge transformation;
- function-preserving gauge canonicalization;
- INT4 quantization operation;
- 40% magnitude-pruning operation;
- finite-budget SGD adaptation capture test;
- constrained parameter-merge capture test;
- explicit pass/fail operation contracts;
- JSON/CSV/text result artifacts;
- optionality visualization;
- viability / capture-basin mathematical formulation;
- prior-art boundary notes.

Current result:

```text
base                         4/4
same-function / bad-gauge    0/4
gauge-fixed                  4/4
```

This is a proof of phenomenon, not yet a useful LLM checkpoint tool.

---

# Track A: scientific validity

## A1. Real Transformer smoke test

Goal: reproduce the hidden-future-state phenomenon on a real decoder-only Transformer.

Start small enough to iterate locally:

- Qwen 0.5B–1.7B class;
- Llama/TinyLlama class;
- optionally a small scratch Transformer where all transformations are fully controlled.

Required experiment:

1. Start with checkpoint `M`.
2. Generate a function-equivalent checkpoint `M'` using exact architecture-valid Transformer symmetries.
3. Verify token/logit equivalence on a held-out calibration set.
4. Test both checkpoints under real downstream operations.
5. Find at least one operation where present behavior is equivalent but future outcome differs materially.
6. Canonicalize/repair `M'` without changing its function and test whether reserve is restored.

This is the first major science gate.

### Transformer symmetry work needed

Architecture-aware exact transformations for:

- attention Q/K gauge transformations;
- attention V/O gauge transformations;
- head permutations where exact;
- RMSNorm/LayerNorm-compatible coordinate transformations;
- MLP/SwiGLU scaling symmetries where valid;
- RoPE restrictions;
- tied embedding / LM-head constraints;
- GQA/MQA constraints.

Never apply a purported gauge unless numerical equivalence is verified.

---

## A2. Replace toy surgery operators with real operators

### Quantization

Integrate at least:

- llama.cpp GGUF `Q8_0`;
- `Q6_K`;
- `Q5_K_M`;
- `Q4_K_M`;
- one aggressive 3-bit/IQ path;
- MLX affine 4-bit on Apple Silicon.

Measure:

- perplexity delta;
- KL/logit distortion;
- task-specific regression;
- instruction/tool-format regressions where available;
- memory and artifact size.

### LoRA / future adaptation

Standardized finite-budget probes:

- LoRA rank 8/16/32;
- fixed data/sample budget;
- fixed optimizer/LR grid;
- protected-capability retention constraint;
- learning-curve AUC and minimum cost to target.

### Merge

Integrate:

- linear merge;
- TIES;
- DARE where appropriate;
- LoRA merge versus dynamic adapter attachment.

Measure both immediate merge success and the **post-merge reserve for later operations**.

### Pruning

Start with:

- magnitude pruning;
- symmetry-aware/path-aware control;
- structured layer/channel cases when architecture support is ready.

### Later operation families

Only after the first three are solid:

- DPO / preference update;
- knowledge editing;
- machine unlearning;
- distillation;
- continued pretraining;
- sparsification / structural compression.

---

## A3. Quantitative reserve instead of binary pass/fail

V0 uses binary capture-basin membership. Finished Theseus needs a margin.

For operation `o`, estimate the minimum intervention cost

```text
J*_o(checkpoint)
```

required to reach the operation target while satisfying protected invariants.

Turn this into a normalized reserve `R_o` using an explicit budget.

Candidate cost axes:

- optimization steps;
- training tokens/examples;
- rank;
- learning rate / update norm;
- bit-width;
- pruning fraction;
- merge coefficient;
- tolerated protected-capability loss;
- compute time / memory.

Report the Pareto frontier when one scalar would hide important tradeoffs.

---

## A4. Artifact optionality vs canonical optionality

For each operation, measure both:

- **artifact optionality**: what the bytes/checkpoint can survive as stored;
- **canonical optionality**: what it can survive after exact function-preserving canonicalization.

Define the gap as repairable lifecycle debt.

Critical validation:

- canonicalization must preserve behavior within declared numerical tolerance;
- it must improve future operation outcome across multiple models/seeds, not only one constructed example;
- compare against random legal gauges and existing operation-specific transforms.

---

## A5. Natural lifecycle histories

Artificial bad gauges are a powerful controlled proof but are not enough.

Generate checkpoints through real histories such as:

```text
SFT → Q4
Q4 → LoRA
SFT → merge → Q4
DPO → merge → Q4
LoRA → merge → LoRA
merge → quantize → later SFT
```

Core hypothesis:

> checkpoints with similar current capability naturally diverge in their future-operation reserve after heterogeneous histories.

Use multiple seeds and matched current-performance pairs.

This is the experiment that turns Theseus from a symmetry curiosity into a lifecycle result.

---

## A6. Predict reserve without executing every surgery

Once there is enough ground-truth operation data, train small numerical predictors.

Inputs can include:

- weight spectra;
- singular-value anisotropy;
- layer norms and scale ratios;
- activation outlier statistics;
- quantization cell margins;
- gradient strength / reliability;
- Fisher/GGN approximations;
- NTK/curvature sketches;
- dormant-unit statistics;
- adapter/task-vector geometry;
- checkpoint history features.

Output:

```text
P(operation succeeds | checkpoint, operation, budget)
```

or predicted minimum surgery cost.

No language model is required. Start with calibrated linear models / gradient-boosted trees before neural predictors.

A prediction is never promoted to a PASS until calibrated against real surgery outcomes.

---

# Track B: usable local-ML tool

## B1. Package and CLI

Turn the experiment into a Python package:

```text
theseus inspect
theseus preflight
theseus prepare
theseus verify
theseus history
```

Suggested package structure:

```text
theseus/
  artifacts/
  architectures/
  diagnostics/
  operations/
  gauges/
  eval/
  lineage/
  reports/
  cli.py
```

`inspect` must be useful without performing destructive operations.

---

## B2. Artifact readers

Priority order:

1. sharded/unsharded safetensors;
2. Hugging Face model directories;
3. GGUF;
4. MLX safetensors/config directories;
5. PEFT/LoRA adapters.

Requirements:

- memory-map/stream tensors where possible;
- avoid full-model GPU loading for static inspection;
- deterministic model identity/hash;
- architecture/config validation;
- preserve private/local operation by default.

---

## B3. Architecture registry

Each architecture adapter declares:

- tensor naming/mapping;
- exact legal symmetries;
- tied weights;
- normalization behavior;
- RoPE/head constraints;
- operation compatibility;
- validation probes.

Initial target families:

1. Qwen;
2. Llama/TinyLlama;
3. Mistral;
4. Gemma.

Fail closed for unknown architectures. Do not guess transformations.

---

## B4. Quick checkpoint health scan

`theseus inspect MODEL --quick`

Static/cheap diagnostics:

- tensor shape/dtype inventory;
- layerwise norms;
- singular-value sketches;
- condition/anisotropy proxies;
- quantization dynamic ranges/outliers;
- MLP and attention scale balance;
- LoRA/task-vector rank statistics;
- known gauge/canonicality metrics;
- lineage metadata if present.

Output both machine-readable JSON and a compact terminal report.

---

## B5. Operation preflight plugins

A stable interface such as:

```python
OperationProbe.assess(checkpoint, budget, invariants)
OperationProbe.execute_probe(...)
OperationProbe.verify(...)
```

First plugins:

- `quantize.gguf` using llama.cpp;
- `quantize.mlx`;
- `adapt.lora`;
- `merge.linear`;
- `merge.mergekit`.

Theseus should orchestrate/measure existing tools rather than reimplementing their kernels.

---

## B6. Function-preserving `prepare` / `repair`

Command:

```text
theseus prepare MODEL --for gguf:q4_k_m
```

Rules:

- search only declared exact/validated symmetry families by default;
- optimize a backend-specific conditioning objective;
- verify logits before and after;
- save a new artifact, never silently overwrite the original;
- emit the transformation manifest and inverse when possible.

Later:

```text
theseus repair MODEL --preserve-function
```

for general canonicalization.

---

## B7. Model passport and lineage

Every inspected/mutated model can carry:

```text
.theseus/passport.json
.theseus/history.jsonl
```

Record:

- content identity/hash;
- parent checkpoint(s);
- operation performed;
- tool/version/config;
- current eval summary;
- reserve vector;
- canonicality/repair debt;
- hardware/runtime used for measured results;
- confidence/coverage of each claim.

Never infer missing lineage as fact.

---

## B8. Verification gates

`theseus verify` should distinguish:

```text
MEASURED
PREDICTED
UNAVAILABLE
FAILED
```

Never convert missing evidence into a PASS.

Verification suite:

- exact/near-exact logit parity for function-preserving transforms;
- task/perplexity regression;
- operation target achieved;
- protected invariants retained;
- reproducibility manifest.

---

## B9. Hardware-aware local mode

Auto-detect:

- CPU/RAM;
- CUDA/VRAM;
- Apple Silicon/unified memory/MLX;
- disk scratch availability.

Use that to choose whether a probe should be:

- static only;
- sampled/layerwise;
- mini-surgery;
- full operation.

A model that cannot be tested on the current machine should report `UNAVAILABLE`, not `FAIL`.

---

## B10. Reports

Three report levels:

### quick
Seconds/minutes, mostly static.

### standard
Calibration forwards + cheap probes.

### deep
Actually executes bounded quantization/adaptation/merge branches.

Primary terminal object:

```text
MODEL OPTIONALITY
Q4_K_M          0.94  measured
LoRA-r16        0.88  measured
TIES merge      0.61  predicted
Prune 20%       0.32  measured
```

Also emit JSON for CI/tool integration and optional standalone HTML later.

---

## B11. Local experiment ledger

Store every surgery as structured data:

```text
before features
operation + budget
history
hardware
result
post-operation reserve
```

This becomes the dataset for reserve predictors.

Optional community sharing should upload only explicit, inspectable telemetry; never weights/prompts/private evaluation data by default.

---

# Milestones

## M1 — Transformer proof

**Exit condition:** function-equivalent real Transformer checkpoints show materially different outcome under at least one real operation, with exact transformation verification and multi-seed replication.

**Status: exit condition met for quantization (2026-08-31); adaptation cells invalidated and re-running under corrected true-LoRA semantics.**
`g3_pow2` — RMSNorm-diagonal gauge on the bf16 exponent lattice, so its logits are *exactly*
identical to base in both fp32 and bf16 compute — loses bounded LoRA capture (0.9731 → 0.1559) and
turns Q8_0 into a 10.7-nat divergence, while an artifact-only canonicalizer returns capture 0.9829
and Q4_K_M at 1.10x base KL divergence. Stress seeds 2 and 3 replicate the stress; the *probe*
seed is replicated separately (`m1/seed_replicate.py`) so the smaller gaps carry error bars.
Merge cells and the natural-history test (A5) are the outstanding items; see `M1_RESULTS.md`.

## M2 — Local preflight alpha

Commands:

```text
theseus inspect
theseus preflight quantize
theseus preflight lora
```

Support safetensors + one architecture family + one GGUF backend.

## M3 — Heterogeneous optionality benchmark

At least three operation families and multiple mixed histories, with matched-current-capability checkpoint pairs.

## M4 — Canonicalization / prepare

Architecture-aware exact gauge transforms that measurably increase one or more future operation reserves without changing current model behavior.

## M5 — Local ML beta

Support:

- Qwen/Llama/Mistral/Gemma;
- safetensors/GGUF/MLX;
- LoRA + merge + quantization;
- passports + lineage;
- quick/standard/deep reports.

## M6 — Predictive optionality

Calibrated no-LLM predictor trained on real surgery ledger, with clear uncertainty and measured-vs-predicted labels.

---

# What not to build first

Do **not** start with:

- an LLM agent choosing post-training recipes;
- a giant web dashboard;
- automatic unlearning/editing before basic operations are validated;
- a single mysterious "health score";
- reimplementations of llama.cpp, MLX, MergeKit, or Unsloth;
- claims that static diagnostics alone prove a surgery will succeed.

The scientific asset is the operation-specific reserve data. The product should expose it with minimal ceremony.

---

# Immediate next implementation

Build **M1** using a small Qwen-family Transformer:

1. safetensors/HF loader;
2. Qwen architecture adapter;
3. exact legal attention/MLP gauge transform;
4. logit-equivalence validator;
5. actual llama.cpp/controlled low-bit quantization test;
6. bounded LoRA adaptation probe;
7. produce the same three-way comparison as V0:

```text
base
function-equivalent stressed representative
canonicalized/repaired representative
```

If the separation survives on a real Transformer, proceed to mixed natural histories.
