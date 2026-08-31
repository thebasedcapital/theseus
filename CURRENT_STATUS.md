# Current status

## Where this repo lives

Standalone GitLab project `thebasedcapital/theseus` (private), split out of
`thebasedcapital/counterpoint` branch `theseus-v0` by `git subtree split --prefix=experiments/theseus`,
so the 15 V0 commits came along. The counterpoint branch is untouched history.

## Research status

**V0 controlled smoke test: PASS** (see `README.md`, `results.json`, `math.md`).
A 98.22 %-accurate ReLU classifier, an exact positive-homogeneity gauge (max logit difference
`5.7e-06`), and a function-preserving gauge-fix:

| checkpoint | Q4 | prune 40 % | finite-budget adaptation | constrained merge | optionality |
|---|---|---|---|---|---|
| base | pass | pass | pass | pass | 4/4 |
| same-function / bad-gauge | fail | fail | fail | fail | 0/4 |
| gauge-fixed | pass | pass | pass | pass | 4/4 |

**M1 real-Transformer proof: complete for constructed gauges.** See `M1_RESULTS.md`,
`M1_TABLE.md` and `CLAIMS.md`.

On `Qwen2.5-0.5B`:

- five architecture-valid gauges pass the fp32 equivalence contract; exponent-lattice G3 is
  bit-identical in fp32 and bf16 compute;
- real native-dtype GGUF quantization separates base from G3: Q8 KLD 0.00094 → 10.69;
- corrected true-LoRA, three seeds per artifact, separates base mean capture 0.9705 from G3
  0.0989 and G7 0.1931. Both exceed the conservative 3σ gate;
- artifact-only lattice repair returns G3 to 0.9753 and G7 to 0.9359 mean capture;
- calibrated merge passes on base. Eleven gauged/prepared representatives fail both operators;
  G5 passes linear at α=0.4 but fails TIES-trim;
- pristine lattice prepare improves Q4 relative ΔPPL from +2.195 % to +2.010 %, but both merge
  operators fail. Reserve is operation-specific, not scalar;
- the provisional static thresholds are REFUTED as a predictor because Q4 has two false negatives
  in ten labelled artifacts. Threshold fitting now refuses to emit below n=20.

The first adaptation/merge panel remains archived and invalidated. The replacement probes globally
freeze the base before adapter insertion; their contract is
`adapt-v2-true-lora-base-frozen`.

## Product and corpus status

- `scan/`: zero-dependency Rust scanner for safetensors, PEFT adapters and GGUF Q8/Q4_K/Q5_K/Q6_K.
  Eleven offline tests pass; Qwen reference J matches within 4.65e-6.
- `ledger/`: append-only evidence store with calibration/environment admission rules, invalidation
  edges and claim explainers. Twelve invariant tests pass; live M1 imports 179 admission-clean cells.
- `analysis/`: Wilson base rates, guarded threshold fitting and matched-lineage null tests. Twenty-one
  tests pass. Current 617 lineage relations produce zero feature-matched measured pairs.
- `harvest/`: 390 public HF artifacts across seven kinds, 264 resolved edges, 88 dangling edges;
  offline reruns are byte-identical.
- `m1/rescue.py`: exact lattice prepare ships only after a power-of-two proof and equivalence gate.
  Full non-lattice mode is diagnostic-only and always REFUSED.
- `ARCHITECTURES.md` / `archcheck/`: 14-architecture audit. Current substring keying silently
  corrupts or drops MoE expert statistics; the probe fails closed on those artifacts.

## What this does not demonstrate yet

- Natural heterogeneous histories (`SFT → merge → Q4` vs `merge → SFT → Q4`) remain untouched.
  K-8 is UNSUPPORTED; the harvested lineage population is selection infrastructure, not evidence.
- One checkpoint family, one scale, one calibration corpus.
- The merge experiment is constructed against a specialist derived from the ungauged base.
- Static preflight thresholds are not validated predictors yet.

## Known deviations

M1 remains Python because it directly drives torch, transformers and llama.cpp. New standalone
tools are Rust (`scan/`, `inspect/`).
