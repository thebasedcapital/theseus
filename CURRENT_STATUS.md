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
- pristine lattice prepare improves Q4 on two disjoint corpus slices: −0.185 pp and −0.478 pp
  relative damage versus base. It still breaks both merge operators;
- Q8 static preflight now uses fitted contract v3 at n=20, recall 1.0 and precision 0.40. Q4
  also reaches n=20 but its fit is refused as uninformative.

The first adaptation/merge panel remains archived and invalidated. The replacement probes globally
freeze the base before adapter insertion; their contract is
`adapt-v2-true-lora-base-frozen`.

## Product and corpus status

- `scan/` and `inspect/`: dense safetensors/GGUF behavior preserved; boundary-aware MoE keying
  separates trusted 2-D expert families and makes fused rank-3 stacks explicitly unavailable.
  Q8 uses contract v3; Q4/export/adaptation remain v2 provisional.
- `ledger/`: append-only evidence store now imports 182 admission-clean cells, including Q8 v3,
  second-corpus K-10 replication and the failed K-8 history attempt.
- `analysis/`: 22 tests pass. Nine CPU/native-bf16 augmentation probes bring Q8/Q4 to n=20;
  contract v3 is idempotent and Q4 emission is refused.
- `harvest/`: 390 public HF artifacts across seven kinds, 264 resolved edges, 88 dangling edges;
  offline reruns are byte-identical.
- `m1/corpus_replication/`: disjoint second slice `[65536,98304)`, sha `c2cc1b4175c60879`;
  base/prepared equivalence is exact and prepared Q4 damage improves by 0.478 pp.
- `m3/`: first real ordered-history attempt built Q4 artifacts for `adapt→merge→Q4` and
  `merge→adapt→Q4`. Present-match failed at KL 0.032311 and top-1 0.88235, so future probes stopped.
- `m1/rescue.py`: exact lattice prepare ships only after a power-of-two proof and equivalence gate.

## What this does not demonstrate yet

- Natural histories remain UNSUPPORTED. One real pair was constructed, but it failed the required
  current-behavior match before future reserve measurement.
- One checkpoint family and scale; K-10 now has two corpora, not a second architecture.
- The merge experiment is constructed against a specialist derived from the ungauged base.
- Q8 v3 is in-sample calibration; Q4/export/adaptation still lack validated thresholds.

## Known deviations

M1 remains Python because it directly drives torch, transformers and llama.cpp. New standalone
tools are Rust (`scan/`, `inspect/`).
