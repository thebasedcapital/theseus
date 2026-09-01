# Current status

## Where this repo lives

Two remotes, full shared history on both:

- **GitHub (public):** `https://github.com/thebasedcapital/theseus` — the public face, MIT licensed.
- **GitLab (private, canonical):** `https://gitlab.com/thebasedcapital/theseus` — `origin`; still where work lands first.

The project was split out of `thebasedcapital/counterpoint` branch `theseus-v0` by
`git subtree split --prefix=experiments/theseus`, so the 15 V0 commits came along. The
counterpoint branch is untouched history.

The third-party eval corpus is **not** redistributed (`m1/data/eval_wikitext.txt` is gitignored);
`cd m1 && python prep_data.py` regenerates it byte-identically, sha256
`f58687faad11242aac5876010246b033b73a632932cd3956129281348de38af8`. Rust `target/` build
artifacts and generated `m1/work/` slices are likewise untracked.

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
  Q8 uses contract v3; Q4/export/adaptation remain v2 provisional. Two independent Rust suites pass
  (13 scanner + 9 inspector) and `cargo fmt --check` is clean.
- `ledger/`: append-only evidence store now imports 182 admission-clean cells and 20 incidents, including Q8 v3 and
  the second-corpus K-10 replication. The K-8 history attempt is carried as a quarantined
  failed-attempt record, not as citable evidence (incident #18).
- `analysis/`: 22 tests pass. Nine CPU/native-bf16 augmentation probes bring Q8/Q4 to n=20;
  contract v3 is idempotent and Q4 emission is refused.
- `harvest/`: 390 public HF artifacts across seven kinds, 264 resolved edges, 88 dangling edges;
  offline reruns are byte-identical.
- `m1/corpus_replication/`: disjoint second slice `[65536,98304)`, sha `c2cc1b4175c60879`;
  base/prepared equivalence is exact and prepared Q4 damage improves by 0.478 pp.
- `m3/`: the first recorded ordered-history attempt (`adapt→merge→Q4` vs `merge→adapt→Q4`) is
  **quarantined, not evidence**: its committed generator cannot run (`train_lora_state` referenced an
  optimizer that was never constructed; incident #18), and its `adapt` dicts lack keys that function
  always returns. The harness is fixed and now executes; K-8 itself is still untested.
- `m1/rescue.py`: exact lattice prepare ships only after a power-of-two proof and equivalence gate.
  The G5 byte-for-byte result is now independently re-derivable:
  `python m1/verify_g5_recovery.py` (rebuild `g5_c8` first) reproduces 0/290 differing tensors and
  sha256 `88c142557820ccad…a0ed342`, with `c_recovered=8.0` from the tie witness alone.
- `analysis/reserve.py`: quantitative reserve vectors (math.md §4) computed from existing cells —
  no GPU. `R_adapt` is the binding of the capture and collateral terms; 37 analysis tests assert it
  agrees with every recorded pass/fail verdict.
- `ledger/verify.py` (`python -m ledger.cli verify`): every cell must name a commit that resolves,
  a script that existed in it, and the fields its writer cannot emit-less. Currently
  `PASS WITH WARNINGS`: the two #18 records are quarantined as expected, and
  `ledger/quarantine.json` is now the single machine-readable record of voided evidence
  (5 entries: #18, #10, #6, #11 and the caveated base reference); the verifier reports each with its
  status and skips hard-voided paths from tallying while still auditing caveated ones. The shipped
  K-3 base calibration cites `ops/base.adapt.json [adapt-v2-true-lora-base-frozen]`, and an
  unversioned fallback is now labelled rather than silently adopted — that closes the comparability
  gap in `m1/work/probes/base_adapt.json`.

## What this does not demonstrate yet

- Two measured gaps are open by choice, not oversight: the seed panel is still 3 seeds per artifact
  (needs ~1 h GPU plus ~5 GB of rebuildable artifacts against 7 GB free on a box with 49 other
  agents), and no remedy baseline (rotation before quantization, alternative calibration corpus) has
  been run, so the effect still has no measured size relative to ordinary fixes.

- **Natural histories have never been tested.** The one recorded attempt is quarantined: its
  generator could not execute (incident #18), so neither its failure nor any inference from it
  stands. K-8 must be re-run under the fixed `m3/` harness.
- Equivalence is now measured on **two** architectures (Qwen2.5-0.5B and Qwen3-0.6B-Base, `m1/work-qwen3/equiv/`), and G2 correctly refuses on the latter. Quantization reserve replicates there too: bf16-export ppl 12.004 on both artifacts, then Q8_0 ppl 1.20e9 and Q4 KLD 18.83 vs base 0.001491/0.091089, with the static cause transferring nearly numerically (J 0.01020→0.02886, dyn 8.66→14.54, flags 0→21); cells in `m1/work-qwen3/`. Adaptation reserve also replicates (base 0.9613±0.0080 vs `g3_pow2` 0.2511±0.0211, gap −0.710 against a 3σ bar of 0.223; repair restores 0.9376), so **K-3 and K-5 are two-architecture claims**; **merge reserve is still Qwen2-only**: the probe refuses on Qwen3 because its specialist degrades rule-holdout perplexity 28.2955 -> 45.46 (gate allows 42.44, measured on the rule task not WikiText), so K-9 is not a two-architecture claim and its hyperparameters need recalibration there first. K-10 has two corpora, not a second architecture.
- The merge experiment is constructed against a specialist derived from the ungauged base. `m1/merge_probe.py --specialist DIR` now accepts an externally sourced adapter and re-measures its rule loss and perplexity, because those are the contract denominators; the default path stamps `specialist_provenance: self-derived` with the caveat inline. The **empirical** re-run against a real public adapter has not been done yet.
- Q8 v3 is in-sample calibration; Q4/export/adaptation still lack validated thresholds.

## Known deviations

M1 remains Python because it directly drives torch, transformers and llama.cpp. New standalone
tools are Rust (`scan/`, `inspect/`).
