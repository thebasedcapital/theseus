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

**M1 real-Transformer proof: phase 1 complete, phase 2 executing.** See `M1_RESULTS.md`
(claims + evidence), `M1_NOTES.md` (symmetry derivations and frozen protocol),
`M1_TABLE.md` (per-checkpoint surgery panel, regenerated as results land),
`m1/PRIOR_ART.md` (novelty boundary).

Established on `Qwen2.5-0.5B`, measured with fp32 forwards over the stored bf16 artifacts:

* five architecture-valid gauges are the same model to `1e-5…2e-4` nats mean KL, 99.5–99.9 %
  teacher-forced greedy agreement and unchanged perplexity — value/output basis change per GQA
  group, the full RoPE-compatible q/k commutant, RMSNorm scale absorption, the SwiGLU up-branch
  diagonal, and residual-stream scaling (the last needs the embedding tie broken and the config
  edit `ε → c²ε` to be exact);
* two permutation controls land at `1.6e-4` max logit difference, proving the harness is not
  flagging byte changes as behaviour changes;
* 13 property tests pass on a tiny Qwen2, including canonicity of every artifact-only
  canonicalizer and a permanent sensitivity control that fails the suite if a forgotten attention
  bias (Qwen2.5 ships `|b_q| = 79`, `|b_k| = 130`) ever goes undetected;
* a precision-floor control separates algebra error from bf16 re-storage (`9.4e-05` vs `2.69`) and
  motivated a lossless exponent-lattice gauge (`G3:pow2` + `canon_g3(snap_pow2=True)`);
* surgery calibration on the untouched checkpoint: llama.cpp `Q4_K_M` costs the *pristine* model
  `+2.27 %` PPL and `0.0319` nats mean KLD, and bounded LoRA r16 captures 97.3 % of the reference
  task gain at `+2.96` PPL collateral — so both operation contracts were rewritten to be
  reference-relative before any variant was measured;
* a static, artifact-only conditioning statistic (32-block max-abs proxy) was snapshotted into
  `m1/work/PREDICTIONS.json` *before* the surgery panel ran; it forecasts `Q4` damage for `G3`
  (debt `+0.0185`) and `G7` (`+0.0037`) and neutrality for the value-subspace and RoPE-plane
  gauges.

## What this does not demonstrate yet

* The surgery panel over the stressed/repaired pairs is still running; no variant-vs-variant
  damage claim is made until those JSONs exist.
* Natural heterogeneous histories (`SFT → merge → Q4` vs `merge → SFT → Q4`) are untouched — that
  is `ROADMAP.md` A5/M3, and it is the difference between a symmetry curiosity and a lifecycle
  result.
* One checkpoint, one scale, base (non-instruct) weights only.
* The merge demo is deliberately constructed (a gauged candidate merged with a specialist derived
  from the ungauge base); real independently-trained finetunes drift in basis too, but ours is a
  worst case.
* No CLI/package yet — Track B is intentionally behind Track A.

## Known deviations worth flagging to a reader

* New code is Python, not the machine convention of Rust for new work: M1 must drive
  `torch`/`transformers`/`llama.cpp` and continue `prototype.py`; the Rust plan applies to the
  Track-B CLI (`theseus inspect/preflight/prepare/verify`).
* The equivalence verdict was amended mid-flight on control evidence (max-|Δlogit| demoted from a
  hard gate to a flag, distributional criteria unchanged); see the comment block in
  `m1/verify_equiv.py` and §1 of `M1_RESULTS.md`.

## Result invalidation in progress

The first adaptation/merge panel was invalidated on 2026-08-31: adapter insertion froze target Linear weights but not embeddings/norms/lm_head, so those runs were full-model training, not LoRA. Twenty-one cells were archived; corrected true-LoRA cells globally freeze the base and are running. Quantization/equivalence/export/static evidence is unaffected.
