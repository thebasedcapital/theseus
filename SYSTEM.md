# SYSTEM — how Theseus is built to be driven

This document is the load-bearing one. `PLAN.md` states what is owed, `SCHEMA.md` states the
record contract, `RUNBOOK.md` states the verbs, `CLAIMS.md` is the live claim register. Design
choices here are each traced to a failure that actually happened while driving this project
(`m1/PIPELINE_FAILURES.md` #1–#10 is the raw log).

---

## 1. The one idea

**The unit of work is a claim, not a run.**

A checkpoint-lifecycle question ("can two function-equivalent checkpoints have different
futures?") is not answered by an experiment; it is answered by a claim reaching a state —
`UNSUPPORTED → PRELIMINARY → CONTROLLED → CONFIRMED`, or `REFUTED` — and each state has
*declared evidence obligations*: which calibration exists, which controls ran, how many
replications, which environment they were all measured in.

Everything in the system exists to satisfy obligations in the cheapest order, and nothing is
allowed to assert a status the ledger has not earned. The planner is not a job runner with a
nice summary; it is an **evidence-obligation solver**. That reframing is what makes the rest of
this document mechanical rather than aspirational.

Corollary for the driver (me, or the next agent): I should never have to hold "what is proven,
by what, and what would change my mind" in working memory. If a human-scale session like M1 costs
an agent ten orchestration incidents and one near-false-result, the missing thing was not care —
it was a place for that state to live.

## 2. The tower

Five layers, each a pure function of the one below. Nothing skips a layer.

```
L0  artifact      content-addressed bytes: sha256(safetensors), config, dtype, tie state
                  + static features (the Rust inspector's numbers: per-family J, dynamic
                    range, row-energy imbalance, f16-range census)
L1  cell          one measurement: (artifact, operation-spec, environment-digest) -> result
                  immutable, self-describing, includes reference cell id + contract version
L2  claim         a sentence with a verdict, its obligations, its evidence (cell ids), and
                  its REFUTER: the query that would flip it and the cell that answers it
L3  view          generated documents: status, tables, passports, figures, the report prose
L4  session       the agent-facing read/write surface: status | explain | plan | admit |
                  annotate — bounded-output, pointer-returning
```

`artifact → cell` is content addressing. `cell → claim` is obligation satisfaction. `claim → view`
is rendering. `view → session` is bounded-context presentation. Every arrow is one-way and
reproducible: deleting L3 regenerates it; deleting L2 is an act of vandalism guarded by write-once.

**Why the layering is the product.** V0 and M1 both produced numbers I trusted and later had to
re-derive under corrected conditions (the f16 source). In a tower, that correction is a *new
environment digest* on a new cell, and the old cells keep saying what they meant under their own
conditions. There is no "update the conclusion" step where an agent can quietly overwrite meaning.

## 3. Invariants (mechanically enforced, not documented-and-hoped)

| # | invariant | failure it kills |
|---|---|---|
| I1 | **Write-once ledger.** A cell id never changes; a re-run produces a new id. Corrections are new cells plus an `invalidates` edge. | #4, #9 (post-hoc editing of what was claimed) |
| I2 | **Frozen code snapshot per cell.** The driver executes a copy (git HEAD + recorded dirty diff hash), never the live tree. Editing a script mid-run is harmless. | #1, #2 |
| I3 | **Environment digest is part of identity.** `(llama.cpp build, torch, transformers, export dtype, corpus byte-range + hash, seqlen, chunk policy, threshold version)`. Cells with different digests are *not comparable*, and any view mixing them is refused. | #5 (f16 vs bf16 confound), #8 |
| I4 | **Calibration gate.** An operation cannot schedule non-reference artifacts until its reference cell exists and is well-formed. Refusal names the missing cell. | #4 (contracts that pristine base fails) |
| I5 | **Control gate.** Every claim names required controls. No claim reaches `CONTROLLED`/`CONFIRMED` without them. The three universal controls are cheap and automatic (below). | #5, #6 |
| I6 | **Resource lease, typed and honest.** A cell declares `vram_bytes`, `disk_bytes`, `cpu_slots`, `wall_s`. The scheduler admits only cells that fit *now*; it never discovers 4.7 GB of free disk at 80 % through a run. Leases carry owner pid + liveness; dead leases are stolen instantly. | #3, #10 |
| I7 | **One key vocabulary.** Cells emit from a closed enum: `pass/fail/unavailable/stale/invalidated`, `measured/predicted`, `artifact/op/claim` ids. Report code reads fields, never guesses shapes. | #8 |
| I8 | **`unavailable ≠ false ≠ 0`.** Absent evidence is never tallied into a score, in either direction. Ω₀ denominators are made of measured cells only, and the count is printed. | #6, #9 |
| I9 | **Docs are generated or frozen.** Any file marked `generated` is a rendering; hand edits are rejected by CI-check. Narrative docs (`SYSTEM.md`, `RUNBOOK.md`) are the only prose a human/agent writes. | the doc-drift I already have between README/ROADMAP/M1_NOTES |
| I10 | **Every number cites.** `177.3286 (cell 7f3a…)`. A number without a cell id fails validation. | #5, #6 |

**The three universal controls** (I5), each near-free, each one a lesson I learned expensively:

* `control.identity-roundtrip` — pristine bytes → writer → export → measure. If this differs
  from the untouched reference, the *pipeline* is the independent variable and nothing else can
  be read. (This is the single control that saved M1's headline.)
* `control.null-gauge` — the same transform with its parameter at identity (φ=0, d=1). Separates
  "gauge effects" from "writer/reader effects".
* `control.permutation` — G4/G6: exact, and by construction inert for a per-block quantizer; the
  harness's proof that it is not flagging byte churn.

## 4. Value-of-information scheduling (the "least resources" half)

Cells are not queued in a fixed list; they are ranked. For each pending cell the planner knows
(i) which claim obligations it discharges, (ii) its declared cost (L6/lease), and
(iii) whether it *could* change a verdict. It then schedules by

```
score = (obligations discharged × marginal belief movement) / wall_seconds
```

with three concrete policies that follow from how M1 actually ran:

* **Escalate, don't batch.** Adaptation gaps are measured at 1 seed; a second and third seed are
  admitted *only if* `|gap| < 3·sd`, i.e. only when the cheap answer is not yet decisive. I ran
  3 seeds × 6 variants on a hunch about a 0.8 pp gap; the ladder should decide that, not me.
* **Climb the quantization ladder and stop.** q8_0 first: it is the cheapest rung and the most
  diagnostic (M1: Q8_0 alone diverged 10.69 nats on the gauged artifact). If q8 already passes and
  the static debt is ~0, q5/q4 add cost without adding belief — unless the claim names a specific
  rung (as M1's does).
* **Static before surgical.** L0 features cost ~2 s and no GPU. In M1, `frac_below_f16_normal`
  and dynamic range predicted every catastrophic outcome before any probe ran. So the plan's
  default first move on any new artifact is: meter → compare to the family baseline → admit only
  the operations whose risk flags fired. That is roughly an order of magnitude fewer GPU-minutes
  per claim than measuring everything, and it is how a predictor (M6) gets trained on labelled
  data instead of being invented after the fact.

The scheduler also refuses to hold state in an agent's memory: the answer to "what next?" is a
`theseus plan` call, not a recollection. Sessions, restarts, and hand-offs stop being lossy.

## 5. Accretion: how the system gets cheaper with use

Four stores grow monotonically and are consulted by every later claim:

1. **Cell ledger** (§SCHEMA) — every measurement, forever, with conditions attached.
2. **Threshold contracts.** Q8 contract v3 is fitted on 20 measured artifacts and stored at
   `analysis/data/evidence/contracts/contract-3.json`; both Rust scanners cite and use it. Q4,
   export and adaptation remain provisional because their gates did not pass. Threshold versions
   are immutable and identical evidence reuses the current version.
3. **Incident store** — machine-readable, auto-appended by the driver when a cell fails twice, a
   lease is stolen, or a view refuses to render (I4/I5/I3 violation attempts). `RUNBOOK.md` is
   generated from it. This is the difference between a project with lessons and a project that
   *applies* them.
4. **Claim history** — a claim's verdict over time, including the moment it was refuted or
   downgraded. This is what makes honesty cheap: the refutation is a first-class object rather
   than an embarrassing edit.

A fifth, negative store matters as much: **dead ends**. "This gauge family is quantization-neutral
(G1, G2, G5 measured) — stop scheduling quant cells for it; spend on merge/adaptation instead."
In M1 that single fact halves the useful work in the remaining matrix, and it is precisely the kind
of thing an agent otherwise re-derives every session.

## 6. A session, played back (this is the acceptance test for the design)

```
$ theseus admit --hf Qwen/Qwen2.5-0.5B
  artifact a41f…: 290 tensors, 357.8M weights, bf16, tied
  features: J 0.01123  dyn 8.83  row_imbal(q_proj) 5.4e4  frac<f16n 0.0028
  family baseline: within 1.1% of median for this arch → no risk flags
  controls: identity-roundtrip c88b… OK, null-gauge 12d0… OK
  → 3 claims newly eligible; cheapest next: c9f2… (export f16 ppl, 40s CPU)

$ theseus gauge --family norm_diag --mode pow2 --seed 1 --out /work/g1
  artifact 7c31…  Δ to a41f…: 168 tensors changed, all by 2^k
  static: J 0.02955 (+163%)  dyn 14.34  frac<f16n 0.0987 → flags: export.f16, quant.q8/q5/q4, adapt
  equivalence c3d1…: fp32 compute dlogit 0.00e+00, KL 0, top1 1.00000
                   bf16 compute dlogit 0.00e+00          ← conditions recorded, both dtypes
  claim K-3 (same function, different future) → CONTROLLED
    obligations: true-LoRA base calibration ✓, identity control ✓,
                 3 seeds each for base/G3/G3-repair/G7/G7-repair ✓

$ theseus plan --claim K-3 --budget 20gpu-min
  complete: G3/G7 capture gaps exceed 3·max within-variant SD
  next: K-8 matched natural histories; threshold refit only after n≥20 per flag

$ theseus explain K-3
  CONTROLLED (not CONFIRMED: one checkpoint family)
  evidence: bit-identical G3 equivalence; Q8 KLD 10.69 vs 0.00094 base;
            capture mean 0.0989 vs 0.9705 base; repaired 0.9753
  refuter: G3/G7 gaps fall below 3σ on a fresh seed panel
  generated views updated: CLAIMS.md, K3.md, passport 7c31…
```

Note what the agent never had to do: remember which cells exist, re-derive a tally, guess whether
a number was measured under f16 or bf16, decide whether to run a third seed, or check free disk.
Every one of those was a failure I actually had.

## 7. What this changes about the science plan

The reframe does not just tidy plumbing; it makes two of M1's hardest problems representable:

* **M3 natural histories become queries, not narratives.** A history is an ancestry edge in L1
  (`sft → merge → q4`), so "find two checkpoints whose L0 features and current-behaviour fields
  match within tolerance and whose histories differ" is a ledger selection — and it is exactly the
  matched-pairs design A5 needs. Hypothesis: `hidden lifecycle state invisible to ordinary
  evaluation` is testable as "pairs indistinguishable at L0 + current behaviour, divergent at L1
  outcomes", i.e. a *negative result at one layer and a positive at another*. The passport (§L1
  `ancestry`) is the record; the ledger makes it queryable.
* **Thresholds become learned, with provenance.** The inspector's risk lines are honest only as
  long as they cite their n. As the family baseline grows, `theseus calibrate` refits them and
  emits a *new contract version*, invalidating (not editing) the verdicts that used the old one —
  I3 and I1 doing real scientific work.
* **The reserve vector is the object.** M1 already showed a single score is wrong (a repaired
  artifact quantization-pristine and adaptation-deficient). So `Ω` is a typed map, views render
  the vector, and any scalarization must name its weights (`p_o` from math.md §4) in the same
  record. "Health score" is rejected by schema.

## 8. Deliberate non-goals

Not a web dashboard (Track B stays CLI + JSON until the ledger is the source of truth). No
reimplementation of llama.cpp/MLX/MergeKit kernels — the ledger's environment digest makes
"orchestrate existing tools" safe, which was the actual worry. No agent-judged evaluation anywhere
in the evidence path (V0's rule, kept). No claim promotion without controls, ever, even when the
number looks obvious — the f16 export looked obvious and was a trap.
