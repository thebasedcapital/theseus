# RUNBOOK — driving Theseus as an agent

Written to be read once and relied on under time pressure. The design rationale is
`SYSTEM.md`; the record shapes are `SCHEMA.md`; what is owed is `PLAN.md`.

## 0. Standing

Theseus is its own project: own history, no import of any sibling checkout's code, history, or data.
`origin` is the private GitLab repo (canonical); `github` is the public MIT mirror. Commit to `origin`
first, then `git push github main` so the two never diverge.
Choose the interpreter once per machine with `THESEUS_PY` (needs torch ≥2.13,
transformers ≥5.16, safetensors; see `requirements.txt`); the scripts default to a suitable
environment that already exists on this workstation rather than duplicating a 6 GB CUDA venv.
`m1/drive.sh` is the only scheduler — never reintroduce parallel driver scripts (incident #13: two
drivers waiting on each other deadlocked the GPU for six hours).

## 1. Verbs

Two entry points exist today: `python -m ledger.cli <verb>` for the evidence layer, and the Rust
binaries `theseus-scan` / `theseus-inspect` for static analysis. **There is no unified `theseus`
executable yet** (ROADMAP M5). The ledger CLI prints `usage: theseus …` because `prog` is set to the
intended future surface; do not read that as a shell command that exists.

| verb | entry point | what it answers | status |
|---|---|---|---|
| `status [--budget N]` | `python -m ledger.cli status` | what is proven, pending, stale, free | implemented |
| `admit` | `python -m ledger.cli admit` | admit an artifact record (write-once) | implemented |
| `cell` | `python -m ledger.cli cell` | admit a cell record, I3/I4 enforced at the door | implemented |
| `plan [--claim K]` | `python -m ledger.cli plan` | ranked next cells with cost, plus what is **refused** and why | implemented |
| `explain <claim-id>` | `python -m ledger.cli explain` | obligation table, evidence cells, the refuter, the cell answering it | implemented |
| `render` | `python -m ledger.cli render` | regenerate `views/` from the ledger | implemented |
| `import-m1 --work <dir>` | `python -m ledger.cli import-m1` | map `m1/work` cell JSON onto the ledger | implemented |
| — | `theseus-scan inspect\|preflight <file> [--json]` | 32-block conditioning, dynamic range, f16-export risk; preflight exits non-zero on risk | implemented (Rust) |
| — | `theseus-inspect inspect\|preflight <file> [--json]` | per-family features incl. boundary-keyed MoE expert families | implemented (Rust) |
| `run <cell-id…>` | *none* | execute a cell in a frozen code snapshot under a lease | **not implemented**; execution is `m1/drive.sh` + per-probe scripts |
| `calibrate --family <arch>` | *none* | refit thresholds, bump contract version, mark dependent verdicts stale | **not implemented**; fitting is `python analysis/thresholds.py --root analysis/data/evidence` |
| `prepare` / `verify` / `history` | *none* | canonicalize an artifact, certify equivalence, read lineage | **not implemented**; see `m1/rescue.py`, `m1/verify_equiv.py`, `m1/passport.py` |

The ledger root defaults to `$THESEUS_LEDGER_ROOT` or `.theseus`; sessions run every command with an
explicit `--root` under `/tmp` so nothing is written into the live repo layout.

Read-only commands (`status`, `plan`, `explain`, `render`, both scanners) are always safe to call;
that is deliberate — the answer to "what is the situation?" must never be a memory task for the
agent driving it.

## 2. Session procedure

1. `status` before touching anything. If a driver is running, do not hand-run its cells.
2. `admit` any new artifact before planning against it. Never compare an artifact to a claim
   whose `environment` differs (I3); if a number looks wrong, that is the first hypothesis.
3. `plan` for the ordering; obey the refusals, they carry reasons (missing calibration, missing
   control, mixed digests, budget).
4. Execute at most one GPU cell concurrently. `m1/drive.sh` is the only scheduler and
   `m1/gguf_probe.py` holds the `m1/work/gpu.lock` lease. VRAM is 8 GB with ~4 GB held by the
   desktop: two concurrent torch jobs plus a Vulkan context is how M1 lost cells to OOM.
5. `explain` the claim you are about to write about, and copy the verdict state verbatim. Do not
   upgrade "CONTROLLED" to "CONFIRMED" in prose.
6. `render` at the end. Never edit a `views/` file; the banner is enforced.

## 3. Triage table (each row is an incident that actually happened)

| symptom | real cause | fix |
|---|---|---|
| a cell disappeared and no JSON exists | script edited while the driver was about to exec it | each cell stamps `code_snapshot` into its environment digest (`ledger/env.py`, I2/I3), so a cell records the code that produced it and mixed snapshots refuse to join; never patch a live driver — write the new file and let the next run pick it up |
| probe died mid tag-list with `TypeError`/`KeyError` | a reference/calibration file mutated under a running probe | calibration files are write-once per contract version (I1/I4) |
| GPU lock held forever | holder crashed; old policy waited out a 30-min staleness | leases carry pid; dead holder is stolen immediately (I6) |
| everything fails including pristine base | contract thresholds absolute | I4: op cannot schedule until its base calibration cell exists and passes |
| "quantization damage" is enormous | damage is in the **export** step (f16 underflow), not the quantizer | K-2; measure export as its own cell; compare only same-`outtype` cells |
| two implementations of one statistic disagree ~5 % | different aggregation conventions | record `convention` on the feature (I7); M1: mean-of-tensors vs ratio-of-sums |
| a report shows UNAVAILABLE for work that ran | key-shape drift between producer and consumer | closed vocabulary + schema validation (I7); regenerate with `render` |
| my summary disagrees with the per-cell files | in-run tally went stale after repairs | summaries are derived views, never written in the hot path (I1) |
| disk hits zero mid-sweep | cell footprint unknown at admission | `lease.disk_bytes` reserved up front (I6) |
| two drivers raced on the same variant dir | wait-predicates did not name each other | one scheduler process (the driver), shell chains retired |

## 4. Scientific hygiene rules that are not optional

* A claim is published with its **refuter**: what observation would downgrade it, and which cell
  would produce it. `explain` prints both; if a claim has no refuter it is not a claim.
* Numbers in prose carry cell ids (`10.690304 nats, 9b02`). `render` fails a document containing a
  number without a citation (I10).
* A partial repair is a result. `g7_rand_rep` restoring quantization but not adaptation is written
  as-is (K-5, K-7); the temptation to report only the clean G3 case is exactly the failure mode this
  project's vocabulary exists to prevent.
* `UNAVAILABLE` is a legal and common answer for a tool: the inspector answers it for merges and
  adapter-on-quantized-base, because those bytes cannot support a verdict yet (I8).
* Equivalence must name the compute dtype. fp32-only equivalence overstated sameness for the
  wide-dynamic-range gauges (top-1 0.998 fp32 vs 0.981 bf16 on `g7_rand`).

## 5. Recovery

`.theseus/ledger/` is the state. If the working tree is lost, `render` rebuilds every view; if a
run is interrupted, `status` shows the incomplete cells (they are recorded at admission, not at
completion — that asymmetry is why a killed driver never leaves a phantom "done"). Cell ids are
content hashes, so re-running an interrupted cell is a no-op deduplicated by I1.
