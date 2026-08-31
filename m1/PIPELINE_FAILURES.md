# M1 pipeline: what broke while it ran, and the rule each break produced

Kept because M3 repeats this at larger scale and every one of these cost measurable compute.
All of these are *orchestration* failures, not scientific ones — but three of them produced
invalid numbers that would have been reported as results if the checks had not caught them.

| # | break | symptom | caught by | rule |
|---|---|---|---|---|
| 1 | Editing a probe script while the driver was about to exec it | process imported a half-written file and died; one `(variant, op)` cell silently missing | missing `ops/*.json` | Never edit a live script. Write `X.new` + `os.replace` (atomic on one fs), and only in the gap between invocations |
| 2 | Mutating `m1/work/quant_ref.json` while a probe was mid-tag-list | every quant cell after base died with `TypeError` (subtracting from a `null` prefix reference) | the driver log showing no output file | A calibration/reference file is append-only for the duration of a run; fix it between runs |
| 3 | Backfill launched with a wait predicate that missed the running panel | two drivers on one GPU, racing on the same variant dirs, one orphan `adapt_probe` outliving its parent | `ps -o ppid=` audit | Every background driver's gate must name *all* live drivers; after killing a driver, sweep for `ppid=1` children |
| 4 | Absolute pass thresholds for quantization and LoRA | pristine base failed its own contract (Q4_K_M `+2.27 %` PPL, `0.0319` KLD; LoRA collateral `+2.96` PPL) | calibrating on base before variants | Contracts are reference-relative unless the reference provably passes |
| 5 | `exact 32-token greedy agreement` as a metric | scored 0.00 for the pristine Q4 row — one divergent token zero-codes a prompt | base calibration | Prefer graded statistics (shared prefix length, KLD percentiles); check that a metric is non-degenerate on the reference |
| 6 | Broken LoRA→weight merge (delta never applied) | "specialist" at ppl 40,694 and rule loss 9.22; six merge cells of pure garbage | the new fail-closed specialist gate the agent added after this | Any artifact that feeds a comparison must be gated against a baseline *inside* the probe that produces it |
| 7 | `state_to_model` per merge alpha | 14 × `from_pretrained` of a 988 MB checkpoint; 342 s per variant | timing accounting | Build once, `load_state_dict` in a loop |
| 8 | VRAM check taken before acquiring the GPU lock | OOM at 140 MiB free while the panel's Vulkan context was resident | `ps`/`nvidia-smi` audit | Lock first, then measure free memory; the lock is what makes the number meaningful |
| 10 | Reference export dtype (f16 GGUF) conflated with quantizer damage | a gauge with bit-identical logits showed ppl 177 before any quantizer ran; f32 export also 177, bf16 export 12.14 | identity round-trip control (base through my own `save_state` → f16 GGUF ppl 12.1399, exactly the pristine number) + three-export decomposition | Measure an operation only against a reference that shares the artifact's native dtype; make export a separately metered operation |
| 9 | `pass: null` (the run that *defines* the reference) tallied as a failure | driver summary showed base failing quantization | `m1/retally.py` | `null` means "no verdict", never `False`; derive summaries from per-cell JSON, not from in-run tallies |

| 11 | Two implementations of one statistic disagreed 5.7 % | pooled block ratio-of-sums vs mean-of-per-tensor-ratios, neither labelled | cross-validation of the Rust inspector against the Python one | Every aggregate carries its `convention`; a number without a convention is not comparable (I7) |
| 12 | "Control" claimed globally | `g4_perm` is quantization-inert and costs 6.4 pp of LoRA capture | the ledger's per-operation verdict vectors | A control must name the operation it controls for (K-7) |

Invariant map: #1,#2 → I2 (frozen code snapshot per cell) · #3,#8 → I6 (typed leases with pid
liveness) · #4 → I4 (calibration gate) · #5,#6 → I3+I5 (environment digest; mandatory controls) ·
#7,#9,#10,#11 → I1+I7 (write-once ledger, closed vocabulary, named conventions) · #12 → K-7
(reserve is a vector). `RUNBOOK.md` §3 is this table from the driver's side.

Cross-cutting: the panel records everything to per-cell JSON files (`m1/work/ops/<variant>.<op>.json`)
and every downstream artifact (`M1_TABLE.md`, `M1_ANALYSIS.md`, `m1_summary.json`,
`m1_prediction_check.json`) is regenerated from those files. That is what made 1-3 and 9
recoverable by re-running a cell instead of the run.
