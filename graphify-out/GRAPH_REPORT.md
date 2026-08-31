# Graph Report - .  (2026-08-31)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1201 nodes · 2379 edges · 86 communities (66 shown, 20 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 87 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `593071f3`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- gguf.rs
- log
- inspect/src/main.rs
- lineage.py
- Ledger
- Track B: usable local-ML tool
- AdaptTests
- import_m1.py
- history_pair.py
- formats.rs
- gauge.py
- baserates.py
- common.py
- make_tmp
- loader.py
- run
- adapt_probe.py
- canonicalize.py
- verify.py
- prototype.py
- scan/src/main.rs
- pairs.py
- VerifyTests
- merge_probe.py
- tests.rs
- thresholds.py
- claims.py
- cli.py
- rescue.py
- status.py
- fixtures.py
- test_ledger.py
- rules.py
- cell_body
- wilson_rate
- probe.py
- test_gauge_math.py
- Theseus: model optionality / checkpoint lifecycle diagnostics
- risk_flags.py
- report.py
- analysis — base rates, threshold fitting, matched pairs
- harvest/README.md
- M1 Prior Art and Novelty Boundary
- RUNBOOK — driving Theseus as an agent
- Claims Register
- Theseus harvest population — GENERATED, do not edit by hand
- analyze.py
- Lock
- 1933 Treasure Coast Hurricane
- passport.py
- phase1.sh
- Rifenburg
- run_pair
- drive.sh
- M1 merge specialist
- verify_equiv.py
- m1/adapt_probe.py
- plot_m1.py
- retally.py
- mk_synthetic_fixtures.py
- explain
- M1 pre-registered predictions (static, artifact-only)
- matrix_parity.sh
- selfcheck.sh
- tests/__init__.py
- M1_ANALYSIS.md
- merge_sd
- M1_NOTES.md
- PIPELINE_FAILURES.md
- M1_TABLE.md
- Architectures Audit
- G1: Value-subspace basis change
- G2: RoPE-plane rotations
- G3: RMSNorm scale absorption
- G5: Global stream scale
- G7: SwiGLU up-branch diagonal
- Result
- Tensor
- Git Re-Basin (arXiv:2209.04836)
- QuIP (arXiv:2307.13304)

## God Nodes (most connected - your core abstractions)
1. `scan_tensor_rows()` - 26 edges
2. `main()` - 21 edges
3. `log()` - 19 edges
4. `Obligation` - 18 edges
5. `build_cells()` - 18 edges
6. `RowSink` - 17 edges
7. `f16_to_f32()` - 16 edges
8. `run()` - 16 edges
9. `Ledger` - 15 edges
10. `make_tmp()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `structural_amax_equals_full_scan_on_randomish_blocks()` --calls--> `f16_to_f32()`  [INFERRED]
  scan/src/tests.rs → inspect/src/main.rs
- `command_runner()` --indirect_call--> `run()`  [INFERRED]
  m1/corpus_replication/run.py → ledger/import_m1.py
- `ppl()` --calls--> `run()`  [INFERRED]
  m1/corpus_replication/run.py → ledger/import_m1.py
- `kld()` --calls--> `run()`  [INFERRED]
  m1/corpus_replication/run.py → ledger/import_m1.py
- `convert()` --calls--> `run()`  [INFERRED]
  m1/corpus_replication/run.py → ledger/import_m1.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Theseus Gauge Families** — g1_gauge, g2_gauge, g3_gauge, g5_gauge, g7_gauge [EXTRACTED 1.00]
- **Quantization Coordinate-Dependency Literature** — m1_prior_art_quip, m1_prior_art_quarot, m1_prior_art_spinquant [EXTRACTED 0.90]
- **M1 Transformer Symmetry Gauges** — m1_prior_art_g1, m1_prior_art_g2, m1_prior_art_g7 [EXTRACTED 1.00]
- **Rifenburg Career and Recognition** — m1_corpus_replication_corpus_second_ppl_rifenburg, m1_corpus_replication_corpus_second_ppl_wben, m1_corpus_replication_corpus_second_ppl_medaille_college, m1_corpus_replication_corpus_second_ppl_webr, m1_corpus_replication_corpus_second_ppl_buffalo_broadcasters_hall_of_fame [EXTRACTED 0.90]
- **1933 Hurricane Florida Impact** — m1_corpus_replication_corpus_second_ppl_1933_treasure_coast_hurricane, m1_corpus_replication_corpus_second_ppl_jupiter_fl, m1_corpus_replication_corpus_second_ppl_west_palm_beach, m1_corpus_replication_corpus_second_ppl_stuart_fl [EXTRACTED 0.95]
- **M1 Exact Gauge Transforms** — g1_gauge, g2_gauge, g3_gauge, g5_gauge, g7_gauge [EXTRACTED 1.00]
- **Theseus System Documentation Tower** — system_md, plan_md, schema_md, claims_md [EXTRACTED 1.00]

## Communities (86 total, 20 thin omitted)

### Community 0 - "gguf.rs"
Cohesion: 0.11
Nodes (45): BufReader, Default, File, Read, Blk, gguf_family_of(), GgufCtx, GType (+37 more)

### Community 1 - "log"
Cohesion: 0.08
Nodes (45): log(), logits_of(), main(), dtype, Path, Tensor, convert(), main() (+37 more)

### Community 2 - "inspect/src/main.rs"
Cohesion: 0.13
Nodes (26): Acc, bf16_to_f32(), block_statistic_matches_hand_computation(), Dtype, entries_have_bias(), entries_names(), f16_range_detection_counts_what_the_f16_export_cannot_hold(), f16_to_f32() (+18 more)

### Community 3 - "lineage.py"
Cohesion: 0.09
Nodes (38): api_listing(), _base_relations(), _blob_sha(), build_reach(), _cache_probe(), _card_bases(), classify(), cmd_fetch() (+30 more)

### Community 4 - "Ledger"
Cohesion: 0.10
Nodes (25): Exception, env_core(), env_digest(), environments_comparable(), mixed_environments(), `ledger/env.py` — environment digest (I3), part of a cell's identity.  Digest is, 12-hex digest of the conditions. `None` when conditions are unknown (I8: never g, Given a list of cell records, return (ok, digest_set, offenders).      `ok` is F (+17 more)

### Community 5 - "Track B: usable local-ML tool"
Cohesion: 0.05
Nodes (39): 0. Current checkpoint: V0 smoke test ✅ / M1 phase 1 ✅ (see CLAIMS.md K-1…K-7), A1. Real Transformer smoke test, A2. Replace toy surgery operators with real operators, A3. Quantitative reserve instead of binary pass/fail, A4. Artifact optionality vs canonical optionality, A5. Natural lifecycle histories, A6. Predict reserve without executing every surgery, B10. Reports (+31 more)

### Community 6 - "AdaptTests"
Cohesion: 0.07
Nodes (20): adapt_reserve(), build(), clip01(), load(), main(), merge_reserve(), Path, quant_reserve() (+12 more)

### Community 7 - "import_m1.py"
Cohesion: 0.14
Nodes (28): _adapt_cell(), _adapt_env(), _adapt_reference(), _ancestry_edges(), _base_env(), build_artifacts(), build_cells(), build_claims() (+20 more)

### Community 8 - "history_pair.py"
Cohesion: 0.12
Nodes (28): main(), one(), relgap(), construct(), feature_map(), frozen_contract(), future_cells(), future_divergence() (+20 more)

### Community 9 - "formats.rs"
Cohesion: 0.18
Nodes (22): AdapterInfo, ArtifactKind, Container, Dtype, family_of(), is_adapter_key(), is_lora_a(), is_lora_b() (+14 more)

### Community 10 - "gauge.py"
Cohesion: 0.17
Nodes (30): Arch, _apply_rot(), apply_spec(), g1_vo_orth(), g2_rope_pairs(), g3_norm_diag(), g4_head_perm(), g5_res_scale() (+22 more)

### Community 11 - "baserates.py"
Cohesion: 0.13
Nodes (26): all_labels(), artifact_fires_c(), build_confusion(), catastrophe(), cells_to_labels(), compute(), _confusion(), family_fires_c() (+18 more)

### Community 12 - "common.py"
Cohesion: 0.14
Nodes (25): cached_logits(), compare_logits(), corpus_tokens(), equivalence_report(), eval_batches(), forward_logits(), load_model(), load_state() (+17 more)

### Community 13 - "make_tmp"
Cohesion: 0.17
Nodes (10): capture(), _inputs_for(), make_tmp(), Prevalence is reported per architecture and never pooled across archs (I3)., Rows without an arch are grouped under 'unknown', not dropped., rm(), TestBaserates, TestLoader (+2 more)

### Community 14 - "loader.py"
Cohesion: 0.14
Nodes (23): artifact_id_from_scan(), _augmentation_probe_rows(), load_all(), load_augmentation(), load_cells(), load_harvest(), load_labels(), load_scans() (+15 more)

### Community 15 - "run"
Cohesion: 0.18
Nodes (20): Build from m1/work, validate every admission, then write append-only records., run(), Path, Minimal but real-shaped m1/work tree exercising every PLAN.md §0 mapping., PLAN §0 mapping on a synthetic m1/work: artifacts, cells, calibration wiring,, Two fresh imports of the same work tree produce identical id sets., import-m1 --dry-run maps records without writing to disk., TestImportM1 (+12 more)

### Community 16 - "adapt_probe.py"
Cohesion: 0.14
Nodes (22): base_metrics(), examples(), LoRALinear, main(), make_data(), merge_lora_state(), Linear, One true-LoRA adaptation. `state` adapts an in-memory artifact (no 1 GB round tr (+14 more)

### Community 17 - "canonicalize.py"
Cohesion: 0.12
Nodes (22): canon_g1(), canon_g2(), canon_g3(), canon_g5(), canon_g7(), _eigen_basis(), Tensor, quant_condition() (+14 more)

### Community 18 - "verify.py"
Cohesion: 0.14
Nodes (23): CompletedProcess, _adapt_records(), _cells_in(), _check_adaptation(), check_quarantined(), commit_exists(), _git(), iter_cells() (+15 more)

### Community 19 - "prototype.py"
Cohesion: 0.17
Nodes (21): acc(), adapt_probe(), bad_gauge(), data(), evaluate(), gauge_fix(), imbalance(), logit_diff() (+13 more)

### Community 20 - "scan/src/main.rs"
Cohesion: 0.22
Nodes (23): SafetensorsHeader, adapter_json(), build_census_json(), fatal(), fmt_f64(), js_esc(), main(), Mode (+15 more)

### Community 21 - "pairs.py"
Cohesion: 0.12
Nodes (21): Inputs, Locations the loader looks at. Every consumer supplies an Inputs; callers who ow, artifact_features(), format_report(), main(), match_within(), null_outcomes(), outcome_map() (+13 more)

### Community 22 - "VerifyTests"
Cohesion: 0.17
Nodes (7): Ledger, head_sha(), Tests for ledger/verify.py.  Every check is paired with a mutation that must tri, The check must catch the actual artefact it was written for, not only fixtures., The #18 signature: the v2 writer returns capture unconditionally., real_script_at(), VerifyTests

### Community 23 - "merge_probe.py"
Cohesion: 0.19
Nodes (19): ensure_specialist(), eval_batches(), evaluate_merge(), examples(), external_specialist(), gate(), LoRALinear, main() (+11 more)

### Community 24 - "tests.rs"
Cohesion: 0.11
Nodes (5): build_synthetic_gguf(), Gb, gguf_header_roundtrip_and_f16_scan(), Vec, structural_amax_equals_full_scan_on_randomish_blocks()

### Community 25 - "thresholds.py"
Cohesion: 0.16
Nodes (20): flag_ranks(), Print ordering used everywhere (CLI tables)., artifact_feature(), candidates(), compute(), emit(), fires_under(), fit_flag() (+12 more)

### Community 26 - "claims.py"
Cohesion: 0.29
Nodes (20): cells_with(), cells_with_exact(), _declared(), _ge(), _merge_ok(), _ob_k1(), _ob_k10(), _ob_k2() (+12 more)

### Community 27 - "cli.py"
Cohesion: 0.22
Nodes (18): cmd_admit(), cmd_cell(), cmd_explain(), cmd_import_m1(), cmd_plan(), cmd_render(), cmd_status(), cmd_verify() (+10 more)

### Community 28 - "rescue.py"
Cohesion: 0.19
Nodes (20): changed_tensors_census(), check_disk(), count_flags(), diff_reports(), disk_free_gb(), equip_metrics(), _family_of(), main() (+12 more)

### Community 29 - "status.py"
Cohesion: 0.17
Nodes (16): all_keys(), Theseus ledger spine — the write-once record store behind the `theseus` CLI.  Ow, calibration_exists(), plan(), _plan_actions(), `ledger/plan.py` — value-of-information scheduling (SYSTEM.md §4) + the I4 calib, Rank actions by belief per gpu-minute within the session budget.      `--op` nam, I4: does the ledger hold a well-formed calibration (reference) cell for this op? (+8 more)

### Community 30 - "fixtures.py"
Cohesion: 0.22
Nodes (14): effect_data(), _fn(), _lineage(), lineage_effect(), lineage_noise(), noise_data(), Same scan shapes as effect_data, but outcomes are assigned at random independent, n_groups 'families', each with a parent family<N> and two children. Children of (+6 more)

### Community 31 - "test_ledger.py"
Cohesion: 0.16
Nodes (10): artifact_body(), env(), Tests for the write-once ledger spine (LedgerSpine) and its invariants.  Stdlib, I1: a correction is a new record with an `invalidates` edge; the old cell stays, I1: ids are sha256(canonical body minus id)[:12]; re-adding is a noop., I1: a caller-supplied id must equal the content hash; one key, one meaning., I3: joining cells across distinct digests refuses and names ids; allow stamps., TestEnv (+2 more)

### Community 32 - "rules.py"
Cohesion: 0.19
Nodes (11): _cap_from_missing(), check_cell(), claim_cap(), _invalidates_of(), `ledger/rules.py` — mechanical enforcement of the invariants (I4, I5, I7, I8, K-, Verdict tally over cells. I8: denominators are made of MEASURED cells only; `pre, I5: derive a claim's capping evidence from its declared obligations.      Return, Ids of cells superseded by a `invalidates` edge (I1: correction = new record + e (+3 more)

### Community 33 - "cell_body"
Cohesion: 0.23
Nodes (7): cell_body(), I4: a pass/fail verdict requires an existing calibration reference., I3: a cell and its reference with differing digests are refused, naming both., I5: a missing required control caps a claim at PRELIMINARY and names the promote, I8: predicted/unavailable are recorded, never counted; basis/reason are mandator, K-7: the schema has no scalar-health field; `render`/admission rejects one., TestCellInvariants

### Community 34 - "wilson_rate"
Cohesion: 0.26
Nodes (6): Wilson score 95% interval for an observed binomial count k of n. Returns (lo, hi, (point estimate, lo, hi)., wilson(), wilson_rate(), Wilson score interval, hand-implemented; cross-checked against its closed-form e, TestWilson

### Community 35 - "probe.py"
Cohesion: 0.27
Nodes (11): fail(), header_class_stats(), header_keying_issues(), load_config(), main(), norm_storage_note(), Return (config_path, config, hf_dir_or_None)., Return [(name, dtype, shape)] from the header ONLY; never touch tensor payload. (+3 more)

### Community 36 - "test_gauge_math.py"
Cohesion: 0.30
Nodes (11): g5_eps_probe(), jdiff(), logits(), main(), make(), Worst relative difference of the static conditioning proxy across tensor familie, The residual-scale gauge is exact only up to RMSNorm's epsilon: n(z)=z/sqrt(mean, Untied tiny Qwen2 whose lm_head starts equal to the embedding, i.e. a materializ (+3 more)

### Community 37 - "Theseus: model optionality / checkpoint lifecycle diagnostics"
Cohesion: 0.20
Nodes (9): Important prior-art boundaries from the search, M1 in one screen (real transformer, real surgery), Mathematical direction, Operation contracts in V0, Run, Smoke-test result, Theseus: model optionality / checkpoint lifecycle diagnostics, V0 in one screen (+1 more)

### Community 38 - "risk_flags.py"
Cohesion: 0.28
Nodes (8): artifact_fires(), artifact_fires_eval(), family_fires(), is_catastrophic(), Stated definition: abs(damage) >= CATASTROPHE_MULTIPLE * abs(damage_ref), with a, Does this single family's features exceed the flag's threshold?     Returns True, Does the artifact-level flag fire? Uses the flag's aggregate rule over per-famil, Same as artifact_fires, but when no family row carries the primary feature and a

### Community 39 - "report.py"
Cohesion: 0.47
Nodes (7): adapt_cells(), gguf_cells(), load(), main(), merge_cells(), op_json(), Path

### Community 40 - "analysis — base rates, threshold fitting, matched pairs"
Cohesion: 0.25
Nodes (7): analysis — base rates, threshold fitting, matched pairs, Current evidence base rates (contract v2), Run against real data, Run in the no-data state, Thresholds gate (PLAN §5 / K-6 refuter), Trap 1 — the f16-export confound (K-2 / incident #10), Trap 2 — the aggregation-convention trap (PIPELINE_FAILURES #11)

### Community 41 - "harvest/README.md"
Cohesion: 0.25
Nodes (7): and consumed. Owned by the HarvestLineage slice; generated files live in harvest/cache/., Commands, HarvestLineage (harvest/README.md): how the declared-lineage population is built, verified,, Honesty rules (enforced), Limitations, Record/edge contract (see SCHEMA.md §1), What this directory is

### Community 42 - "M1 Prior Art and Novelty Boundary"
Cohesion: 0.32
Nodes (8): AdamW Non-Equivariance (arXiv:2410.19964), G1: Per-GQA-group value/output orthogonal basis change, G2: RoPE-plane rotations, G7: SwiGLU up-branch diagonal, LoRA-RITE (arXiv:2410.20625), M1 Prior Art and Novelty Boundary, QuaRot (arXiv:2404.00456), SpinQuant (arXiv:2405.16406)

### Community 43 - "RUNBOOK — driving Theseus as an agent"
Cohesion: 0.25
Nodes (7): 0. Standing, 1. Verbs, 2. Session procedure, 3. Triage table (each row is an incident that actually happened), 4. Scientific hygiene rules that are not optional, 5. Recovery, RUNBOOK — driving Theseus as an agent

### Community 44 - "Claims Register"
Cohesion: 0.33
Nodes (4): Claims Register, Mathematical Formulation, Technical Report, System Design

### Community 45 - "Theseus harvest population — GENERATED, do not edit by hand"
Cohesion: 0.29
Nodes (6): 1. Composition by kind, 2. Resolvable parent (what enables matched pairs), 3. Top architectures, 4. Selection bias of this sampling (self-declared; analysis is the next slice), 5. Sample drops (top reasons), Theseus harvest population — GENERATED, do not edit by hand

### Community 46 - "analyze.py"
Cohesion: 0.48
Nodes (6): jtotal(), main(), prediction_check(), rank(), Score the pre-registered static predictions against measured Q4_K_M damage., spearman()

### Community 47 - "Lock"
Cohesion: 0.33
Nodes (3): Lock, True when the recorded owner process no longer exists (or its cmdline no longer, mkdir-based advisory mutex so sibling processes serialize GPU/CPU-heavy phases.

### Community 48 - "1933 Treasure Coast Hurricane"
Cohesion: 0.29
Nodes (7): 1933 Treasure Coast Hurricane, Bahamas, Florida, Jupiter, Florida, Stuart, Florida, Tampa Electric Co., West Palm Beach

### Community 49 - "passport.py"
Cohesion: 0.57
Nodes (6): build(), _head(), load(), main(), Path, reserve_entry()

### Community 50 - "phase1.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF, phase1.sh script, TSX_CPU, TSX_THREADS

### Community 51 - "Rifenburg"
Cohesion: 0.33
Nodes (6): Buffalo Broadcasters Hall of Fame, Jane Rifenburg, Medaille College, Rifenburg, WBEN, WEBR (WDCZ)

### Community 52 - "run_pair"
Cohesion: 0.60
Nodes (4): main(), dtype, Path, run_pair()

### Community 53 - "drive.sh"
Cohesion: 0.40
Nodes (4): OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF, drive.sh script, TSX_THREADS

### Community 54 - "M1 merge specialist"
Cohesion: 0.40
Nodes (4): Correct BA merge, M1 merge specialist, Specialist recipe, Specialist self-gate

### Community 55 - "verify_equiv.py"
Cohesion: 0.70
Nodes (4): git_head(), main(), Path, verify()

### Community 56 - "m1/adapt_probe.py"
Cohesion: 0.50
Nodes (4): m1/adapt_probe.py, m3/history_pair.py, K-8 Harness, Exploratory Screens README

### Community 57 - "plot_m1.py"
Cohesion: 0.83
Nodes (3): main(), panel(), role_of()

### Community 58 - "retally.py"
Cohesion: 0.67
Nodes (3): load(), main(), Path

### Community 60 - "explain"
Cohesion: 0.67
Nodes (3): explain(), obligations_for(), Derive and check a claim's state from the ledger (the `theseus explain` body).

## Knowledge Gaps
- **106 isolated node(s):** `selfcheck.sh script`, `M1 analysis — static conditioning vs measured surgery damage`, `What this forecasts for the panel`, `M1 reserve table`, `M1 in one screen (real transformer, real surgery)` (+101 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `log()` connect `log` to `canonicalize.py`, `common.py`, `Lock`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Why does `run()` connect `run` to `cli.py`, `test_ledger.py`, `import_m1.py`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Obligation` (e.g. with `TestCellInvariants` and `TestEnv`) actually correct?**
  _`Obligation` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `selfcheck.sh script`, `M1 analysis — static conditioning vs measured surgery damage`, `What this forecasts for the panel` to the rest of the system?**
  _106 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `gguf.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.10675990675990676 - nodes in this community are weakly interconnected._
- **Should `log` be split into smaller, more focused modules?**
  _Cohesion score 0.08127721335268505 - nodes in this community are weakly interconnected._
- **Should `inspect/src/main.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.1322849213691027 - nodes in this community are weakly interconnected._