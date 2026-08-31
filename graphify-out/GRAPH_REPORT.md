# Graph Report - theseus  (2026-08-31)

## Corpus Check
- 82 files · ~177,154 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1044 nodes · 2162 edges · 89 communities (57 shown, 32 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 101 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a0cc8f8f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- GGUF File Reader
- Tensor Basis Canonicalization
- Claim and Obligation Logic
- Tensor Inspection and Decoding
- Lineage and Blob Management
- Artifact and Cell Builders
- Logits and Precision Analysis
- Model Format Adapters
- Scan Statistics Reporting
- Baserate and Confusion Metrics
- Architecture Prevalence Testing
- Model Loading and Perplexity
- Ledger Record Index
- Probe Evaluation and Merging
- Environment Digest and Ledger
- Data Loading and Manifests
- System Metrics and Rescue
- GGUF Header Unit Tests
- Outcome Divergence Scoring
- Flag Thresholds and Ranking
- GGUF Backend Probing
- Ledger Rules and Rendering
- Cell Invariant Validation
- LoRA Specialist Merging
- Command Line Interface
- Synthetic Lineage Fixtures
- Wilson Score Intervals
- Linear Probe Adaptation
- Import Workflow Testing
- Safetensors Header Audit
- Gauge Math and Residuals
- Logit Equivalence Checking
- Risk Flag Evaluation
- Cell Report Generation
- Project Documentation
- Prediction Rank Analysis
- GPU Advisory Mutex
- Passport Entry Management
- Phase 1 Execution Script
- Dtype Compatibility Check
- Drive Execution Script
- Git Head Verification
- Loader Input Locations
- Lissamphibia Taxonomy
- M1 Plotting Panels
- Ledger Retally Utility
- Synthetic Fixture Generation
- Classical Poetry
- Self-Check Script
- Ledger Test Suite
- Dvorak Technique
- Imagism Poetry
- Francis Bacon Art
- Naval Warships
- Operation Eastern Exit
- Medieval Clergy
- Value-Subspace Basis
- RoPE-Plane Rotations
- RMSNorm Scale Absorption
- Global Stream Scale
- SwiGLU Diagonal Branch
- 1933 Hurricane
- Ben Amos
- Clayton Kershaw
- Dick Rifenburg
- Hed PE
- Josepha Petrick Kemarre
- One Direction Song
- Labyrinthodontia
- New York Route 31B
- George Steiner Novel
- Robert Boulter
- Battle of Naktong Bulge
- Stereospondyli
- Store Content Hashing
- common.py
- analysis — base rates, threshold fitting, matched pairs
- harvest/README.md
- RUNBOOK — driving Theseus as an agent
- Current status
- Theseus harvest population — GENERATED, do not edit by hand
- M1 merge specialist
- M1 pre-registered predictions (static, artifact-only)
- M1_ANALYSIS.md
- PIPELINE_FAILURES.md
- M1_TABLE.md

## God Nodes (most connected - your core abstractions)
1. `Ledger` - 31 edges
2. `LedgerError` - 30 edges
3. `scan_tensor_rows()` - 25 edges
4. `log()` - 23 edges
5. `Arch` - 22 edges
6. `main()` - 21 edges
7. `env_digest()` - 19 edges
8. `StatAcc` - 19 edges
9. `Obligation` - 18 edges
10. `build_cells()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `structural_amax_equals_full_scan_on_randomish_blocks()` --calls--> `f16_to_f32()`  [INFERRED]
  scan/src/tests.rs → scan/src/stats.rs
- `artifact_features()` --calls--> `family_rows_by_artifact()`  [INFERRED]
  analysis/pairs.py → analysis/baserates.py
- `fit_flag()` --calls--> `family_rows_by_artifact()`  [INFERRED]
  analysis/thresholds.py → analysis/baserates.py
- `invalidation_set()` --calls--> `family_rows_by_artifact()`  [INFERRED]
  analysis/thresholds.py → analysis/baserates.py
- `artifact_features()` --calls--> `total_rows_by_artifact()`  [INFERRED]
  analysis/pairs.py → analysis/baserates.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **M1 Exact Gauge Transforms** — g1_gauge, g2_gauge, g3_gauge, g5_gauge, g7_gauge [EXTRACTED 1.00]
- **Theseus System Documentation Tower** — system_md, plan_md, schema_md, claims_md [EXTRACTED 1.00]
- **Historical Military and Natural Events** — m1_data_eval_wikitext_1933_treasure_coast_hurricane, m1_data_eval_wikitext_second_battle_of_naktong_bulge, m1_data_eval_wikitext_ise_class_battleship [INFERRED 0.70]
- **Literary and Artistic Analysis** — m1_data_eval_wikitext_du_fu, m1_data_eval_wikitext_little_gidding_poem, m1_data_eval_wikitext_portage_to_san_cristobal_novella [EXTRACTED 0.90]
- **Temnospondyl Evolution and Lissamphibian Origins** — m1_data_eval_wikitext_temnospondyli, m1_data_eval_wikitext_lissamphibia, m1_data_eval_wikitext_doleserpeton [EXTRACTED 0.90]
- **Modernist Poetry and Imagism** — m1_data_eval_wikitext_imagism, m1_data_eval_wikitext_ezra_pound [EXTRACTED 0.95]

## Communities (89 total, 32 thin omitted)

### Community 0 - "GGUF File Reader"
Cohesion: 0.14
Nodes (42): BufReader, Default, File, Read, Blk, gguf_family_of(), GgufCtx, GType (+34 more)

### Community 1 - "Tensor Basis Canonicalization"
Cohesion: 0.09
Nodes (48): canon_g1(), canon_g2(), canon_g3(), canon_g5(), canon_g7(), _eigen_basis(), Tensor, quant_condition() (+40 more)

### Community 2 - "Claim and Obligation Logic"
Cohesion: 0.24
Nodes (23): cells_with(), cells_with_exact(), _declared(), explain(), _ge(), _merge_ok(), _ob_k1(), _ob_k10() (+15 more)

### Community 3 - "Tensor Inspection and Decoding"
Cohesion: 0.14
Nodes (24): Acc, bf16_to_f32(), block_statistic_matches_hand_computation(), Dtype, entries_have_bias(), entries_names(), f16_range_detection_counts_what_the_f16_export_cannot_hold(), f16_to_f32() (+16 more)

### Community 4 - "Lineage and Blob Management"
Cohesion: 0.09
Nodes (38): api_listing(), _base_relations(), _blob_sha(), build_reach(), _cache_probe(), _card_bases(), classify(), cmd_fetch() (+30 more)

### Community 5 - "Artifact and Cell Builders"
Cohesion: 0.14
Nodes (28): _adapt_cell(), _adapt_env(), _adapt_reference(), _ancestry_edges(), _base_env(), build_artifacts(), build_cells(), build_claims() (+20 more)

### Community 6 - "Logits and Precision Analysis"
Cohesion: 0.08
Nodes (45): log(), logits_of(), main(), dtype, Path, Tensor, convert(), main() (+37 more)

### Community 7 - "Model Format Adapters"
Cohesion: 0.18
Nodes (21): AdapterInfo, ArtifactKind, Container, Dtype, family_of(), is_adapter_key(), is_lora_a(), is_lora_b() (+13 more)

### Community 8 - "Scan Statistics Reporting"
Cohesion: 0.15
Nodes (22): adapter_json(), build_census_json(), fatal(), fmt_f64(), js_esc(), main(), Mode, ops_matrix() (+14 more)

### Community 9 - "Baserate and Confusion Metrics"
Cohesion: 0.14
Nodes (24): all_labels(), artifact_fires_c(), build_confusion(), catastrophe(), cells_to_labels(), compute(), family_fires_c(), family_rows_by_artifact() (+16 more)

### Community 10 - "Architecture Prevalence Testing"
Cohesion: 0.17
Nodes (10): capture(), _inputs_for(), make_tmp(), Prevalence is reported per architecture and never pooled across archs (I3)., Rows without an arch are grouped under 'unknown', not dropped., rm(), TestBaserates, TestLoader (+2 more)

### Community 11 - "Model Loading and Perplexity"
Cohesion: 0.15
Nodes (15): corpus_tokens(), eval_batches(), load_model(), load_state(), load_tokenizer(), Path, State dict (safetensors shards merged), copied out and cast., Write a loadable HF dir: weights + config/tokenizer copied from `ref_dir`. (+7 more)

### Community 12 - "Ledger Record Index"
Cohesion: 0.16
Nodes (10): Ledger, load_record(), Path, Add a record. Returns (id, action) where action in {"added","noop"}.          `i, Map a human label (e.g. claim "K-3") to the stored record (by key registry)., Return records whose top-level fields match all given `fields` (exact equality)., Re-read every record and recompute its content hash. Returns [(path, error)]., Write-once store rooted at `<root>/ledger/`.      All mutation goes through `add (+2 more)

### Community 13 - "Probe Evaluation and Merging"
Cohesion: 0.17
Nodes (21): acc(), adapt_probe(), bad_gauge(), data(), evaluate(), gauge_fix(), imbalance(), logit_diff() (+13 more)

### Community 14 - "Environment Digest and Ledger"
Cohesion: 0.14
Nodes (23): Exception, all_keys(), env_core(), env_digest(), environments_comparable(), mixed_environments(), `ledger/env.py` — environment digest (I3), part of a cell's identity.  Digest is, 12-hex digest of the conditions. `None` when conditions are unknown (I8: never g (+15 more)

### Community 15 - "Data Loading and Manifests"
Cohesion: 0.16
Nodes (18): artifact_id_from_scan(), load_all(), load_cells(), load_harvest(), load_labels(), load_scans(), note_missing(), Returns dict(family=[...], total=[...], context={id: ctx}). Empty on absence. (+10 more)

### Community 16 - "System Metrics and Rescue"
Cohesion: 0.19
Nodes (20): changed_tensors_census(), check_disk(), count_flags(), diff_reports(), disk_free_gb(), equip_metrics(), _family_of(), main() (+12 more)

### Community 17 - "GGUF Header Unit Tests"
Cohesion: 0.13
Nodes (5): build_synthetic_gguf(), Gb, gguf_header_roundtrip_and_f16_scan(), Vec, structural_amax_equals_full_scan_on_randomish_blocks()

### Community 18 - "Outcome Divergence Scoring"
Cohesion: 0.16
Nodes (18): artifact_features(), format_report(), main(), match_within(), null_outcomes(), outcome_map(), pair_score(), Divergence score for one flag between two outcome records. Returns (diverges: bo (+10 more)

### Community 19 - "Flag Thresholds and Ranking"
Cohesion: 0.17
Nodes (18): flag_ranks(), Print ordering used everywhere (CLI tables)., artifact_feature(), candidates(), compute(), emit(), fires_under(), fit_flag() (+10 more)

### Community 20 - "GGUF Backend Probing"
Cohesion: 0.05
Nodes (39): 0. Current checkpoint: V0 smoke test ✅ / M1 phase 1 ✅ (see CLAIMS.md K-1…K-7), A1. Real Transformer smoke test, A2. Replace toy surgery operators with real operators, A3. Quantitative reserve instead of binary pass/fail, A4. Artifact optionality vs canonical optionality, A5. Natural lifecycle histories, A6. Predict reserve without executing every surgery, B10. Reports (+31 more)

### Community 21 - "Ledger Rules and Rendering"
Cohesion: 0.19
Nodes (12): _cap_from_missing(), check_artifact(), check_cell(), claim_cap(), _invalidates_of(), `ledger/rules.py` — mechanical enforcement of the invariants (I4, I5, I7, I8, K-, Verdict tally over cells. I8: denominators are made of MEASURED cells only; `pre, I5: derive a claim's capping evidence from its declared obligations.      Return (+4 more)

### Community 22 - "Cell Invariant Validation"
Cohesion: 0.15
Nodes (11): artifact_body(), cell_body(), I4: a pass/fail verdict requires an existing calibration reference., I3: a cell and its reference with differing digests are refused, naming both., I5: a missing required control caps a claim at PRELIMINARY and names the promote, I8: predicted/unavailable are recorded, never counted; basis/reason are mandator, K-7: the schema has no scalar-health field; `render`/admission rejects one., I1: ids are sha256(canonical body minus id)[:12]; re-adding is a noop. (+3 more)

### Community 23 - "LoRA Specialist Merging"
Cohesion: 0.24
Nodes (14): ensure_specialist(), eval_batches(), evaluate_merge(), examples(), gate(), LoRALinear, main(), make_data() (+6 more)

### Community 24 - "Command Line Interface"
Cohesion: 0.34
Nodes (15): cmd_admit(), cmd_cell(), cmd_explain(), cmd_import_m1(), cmd_plan(), cmd_render(), cmd_status(), _load_json() (+7 more)

### Community 25 - "Synthetic Lineage Fixtures"
Cohesion: 0.22
Nodes (14): effect_data(), _fn(), _lineage(), lineage_effect(), lineage_noise(), noise_data(), Same scan shapes as effect_data, but outcomes are assigned at random independent, n_groups 'families', each with a parent family<N> and two children. Children of (+6 more)

### Community 26 - "Wilson Score Intervals"
Cohesion: 0.21
Nodes (8): _confusion(), predicted: list of bool; outcome_pos: parallel list of bool (True=fail). Returns, Wilson score 95% interval for an observed binomial count k of n. Returns (lo, hi, (point estimate, lo, hi)., wilson(), wilson_rate(), Wilson score interval, hand-implemented; cross-checked against its closed-form e, TestWilson

### Community 27 - "Linear Probe Adaptation"
Cohesion: 0.30
Nodes (11): base_metrics(), examples(), LoRALinear, main(), make_data(), Linear, replace_targets(), run_variant() (+3 more)

### Community 28 - "Import Workflow Testing"
Cohesion: 0.28
Nodes (8): Build from m1/work, validate every admission, then write append-only records., run(), Path, Minimal but real-shaped m1/work tree exercising every PLAN.md §0 mapping., PLAN §0 mapping on a synthetic m1/work: artifacts, cells, calibration wiring,, Two fresh imports of the same work tree produce identical id sets., import-m1 --dry-run maps records without writing to disk., TestImportM1

### Community 29 - "Safetensors Header Audit"
Cohesion: 0.27
Nodes (11): fail(), header_class_stats(), header_keying_issues(), load_config(), main(), norm_storage_note(), Return (config_path, config, hf_dir_or_None)., Return [(name, dtype, shape)] from the header ONLY; never touch tensor payload. (+3 more)

### Community 30 - "Gauge Math and Residuals"
Cohesion: 0.30
Nodes (11): g5_eps_probe(), jdiff(), logits(), main(), make(), Worst relative difference of the static conditioning proxy across tensor familie, The residual-scale gauge is exact only up to RMSNorm's epsilon: n(z)=z/sqrt(mean, Untied tiny Qwen2 whose lm_head starts equal to the embedding, i.e. a materializ (+3 more)

### Community 31 - "Logit Equivalence Checking"
Cohesion: 0.27
Nodes (11): cached_logits(), compare_logits(), equivalence_report(), forward_logits(), pick_device(), ppl_from_logits(), Tensor, Equivalence evidence on identical tokens (fp32 forwards, chunked compare). (+3 more)

### Community 32 - "Risk Flag Evaluation"
Cohesion: 0.28
Nodes (8): artifact_fires(), artifact_fires_eval(), family_fires(), is_catastrophic(), Same as artifact_fires, but when no family row carries the primary feature and a, Stated definition: abs(damage) >= CATASTROPHE_MULTIPLE * abs(damage_ref), with a, Does this single family's features exceed the flag's threshold?     Returns True, Does the artifact-level flag fire? Uses the flag's aggregate rule over per-famil

### Community 33 - "Cell Report Generation"
Cohesion: 0.47
Nodes (7): adapt_cells(), gguf_cells(), load(), main(), merge_cells(), op_json(), Path

### Community 35 - "Prediction Rank Analysis"
Cohesion: 0.48
Nodes (6): jtotal(), main(), prediction_check(), rank(), Score the pre-registered static predictions against measured Q4_K_M damage., spearman()

### Community 36 - "GPU Advisory Mutex"
Cohesion: 0.33
Nodes (3): Lock, True when the recorded owner process no longer exists (or its cmdline no longer, mkdir-based advisory mutex so sibling processes serialize GPU/CPU-heavy phases.

### Community 37 - "Passport Entry Management"
Cohesion: 0.57
Nodes (6): build(), _head(), load(), main(), Path, reserve_entry()

### Community 38 - "Phase 1 Execution Script"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF, phase1.sh script, TSX_CPU, TSX_THREADS

### Community 39 - "Dtype Compatibility Check"
Cohesion: 0.60
Nodes (4): main(), dtype, Path, run_pair()

### Community 40 - "Drive Execution Script"
Cohesion: 0.40
Nodes (4): OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF, drive.sh script, TSX_THREADS

### Community 41 - "Git Head Verification"
Cohesion: 0.70
Nodes (4): git_head(), main(), Path, verify()

### Community 42 - "Loader Input Locations"
Cohesion: 0.50
Nodes (3): Inputs, Locations the loader looks at. Every consumer supplies an Inputs; callers who ow, object

### Community 43 - "Lissamphibia Taxonomy"
Cohesion: 0.50
Nodes (4): Doleserpeton, Lissamphibia, Stegocephalia, Temnospondyli

### Community 44 - "M1 Plotting Panels"
Cohesion: 0.08
Nodes (25): main(), panel(), role_of(), 10. Prior work that constrains our novelty claims, 1. State is a checkpoint artifact, not only a function, 2. Each model surgery is a controlled dynamical system, 3. Operation-specific capture basin, 4. Replace binary membership with a quantitative reserve (+17 more)

### Community 45 - "Ledger Retally Utility"
Cohesion: 0.67
Nodes (3): load(), main(), Path

### Community 47 - "Classical Poetry"
Cohesion: 0.67
Nodes (3): Du Fu, Li Bai, Little Gidding (Poem)

### Community 77 - "Store Content Hashing"
Cohesion: 0.18
Nodes (15): Theseus ledger spine — the write-once record store behind the `theseus` CLI.  Ow, calibration_exists(), plan(), _plan_actions(), `ledger/plan.py` — value-of-information scheduling (SYSTEM.md §4) + the I4 calib, Rank actions by belief per gpu-minute within the session budget.      `--op` nam, I4: does the ledger hold a well-formed calibration (reference) cell for this op?, Name the calibration cell that must exist before any non-reference cell of `op_f (+7 more)

### Community 78 - "common.py"
Cohesion: 0.18
Nodes (4): head_of(), merge_sd(), kv-group serving query head h (matches HF `repeat_kv` interleaving)., Task-vector merge of candidate `a` with specialist `b`: a + alpha*(b - a).

### Community 79 - "analysis — base rates, threshold fitting, matched pairs"
Cohesion: 0.25
Nodes (7): analysis — base rates, threshold fitting, matched pairs, Current evidence base rates (contract v2), Run against real data, Run in the no-data state, Thresholds gate (PLAN §5 / K-6 refuter), Trap 1 — the f16-export confound (K-2 / incident #10), Trap 2 — the aggregation-convention trap (PIPELINE_FAILURES #11)

### Community 80 - "harvest/README.md"
Cohesion: 0.25
Nodes (7): and consumed. Owned by the HarvestLineage slice; generated files live in harvest/cache/., Commands, HarvestLineage (harvest/README.md): how the declared-lineage population is built, verified,, Honesty rules (enforced), Limitations, Record/edge contract (see SCHEMA.md §1), What this directory is

### Community 81 - "RUNBOOK — driving Theseus as an agent"
Cohesion: 0.25
Nodes (7): 0. Standing, 1. Verbs, 2. Session procedure, 3. Triage table (each row is an incident that actually happened), 4. Scientific hygiene rules that are not optional, 5. Recovery, RUNBOOK — driving Theseus as an agent

### Community 82 - "Current status"
Cohesion: 0.29
Nodes (6): Current status, Known deviations, Product and corpus status, Research status, What this does not demonstrate yet, Where this repo lives

### Community 83 - "Theseus harvest population — GENERATED, do not edit by hand"
Cohesion: 0.29
Nodes (6): 1. Composition by kind, 2. Resolvable parent (what enables matched pairs), 3. Top architectures, 4. Selection bias of this sampling (self-declared; analysis is the next slice), 5. Sample drops (top reasons), Theseus harvest population — GENERATED, do not edit by hand

### Community 84 - "M1 merge specialist"
Cohesion: 0.40
Nodes (4): Correct BA merge, M1 merge specialist, Specialist recipe, Specialist self-gate

## Ambiguous Edges - Review These
- `Du Fu` → `Little Gidding (Poem)`  [AMBIGUOUS]
  m1/data/eval_wikitext.txt · relation: conceptually_related_to

## Knowledge Gaps
- **134 isolated node(s):** `selfcheck.sh script`, `Mode`, `drive.sh script`, `TSX_THREADS`, `OMP_NUM_THREADS` (+129 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **32 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Du Fu` and `Little Gidding (Poem)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `log()` connect `Logits and Precision Analysis` to `System Metrics and Rescue`, `Tensor Basis Canonicalization`, `GPU Advisory Mutex`, `common.py`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `Ledger` connect `Ledger Record Index` to `Command Line Interface`, `Import Workflow Testing`, `Environment Digest and Ledger`, `Cell Invariant Validation`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Why does `Arch` connect `Tensor Basis Canonicalization` to `Model Loading and Perplexity`, `common.py`, `Logits and Precision Analysis`?**
  _High betweenness centrality (0.009) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `Ledger` (e.g. with `TestCellInvariants` and `TestEnv`) actually correct?**
  _`Ledger` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `LedgerError` (e.g. with `Obligation` and `cmd_admit()`) actually correct?**
  _`LedgerError` has 10 INFERRED edges - model-reasoned connections that need verification._
- **What connects `selfcheck.sh script`, `Mode`, `drive.sh script` to the rest of the system?**
  _134 weakly-connected nodes found - possible documentation gaps or missing edges._