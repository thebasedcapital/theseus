# Graph Report - .  (2026-08-31)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1121 nodes · 2259 edges · 96 communities (59 shown, 37 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 91 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `462bd007`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- gguf.rs
- gauge.py
- log
- claims.py
- inspect/src/main.rs
- lineage.py
- Track B: usable local-ML tool
- import_m1.py
- formats.rs
- history_pair.py
- scan/src/main.rs
- baserates.py
- Theseus V0 - mathematical formulation
- make_tmp
- loader.py
- run
- pairs.py
- LedgerError
- prototype.py
- tests.rs
- Ledger
- rescue.py
- common.py
- thresholds.py
- status.py
- merge_probe.py
- cli.py
- fixtures.py
- adapt_probe.py
- Path
- rules.py
- wilson_rate
- probe.py
- test_gauge_math.py
- risk_flags.py
- report.py
- analysis — base rates, threshold fitting, matched pairs
- harvest/README.md
- RUNBOOK — driving Theseus as an agent
- Theseus harvest population — GENERATED, do not edit by hand
- analyze.py
- Lock
- 1933 Treasure Coast Hurricane
- passport.py
- phase1.sh
- K-1: Five exact gauges
- Rifenburg
- run_pair
- drive.sh
- M1 merge specialist
- verify_equiv.py
- Lissamphibia
- retally.py
- mk_synthetic_fixtures.py
- Du Fu
- M1 pre-registered predictions (static, artifact-only)
- K-10: Lattice prepare
- K-3: Adaptation reserve
- selfcheck.sh
- tests/__init__.py
- M1_ANALYSIS.md
- Dvorak Technique
- Ezra Pound
- Francis Bacon
- Ironclad Warship
- James K. Bishop
- Osbert de Bayeux
- M1_NOTES.md
- PIPELINE_FAILURES.md
- M1_TABLE.md
- K-2: Export format damage
- K-7: Reserve is a vector
- G1: Value-subspace basis change
- G2: RoPE-plane rotations
- G3: RMSNorm scale absorption
- G5: Global stream scale
- G7: SwiGLU up-branch diagonal
- 1933 Treasure Coast Hurricane
- Ben Amos
- Clayton Kershaw
- Dick Rifenburg
- Hed PE
- Josepha Petrick Kemarre
- Kiss You (One Direction Song)
- Labyrinthodontia
- New York State Route 31B
- The Portage to San Cristobal of A.H.
- Robert Boulter
- Second Battle of Naktong Bulge
- Stereospondyli
- Qwen2 Architecture

## God Nodes (most connected - your core abstractions)
1. `scan_tensor_rows()` - 26 edges
2. `Arch` - 22 edges
3. `main()` - 21 edges
4. `log()` - 20 edges
5. `StatAcc` - 19 edges
6. `LedgerError` - 18 edges
7. `Obligation` - 18 edges
8. `build_cells()` - 18 edges
9. `run()` - 18 edges
10. `Ledger` - 17 edges

## Surprising Connections (you probably didn't know these)
- `command_runner()` --indirect_call--> `run()`  [INFERRED]
  m1/corpus_replication/run.py → ledger/import_m1.py
- `convert()` --calls--> `run()`  [INFERRED]
  m1/corpus_replication/run.py → ledger/import_m1.py
- `kld()` --calls--> `run()`  [INFERRED]
  m1/corpus_replication/run.py → ledger/import_m1.py
- `main()` --calls--> `run()`  [INFERRED]
  m1/corpus_replication/run.py → ledger/import_m1.py
- `measure_artifact()` --calls--> `run()`  [INFERRED]
  m1/corpus_replication/run.py → ledger/import_m1.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Theseus Gauge Families** — gauge_g1, gauge_g2, gauge_g3, gauge_g5, gauge_g7 [EXTRACTED 1.00]
- **M1 Core Claims** — claim_k1, claim_k2, claim_k3, claim_k4, claim_k5, claim_k7, claim_k10 [EXTRACTED 1.00]
- **Rifenburg Career and Recognition** — m1_corpus_replication_corpus_second_ppl_rifenburg, m1_corpus_replication_corpus_second_ppl_wben, m1_corpus_replication_corpus_second_ppl_medaille_college, m1_corpus_replication_corpus_second_ppl_webr, m1_corpus_replication_corpus_second_ppl_buffalo_broadcasters_hall_of_fame [EXTRACTED 0.90]
- **1933 Hurricane Florida Impact** — m1_corpus_replication_corpus_second_ppl_1933_treasure_coast_hurricane, m1_corpus_replication_corpus_second_ppl_jupiter_fl, m1_corpus_replication_corpus_second_ppl_west_palm_beach, m1_corpus_replication_corpus_second_ppl_stuart_fl [EXTRACTED 0.95]
- **M1 Exact Gauge Transforms** — g1_gauge, g2_gauge, g3_gauge, g5_gauge, g7_gauge [EXTRACTED 1.00]
- **Theseus System Documentation Tower** — system_md, plan_md, schema_md, claims_md [EXTRACTED 1.00]
- **Historical Military and Natural Events** — m1_data_eval_wikitext_1933_treasure_coast_hurricane, m1_data_eval_wikitext_second_battle_of_naktong_bulge, m1_data_eval_wikitext_ise_class_battleship [INFERRED 0.70]
- **Literary and Artistic Analysis** — m1_data_eval_wikitext_du_fu, m1_data_eval_wikitext_little_gidding_poem, m1_data_eval_wikitext_portage_to_san_cristobal_novella [EXTRACTED 0.90]
- **Temnospondyl Evolution and Lissamphibian Origins** — m1_data_eval_wikitext_temnospondyli, m1_data_eval_wikitext_lissamphibia, m1_data_eval_wikitext_doleserpeton [EXTRACTED 0.90]
- **Modernist Poetry and Imagism** — m1_data_eval_wikitext_imagism, m1_data_eval_wikitext_ezra_pound [EXTRACTED 0.95]

## Communities (96 total, 37 thin omitted)

### Community 0 - "gguf.rs"
Cohesion: 0.13
Nodes (43): BufReader, Default, File, Read, Blk, gguf_family_of(), GgufCtx, GType (+35 more)

### Community 1 - "gauge.py"
Cohesion: 0.08
Nodes (51): canon_g1(), canon_g2(), canon_g3(), canon_g5(), canon_g7(), _eigen_basis(), Tensor, quant_condition() (+43 more)

### Community 2 - "log"
Cohesion: 0.08
Nodes (45): log(), logits_of(), main(), dtype, Path, Tensor, convert(), main() (+37 more)

### Community 3 - "claims.py"
Cohesion: 0.09
Nodes (40): cells_with(), cells_with_exact(), _declared(), explain(), _ge(), _merge_ok(), _ob_k1(), _ob_k10() (+32 more)

### Community 4 - "inspect/src/main.rs"
Cohesion: 0.13
Nodes (26): Acc, bf16_to_f32(), block_statistic_matches_hand_computation(), Dtype, entries_have_bias(), entries_names(), f16_range_detection_counts_what_the_f16_export_cannot_hold(), f16_to_f32() (+18 more)

### Community 5 - "lineage.py"
Cohesion: 0.09
Nodes (38): api_listing(), _base_relations(), _blob_sha(), build_reach(), _cache_probe(), _card_bases(), classify(), cmd_fetch() (+30 more)

### Community 6 - "Track B: usable local-ML tool"
Cohesion: 0.05
Nodes (39): 0. Current checkpoint: V0 smoke test ✅ / M1 phase 1 ✅ (see CLAIMS.md K-1…K-7), A1. Real Transformer smoke test, A2. Replace toy surgery operators with real operators, A3. Quantitative reserve instead of binary pass/fail, A4. Artifact optionality vs canonical optionality, A5. Natural lifecycle histories, A6. Predict reserve without executing every surgery, B10. Reports (+31 more)

### Community 7 - "import_m1.py"
Cohesion: 0.14
Nodes (28): _adapt_cell(), _adapt_env(), _adapt_reference(), _ancestry_edges(), _base_env(), build_artifacts(), build_cells(), build_claims() (+20 more)

### Community 8 - "formats.rs"
Cohesion: 0.18
Nodes (22): AdapterInfo, ArtifactKind, Container, Dtype, family_of(), is_adapter_key(), is_lora_a(), is_lora_b() (+14 more)

### Community 9 - "history_pair.py"
Cohesion: 0.14
Nodes (26): construct(), feature_map(), frozen_contract(), future_cells(), future_divergence(), gguf_ppl(), git_head(), inspect() (+18 more)

### Community 10 - "scan/src/main.rs"
Cohesion: 0.15
Nodes (22): adapter_json(), build_census_json(), fatal(), fmt_f64(), js_esc(), main(), Mode, ops_matrix() (+14 more)

### Community 11 - "baserates.py"
Cohesion: 0.13
Nodes (26): all_labels(), artifact_fires_c(), build_confusion(), catastrophe(), cells_to_labels(), compute(), _confusion(), family_fires_c() (+18 more)

### Community 12 - "Theseus V0 - mathematical formulation"
Cohesion: 0.08
Nodes (25): main(), panel(), role_of(), 10. Prior work that constrains our novelty claims, 1. State is a checkpoint artifact, not only a function, 2. Each model surgery is a controlled dynamical system, 3. Operation-specific capture basin, 4. Replace binary membership with a quantitative reserve (+17 more)

### Community 13 - "make_tmp"
Cohesion: 0.17
Nodes (10): capture(), _inputs_for(), make_tmp(), Prevalence is reported per architecture and never pooled across archs (I3)., Rows without an arch are grouped under 'unknown', not dropped., rm(), TestBaserates, TestLoader (+2 more)

### Community 14 - "loader.py"
Cohesion: 0.14
Nodes (23): artifact_id_from_scan(), _augmentation_probe_rows(), load_all(), load_augmentation(), load_cells(), load_harvest(), load_labels(), load_scans() (+15 more)

### Community 15 - "run"
Cohesion: 0.18
Nodes (20): Build from m1/work, validate every admission, then write append-only records., run(), Path, Minimal but real-shaped m1/work tree exercising every PLAN.md §0 mapping., PLAN §0 mapping on a synthetic m1/work: artifacts, cells, calibration wiring,, Two fresh imports of the same work tree produce identical id sets., import-m1 --dry-run maps records without writing to disk., TestImportM1 (+12 more)

### Community 16 - "pairs.py"
Cohesion: 0.12
Nodes (22): Inputs, Locations the loader looks at. Every consumer supplies an Inputs; callers who ow, artifact_features(), format_report(), main(), match_within(), null_outcomes(), outcome_map() (+14 more)

### Community 17 - "LedgerError"
Cohesion: 0.14
Nodes (19): Exception, all_keys(), env_core(), env_digest(), environments_comparable(), mixed_environments(), `ledger/env.py` — environment digest (I3), part of a cell's identity.  Digest is, 12-hex digest of the conditions. `None` when conditions are unknown (I8: never g (+11 more)

### Community 18 - "prototype.py"
Cohesion: 0.17
Nodes (21): acc(), adapt_probe(), bad_gauge(), data(), evaluate(), gauge_fix(), imbalance(), logit_diff() (+13 more)

### Community 19 - "tests.rs"
Cohesion: 0.11
Nodes (5): build_synthetic_gguf(), Gb, gguf_header_roundtrip_and_f16_scan(), Vec, structural_amax_equals_full_scan_on_randomish_blocks()

### Community 20 - "Ledger"
Cohesion: 0.19
Nodes (8): Ledger, load_record(), Path, Add a record. Returns (id, action) where action in {"added","noop"}.          `i, Map a human label (e.g. claim "K-3") to the stored record (by key registry)., Return records whose top-level fields match all given `fields` (exact equality)., Re-read every record and recompute its content hash. Returns [(path, error)]., Write-once store rooted at `<root>/ledger/`.      All mutation goes through `add

### Community 21 - "rescue.py"
Cohesion: 0.19
Nodes (20): changed_tensors_census(), check_disk(), count_flags(), diff_reports(), disk_free_gb(), equip_metrics(), _family_of(), main() (+12 more)

### Community 22 - "common.py"
Cohesion: 0.16
Nodes (13): cached_logits(), compare_logits(), equivalence_report(), forward_logits(), merge_sd(), pick_device(), ppl_from_logits(), Tensor (+5 more)

### Community 23 - "thresholds.py"
Cohesion: 0.18
Nodes (18): artifact_feature(), candidates(), compute(), emit(), fires_under(), fit_flag(), _flag_core(), format_report() (+10 more)

### Community 24 - "status.py"
Cohesion: 0.18
Nodes (15): Theseus ledger spine — the write-once record store behind the `theseus` CLI.  Ow, calibration_exists(), plan(), _plan_actions(), `ledger/plan.py` — value-of-information scheduling (SYSTEM.md §4) + the I4 calib, Rank actions by belief per gpu-minute within the session budget.      `--op` nam, I4: does the ledger hold a well-formed calibration (reference) cell for this op?, Name the calibration cell that must exist before any non-reference cell of `op_f (+7 more)

### Community 25 - "merge_probe.py"
Cohesion: 0.24
Nodes (14): ensure_specialist(), eval_batches(), evaluate_merge(), examples(), gate(), LoRALinear, main(), make_data() (+6 more)

### Community 26 - "cli.py"
Cohesion: 0.34
Nodes (15): cmd_admit(), cmd_cell(), cmd_explain(), cmd_import_m1(), cmd_plan(), cmd_render(), cmd_status(), _load_json() (+7 more)

### Community 27 - "fixtures.py"
Cohesion: 0.22
Nodes (14): effect_data(), _fn(), _lineage(), lineage_effect(), lineage_noise(), noise_data(), Same scan shapes as effect_data, but outcomes are assigned at random independent, n_groups 'families', each with a parent family<N> and two children. Children of (+6 more)

### Community 28 - "adapt_probe.py"
Cohesion: 0.30
Nodes (11): base_metrics(), examples(), LoRALinear, main(), make_data(), Linear, replace_targets(), run_variant() (+3 more)

### Community 29 - "Path"
Cohesion: 0.18
Nodes (14): corpus_tokens(), eval_batches(), load_model(), load_state(), load_tokenizer(), Path, State dict (safetensors shards merged), copied out and cast., Write a loadable HF dir: weights + config/tokenizer copied from `ref_dir`. (+6 more)

### Community 30 - "rules.py"
Cohesion: 0.19
Nodes (12): _cap_from_missing(), check_artifact(), check_cell(), claim_cap(), _invalidates_of(), `ledger/rules.py` — mechanical enforcement of the invariants (I4, I5, I7, I8, K-, Verdict tally over cells. I8: denominators are made of MEASURED cells only; `pre, I5: derive a claim's capping evidence from its declared obligations.      Return (+4 more)

### Community 31 - "wilson_rate"
Cohesion: 0.26
Nodes (6): Wilson score 95% interval for an observed binomial count k of n. Returns (lo, hi, (point estimate, lo, hi)., wilson(), wilson_rate(), Wilson score interval, hand-implemented; cross-checked against its closed-form e, TestWilson

### Community 32 - "probe.py"
Cohesion: 0.27
Nodes (11): fail(), header_class_stats(), header_keying_issues(), load_config(), main(), norm_storage_note(), Return (config_path, config, hf_dir_or_None)., Return [(name, dtype, shape)] from the header ONLY; never touch tensor payload. (+3 more)

### Community 33 - "test_gauge_math.py"
Cohesion: 0.30
Nodes (11): g5_eps_probe(), jdiff(), logits(), main(), make(), Worst relative difference of the static conditioning proxy across tensor familie, The residual-scale gauge is exact only up to RMSNorm's epsilon: n(z)=z/sqrt(mean, Untied tiny Qwen2 whose lm_head starts equal to the embedding, i.e. a materializ (+3 more)

### Community 34 - "risk_flags.py"
Cohesion: 0.28
Nodes (8): artifact_fires(), artifact_fires_eval(), family_fires(), is_catastrophic(), Stated definition: abs(damage) >= CATASTROPHE_MULTIPLE * abs(damage_ref), with a, Does this single family's features exceed the flag's threshold?     Returns True, Does the artifact-level flag fire? Uses the flag's aggregate rule over per-famil, Same as artifact_fires, but when no family row carries the primary feature and a

### Community 35 - "report.py"
Cohesion: 0.47
Nodes (7): adapt_cells(), gguf_cells(), load(), main(), merge_cells(), op_json(), Path

### Community 36 - "analysis — base rates, threshold fitting, matched pairs"
Cohesion: 0.25
Nodes (7): analysis — base rates, threshold fitting, matched pairs, Current evidence base rates (contract v2), Run against real data, Run in the no-data state, Thresholds gate (PLAN §5 / K-6 refuter), Trap 1 — the f16-export confound (K-2 / incident #10), Trap 2 — the aggregation-convention trap (PIPELINE_FAILURES #11)

### Community 37 - "harvest/README.md"
Cohesion: 0.25
Nodes (7): and consumed. Owned by the HarvestLineage slice; generated files live in harvest/cache/., Commands, HarvestLineage (harvest/README.md): how the declared-lineage population is built, verified,, Honesty rules (enforced), Limitations, Record/edge contract (see SCHEMA.md §1), What this directory is

### Community 38 - "RUNBOOK — driving Theseus as an agent"
Cohesion: 0.25
Nodes (7): 0. Standing, 1. Verbs, 2. Session procedure, 3. Triage table (each row is an incident that actually happened), 4. Scientific hygiene rules that are not optional, 5. Recovery, RUNBOOK — driving Theseus as an agent

### Community 39 - "Theseus harvest population — GENERATED, do not edit by hand"
Cohesion: 0.29
Nodes (6): 1. Composition by kind, 2. Resolvable parent (what enables matched pairs), 3. Top architectures, 4. Selection bias of this sampling (self-declared; analysis is the next slice), 5. Sample drops (top reasons), Theseus harvest population — GENERATED, do not edit by hand

### Community 40 - "analyze.py"
Cohesion: 0.48
Nodes (6): jtotal(), main(), prediction_check(), rank(), Score the pre-registered static predictions against measured Q4_K_M damage., spearman()

### Community 41 - "Lock"
Cohesion: 0.33
Nodes (3): Lock, True when the recorded owner process no longer exists (or its cmdline no longer, mkdir-based advisory mutex so sibling processes serialize GPU/CPU-heavy phases.

### Community 42 - "1933 Treasure Coast Hurricane"
Cohesion: 0.29
Nodes (7): 1933 Treasure Coast Hurricane, Bahamas, Florida, Jupiter, Florida, Stuart, Florida, Tampa Electric Co., West Palm Beach

### Community 43 - "passport.py"
Cohesion: 0.57
Nodes (6): build(), _head(), load(), main(), Path, reserve_entry()

### Community 44 - "phase1.sh"
Cohesion: 0.29
Nodes (6): MKL_NUM_THREADS, OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF, phase1.sh script, TSX_CPU, TSX_THREADS

### Community 45 - "K-1: Five exact gauges"
Cohesion: 0.33
Nodes (6): K-1: Five exact gauges, G1: Value-subspace basis change, G2: RoPE-plane rotations, G3: RMSNorm scale absorption, G5: Global stream scale, G7: SwiGLU up-branch diagonal

### Community 46 - "Rifenburg"
Cohesion: 0.33
Nodes (6): Buffalo Broadcasters Hall of Fame, Jane Rifenburg, Medaille College, Rifenburg, WBEN, WEBR (WDCZ)

### Community 48 - "run_pair"
Cohesion: 0.60
Nodes (4): main(), dtype, Path, run_pair()

### Community 49 - "drive.sh"
Cohesion: 0.40
Nodes (4): OMP_NUM_THREADS, PYTORCH_CUDA_ALLOC_CONF, drive.sh script, TSX_THREADS

### Community 50 - "M1 merge specialist"
Cohesion: 0.40
Nodes (4): Correct BA merge, M1 merge specialist, Specialist recipe, Specialist self-gate

### Community 51 - "verify_equiv.py"
Cohesion: 0.70
Nodes (4): git_head(), main(), Path, verify()

### Community 52 - "Lissamphibia"
Cohesion: 0.50
Nodes (4): Doleserpeton, Lissamphibia, Stegocephalia, Temnospondyli

### Community 53 - "retally.py"
Cohesion: 0.67
Nodes (3): load(), main(), Path

### Community 55 - "Du Fu"
Cohesion: 0.67
Nodes (3): Du Fu, Li Bai, Little Gidding (Poem)

## Ambiguous Edges - Review These
- `Du Fu` → `Little Gidding (Poem)`  [AMBIGUOUS]
  m1/data/eval_wikitext.txt · relation: conceptually_related_to

## Knowledge Gaps
- **147 isolated node(s):** `selfcheck.sh script`, `drive.sh script`, `TSX_THREADS`, `OMP_NUM_THREADS`, `PYTORCH_CUDA_ALLOC_CONF` (+142 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **37 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Du Fu` and `Little Gidding (Poem)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `run()` connect `run` to `cli.py`, `claims.py`, `rules.py`, `import_m1.py`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `log()` connect `log` to `gauge.py`, `common.py`, `Lock`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `Ledger` connect `Ledger` to `LedgerError`, `cli.py`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **What connects `selfcheck.sh script`, `drive.sh script`, `TSX_THREADS` to the rest of the system?**
  _147 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `gguf.rs` be split into smaller, more focused modules?**
  _Cohesion score 0.13408521303258145 - nodes in this community are weakly interconnected._
- **Should `gauge.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08245981830887492 - nodes in this community are weakly interconnected._