# PLAN — obligations, not tasks

`ROADMAP.md` remains the scientific wish-list (which operations, which architectures). This file is
the *executable* plan: the claim set, each claim's evidence obligations, the budget, and the
ordering rules that the driver uses (`SYSTEM.md` §1, §4). Status here is derived from the ledger;
when the two disagree, the ledger is right and this file is stale.

## 0. Immediate migration (this is not a rewrite, it is a re-describing)

The M1 run already contains the evidence; what it lacks is a place for the *conditions* to live.
`theseus import-m1` maps today's files onto `SCHEMA.md` records, so no measurement is lost:

| today | becomes |
|---|---|
| `m1/work/VARIANTS.json` (spec, gauge manifests, sha256, untied, eps patch) | `artifact` records + `ancestry` edges |
| `m1/work/equiv/<v>.json` (metrics, gate, verdict, flags, cond_a/cond_b) | `artifact.features` + an equivalence `cell` per compute dtype |
| `m1/work/ops/<v>.<op>.json` | measurement `cell`s, with `environment` rebuilt from the embedded `versions`/`corpus`/`export`/`pass_contract` fields |
| `m1/work/quant_ref.json`, `ref_capture.json` | the `reference_cell` targets that satisfy I4 for each op |
| `m1/work/PREDICTIONS*.json`, `debts_lattice.json` | `status: predicted` cells attached to claims (never tallied) |
| `M1_TABLE.md`, `M1_ANALYSIS.md`, `m1_optionality.svg` | `views/` (generated; they already are, which is why they survived 4 driver restarts) |
| `m1/PIPELINE_FAILURES.md` #1–#11 | `incident` records, each linked to the invariant that now prevents it |

Anything that cannot be mapped without inventing conditions is imported as `environment: unknown`
and is excluded from cross-cell comparison by I3. That is the honest outcome: a number whose export
dtype I did not record is a number I cannot compare, and the schema says so instead of me
remembering it.

## 1. Claim set, with obligations

`replication` counts seeds of the *probe*, not only of the stress — the distinction that made me
write `seed_replicate.py` after the fact. `controls` are the three universal ones from `SYSTEM.md` §3.

| id | claim | equivalence | calibration | controls | replication | escalation | state today |
|---|---|---|---|---|---|---|---|
| **K-1** | Qwen2.5-0.5B admits ≥5 exact gauges that ordinary tooling does not quotient out | fp32+bf16 | n/a | null-gauge | 3 stress seeds | n/a | CONTROLLED (fp32 complete; bf16 measured for G3/G7 only) |
| **K-2** | A gauged checkpoint can be exported to bf16 GGUF with perplexity identical to base while its f16/f32 export collapses | fp32+bf16 | base export cell | identity-roundtrip ★ | 1 | n/a | CONTROLLED |
| **K-3** | Function-equivalent checkpoints have materially different **adaptation** reserve | fp32; G3 bf16 bit-identical | true-LoRA base cell ✓ | identity ✓ | 3 seeds on base/G3/G7 + repairs | 3σ gate | **CONTROLLED**: G3 −87 pp, G7 −78 pp; both repair |
| **K-4** | …and different **quantization** reserve, at the same bit-width, same corpus | fp32 | quant ladder refs ✓ | identity, permutation ✓ | 1 | ladder-stop armed | **CONTROLLED**: G3 Q8 10.69 vs base 0.00094 |
| **K-5** | Artifact-only lattice canonicalization restores G3/G7 adaptation reserve | fp32; G3 bf16 | LoRA base ✓ | power-of-two proof | 3 seeds | repaired gap >3σ refuter | **CONTROLLED** |
| **K-6** | Fitted static threshold predicts Q8 risk; Q4 remains unsupported | n/a | Q8 n=20 | n/a | in-sample | out-of-sample recall | **PARTIAL**: Q8 v3 recall 1.0, precision 0.40; Q4 refused |
| **K-7** | Reserve is a vector: no scalar summarizes it | fp32 | per-op refs | n/a | two Q4 corpora | operation-independent ordering | **CONTROLLED**: prepare helps Q4, hurts merge |
| **K-8** | Natural post-training histories, not constructed gauges, produce divergent reserves | Q4 present-match | per-op refs | identity | 2 matched pairs | ≥200-shuffle null | **UNSUPPORTED**: first real pair failed KL/top1 match |
| **K-9** | Merge compatibility is gauge-dependent | fp32 | base linear/TIES refs ✓ | key/tie normalization ✓ | ≥2 gauge representatives | base rerun / passing gauged pairs | **CONTROLLED**: 11 fail both; G5 passes linear only |
| **K-10** | Pristine lattice prepare improves Q4 across corpora but hurts merge | fp32 exact | own bf16 refs | lattice proof | 2 disjoint corpora | third corpus/arch | **CONTROLLED** |

★ K-2's obligations are uninteresting until the identity round-trip passes; that single control is
what prevented M1's headline from being an importer artifact (incident #10).

## 2. Budget and admission

Declared per cell type from measured costs in this run (not guesses):

| cell | wall | vram | disk | notes |
|---|---|---|---|---|
| `inspect` (Rust, static) | 2–4 s | 0 | 1 GB transient | **always first** |
| `export.gguf.f16/bf16/f32` | 60–90 s | 0 | 1 GB | CPU-tolerable |
| `quantize.gguf.q8_0` | 60 s | 2.4 GB | 0.5 GB | ladder entry point |
| `quantize.gguf.q5_k_m/q4_k_m` | 120 s | 2.4 GB | 0.5 GB | only if q8 undecided |
| `adapt.lora.r16` | 90 s/seed | 3.0 GB | 1 GB | escalate on \|gap\|<3sd |
| `merge.linear/ties` | 60 s (post-fix) | 2.4 GB | 1 GB | was 342 s: incident #7 |
| `equivalence` | 60–120 s/dtype | 2.4 GB | 1 GB | two dtypes = two cells |

Session budget guard: `--budget 60gpu-min`, admission refuses anything that does not fit the
current leases, and disk is reserved (not checked) before a cell starts — three of eleven incidents
this session were disk or VRAM arrivals, not bugs.

## 3. Ordering (derived, then pinned for the current run)

Next highest-value work, ordered by evidence gain:

1. **K-8 pair construction v2** — tune ordinary history budgets until present KL/top-1 match before future surgery.
2. **Q8 v3 out-of-sample validation** — the emitted threshold is calibrated in-sample.
3. **Fused MoE stack metering** — keying is fixed; rank-3 expert stacks remain unavailable.
4. **Q4 threshold evidence** — n=20 exists, but the current feature is not informative enough.
5. **Third K-10 corpus or second architecture** — two corpora currently agree.

## 4. Track B, gated on Track A (unchanged in spirit, sharper in shape)

`theseus inspect|preflight|verify|prepare|history` are *views over the ledger*, not new machinery:
`inspect` = L0 render, `preflight` = obligation/verdict render, `verify` = claim render,
`prepare` = a cell type that emits an artifact with an `ancestry` edge, `history` = an ancestry
traversal. This is the reason to build the ledger before the CLI: the CLI then costs almost nothing,
and every command's output is citable to a cell. The Rust inspector is the one Track-B piece that
already pays for itself (2 s, zero deps, cross-validated to 4.4e-09) and stays.

## 5. What is explicitly not planned

No dashboard; no new quantizer; no learned predictor before the ledger has ≥20 labelled cells
(K-6's refuter is the gate); no claim promoted without the three controls; no scalar health score
(schema rejects it); no comparing cells with different environment digests (I3 refuses to render).
