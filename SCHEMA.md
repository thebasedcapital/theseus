# SCHEMA — the record contract

Four record types. All are JSON objects with a stable `kind`, content-addressed by
`id = sha256(canonical_json(body minus id))[:12]`, stored append-only under `.theseus/ledger/`.
Any consumer that cannot find a field reads `null` and must treat `null` as *no evidence* (I8).

## 1. `artifact`

```jsonc
{ "kind": "artifact", "id": "a41f3c…",
  "origin": {"hf": "Qwen/Qwen2.5-0.5B", "revision": "main", "blob_sha256": "88c1425…"},
  "container": {"file": "model.safetensors", "bytes": 988097824, "sha256": "88c1425…",
                 "tensors": 290, "weights": 357826560, "dtype": "BF16", "tied": true},
  "config": {"arch": "Qwen2ForCausalLM", "hidden": 896, "q_heads": 14, "kv_heads": 2,
              "head_dim": 64, "layers": 24, "intermediate": 4864, "rms_norm_eps": 1e-6},
  "features": {                        // static, cheap, from the inspector (L0)
     "per_family": {"q_proj": {"q4_block_mse": 0.01132, "dyn_range_log10": 7.922,
                                "row_energy_imbalance": 5.4e4, "amax_over_rms": 63.0,
                                "frac_below_f16_normal": 0.00323}, …},
     "total": {"q4_block_mse": 0.01123, "convention": "mean_of_per_tensor_ratios"},
     "inspector": {"version": "theseus-inspect 0.1.0", "code_cell": "9d21…"}},
  "ancestry": [                        // how this artifact came to exist (M3's substrate)
     {"from": "a41f3c…", "op": "gauge.norm_diag", "params": {"mode": "pow2", "seed": 1}}],
  "written_by": {"cell": "…", "code_snapshot": "git 6503f8c+dirty:5f2a"}}
```

An `artifact` is a *noun*. `origin` may be null (synthetic), `ancestry` may be empty (imported).
`features.total.convention` is mandatory: pooled and per-tensor means of the same blocks differ by
up to 5.7 % and M1 was briefly fooled by that (PIPELINE_FAILURES #11).

## 2. `cell` — one measurement, immutable

```jsonc
{ "kind": "cell", "id": "9b02c7…",
  "op": {"name": "quantize.gguf", "spec": {"tag": "q8_0"}, "contract": "v2",
          "reference_cell": "c1de…"},              // the calibration it is judged against
  "subject": "7c31…",                               // artifact under test
  "environment": {                                   // I3: part of identity
     "code_snapshot": "git 6503f8c+dirty:5f2a",
     "llama_cpp": "9851 (0eca4d490)", "torch": "2.13.0+cu130",
     "export_outtype": "bf16", "compute_dtype": "fp32",
     "corpus": {"file": "eval_wikitext.txt", "byte_range": [0, 32768],
                 "sha256_16": "b41e…", "seqlen": 512, "chunks": 4}},
  "lease": {"vram_bytes": 2400000000, "disk_bytes": 2400000000, "cpu_slots": 4, "wall_s": 174},
  "result": {
     "status": "measured",                            // measured | predicted | unavailable
     "verdict": "fail",                               // pass | fail | unavailable | null
     "metrics": {"ppl": 633431.0375, "ppl_reference": 12.1351, "rel_dppl": 52173.9,
                  "kl_mean_nats": 10.690304, "kl_p999": 24.574484, "greedy_prefix_agree": 0.0,
                  "artifact_size_mb": 506.5},
     "obligation": "K-3.quantize"},
  "invalidates": null,                                // cell id, when a correction supersedes
  "notes": []}
```

Rules: `verdict: pass|fail` requires a `reference_cell` whose `subject` is the calibration
artifact and whose `environment` equals this cell's (I4, I3). `status: predicted` cells carry a
`basis` (claim id + static feature) and may never be counted in a verdict tally (I8). A cell that
fails to run is recorded with `status: unavailable` and a `reason` — never omitted, because
omission is what makes a report silently optimistic (#6).

## 3. `claim` — a sentence with obligations and a refuter

```jsonc
{ "kind": "claim", "id": "K-3",
  "text": "A function-equivalent real Transformer checkpoint can have materially reduced
           adaptation reserve, restorable by an artifact-only canonicalizer.",
  "state": "CONTROLLED",                     // unsupported|preliminary|controlled|confirmed|refuted
  "state_history": [{"at": "…", "state": "PRELIMINARY", "cells": ["c3d1…","41aa…"]},
                     {"at": "…", "state": "CONTROLLED", "cells": ["9b02…"]}],
  "obligations": {
     "equivalence": {"required": ["fp32", "bf16"], "cells": ["c3d1…","e77a…"]},
     "calibration": {"cells": ["c1de…"]},
     "controls": {"identity_roundtrip": "8ab2…", "null_gauge": "f019…", "permutation": "2ce4…"},
     "replication": {"required": 2, "done": 3, "cells": ["41aa…","77b1…","0c93…"]},
     "escalation": {"rule": "add seed iff |gap| < 3·sd", "fired": true}},
  "refuter": {"query": "capture gap between subject and repaired subject < 3·sd across seeds",
               "would_drop_to": "PRELIMINARY", "answering_cells": ["c77e…"]},
  "numbers": [{"value": 0.1559, "cite": "41aa"}, {"value": 10.690304, "cite": "9b02"}]}
```

`numbers[].cite` is mandatory: I10. `refuter` is mandatory before a claim can pass
`PRELIMINARY` — a claim nobody could disprove is not a scientific object, and it is also the
planner's cheapest source of new work: the refuter names the decisive cell.

## 4. `session`, `lease`, `incident` — operational records

```jsonc
{"kind": "incident", "id": "I-14", "at": "…", "severity": "result-threatening",
 "what": "f16 GGUF export of a gauged artifact gave ppl 177 vs bf16 12.14; quant damage numbers
          measured from an f16 source were export damage",
 "caught_by": "ad-hoc identity round-trip control", "rule_now": "I3 + I5 (controls are mandatory)",
 "prevents": "the headline result being an importer artifact"}
```

Incidents are appended by the driver automatically on: a cell failing twice, a stolen lease, a
view refusing to render (mixed environment digests), a claim blocked by a missing control.
`RUNBOOK.md`'s triage table is generated from them.

## 5. Directory layout and lifecycle

```
.theseus/ledger/{artifact,cell,claim,incident}/<id>.json      write-once
.theseus/baseline/<arch>.json                                  generated: feature+damage stats
.theseus/runs/<snapshot-id>/{src,env.json}                      frozen code per run (I2)
views/CLAIMS.md views/<K-id>.md views/table.md views/passport-<artifact>.md
```

`views/` is disposable and regenerated by `theseus render`; nothing outside `.theseus/ledger/` is
authoritative. Regenerating views is always safe, which makes the "generated" banner on those files
true rather than decorative (I9).
