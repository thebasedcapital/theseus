# HarvestLineage (harvest/README.md): how the declared-lineage population is built, verified,
# and consumed. Owned by the HarvestLineage slice; generated files live in harvest/cache/.

# What this directory is

This is the Theseus artifact-population harvest: a **metadata-first** census of PUBLIC
Hugging Face checkpoints that already carry **declared lineage** (finetune->base, merge
parents, quantized variants, adapters). It feeds the base-rate and calibrator work (LedgerSpine,
BaseRates): the `POPULATION.md` bias notes are written for the analysis slice, not by it.

The harvest is deliberately read-mostly. We hold **declared** relations (what the publisher
wrote), never guesses about hidden lineage — a filename-inferred relation is tagged
`declared_in: filename` and is a HYPOTHESIS, never elevated to card/config.

```
harvest/
  lineage.py          the enumerator (stdlib only) — list | fetch | --selfcheck
  selfcheck.sh        integrity + idempotency proof
  README.md           this file
  .gitignore          cache raw/probes/w are regenerable, never committed
  cache/
    manifest.jsonl    Contract manifest records (one JSON object per line)
    edges.jsonl       lineage edges, child/parent manifest ids, or dangling-with-reason
    usage.json        cache byte accounting + df capture
    POPULATION.md     composition, resolvable-parent fraction, archs, self-bias (generated)
    reach.json        per-repo weight-file authorization (gated/private/public)
    raw/<repo>.json   cached HF /api/models/<repo>?blobs=true payloads (audit trail)
    probes/*.json     cached HF listing/HEAD responses (idempotency, politeness)
    w/                downloaded weights (--fetch only; LRU, hard budget)
```

# Record/edge contract (see SCHEMA.md §1)

`manifest.jsonl` line:
`{"id": <sha256(content)[:12]>, "repo": "ns/name", "revision": <commit sha or null>,
  "kind": base|finetune|instruct|merge|quant|adapter|mlx,
  "declared_base": [repo,...]|null, "lineage_source": card|config|filename|none,
  "arch": str|null, "params": int|null, "dtype": str|null, "quant": str|null,
  "files": {<weight filename>: bytes}, "weights_present": bool, "sha256_16": str|null,
  "downloads": int|null, "created": str}`

`id` is the content hash of the record body **minus** `id` (write-once semantics: a changed
body changes the id, so a re-bump of `downloads` mints a new id — key consumers on `repo`).

`edges.jsonl` line:
`{"child": <id>, "parent": <id>|null, "relation": finetune_of|merge_of|quant_of|adapter_on,
  "declared_in": card|config|filename|none}` — plus, for dangling parents,
`"parent_repo": <repo>, "reason": "declared parent has no manifest record"`.

Source priority (strongest first, all *declared* by the publisher):
1. HF `base_model:` tags (`base_model:finetune:|:quantized:|:adapter:|:merge:`)      -> card
2. `cardData.base_model` / mergekit `base_models`                                    -> card
3. `config.peft.base_model_name_or_path` (PEFT)                                      -> config
4. raw `config.json` `_name_or_path`                                                  -> config
5. quant filename matching an in-set base exactly                                     -> filename (hypothesis)

# Commands

```bash
/usr/bin/python3 harvest/lineage.py list            # build/refresh (offline after first run)
/usr/bin/python3 harvest/lineage.py --selfcheck      # validate cache contents
bash harvest/selfcheck.sh                           # selfcheck + idempotency + disk cap proof
/usr/bin/python3 harvest/lineage.py --fetch <id...> --budget-gb 0.8   # weights, LRU, refusals
/usr/bin/python3 harvest/lineage.py fetch --vacate  # drop all fetched weights
```

- `--fetch` computes byte budget = current cache/w + incoming; it LRU-evicts the oldest
  downloaded files only as needed and **refuses** (non-zero exit, explicit message) if the
  requested set still would exceed `--budget-gb`. Downloads are sha256-verified against the
  `lfs.sha256` recorded in `raw/<repo>.json`; a mismatch deletes the file and refuses.
- `budget-gb` is a *hard* cap; the default `0.8` fits tiny models (e.g. SmolLM2-135M ~ 270 MB)
  and rejects 0.5B+ weights. Watch `files`/`usedStorage` in the manifest before asking for more.

# Honesty rules (enforced)

- metadata-only by default: nothing is downloaded except with `--fetch`;
- no evidence -> `null`, never a fabricated number (fail closed);
- filename relations stay `declared_in: filename`;
- every cached response is keyed by its URL, so re-runs are deterministic and offline;
- the hub is the sole source of truth for repo existence/size/sha — if it refuses (429/5xx for
  > 60 s) we stop and report; nothing is invented.

# Limitations

- bias of popularity: `sort=downloads` over-represents the big English instruct bases (see
  POPULATION.md §4);
- `instruct` vs `finetune` is a name heuristic (HF tags both `base_model:finetune:`);
- one record per repo; a GGUF repo hosting many quants is a single record whose `files` map
  lists every `.gguf`;
- gated/private repos keep metadata but get `weights_present: false`; `reach.json` states why.
