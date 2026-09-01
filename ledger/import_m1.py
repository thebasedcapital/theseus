"""`ledger/import_m1.py` — PLAN.md §0: map today's `m1/work` files onto ledger records.

| today | becomes |
|---|---|
| VARIANTS.json | `artifact` records + `ancestry` edges |
| equiv/<v>.json | an equivalence `cell` per compute dtype (fp32 here; bf16 cells come from compute_dtype_check.json and obligations stay OPEN) |
| ops/<v>.{gguf,adapt,merge}.json | measurement `cell`s, environment rebuilt from embedded versions/corpus/export/pass_contract |
| M1_OPS.json (aggregate) | fallback for (variant, op) cells whose per-cell files the live panel recycled |
| quant_ref.json, ref_capture.json | the `reference_cell` targets satisfying I4 (verified against base calibration cells) |
| PREDICTIONS*.json, debts_lattice.json | `status: predicted` cells attached to K-6 (never tallied, I8) |
| M1_OPS.old-f16.json | pre-amendment gguf cells with `environment.unknown: true` — EXCLUDED from comparisons |
| PIPELINE_FAILURES.md | `incident` records linked to the invariant that now prevents each |

Every imported record carries `provenance {source_file, imported_at, mapping: "PLAN.md §0"}`.
Where conditions cannot be recovered the environment is marked `unknown: true` and the digest is
None: those numbers are preserved but never compared (never guess an outtype). All record ids are
content hashes — identical re-import is a no-op; ids are stable across time because the import
epoch is fixed and environment contract tags are source-independent (a cell from `ops/*.json` and
the same cell re-read from the `M1_OPS.json` aggregate share one environment digest, I3).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .env import env_digest
from .store import content_id, KINDS
from . import rules

IMPORT_EPOCH = "2026-08-31"
CODE_SNAPSHOT_IMPORT = f"m1-import-{IMPORT_EPOCH}"   # frozen-code snapshot (I2) for the legacy run
J_CONVENTION = "mean_of_per_tensor_ratios"           # debts_lattice/PREDICTIONS_new definition
# Stable contract-version tags so cells sourced from ops/*.json and from the recycled M1_OPS.json
# aggregate share one environment digest (I3). Grounded in the contract text the panel recorded
# ("reference-relative", amended 2026-08-30 before any variant was measured).
GGUF_CONTRACT = content_id({"op": "quantize.gguf", "contract": "reference-relative amended 2026-08-30"})
ADAPT_CONTRACT = content_id({"op": "adapt.lora.r16", "contract": "reference-relative v1"})
TORCH_M1 = "2.13.0+cu130"   # recorded torch for this run in every ops/*.json (29/30 files)

# PIPELINE_FAILURES.md → invariant that now prevents it (footer map of the doc, plus SYSTEM §3
ADAPT_V2 = "adapt-v2-true-lora-base-frozen"

PF_INVARIANT = {1: "I2", 2: "I2", 3: "I6", 4: "I4", 5: "I3+I5", 6: "I3+I5",
                7: "I1+I7", 8: "I6", 9: "I1+I7", 10: "I1+I7", 11: "I1+I7",
                12: "K-7", 13: "I6", 14: "I1+I7", 15: "I2+I5", 16: "I1+I7",
                17: "I2+I5", 18: "I2+I5+I10", 19: "I2+I5+I10", 20: "I3+I7", 21: "I4", 22: "I6+I9"}
PF_SEVERITY = {4: "result-threatening", 5: "result-threatening", 6: "result-threatening",
               8: "result-threatening", 10: "result-threatening", 11: "result-threatening",
               14: "result-threatening", 15: "result-threatening", 17: "result-threatening",
               18: "result-threatening", 19: "result-threatening"}


class ImportReport:
    def __init__(self):
        self.files = {}
        self.total = {"artifact": 0, "cell": 0, "claim": 0, "incident": 0}
        self.confirmed_refs = []

    def file(self, name):
        return self.files.setdefault(name, {"records": 0, "unknown": 0, "malformed": 0, "notes": []})

    def malformed(self, name, reason):
        self.file(name)["malformed"] += 1
        self.file(name)["notes"].append(f"malformed: {reason}")

    def record(self, name, kind):
        self.file(name)["records"] += 1
        self.total[kind] += 1

    def unknown(self, name):
        self.file(name)["unknown"] += 1

    def note(self, name, msg):
        self.file(name)["notes"].append(msg)


def _load(w, name, report):
    p = w / name
    if not p.exists():
        report.note(name, "file absent — skipped")
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        report.malformed(name, f"unparseable JSON: {exc}")
        return None


def _provenance(src):
    return {"source_file": src, "imported_at": IMPORT_EPOCH, "mapping": "PLAN.md §0"}


# ---------------------------------------------------------------------------
# ARTIFACTS
# ---------------------------------------------------------------------------

def _config_fields(config):
    if not isinstance(config, dict):
        return {}
    heads = config.get("num_attention_heads")
    hidden = config.get("hidden_size")
    head_dim = config.get("head_dim")
    if head_dim is None and heads and hidden:
        head_dim = hidden // heads
    return {
        "arch": (config.get("architectures") or [None])[0] or None,
        "hidden": hidden, "layers": config.get("num_hidden_layers"),
        "q_heads": heads, "kv_heads": config.get("num_key_value_heads"),
        "head_dim": head_dim, "intermediate": config.get("intermediate_size"),
        "rms_norm_eps": config.get("rms_norm_eps"),
    }


def _variant_config(w, name):
    d = w / name
    p = d / "config.json" if d.is_dir() else None
    if p and p.exists():
        try:
            return _config_fields(json.loads(p.read_text()))
        except (OSError, ValueError):
            return {}
    return {}


def _features_from_variants(debt, eq, pred_new, census):
    """Best available L0 static features, with the mandatory convention (I7/PF#11)."""
    fam, j_var = None, None
    if debt and debt.get("per_tensor"):
        fam = dict(debt["per_tensor"])
        j_var = debt.get("J_var", debt.get("J"))
    elif eq and eq.get("cond_b"):
        fam = dict(eq["cond_b"])
        j_var = sum(fam.values()) / len(fam) if fam else None
    elif pred_new and pred_new.get("per_tensor"):
        fam = dict(pred_new["per_tensor"])
        j_var = pred_new.get("J_var")
    features = {"total": {"q4_block_mse": j_var, "convention": J_CONVENTION},
                "per_family": fam or {}}
    if census:
        c = {k: census.get(k) for k in ("below_f16_normal", "frac_below_f16_normal",
                                        "min_abs_weight", "weights")}
        features["census"] = {k: v for k, v in c.items() if v is not None}
    return features


def _ancestry_edges(variant, name, id_of):
    edges = []
    gauge = variant.get("gauge") or {}
    for t in gauge.get("transforms") or []:
        fam = t.get("family")
        mode = t.get("mode") or ""
        params = {k: t[k] for k in ("seed", "decades", "groups", "c") if k in t}
        edges.append({"from": id_of.get("base"),
                      "op": f"gauge.{fam}.{mode}".rstrip("."), "params": params})
    canon = variant.get("canon") or variant.get("canonicalize")
    if canon:
        src = name[:-4] if name.endswith("_rep") and name[:-4] in id_of else "base"
        params = {}
        steps = (canon.get("canonicalize") or []) if isinstance(canon, dict) else []
        for step in steps:
            if isinstance(step, dict):
                params.setdefault("family", step.get("canon"))
                if "target" in step:
                    params["target"] = step["target"]
        edges.append({"from": id_of.get(src), "op": "canonicalize.prepare", "params": params})
    if variant.get("canonicalize") and not edges:
        src = name[:-4] if name.endswith("_rep") and name[:-4] in id_of else "base"
        edges.append({"from": id_of.get(src), "op": "canonicalize.prepare",
                      "params": {"canonicalize": str(variant.get("canonicalize"))}})
    return edges


def build_artifacts(w, variants, eq, debts, pred_new, export_damage, report):
    """Returns (records, name_to_id): artifacts first so cells can reference their ids."""
    name_to_id = {}
    records = []

    def admit(name, origin, container, config, features, ancestry):
        body = {"kind": "artifact", "origin": origin, "container": container,
                "config": config, "features": features, "ancestry": ancestry,
                "written_by": {"cell": None, "code_snapshot": CODE_SNAPSHOT_IMPORT},
                "provenance": _provenance(f"VARIANTS.json→{name}")}
        name_to_id[name] = content_id(body)
        records.append((name, body))

    cache_cfg = Path("/home/admin/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/"
                     "060db6499f32faf8b98477b0a26969ef7d8b9987/config.json")
    base_config = {}
    if cache_cfg.exists():
        try:
            base_config = _config_fields(json.loads(cache_cfg.read_text()))
        except (OSError, ValueError):
            pass
    debt = debts.get("base") or {}
    eq_base = eq.get("base") or {}
    census = (export_damage.get("base") or {}).get("census") or {}
    # blob hash: g5_c8_rep reproduced the pristine file byte-for-byte (CLAIMS.md K-5) with the
    # HF blob sha256 recorded in VARIANTS.json.
    repr_hash = (variants.get("g5_c8_rep") or {}).get("sha256")
    admit("base",
          {"hf": "Qwen/Qwen2.5-0.5B", "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
           "blob_sha256": repr_hash},
          {"file": "model.safetensors", "bytes": 988097824, "sha256": repr_hash,
           "tensors": 290, "weights": census.get("weights") or 357868417,
           "dtype": "BF16", "tied": True},
          base_config,
          _features_from_variants(debt, eq_base, None, census),
          [])
    report.record("VARIANTS.json", "artifact")

    order = [n for n in variants if not n.endswith("_rep")]
    order += [n for n in variants if n.endswith("_rep")]
    for name in order:
        v = variants[name]
        cfg = _variant_config(w, name)
        eq_v = eq.get(name) or {}
        debt_v = debts.get(name) or {}
        pred_v = (pred_new.get("variants") or {}).get(name) or {}
        cens = (export_damage.get(name) or {}).get("census") or {}
        container = {"file": "model.safetensors", "bytes": v.get("bytes"),
                     "sha256": v.get("sha256"), "tensors": None, "weights": None,
                     "dtype": "BF16", "tied": (not bool(v.get("untied")))}
        admit(name, None, container, cfg or None,
              _features_from_variants(debt_v, eq_v, pred_v, cens),
              _ancestry_edges(v, name, name_to_id))
        report.record("VARIANTS.json", "artifact")

    # measured variants with equivalence cells but no VARIANTS entry (dirs freed for disk):
    for name in eq:
        if name in name_to_id or name == "base":
            continue
        body = {"kind": "artifact", "origin": None,
                "container": {"file": "model.safetensors", "bytes": None, "sha256": None,
                              "tensors": None, "weights": None, "dtype": None, "tied": None},
                "config": None,
                "features": _features_from_variants(debts.get(name) or {}, eq[name] or {},
                                                    None, None),
                "ancestry": [],
                "written_by": {"cell": None, "code_snapshot": CODE_SNAPSHOT_IMPORT},
                "provenance": _provenance(f"equiv/{name}.json (container UNAVAILABLE: dir freed)")}
        name_to_id[name] = content_id(body)
        records.append((name, body))
        report.record("equiv/*.json", "artifact")
        report.note("equiv/*.json", f"{name}: metadata-only artifact (dir freed)")
    return [b for _, b in records], name_to_id


# ---------------------------------------------------------------------------
# CELLS
# ---------------------------------------------------------------------------

def _corpus_sha(w):
    src = w.parent / "data" / "eval_wikitext.txt"
    if not src.exists():
        return None
    return hashlib.sha256(src.read_bytes()[:32768]).hexdigest()[:16]


def _base_env(corpus_sha, torch_v, export_outtype, seqlen, chunks, contract_version):
    env = {"code_snapshot": CODE_SNAPSHOT_IMPORT, "torch": torch_v,
           "export_outtype": export_outtype, "compute_dtype": "fp32",
           "corpus": {"file": "eval_wikitext.txt", "byte_range": [0, 32768],
                      "sha256_16": corpus_sha},
           "seqlen": seqlen, "chunk_policy": f"chunks={chunks},ngl=0",
           "contract_version": contract_version}
    env["digest"] = env_digest(env)
    return env


def _adapt_env(var):
    env = {"code_snapshot": CODE_SNAPSHOT_IMPORT, "torch": None,
           "compute_dtype": "fp32", "export_outtype": None, "corpus": None,
           "seqlen": var.get("seq_len") or 128, "chunk_policy": None,
           "contract_version": ADAPT_CONTRACT}
    env["digest"] = env_digest(env)
    return env


def _cell(inner):
    body = {"kind": "cell"}
    body.update(inner)
    return body
def _verdict(value):
    if value is True:
        return "pass"
    if value is False:
        return "fail"
    return value if value in ("pass", "fail", "unavailable", "stale", "invalidated") else None




def _legacy_env(torch_v):
    env = {"code_snapshot": CODE_SNAPSHOT_IMPORT, "torch": torch_v,
           "export_outtype": None, "compute_dtype": None, "corpus": None,
           "seqlen": None, "chunk_policy": None, "contract_version": None,
           "unknown": True}
    env["digest"] = None          # I3: never comparable — conditions unrecoverable
    return env


def _gguf_cell(corpus_sha, tag, rung, art_id, r, torch_v, export_ot, lease_s,
               obligation, prov, contract_version=GGUF_CONTRACT, unknown=False):
    env = (_unknown_env(torch_v) if unknown
           else _base_env(corpus_sha, torch_v, export_ot, 512, 4, contract_version))
    return _cell({
        "op": {"name": f"quantize.gguf.{rung}", "spec": {"tag": rung},
               "contract": "v2", "reference_cell": None,
               "role": "reference_calibration" if obligation.startswith("K-4.calibration.") else None},
        "subject": art_id, "environment": env,
        "lease": {"wall_s": lease_s},
        "result": {"status": "measured", "verdict": _verdict(r.get("pass")),
                   "metrics": {"ppl": r.get("ppl"), "ppl_f16": r.get("ppl_f16"),
                               "rel_dppl": r.get("rel_dppl"), "kl_mean": r.get("kl_mean"),
                               "kl_p999": r.get("kl_p999"),
                               "prefix_agree": r.get("prefix_agree"),
                               "artifact_size_mb": r.get("size_mb")}},
        "invalidates": None, "notes": [],
        "obligation": obligation, "provenance": prov})


def _unknown_env(torch_v):
    env = _legacy_env(torch_v)
    return env


def _adapt_cell(art_id, var, cal_id, obligation, fname, torch_v=None):
    env = _adapt_env(var)
    return _cell({
        "op": {"name": "adapt.lora.r16",
               "spec": {"rank": var.get("lora_rank") or 16, "seed": var.get("seed"),
                        "steps": var.get("steps")},
               "contract": "v1", "reference_cell": cal_id,
               "role": "reference_calibration" if cal_id is None else None},
        "subject": art_id, "environment": env,
        "lease": {"wall_s": var.get("runtime_s") or var.get("runtime_s")},
        "result": {"status": "measured", "verdict": _verdict(var.get("pass")),
                   "metrics": {"capture": var.get("capture"),
                               "capture_ref": var.get("capture_ref"),
                               "capture_threshold": var.get("capture_threshold"),
                               "protected_dppl": var.get("protected_dppl"),
                               "protected_dppl_ref": var.get("protected_dppl_ref"),
                               "train_examples": var.get("train_examples"),
                               "heldout_examples": var.get("heldout_examples")}},
        "invalidates": None, "notes": [],
        "obligation": obligation, "provenance": _provenance(fname)})


def build_cells(w, name_to_id, ops, eq, cd_check, export_damage, pred, pred_new, debts, report):
    records = []
    seed_rep = _load(w, "seed_replicate.json", report) or {}
    corpus_sha = _corpus_sha(w)

    # ---- equivalence cells (fp32 compute) ----
    for vname, ev in eq.items():
        if not isinstance(ev, dict):
            continue
        gv = ev.get("gate") or {}
        contract_version = content_id(gv) if gv else None
        env = {"code_snapshot": CODE_SNAPSHOT_IMPORT, "torch": ev.get("torch"),
               "compute_dtype": "fp32", "export_outtype": None,
               "corpus": None, "seqlen": (ev.get("metrics") or {}).get("seqlen"),
               "chunk_policy": None, "contract_version": contract_version}
        env["digest"] = env_digest(env)
        m = ev.get("metrics") or {}
        map_v = {"EQUIVALENT": "pass", "NOT_EQUIVALENT": "fail"}
        records.append(_cell({
            "op": {"name": "equivalence.verify", "spec": {"gate": gv},
                   "contract": "v2", "reference_cell": None},
            "subject": name_to_id.get(vname), "environment": env,
            "lease": {"wall_s": ev.get("duration_s")},
            "result": {"status": "measured", "verdict": map_v.get(ev.get("verdict")),
                       "metrics": {"kl_mean_nats": m.get("kl_mean_nats"),
                                   "max_dlogit": m.get("max_dlogit"),
                                   "top1_agree": m.get("top1_agree"),
                                   "ppl_a": m.get("ppl_a"), "ppl_b": m.get("ppl_b"),
                                   "rel_ppl": ev.get("rel_ppl"),
                                   "n_positions": m.get("n_positions")}},
            "invalidates": None, "notes": [],
            "obligation": f"K-1.equivalence.{vname}",
            "provenance": _provenance(f"equiv/{vname}.json")}))
        report.record("equiv/*.json", "cell")

    # ---- compute_dtype_check.json: fp32 + bf16 equivalence, distinct digests ----
    for vname, entry in (cd_check or {}).items():
        runs = entry.get("runs") or {}
        for run_name, run in runs.items():
            if not isinstance(run, dict):
                continue
            dtype = "bf16" if run_name == "bf16_compute" else "fp32"
            env = {"code_snapshot": CODE_SNAPSHOT_IMPORT, "torch": None,
                   "compute_dtype": dtype, "export_outtype": None, "corpus": None,
                   "seqlen": entry.get("seqlen"), "chunk_policy": None,
                   "contract_version": content_id({"cd_check": run_name})}
            env["digest"] = env_digest(env)
            tag = (f"K-1.equivalence_bf16.{vname}" if dtype == "bf16"
                   else f"K-1.equivalence_fp32cd.{vname}")
            records.append(_cell({
                "op": {"name": "equivalence.compute_dtype", "spec": {"compute": dtype},
                       "contract": "v2", "reference_cell": None},
                "subject": name_to_id.get(vname), "environment": env,
                "lease": {"wall_s": None},
                "result": {"status": "measured",
                           "verdict": "pass" if run.get("top1_agree", 0) >= 0.995 else None,
                           "metrics": {"kl_mean_nats": run.get("kl_mean_nats"),
                                       "max_dlogit": run.get("max_dlogit"),
                                       "top1_agree": run.get("top1_agree"),
                                       "ppl_a": run.get("ppl_a"),
                                       "ppl_b": run.get("ppl_b"),
                                       "n_positions": entry.get("tokens")}},
                "invalidates": None, "notes": [],
                "obligation": tag, "provenance": _provenance("compute_dtype_check.json")}))
            report.record("compute_dtype_check.json", "cell")

    # ---- gguf probes: base calibration rungs + base export reference first ----
    gc_ids = {}
    for fname in sorted(ops):
        if not fname.endswith(".gguf.json") or not fname.startswith("base."):
            continue
        d = ops[fname]
        export_ot = (d.get("export") or {}).get("outtype")     # known bf16 for the amended run
        res = d.get("results") or {}
        base_f16 = res.get("f16") or {}
        if base_f16:
            env = _base_env(corpus_sha, d.get("torch"), "bf16", 512, 4, GGUF_CONTRACT)
            records.append(_cell({
                "op": {"name": "export.gguf.bf16", "spec": {"outtype": "bf16"},
                       "contract": "v2", "reference_cell": None},
                "subject": name_to_id["base"], "environment": env,
                "lease": {"wall_s": d.get("duration_s")},
                "result": {"status": "measured", "verdict": None,
                           "metrics": {"ppl": base_f16.get("ppl")}},
                "invalidates": None, "notes": [],
                "obligation": "K-2.export.base.bf16",
                "provenance": _provenance(f"ops/{fname}")}))
            report.record(fname, "cell")
        for rung in ("q8_0", "q5_k_m", "q4_k_m"):
            r = res.get(rung) or {}
            if not r:
                continue
            body = _gguf_cell(corpus_sha, "base", rung, name_to_id["base"], r, d.get("torch"),
                              export_ot, d.get("duration_s"), f"K-4.calibration.{rung}",
                              _provenance(f"ops/{fname}"))
            gc_ids[rung] = content_id(body)
            records.append(body)
            report.record(fname, "cell")

    for fname in sorted(ops):
        if not fname.endswith(".gguf.json") or fname.startswith("base."):
            continue
        tag = fname[: -len(".gguf.json")]
        d = ops[fname]
        art = name_to_id.get(tag)
        if art is None:
            report.note(fname, f"variant {tag} has no artifact metadata — UNAVAILABLE")
            continue
        export_ot = (d.get("export") or {}).get("outtype")
        res = d.get("results") or {}
        unknown = bool(d.get("_missing_export"))    # aggregate row without a recorded outtype
        f16row = res.get("f16") or {}
        if f16row and not unknown:
            env = _base_env(corpus_sha, d.get("torch"), "bf16", 512, 4, GGUF_CONTRACT)
            records.append(_cell({
                "op": {"name": "export.gguf.bf16", "spec": {"outtype": "bf16"},
                       "contract": "v2", "reference_cell": None},
                "subject": art, "environment": env,
                "lease": {"wall_s": d.get("duration_s")},
                "result": {"status": "measured", "verdict": None,
                           "metrics": {"ppl": f16row.get("ppl")}},
                "invalidates": None, "notes": [],
                "obligation": f"K-2.export.{tag}.bf16",
                "provenance": _provenance(f"ops/{fname}")}))
            report.record(fname, "cell")
        for rung in ("q8_0", "q5_k_m", "q4_k_m"):
            r = res.get(rung) or {}
            if not r:
                continue
            ref = gc_ids.get(rung) if (export_ot == "bf16" and not unknown) else None
            body = _gguf_cell(corpus_sha, tag, rung, art, r, d.get("torch"),
                              export_ot, d.get("duration_s"), f"K-4.quantize.{tag}.{rung}",
                              _provenance(f"ops/{fname}"), unknown=unknown)
            if unknown:
                body["op"]["reference_cell"] = None
                body["environment"] = _unknown_env(d.get("torch"))
                body["obligation"] = f"K-4.legacy-unknown.{tag}.{rung}"
                body["result"]["status"] = "unavailable"
                body["result"]["verdict"] = None
                body["result"]["reason"] = "export outtype and comparison conditions unrecoverable"
                report.unknown(fname)
            else:
                body["op"]["reference_cell"] = ref
            records.append(body)
            report.record(fname, "cell")

    # ---- adapt / LoRA cells (base calibration first, then variants) ----
    lora_cal_id = None
    base_var = None
    base_ref_note = "ops/base.adapt.json"
    for fname in ("base.adapt.json", "probes/base_adapt.json"):
        var = ((ops.get(fname) or {}).get("results") or {}).get("variant") or {}
        if var.get("contract_version") == ADAPT_V2:
            base_var = var
            base_ref_note = f"ops/{fname} [{ADAPT_V2}]"
            break
        if var and base_var is None:
            base_var = var
            base_ref_note = (f"ops/{fname} (unversioned: no contract_version, so v2 verdicts "
                             "calibrated against it are marked, not silently joined)")
    if base_var is None:
        fb = _adapt_reference(w)
        if fb:
            base_var = fb
            base_ref_note = f"fallback {fb.get('base_reference_file', 'ref_capture.json')} " \
                            f"[{fb.get('base_reference_status')}]"
    if base_var:
        body = _adapt_cell(name_to_id.get("base"), base_var, None,
                           "K-3.calibration.lora", base_ref_note,
                           torch_v=ops.get("base.adapt.json", {}).get("torch"))
        lora_cal_id = content_id(body)
        records.append(body)
        report.record("ops/base.adapt.json", "cell")

    for fname in sorted(ops):
        if not fname.endswith(".adapt.json") or fname.startswith("base."):
            continue
        tag = fname[: -len(".adapt.json")]
        d = ops[fname]
        var = (d.get("results") or {}).get("variant") or {}
        if not var:
            env = _unknown_env(d.get("torch"))
            records.append(_cell({
                "op": {"name": "adapt.lora.r16", "spec": {"rank": 16},
                       "contract": "v1", "reference_cell": lora_cal_id},
                "subject": name_to_id.get(tag), "environment": env,
                "lease": {"wall_s": None},
                "result": {"status": "unavailable", "verdict": None, "reason": "no result row"},
                "invalidates": None, "notes": [],
                "obligation": f"K-3.lora.{tag}",
                "provenance": _provenance(fname)}))
            report.record(fname, "cell")
            continue
        body = _adapt_cell(name_to_id.get(tag), var, lora_cal_id, f"K-3.lora.{tag}", fname,
                           torch_v=d.get("torch"))
        records.append(body)
        report.record(fname, "cell")
    # ---- corrected true-LoRA seed replication ----
    seed_contract = seed_rep.get("_contract") or {}
    seed_env = _adapt_env({"seq_len": 128})
    for tag, entry in sorted(seed_rep.items()):
        if tag.startswith("_") or not isinstance(entry, dict):
            continue
        for seed, row in sorted((entry.get("seeds") or {}).items(), key=lambda x: int(x[0])):
            records.append(_cell({
                "op": {"name": "adapt.lora.r16.seed_replication",
                       "spec": {"rank": 16, "seed": int(seed)},
                       "contract": seed_contract.get("version"),
                       "reference_cell": lora_cal_id},
                "subject": name_to_id.get(tag), "environment": dict(seed_env),
                "lease": {"wall_s": sum(x.get("runtime_s", 0) for x in row.get("grid", []))},
                "result": {"status": "measured", "verdict": None,
                           "metrics": {"capture": row.get("capture"),
                                       "selected_lr": row.get("selected_lr")}},
                "invalidates": None, "notes": [],
                "obligation": f"K-3.replication.{tag}.{seed}",
                "provenance": _provenance("seed_replicate.json")}))
            report.record("seed_replicate.json", "cell")
    summary = seed_rep.get("_summary") or {}
    if summary:
        records.append(_cell({
            "op": {"name": "adapt.lora.r16.seed_summary", "spec": {"rank": 16},
                   "contract": seed_contract.get("version"), "reference_cell": lora_cal_id},
            "subject": name_to_id.get("base"), "environment": dict(seed_env),
            "lease": {"wall_s": None},
            "result": {"status": "measured", "verdict": None, "metrics": summary},
            "invalidates": None, "notes": [], "obligation": "K-3.replication.summary",
            "provenance": _provenance("seed_replicate.json")}))
        report.record("seed_replicate.json", "cell")

    # ---- merge cells: base calibration first, then every variant references it (I4) ----
    merge_cal_id = None
    merge_files = sorted((f for f in ops if f.endswith(".merge.json")),
                         key=lambda f: (not f.startswith("base."), f))
    for fname in merge_files:
        tag = fname[: -len(".merge.json")]
        d = ops[fname]
        if not isinstance(d, dict):
            d = {"_m1ops_status": str(d)}
        results = d.get("results") or {}
        contract = (results.get("contract") or {}).get("version") or "merge-v2-base-calibrated"
        if d.get("_m1ops_status"):
            reason = f"M1_OPS aggregate merge status: {d['_m1ops_status']}"
            status, verdict = "unavailable", None
        else:
            err = d.get("error") or {}
            if isinstance(err, str):
                err = {"message": err}
            if err:
                reason, status, verdict = (err.get("message") or "merge probe error"), "unavailable", None
            elif results:
                rows = [r for r in (results.get("linear") or {}).get("matrix", [])] + \
                       [r for r in (results.get("ties") or {}).get("matrix", [])]
                verdicts = {r.get("pass") for r in rows if isinstance(r, dict)}
                verdict = ("pass" if True in verdicts else "fail") if verdicts else None
                status, reason = "measured", None
            else:
                reason, status, verdict = "no merge result recorded (M1_OPS empty)", "unavailable", None
        if tag != "base" and merge_cal_id is None and status == "measured":
            reason, status, verdict = "missing PASSING base merge calibration reference (I4)", "unavailable", None
        env = {"code_snapshot": CODE_SNAPSHOT_IMPORT, "torch": d.get("torch") or TORCH_M1,
               "compute_dtype": "fp32", "export_outtype": None, "corpus": None,
               "seqlen": None, "chunk_policy": None, "contract_version": contract}
        env["digest"] = env_digest(env)
        obligation = "K-9.calibration.merge" if tag == "base" else f"K-9.merge.{tag}"
        linear = results.get("linear") or {}
        ties = results.get("ties") or {}
        linear_rows = [r for r in linear.get("matrix", []) if isinstance(r, dict)]
        ties_rows = [r for r in ties.get("matrix", []) if isinstance(r, dict)]
        metrics = {
            "base_eval_ppl": results.get("base_eval_ppl"),
            "candidate_ppl": results.get("candidate_ppl"),
            "base_rule_loss": results.get("base_rule_loss"),
            "linear_smallest_passing_alpha": linear.get("smallest_passing_alpha"),
            "ties_smallest_passing_alpha": ties.get("smallest_passing_alpha"),
            "linear_min_ppl_ratio": min((r.get("ppl_ratio") for r in linear_rows
                                          if r.get("ppl_ratio") is not None), default=None),
            "ties_min_ppl_ratio": min((r.get("ppl_ratio") for r in ties_rows
                                        if r.get("ppl_ratio") is not None), default=None),
        }
        result = {"status": status, "verdict": verdict}
        if reason:
            result["reason"] = reason
        if any(v is not None for v in metrics.values()):
            result["metrics"] = metrics
        body = _cell({
            "op": {"name": "merge.linear", "spec": {"compat": "M1 merge probe"},
                   "contract": contract, "reference_cell": None if tag == "base" else merge_cal_id,
                   "role": "reference_calibration" if tag == "base" else None},
            "subject": name_to_id.get(tag), "environment": env,
            "lease": {"wall_s": d.get("duration_s")}, "result": result,
            "invalidates": None, "notes": [], "obligation": obligation,
            "provenance": _provenance(fname)})
        records.append(body)
        if tag == "base" and status == "measured" and verdict == "pass":
            merge_cal_id = content_id(body)
        report.record(fname, "cell")

    # ---- export-damage cells (f16/f32 export op) + identity round-trip control ----
    for vname, ed in (export_damage or {}).items():
        if not isinstance(ed, dict):
            continue
        ppl = ed.get("ppl") or {}
        ratio = ed.get("export_damage_ratio")
        for ot, pv in (("f16", ppl.get("f16")), ("f32", ppl.get("f32"))):
            if pv is None:
                continue
            env = _base_env(corpus_sha, None, ot, 512, 4, content_id({"export_damage": ot}))
            records.append(_cell({
                "op": {"name": f"export.gguf.{ot}", "spec": {"outtype": ot},
                       "contract": "v2", "reference_cell": None},
                "subject": name_to_id.get(vname), "environment": env,
                "lease": {"wall_s": None},
                "result": {"status": "measured", "verdict": None,
                           "metrics": {"ppl": pv, "ppl_bf16": ppl.get("bf16"),
                                       "export_damage_ratio": ratio}},
                "invalidates": None, "notes": [],
                "obligation": f"K-2.export.{vname}.{ot}",
                "provenance": _provenance("export_damage.json")}))
            report.record("export_damage.json", "cell")
    if export_damage and isinstance(export_damage.get("base"), dict):
        b = export_damage["base"].get("ppl") or {}
        ok = bool(b.get("bf16")) and abs(1 - (b.get("f16") or 0) / b["bf16"]) < 0.02
        env = _base_env(corpus_sha, None, "f16", 512, 4,
                        content_id({"control": "identity_roundtrip"}))
        records.append(_cell({
            "op": {"name": "control.identity_roundtrip", "spec": {"kind": "identity"},
                   "contract": "v1", "reference_cell": None},
            "subject": name_to_id.get("base"), "environment": env,
            "lease": {"wall_s": None},
            "result": {"status": "measured", "verdict": "pass" if ok else None,
                       "metrics": {"ppl_bf16": b.get("bf16"), "ppl_f16": b.get("f16"),
                                   "ratio": export_damage["base"].get("export_damage_ratio")}},
            "invalidates": None, "notes": [],
            "obligation": "K-2.control.identity_roundtrip",
            "provenance": _provenance("export_damage.json")}))
        report.record("export_damage.json", "cell")

    # ---- pre-amendment f16 numbers: environment.unknown ⇒ excluded from comparisons ----
    for vname, entry in sorted(_load_old_f16(w).items()):
        gg = (entry.get("ops") or {}).get("gguf") or {}
        res = gg.get("results") or {}
        for rung in ("q8_0", "q5_k_m", "q4_k_m"):
            r = res.get(rung) or {}
            if not r:
                continue
            body = _gguf_cell(corpus_sha, vname, rung, name_to_id.get(vname), r, None,
                              None, None, f"K-4.legacy-unknown.{vname}.{rung}",
                              _provenance("M1_OPS.old-f16.json"), unknown=True)
            body["result"]["status"] = "unavailable"
            body["result"]["verdict"] = None
            body["result"]["reason"] = "pre-amendment f16 conditions unrecoverable; preserved for archaeology only"
            records.append(body)
            report.record("M1_OPS.old-f16.json", "cell")
            report.unknown("M1_OPS.old-f16.json")

    # ---- predicted cells (status: predicted — recorded, never tallied, I8) ----
    sources = [("PREDICTIONS_new.json", pred_new, "variants"),
               ("PREDICTIONS.json", pred, None)]
    for srcname, srcdata, sub in sources:
        if not isinstance(srcdata, dict):
            continue
        variants_map = srcdata.get(sub) if sub else srcdata
        if not isinstance(variants_map, dict):
            continue
        for vname, pv in variants_map.items():
            if not isinstance(pv, dict):
                continue
            debt = pv.get("debt")
            vv = "fail" if (debt is not None and debt > 1.0e-3) else "pass"
            records.append(_cell({
                "op": {"name": "predict.quantize", "spec": {"feature": "debt"},
                       "contract": "K-6.prediction.v1", "reference_cell": None},
                "subject": name_to_id.get(vname),
                "environment": {"code_snapshot": CODE_SNAPSHOT_IMPORT,
                                "compute_dtype": None, "export_outtype": None,
                                "corpus": None, "seqlen": None, "chunk_policy": None,
                                "contract_version": content_id({"K-6": "prediction-v1"})},
                "basis": {"claim": "K-6", "static_feature": "debt = J_var − J_base"},
                "result": {"status": "predicted", "verdict": vv,
                           "metrics": {"J_base": pv.get("J_base"), "J_var": pv.get("J_var"),
                                       "debt": debt}},
                "invalidates": None,
                "notes": [] if srcname.startswith("PREDICTIONS") else
                         [f"declared equivalence: {pv.get('equiv')}"],
                "obligation": f"K-6.prediction.{vname}",
                "provenance": _provenance(srcname)}))
            report.record("PREDICTIONS*/debts_lattice.json", "cell")
    for vname, pv in (debts or {}).items():
        if not isinstance(pv, dict):
            continue
        debt = pv.get("debt")
        vv = "fail" if (debt is not None and debt > 1.0e-3) else "pass"
        records.append(_cell({
            "op": {"name": "predict.quantize", "spec": {"feature": "debt"},
                   "contract": "K-6.prediction.v1", "reference_cell": None},
            "subject": name_to_id.get(vname),
            "environment": {"code_snapshot": CODE_SNAPSHOT_IMPORT,
                            "compute_dtype": None, "export_outtype": None,
                            "corpus": None, "seqlen": None, "chunk_policy": None,
                            "contract_version": content_id({"K-6": "prediction-v1"})},
            "basis": {"claim": "K-6", "static_feature": "debt = J_var − J_base"},
            "result": {"status": "predicted", "verdict": vv,
                       "metrics": {"J_base": pv.get("J_base"), "J_var": pv.get("J_var"),
                                   "debt": debt}},
            "invalidates": None, "notes": [],
            "obligation": f"K-6.prediction.{vname}",
            "provenance": _provenance("debts_lattice.json")}))
        report.record("PREDICTIONS*/debts_lattice.json", "cell")

    # ---- calibration-reference confirmation (no new records) ----
    for rung, meta in (_load(w, "quant_ref.json", report) or {}).items():
        cid = gc_ids.get(rung)
        if cid is None:
            report.note("quant_ref.json", f"rung {rung}: UNAVAILABLE (base cell absent)")
            continue
        report.confirmed_refs.append({"source": "quant_ref.json", "rung": rung,
                                      "reference_cell": cid, "declared_tag": meta.get("tag")})
        report.note("quant_ref.json", f"{rung} → reference cell {cid} (base {meta.get('tag')})")
    if lora_cal_id is not None:
        report.confirmed_refs.append({"source": "ref_capture.json", "reference_cell": lora_cal_id})
        report.note("ref_capture.json", f"LoRA base reference → cell {lora_cal_id}")
    # ---- next-milestone evidence outside m1/work ----
    repo = w.parent.parent
    threshold_path = repo / "analysis/data/evidence/contracts/contract-3.json"
    if threshold_path.exists():
        try:
            tc = json.loads(threshold_path.read_text())
            q8 = (tc.get("flags") or {}).get("quant.q8_0") or {}
            fit_path = repo / "analysis/data/augmentation/fit_result.json"
            fit = json.loads(fit_path.read_text()) if fit_path.exists() else {}
            q4_fit = ((fit.get("flags") or {}).get("quant.q4_k_m") or {})
            records.append(_cell({
                "op": {"name": "calibrate.preflight.quant.q8_0",
                       "spec": {"feature": "q4_block_mse"},
                       "contract": f"threshold-v{tc.get('version')}", "reference_cell": None},
                "subject": name_to_id.get("base"),
                "environment": {"code_snapshot": CODE_SNAPSHOT_IMPORT,
                                "compute_dtype": None, "export_outtype": "bf16",
                                "corpus": None, "seqlen": None, "chunk_policy": None,
                                "contract_version": f"threshold-v{tc.get('version')}"},
                "lease": {"wall_s": None},
                "result": {"status": "measured", "verdict": "pass",
                           "metrics": {**{k: q8.get(k) for k in
                                           ("n", "threshold", "precision", "recall", "specificity", "f1")},
                                       "q4_n": q4_fit.get("n"), "q4_gate": q4_fit.get("gate")}},
                "invalidates": None, "notes": ["Q8 v3 emitted; Q4 refit gate refused"],
                "obligation": "K-6.threshold.q8_v3",
                "provenance": _provenance("analysis/data/evidence/contracts/contract-3.json")}))
            report.record("threshold contract v3", "cell")
        except (OSError, ValueError):
            report.malformed("threshold contract v3", "unparseable JSON")

    corpus_path = w.parent / "corpus_replication/results.json"
    if corpus_path.exists():
        try:
            cr = json.loads(corpus_path.read_text())
            records.append(_cell({
                "op": {"name": "replicate.quant.q4_k_m.corpus2",
                       "spec": {"corpus_sha256_16": (cr.get("corpus") or {}).get("sha256_16")},
                       "contract": "K-10-corpus2-v1", "reference_cell": None},
                "subject": name_to_id.get("prep_base_exact"),
                "environment": {"code_snapshot": CODE_SNAPSHOT_IMPORT, "llama_cpp": cr.get("llama_commit"),
                                "compute_dtype": "fp32", "export_outtype": "bf16",
                                "corpus": cr.get("corpus"), "seqlen": 512, "chunk_policy": "ngl=0",
                                "contract_version": "K-10-corpus2-v1"},
                "lease": cr.get("timings") or {},
                "result": {"status": "measured" if cr.get("status") == "OK" else "unavailable",
                           "verdict": _verdict(cr.get("verdict")),
                           "reason": cr.get("blocker"),
                           "metrics": {"q4_relative_delta_prepared_minus_base":
                                       cr.get("q4_relative_delta_prepared_minus_base"),
                                       "equivalence": cr.get("equivalence")}},
                "invalidates": None, "notes": [], "obligation": "K-10.replication.corpus2",
                "provenance": _provenance("m1/corpus_replication/results.json")}))
            report.record("corpus replication", "cell")
        except (OSError, ValueError):
            report.malformed("corpus replication", "unparseable JSON")

    history_path = repo / "m3/results.json"
    if history_path.exists():
        try:
            hr = json.loads(history_path.read_text())
            records.append(_cell({
                "op": {"name": "history.order_pair.q4_k_m", "spec": (hr.get("contract") or {}).get("history_pair"),
                       "contract": "K-8-attempt-v1", "reference_cell": None},
                "subject": name_to_id.get("base"),
                "environment": {"code_snapshot": CODE_SNAPSHOT_IMPORT, "compute_dtype": "bf16",
                                "export_outtype": "q4_k_m", "corpus": (hr.get("contract") or {}).get("corpus"),
                                "seqlen": 512, "chunk_policy": None, "contract_version": "K-8-attempt-v1"},
                "lease": {"wall_s": hr.get("duration_s")},
                "result": {"status": "unavailable", "verdict": None,
                           "reason": hr.get("blocker") or "present-match gate failed",
                           "metrics": hr.get("present_match_gate") or {}},
                "invalidates": None, "notes": [], "obligation": "K-8.attempt.qwen25_order_pair",
                "provenance": _provenance("m3/results.json")}))
            report.record("K-8 history attempt", "cell")
        except (OSError, ValueError):
            report.malformed("K-8 history attempt", "unparseable JSON")

    return records


def _adapt_reference(w):
    """Fallback LoRA base reference (ref_capture.json / probes/base_adapt.json)."""
    found = []
    for cand in ("ref_capture.json", "probes/base_adapt.json"):
        if not (w / cand).exists():
            continue
        try:
            data = json.loads((w / cand).read_text())
        except (OSError, ValueError):
            continue
        var = None
        if isinstance(data, dict):
            var = (data.get("results") or {}).get("variant")
            if var is None and data.get("capture") is not None:
                var = data
        if var:
            found.append((cand, {"capture": var.get("capture"),
                                 "protected_dppl": var.get("protected_dppl"),
                                 "seed": var.get("seed"),
                                 "rank": var.get("rank") or var.get("lora_rank"),
                                 "steps": var.get("steps"), "device": var.get("device"),
                                 "seq_len": var.get("seq_len"),
                                 "pass_contract": var.get("pass_contract"),
                                 "contract_version": var.get("contract_version")}))
    # Prefer a record that declares its contract. An unversioned cell is a legitimate last resort,
    # but using it as the yardstick for adapt-v2 verdicts is the I4 gap incident #19 exposed, so it
    # is labelled rather than adopted silently.
    for cand, norm in found:
        if norm.get("contract_version") == ADAPT_V2:
            norm["base_reference_status"] = "versioned"
            return norm
    if found:
        cand, norm = found[0]
        norm["base_reference_status"] = "unversioned-fallback"
        norm["base_reference_file"] = cand
        return norm
    return None


def _load_old_f16(w):
    p = w / "M1_OPS.old-f16.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# CLAIMS + INCIDENTS
# ---------------------------------------------------------------------------

def build_claims(report) -> list:
    from .claims import CLAIM_SEEDS
    records = []
    for key, text, declared, blocked, partial, refuter, numbers in CLAIM_SEEDS:
        body = {
            "kind": "claim", "key": key, "text": text, "state": declared,
            "blocked": blocked, "partial": partial,
            "state_history": [{"at": IMPORT_EPOCH, "state": declared.upper(), "cells": []}],
            "obligations": {},
            "refuter": refuter,
            "numbers": [{"value": v, "obligation": ob} for v, ob in numbers],
            "provenance": _provenance("CLAIMS.md + PLAN.md §1"),
        }
        records.append(body)
        report.record("CLAIMS.md/PLAN.md §1", "claim")
    return records


def build_incidents(w, report) -> list:
    p = w / "PIPELINE_FAILURES.md"
    if not p.exists():
        p = w.parent / "PIPELINE_FAILURES.md"   # lives beside m1/work/, not inside it
    if not p.exists():
        report.note("PIPELINE_FAILURES.md", "absent — 0 incidents imported")
        return []
    text = p.read_text()
    rows = re.findall(r"^\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|",
                      text, flags=re.MULTILINE)
    records = []
    seen = set()
    for num_raw, brk, sym, caught, rule in rows:
        num = int(num_raw)
        if num in seen:
            continue
        seen.add(num)
        records.append({
            "kind": "incident", "key": f"PF-{num}", "at": IMPORT_EPOCH,
            "severity": PF_SEVERITY.get(num, "operational"),
            "what": brk.strip(),
            "caught_by": caught.strip(),
            "rule_now": f"{PF_INVARIANT.get(num, 'I1')}: {rule.strip()}",
            "prevents": "the invariant in SYSTEM.md that now prevents recurrence",
            "provenance": _provenance("PIPELINE_FAILURES.md"),
        })
        report.record("PIPELINE_FAILURES.md", "incident")
    report.note("PIPELINE_FAILURES.md", f"{len(records)} incidents parsed from the table")
    return records


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def gather(work: str, report: ImportReport):
    """Build every record from `m1/work` WITHOUT writing. Returns (records_by_kind, name_to_id)."""
    w = Path(work)
    variants = _load(w, "VARIANTS.json", report) or {}
    eq = {}
    if (w / "equiv").is_dir():
        for f in sorted((w / "equiv").glob("*.json")):
            e = _load(w, f"equiv/{f.name}", report)
            if e is not None and isinstance(e, dict):
                eq[f.name[:-5]] = e
    ops = {}
    if (w / "ops").is_dir():
        for f in sorted((w / "ops").glob("*.json")):
            o = _load(w, f"ops/{f.name}", report)
            if o is not None and isinstance(o, dict):
                ops[f.name] = o

    # The live panel recycles per-cell files (`ops/*.json`) as it progresses; M1_OPS.json is the
    # aggregate it regenerates, so use it to recover (variant, op) cells whose files are gone.
    agg = _load(w, "M1_OPS.json", report)
    if isinstance(agg, dict):
        m1ops_n = 0
        for vname, entry in agg.items():
            if not isinstance(entry, dict):
                continue
            for op, blob in (entry.get("ops") or {}).items():
                fname = f"{vname}.{op}.json"
                if fname in ops:
                    continue
                if isinstance(blob, str):
                    ops[fname] = {"_m1ops_status": blob}   # "error" | "empty" | other
                    m1ops_n += 1
                    continue
                if not isinstance(blob, dict) or not (blob.get("results") or blob.get("error")):
                    continue
                merged = dict(blob)
                merged["_source"] = "M1_OPS.json"
                if op == "gguf" and (blob.get("export") or {}).get("outtype") is None:
                    merged["_missing_export"] = True
                ops[fname] = merged
                m1ops_n += 1
        if m1ops_n:
            report.note("M1_OPS.json", f"recovered {m1ops_n} recycled cell(s) from the aggregate")

    debts = _load(w, "debts_lattice.json", report) or {}
    pred_new = _load(w, "PREDICTIONS_new.json", report) or {}
    pred = _load(w, "PREDICTIONS.json", report) or {}
    cd_check = _load(w, "compute_dtype_check.json", report) or {}
    export_damage = _load(w, "export_damage.json", report) or {}

    artifacts, name_to_id = build_artifacts(w, variants, eq, debts, pred_new,
                                            export_damage, report)
    cells = build_cells(w, name_to_id, ops, eq, cd_check, export_damage,
                        pred, pred_new, debts, report)
    claims = build_claims(report)
    incidents = build_incidents(w, report)
    return {"artifact": artifacts, "cell": cells, "claim": claims,
            "incident": incidents}, name_to_id


def run(store, work: str, *, dry_run: bool = False, verify: bool = False) -> ImportReport:
    """Build from m1/work, validate every admission, then write append-only records."""
    report = ImportReport()
    records, _ = gather(work, report)
    if dry_run:
        for kind, rs in records.items():
            report.total[kind] = len(rs)
        return report
    for kind in KINDS:
        for body in records[kind]:
            if kind == "artifact":
                problems = rules.check_artifact(body, ledger=store)
            elif kind == "cell":
                problems = rules.check_cell(body, ledger=store)
            else:
                problems = []
            if problems:
                raise RuntimeError(f"import admission refused {kind} "
                                   f"{body.get('obligation') or body.get('key') or body.get('id')}: "
                                   f"{'; '.join(problems)}")
            store.add(kind, body)
    if verify:
        problems = store.verify()
        if problems:
            raise RuntimeError(f"ledger --verify failed with {len(problems)} problem(s): "
                               f"{problems[:5]}")
    return report
