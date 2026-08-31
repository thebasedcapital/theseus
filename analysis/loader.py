# loader.py — read-only ingestion for the analysis pipeline. Reads (i) Inspector schema v1 JSON
# scans, (ii) harvest manifest.jsonl + edges.jsonl if present, (iii) ledger cell files if present,
# (iv) an explicit labels.jsonl. Missing inputs print `no data yet: <path>` and yield empty frames
# — never an exception, never invented rows (I8). Frames are plain lists of dicts: lossless with
# None = no evidence. Owned by BaseRates.

import json
import re
import sys
from pathlib import Path

FEATURE_KEYS = ("q4_block_mse", "q4_block_mse_pooled", "dyn_range_log10",
                "row_energy_imbalance", "amax_over_rms", "frac_below_f16_normal")
AUGMENT_CORPUS = {"ppl_bytes": 32768, "kl_bytes": 8192}
AUGMENT_PASS_CONTRACT = {"mode": "reference-relative", "rel_dppl_slack": 0.010,
                         "kl_mean_slack": 0.005}



def note_missing(path):
    print(f"no data yet: {path}")


class Inputs(object):
    """Locations the loader looks at. Every consumer supplies an Inputs; callers who own a
    directory (e.g. the fixtures or the evidence freeze) just point root at it."""

    def __init__(self, root="analysis/data", scans=None, manifest=None, edges=None,
                 ledger=None, labels=None, augmentation=None):
        self.root = Path(root)
        # dir of Inspector v1 JSONs (one file per artifact) and/or a inline scans.jsonl
        self.scans = Path(scans) if scans else self.root / "scans"
        # harvest slice outputs
        self.manifest = Path(manifest) if manifest else self.root / "harvest" / "manifest.jsonl"
        self.edges = Path(edges) if edges else self.root / "harvest" / "edges.jsonl"
        # ledger cell files (kind: cell records)
        self.ledger = Path(ledger) if ledger else self.root / "ledger"
        # explicit labelled rows (measurements already joined to features)
        self.labels = Path(labels) if labels else self.root / "labels.jsonl"
        # evidence freezes may use a sibling augmentation directory under data/
        default_aug = self.root / "augmentation"
        if not default_aug.is_dir() and self.root.name == "evidence":
            default_aug = self.root.parent / "augmentation"
        self.augmentation = Path(augmentation) if augmentation else default_aug


def _augmentation_probe_rows(path):
    """Load one real CPU gguf_probe JSON into measured quant labels."""
    doc = _read_json(path)
    if not isinstance(doc, dict) or doc.get("status") != "OK":
        return [], None
    backend = doc.get("backend") or {}
    if backend.get("device") != "cpu" or str(backend.get("ngl")) != "0":
        return [], None
    corpus = doc.get("corpus") or {}
    if any(corpus.get(k) != v for k, v in AUGMENT_CORPUS.items()):
        return [], None
    if corpus.get("source") and not str(corpus["source"]).endswith("eval_wikitext.txt"):
        return [], None
    contract = doc.get("pass_contract") or {}
    if any(contract.get(k) != v for k, v in AUGMENT_PASS_CONTRACT.items()):
        return [], None
    versions = (doc.get("versions") or {}).get("llama_cpp") or []
    if not any("9851" in str(v) for v in versions):
        return [], None
    tag = doc.get("tag")
    if not tag or (doc.get("export") or {}).get("outtype") != "bf16":
        return [], None
    ref = doc.get("quant_ref") or {}
    if tag == "base" and not ref:
        ref = {q: {"rel_dppl": ((doc.get("results") or {}).get(q) or {}).get("rel_dppl")}
               for q in ("q8_0", "q4_k_m")}
    labels = []
    for quant, flag in (("q8_0", "quant.q8_0"), ("q4_k_m", "quant.q4_k_m")):
        ent = (doc.get("results") or {}).get(quant) or {}
        damage = ent.get("rel_dppl")
        ref_ent = ref.get(quant) or {}
        if ent.get("status") != "OK" or damage is None or ref_ent.get("rel_dppl") is None:
            continue
        outcome = "pass" if tag == "base" else ("pass" if ent.get("pass") is not False else "fail")
        labels.append({"artifact": tag, "flag": flag, "outcome": outcome,
                       "status": "measured", "operation": quant,
                       "damage": damage, "damage_ref": ref_ent["rel_dppl"],
                       "damage_unit": "rel_dppl_frac", "environment": {
                           "backend": "cpu", "ngl": "0", "git_head": doc.get("git_head"),
                           "llama_cpp": (doc.get("versions") or {}).get("llama_cpp"),
                           "corpus": doc.get("corpus"), "export": doc.get("export")},
                       "src": str(path)})
    return labels, {"artifact": tag, "arch": None, "verdicts": [], "preflight": [],
                    "skipped": [], "source": str(path)}


def artifact_id_from_scan(doc, filename=None):
    """Best-effort canonical artifact id for a scan JSON: an explicit 'id', else the scan file's
    name stripped of .json. Preferring the file name keeps scan caches joinable to lineage/ledger
    ids, which are short slugs; document the mapping where the path differs."""
    if doc and doc.get("id"):
        return doc["id"]
    if filename:
        stem = Path(filename).stem
        stem = re.sub(r"\.jsonl$", "", stem)
        if stem and stem != "scans":
            return stem
    path = (doc or {}).get("path") or ""
    base = re.sub(r"\.(safetensors|bin|gguf|json)$", "", str(Path(path).name))
    if base in ("model", "pytorch_model"):
        return str(Path(path).parent.name)
    return base or "artifact"


def _read_json(path):
    if not Path(path).exists():
        note_missing(path)
        return None
    try:
        return json.loads(Path(path).read_text())
    except json.JSONDecodeError:
        print(f"skip malformed json: {path}")
        return None


# ---- scans ---------------------------------------------------------------------------------

def _scan_json_rows(path):
    """One inspector v1 scan file -> (family rows, total row, scan ctx). None if unreadable."""
    doc = _read_json(path)
    if doc is None or not isinstance(doc, dict) or "families" not in doc:
        return None
    artid = artifact_id_from_scan(doc, Path(path).name)
    ctx = {"artifact": artid, "arch": doc.get("arch"),
           "verdicts": doc.get("verdicts") or [], "preflight": doc.get("preflight") or [],
           "skipped": doc.get("skipped") or [], "source": str(path)}
    fam_rows, total_row = [], None
    for fam, feats in (doc.get("families") or {}).items():
        row = {"artifact": artid, "level": "family", "family": fam, "arch": ctx["arch"],
               "source": str(path)}
        for k in FEATURE_KEYS:
            row[k] = feats.get(k) if isinstance(feats, dict) else None
        fam_rows.append(row)
    tot = doc.get("total") or {}
    if tot:
        total_row = {"artifact": artid, "level": "total", "family": "TOTAL",
                     "arch": ctx["arch"], "source": str(path)}
        for k in FEATURE_KEYS:
            total_row[k] = tot.get(k)
    return fam_rows, total_row, ctx


def _scan_line_rows(line_doc):
    """A flat scan record (evidence/fixtures scans.jsonl) -> family row or total row."""
    row = dict(line_doc)
    row.setdefault("level", "family")
    for k in FEATURE_KEYS:
        row.setdefault(k, None)
    row.setdefault("artifact", row.get("id"))
    return row


def _scan_json_rows_from_doc(doc, source):
    artid = artifact_id_from_scan(doc, Path(source).name)
    ctx = {"artifact": artid, "arch": doc.get("arch"),
           "verdicts": doc.get("verdicts") or [], "preflight": doc.get("preflight") or [],
           "skipped": doc.get("skipped") or [], "source": source}
    fam_rows, total_row = [], None
    for fam, feats in (doc.get("families") or {}).items():
        row = {"artifact": artid, "level": "family", "family": fam, "arch": ctx["arch"],
               "source": source}
        for k in FEATURE_KEYS:
            row[k] = feats.get(k) if isinstance(feats, dict) else None
        fam_rows.append(row)
    tot = doc.get("total") or {}
    if tot:
        total_row = {"artifact": artid, "level": "total", "family": "TOTAL",
                     "arch": ctx["arch"], "source": source}
        for k in FEATURE_KEYS:
            total_row[k] = tot.get(k)
    return fam_rows, total_row, ctx


def load_scans(inputs):
    """Returns dict(family=[...], total=[...], context={id: ctx}). Empty on absence."""
    out = {"family": [], "total": [], "context": {}}
    scan_dir = Path(inputs.scans)
    found = False
    if scan_dir.is_dir():
        for p in sorted(scan_dir.glob("*.json")):
            doc = _read_json(p)
            if not isinstance(doc, dict) or "families" not in doc:
                continue
            res = _scan_json_rows_from_doc(doc, str(p))
            fam_rows, total_row, ctx = res
            found = True
            out["family"].extend(fam_rows)
            if total_row is not None:
                out["total"].append(total_row)
            out["context"][ctx["artifact"]] = ctx
    if not found:
        candidates = []
        if str(scan_dir).endswith(".jsonl"):
            candidates.append(scan_dir)
        candidates.append(Path(inputs.root) / "scans.jsonl")
        for jl in dict.fromkeys(candidates):
            if not jl.exists():
                continue
            found = True
            with open(jl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = _scan_line_rows(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"skip malformed scan line in {jl}")
                        continue
                    (out["total"] if row.get("level") == "total" else out["family"]).append(row)
            break
    if not found:
        note_missing(scan_dir)
    ad = Path(inputs.augmentation)
    if ad.is_dir():
        known = set(out["context"])
        for p in sorted(ad.glob("*.scan.json")):
            doc = _read_json(p)
            if not isinstance(doc, dict) or "families" not in doc:
                continue
            doc = dict(doc)
            doc.setdefault("id", p.name.removesuffix(".scan.json"))
            fam_rows, total_row, ctx = _scan_json_rows_from_doc(doc, str(p))
            if ctx["artifact"] in known:
                continue
            known.add(ctx["artifact"])
            out["family"].extend(fam_rows)
            if total_row is not None:
                out["total"].append(total_row)
            out["context"][ctx["artifact"]] = ctx
    return out


# ---- harvest + lineage ----------------------------------------------------------------------

def load_harvest(inputs):
    """manifest + edges. Edges absent -> synthesized from declared_base (the harvest contract's
    own lineage source). Returns dict of frames."""
    frames = {"manifest": [], "edges": []}
    mp = Path(inputs.manifest)
    if mp.exists():
        with open(mp) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        frames["manifest"].append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"skip malformed manifest line in {mp}")
    else:
        note_missing(mp)
    ep = Path(inputs.edges)
    if ep.exists():
        with open(ep) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        frames["edges"].append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"skip malformed edge line in {ep}")
    else:
        note_missing(ep)
    if not frames["edges"]:
        for m in frames["manifest"]:
            bases = m.get("declared_base") or []
            for b in bases:
                frames["edges"].append({"parent": b, "child": m.get("id"),
                                        "op": "declared." + (m.get("kind") or "unknown"),
                                        "source": "manifest.declared_base"})
    return frames


def _dig(obj, *keys):
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


# ---- augmentation probe import --------------------------------------------------------------

def load_augmentation(inputs):
    """Load durable CPU probe JSONs into measured quant labels."""
    labels, contexts = [], {}
    ad = Path(inputs.augmentation)
    if not ad.is_dir():
        return labels, contexts
    for p in sorted(ad.glob("*.gguf.json")):
        probe_labels, ctx = _augmentation_probe_rows(p)
        labels.extend(probe_labels)
        if ctx:
            contexts[ctx["artifact"]] = ctx
    return labels, contexts


# ---- ledger cells ---------------------------------------------------------------------------

def load_cells(inputs):
    """Ledger kind=cell records flattened to rows: subject, op, status, verdict, metrics,
    reference_cell. Missing ledger dir -> empty frame + no-data line."""
    rows = []
    ld = Path(inputs.ledger)
    if not ld.is_dir():
        note_missing(ld)
        return rows
    cells = sorted(ld.glob("*.json"))
    for p in cells:
        if p.name.startswith("index"):
            continue
        doc = _read_json(p)
        if doc is None or not isinstance(doc, dict):
            continue
        if doc.get("kind") not in (None, "cell"):
            continue
        op = doc.get("op") or {}
        result = doc.get("result") or {}
        rows.append({"id": doc.get("id") or p.stem, "subject": doc.get("subject"),
                     "op": op.get("name"), "spec": op.get("spec"),
                     "reference_cell": op.get("reference_cell"),
                     "status": result.get("status"), "verdict": result.get("verdict"),
                     "metrics": result.get("metrics") or {}, "source": str(p)})
    if not cells:
        note_missing(ld)
    return rows


# ---- explicit labels ------------------------------------------------------------------------

def load_labels(inputs):
    """Explicit labelled rows; absent file -> empty list + no-data line."""
    rows = []
    lp = Path(inputs.labels)
    if not lp.exists():
        note_missing(lp)
        return rows
    with open(lp) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"skip malformed label line in {lp}")
    return rows


def load_all(inputs):
    """Returns dict with scans, harvest, cells, labels frames."""
    aug_labels, _ = load_augmentation(inputs)
    labels = load_labels(inputs)
    existing = {(r.get("artifact"), r.get("flag")) for r in labels}
    labels.extend(r for r in aug_labels if (r.get("artifact"), r.get("flag")) not in existing)
    return {"scans": load_scans(inputs), "harvest": load_harvest(inputs),
            "cells": load_cells(inputs), "labels": labels}

def summarize(frames, stream=sys.stdout):
    s = frames["scans"]
    h = frames["harvest"]
    print(f"scans: {len(s['family'])} family rows, {len(s['total'])} total rows, "
          f"{len(s['context'])} artifacts; harvest: {len(h['manifest'])} manifest, "
          f"{len(h['edges'])} edges; cells: {len(frames['cells'])}; "
          f"labels: {len(frames['labels'])}", file=stream)
