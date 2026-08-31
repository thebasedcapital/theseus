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


def note_missing(path):
    print(f"no data yet: {path}")


class Inputs(object):
    """Locations the loader looks at. Every consumer supplies an Inputs; callers who own a
    directory (e.g. the fixtures or the evidence freeze) just point root at it."""

    def __init__(self, root="analysis/data", scans=None, manifest=None, edges=None,
                 ledger=None, labels=None):
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


def load_scans(inputs):
    """Returns dict(family=[...], total=[...], context={id: ctx}). Empty on absence."""
    out = {"family": [], "total": [], "context": {}}
    scan_dir = Path(inputs.scans)
    found = False
    if scan_dir.is_dir():
        for p in sorted(scan_dir.glob("*.json")):
            res = _scan_json_rows(p)
            if res is None:
                continue
            fam_rows, total_row, ctx = res
            found = True
            out["family"].extend(fam_rows)
            if total_row is not None:
                out["total"].append(total_row)
            key = ctx.get("artifact")
            if key:
                out["context"][key] = ctx
    if not found:
        # flat scans.jsonl (evidence-style / future scan-slice cache), explicit or under root
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
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"skip malformed scan line in {jl}")
                        continue
                    row = _scan_line_rows(doc)
                    (out["total"] if row.get("level") == "total" else out["family"]).append(row)
            break
    if not found:
        note_missing(scan_dir)
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
        rows.append({
            "id": doc.get("id") or p.stem,
            "subject": doc.get("subject"),
            "op": op.get("name"),
            "spec": op.get("spec"),
            "reference_cell": op.get("reference_cell"),
            "status": result.get("status"),
            "verdict": result.get("verdict"),
            "metrics": result.get("metrics") or {},
            "source": str(p),
        })
    if not cells:
        note_missing(ld)
    return rows


# ---- explicit labels ------------------------------------------------------------------------

def load_labels(inputs):
    """Explicit labelled rows: {artifact, flag, outcome pass|fail|unavailable, status,
    damage, damage_ref, ...}. Absent file -> empty list + no-data line."""
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
    return {
        "scans": load_scans(inputs),
        "harvest": load_harvest(inputs),
        "cells": load_cells(inputs),
        "labels": load_labels(inputs),
    }


def summarize(frames, stream=sys.stdout):
    s = frames["scans"]
    h = frames["harvest"]
    print(f"scans: {len(s['family'])} family rows, {len(s['total'])} total rows, "
          f"{len(s['context'])} artifacts; harvest: {len(h['manifest'])} manifest, "
          f"{len(h['edges'])} edges; cells: {len(frames['cells'])}; "
          f"labels: {len(frames['labels'])}", file=stream)
