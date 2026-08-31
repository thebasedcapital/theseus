# thresholds.py — choose each risk flag's threshold from measured, labelled rows by a STATED
# objective; print precision/recall for >=5 candidate cuts; and when the evidence passes the
# honesty gates, EMIT a NEW contract version plus the set of prior verdicts it invalidates
# (PLAN.md section 5: no learned predictor before the ledger has >=20 labelled cells; K-6's
# refuter is the gate). History is never rewritten: the new threshold is a new file under
# analysis/data/contracts/, never an edit of the past. Owned by BaseRates.
#
# Objective (stated): for a preflight tool a FALSE NEGATIVE is the only direction that hurts
# (K-6 refuter), so we choose the *largest* cut (fewest false alarms) whose recall >= 0.95;
# ties broken by higher precision. Emission additionally requires the chosen cut to beat the
# flag's own fail base rate on precision (the flag adds information), otherwise `contract
# unchanged` — a silently meaningless threshold is worse than none.

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from risk_flags import (FLAG_DEFS, CONTRACT_VERSION, flag_ranks, artifact_fires_eval)
from loader import Inputs, load_all
from baserates import (family_rows_by_artifact, total_rows_by_artifact, _confusion,
                       all_labels, measured_labels)

RECALL_FLOOR = 0.95          # stated: catch >=95% of true fails
MIN_LABELLED = 20            # PLAN section 5 / K-6 refuter: no predictor before >=20 cells
BEAT_RATE_MULTIPLE = 1.25    # stated: a cut must concentrate fails >=1.25x the flag's fail base
                             # rate on precision, else the flag is a rumor, not a predictor
CONTRACT_RE = re.compile(r"contract-(\d+)\.json")


# ---- feature and verdict evaluation under a threshold ----------------------------------------

def artifact_feature(flag, fam_rows, total_row):
    """The scalar a flag's threshold cuts on: worst family value of the primary feature,
    falling back to the total row when no family carries the feature (proxy, marked as such)."""
    d = FLAG_DEFS[flag]
    feat = d["feature"]
    vals = [r.get(feat) for r in fam_rows if r.get(feat) is not None]
    if vals:
        return max(vals), "worst_family"
    if total_row is not None and total_row.get(feat) is not None:
        return total_row[feat], "total_proxy"
    return None, None


def fires_under(flag, spec, fam_rows, total_row):
    """artifact-level firing under an arbitrary spec (one specialty: 'threshold' replaced by the
    candidate cut). Mirrors risk_flags.artifact_fires but uses the passed spec."""
    any_fire = None
    for r in fam_rows:
        v = r.get(spec["feature"])
        if v is None:
            continue
        over = v > spec["threshold"]
        if over and spec.get("secondary"):
            s = r.get(spec["secondary"])
            over = over or (s is not None and s > spec.get("secondary_threshold", 0.0))
        if over:
            any_fire = True
        elif any_fire is None:
            any_fire = False
    if any_fire is False and spec.get("total_threshold") is not None and total_row is not None:
        t = total_row.get(spec["feature"])
        if t is not None and t > spec["total_threshold"]:
            any_fire = True
    return any_fire


# ---- candidate enumeration -------------------------------------------------------------------

def candidates(values, n_min=5):
    """Sorted unique values plus interpolated cuts; always at least n_min distinct cuts."""
    vals = sorted({float(v) for v in values if v is not None})
    if not vals:
        return []
    if len(vals) < n_min:
        lo, hi = vals[0], vals[-1]
        if hi > lo:
            extra = [lo + (hi - lo) * i / float(n_min) for i in range(1, n_min)]
            vals = sorted({v for v in vals + extra if lo <= v <= hi})
    while len(vals) < n_min:
        gap_at = max((vals[i + 1] - vals[i], i) for i in range(len(vals) - 1))
        gap, i = gap_at
        if gap <= 0:
            break
        vals.insert(i + 1, (vals[i] + vals[i + 1]) / 2.0)
    cuts = list(vals)
    # a floor cut that fires on everything (conservative reference point for the candidate table)
    cuts.append(vals[0] - max(1.0, abs(vals[0])) * 1e-3)
    return sorted(set(cuts))


# ---- per-flag fit -----------------------------------------------------------------------------

def fit_flag(flag, labels, fam_rows, total_rows, table=False):
    """Returns (chosen_cut_dict|None, gate_reason, n_rows[, candidate_table])."""
    d = FLAG_DEFS[flag]
    total_by = total_rows_by_artifact(total_rows)
    fam_by = family_rows_by_artifact(fam_rows)
    rows = []
    for r in measured_labels([x for x in labels if x.get("flag") == flag]):
        art = r.get("artifact")
        feat, how = artifact_feature(flag, fam_by.get(art) or [], total_by.get(art))
        if feat is None:
            continue
        rows.append({"artifact": art, "feature": feat, "proxy": how,
                     "outcome": r["outcome"] == "fail"})
    if not rows:
        return None, "unavailable: no labelled rows", 0, ([] if table else None)
    n_fail = sum(1 for r in rows if r["outcome"])

    tbl = []
    for c in candidates([r["feature"] for r in rows]):
        m = _confusion([r["feature"] > c for r in rows], [r["outcome"] for r in rows])
        prec = m["tp"] / float(m["tp"] + m["fp"]) if m["tp"] + m["fp"] else None
        rec, spec = m["sens"], m["spec"]
        f1 = None
        if prec is not None and rec is not None and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        tbl.append({"c": c, **m, "precision": prec, "recall": rec,
                    "specificity": spec, "f1": f1})

    eligible = [t for t in tbl if t["recall"] is not None and t["recall"] >= RECALL_FLOOR]
    if eligible:
        chosen = max(eligible, key=lambda t: (t["c"], t["precision"] or 0.0))
    else:
        chosen = max(tbl, key=lambda t: ((t["recall"] or 0.0), (t["precision"] or 0.0)))
        if (chosen["recall"] or 0.0) < RECALL_FLOOR:
            return None, (f"recall floor {RECALL_FLOOR:g} unreachable "
                          f"(best {chosen['recall']:.3f})"), len(rows), (tbl if table else None)

    n = len(rows)
    fail_prev = n_fail / float(n)
    bar = BEAT_RATE_MULTIPLE * fail_prev
    if n < MIN_LABELLED:
        return None, f"n={n} < {MIN_LABELLED} labelled cells", n, (tbl if table else None)
    if chosen["precision"] is None or chosen["precision"] < bar:
        a = chosen["precision"]
        return None, (f"not informative: precision {a:.3f} < {BEAT_RATE_MULTIPLE:g}x fail "
                      f"base rate {fail_prev:.3f} ({bar:.3f})"), n, (tbl if table else None)
    return chosen, "emitted", n, (tbl if table else None)


def next_version(contracts_dir):
    cur = CONTRACT_VERSION
    if Path(contracts_dir).is_dir():
        for p in Path(contracts_dir).glob("contract-*.json"):
            m = CONTRACT_RE.match(p.name)
            if m:
                cur = max(cur, int(m.group(1)))
    return cur + 1


# ---- invalidation set -------------------------------------------------------------------------

def _stored_verdict(flag, scan_doc):
    for op, verdict, _reason in scan_doc.get("preflight") or []:
        if op == FLAG_DEFS[flag]["op"]:
            return "OK" if verdict == "OK" else (
                "AT_RISK" if verdict in ("AT_RISK",) else None)
    return None


def invalidation_set(flag, chosen_c, fam_rows, total_rows, scan_ctx):
    """Prior verdicts under the active contract that flip under the new threshold. Nothing is
    rewritten: this list ships inside the new contract file."""
    total_by = total_rows_by_artifact(total_rows)
    fam_by = family_rows_by_artifact(fam_rows)
    spec = dict(FLAG_DEFS[flag])
    spec["threshold"] = chosen_c
    out = []
    for art, fr in fam_by.items():
        tr = total_by.get(art)
        newv = fires_under(flag, spec, fr, tr)
        new_s = "AT_RISK" if newv else "OK" if newv is False else None
        if new_s is None:
            continue
        scan_doc = (scan_ctx or {}).get(art, {})
        old_s = _stored_verdict(flag, scan_doc)
        if old_s is None:
            oldv = artifact_fires_eval(flag, fr, tr)
            old_s = "AT_RISK" if oldv else "OK" if oldv is False else None
            src = "recomputed"
        else:
            src = "scantime"
        if old_s is None:
            continue
        if old_s != new_s:
            out.append({"artifact": art, "flag": flag, "old_verdict": old_s,
                        "new_verdict": new_s, "source": src})
    return out


# ---- main -------------------------------------------------------------------------------------

def compute(inputs):
    frames = load_all(inputs)
    labels = all_labels(frames)
    fam_rows, total_rows = frames["scans"]["family"], frames["scans"]["total"]
    ctx = frames["scans"]["context"]
    contracts_dir = Path(inputs.root) / "contracts"
    version = next_version(contracts_dir)
    flags_out, any_emitted = {}, False
    for flag in flag_ranks():
        chosen, why, n, tbl = fit_flag(flag, labels, fam_rows, total_rows, table=True)
        flags_out[flag] = {"n": n, "gate": why,
                           "candidates": [{"c": t["c"], "tp": t["tp"], "fp": t["fp"],
                                           "tn": t["tn"], "fn": t["fn"],
                                           "precision": t["precision"], "recall": t["recall"],
                                           "specificity": t["specificity"], "f1": t["f1"]}
                                          for t in (tbl or [])]}
        if chosen is None:
            continue
        any_emitted = True
        flags_out[flag].update({
            "emitted": True,
            "threshold": chosen["c"],
            "precision": chosen["precision"],
            "recall": chosen["recall"],
            "specificity": chosen["specificity"],
            "f1": chosen["f1"],
            "invalidates": invalidation_set(flag, chosen["c"], fam_rows, total_rows, ctx),
        })
    res = {"version": version, "emitted": any_emitted, "flags": flags_out}
    return res, contracts_dir

def _latest_contract(contracts_dir):
    found = []
    if Path(contracts_dir).is_dir():
        for p in Path(contracts_dir).glob("contract-*.json"):
            m = CONTRACT_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
    if not found:
        return None, None
    _, path = max(found)
    try:
        return path, json.loads(path.read_text())
    except (OSError, ValueError):
        return path, None

def _flag_core(flags):
    keep = ("n", "threshold", "precision", "recall", "specificity", "f1")
    return {name: {k: spec.get(k) for k in keep} for name, spec in (flags or {}).items()}



def emit(res, contracts_dir):
    if not res["emitted"]:
        return None
    Path(contracts_dir).mkdir(parents=True, exist_ok=True)
    flags_clean = {}
    for flag, d in res["flags"].items():
        if d.get("emitted"):
            flags_clean[flag] = {k: v for k, v in d.items() if k != "candidates"}
    invalidates = [inv for f in res["flags"].values() for inv in f.get("invalidates", [])]
    latest_path, latest = _latest_contract(contracts_dir)
    if latest and _flag_core(latest.get("flags")) == _flag_core(flags_clean):
        res["version"] = latest["version"]
        res["reused"] = True
        return latest_path
    doc = {
        "kind": "thresholds.contract",
        "version": res["version"],
        "prev_version": res["version"] - 1,
        "emitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gate": {"min_labelled": MIN_LABELLED, "recall_floor": RECALL_FLOOR,
                 "beat_rate_multiple": BEAT_RATE_MULTIPLE,
                 "note": "PLAN section 5 / K-6 refuter: no predictor learned before the ledger "
                         "has >=20 labelled cells; a cut must catch >=95% of true fails and "
                         f"reach precision >= {BEAT_RATE_MULTIPLE:g}x the flag's fail base rate "
                         "or it is not emitted (else the flag is a rumor, not a predictor)."},
        "flags": flags_clean,
        "invalidates": invalidates,
        "invariants": [
            "history is never rewritten: a new version is a new file; prior verdicts it "
            "invalidates are listed, not edited",
            "feature conventions never change within a version: q4_block_mse is the mean of "
            "per-tensor ratios and q4_block_mse_pooled is the ratio of sums — they differ by up "
            "to 5.7% and must never be mixed in one threshold",
        ],
    }
    path = Path(contracts_dir) / f"contract-{res['version']}.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path

def format_report(res, contracts_path):
    lines = []
    lines.append((f"active contract: v{res['version']} (identical evidence reused)"
                  if res.get("reused") else f"contract next version: v{res['version']}"))
    for flag in flag_ranks():
        d = res["flags"][flag]
        lines.append("")
        lines.append(f"== {flag} (feature {FLAG_DEFS[flag]['feature']}) ==")
        if d["gate"] == "unavailable: no labelled rows":
            lines.append("  no labelled rows -> unavailable")
            continue
        lines.append(f"  n={d['n']} labelled; gate: {d['gate']}")
        if d["gate"].startswith("not informative") or d["gate"].startswith("recall floor"):
            lines.append("  (candidate table printed below; nothing emitted because the gate "
                         "refused a useless cut)")
        for t in d.get("candidates") or []:
            lines.append(f"    c={t['c']:.5g}  TP={t['tp']} FP={t['fp']} TN={t['tn']} "
                         f"FN={t['fn']}  prec={(t['precision'] if t['precision'] is not None else -1):.3f} "
                         f"rec={(t['recall'] if t['recall'] is not None else -1):.3f} "
                         f"spec={(t['specificity'] if t['specificity'] is not None else -1):.3f} "
                         f"f1={(t['f1'] if t['f1'] is not None else -1):.3f}")
        if not d.get("emitted"):
            continue
        lines.append(f"  CHOSEN c={d['threshold']:.5g} "
                     f"(prec {d['precision']:.3f}, rec {d['recall']:.3f}, "
                     f"spec {d['specificity']:.3f})")
        inv = d.get("invalidates", [])
        lines.append(f"  prior verdicts invalidated by v{res['version']}: {len(inv)}")
        for i in inv[:40]:
            lines.append(f"    {i['artifact']}: {i['old_verdict']} -> {i['new_verdict']} "
                         f"({i['source']})")
    if res["emitted"]:
        action = "reused" if res.get("reused") else "written"
        lines.append(f"\ncontract v{res['version']} {action} at {contracts_path}")
    else:
        lines.append("\ncontract unchanged: no flag met the emission gate; "
                     "no new contract version was written.")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="fit risk-flag thresholds and (when gated) emit a "
                                            "new contract version with its invalidation set")
    ap.add_argument("--root", default="analysis/data")
    ap.add_argument("--scans", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--edges", default=None)
    ap.add_argument("--ledger", default=None)
    args = ap.parse_args(argv)

    inputs = Inputs(root=args.root, scans=args.scans, labels=args.labels,
                    manifest=args.manifest, edges=args.edges, ledger=args.ledger)
    res, contracts_dir = compute(inputs)
    path = emit(res, contracts_dir)
    print("\n".join(format_report(res, path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
