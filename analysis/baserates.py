# baserates.py — turn scans + measured outcomes into PER-RISK-FLAG, PER-FAMILY BASE RATES with
#   * prevalence    P(static flag fires) over the scanned population, per family, per
#                   architecture and per artifact;
#   * confusion     predicted flag vs measured verdict (sensitivity/specificity + intervals)
#                   where measured outcomes exist;
#   * catastrophic  P(damage >= 100 x the operation's reference measurement) — the stated
#                   multiple lives in risk_flags.CATASTROPHE_MULTIPLE.
# With zero measured outcomes the correct output is prevalence + `outcomes: unavailable (reason)`
# — UNAVAILABLE is a first-class value and null never means False (I8). Wilson is hand-implemented
# (stdlib + numpy only) and cross-checked against its closed-form k=0/n and k=n/n endpoints.
# Owned by BaseRates.

import argparse
import json
import sys
from math import sqrt

from risk_flags import (FLAG_DEFS, CONTRACT_VERSION, FEATURE_KEYS, FAMILIES, flag_ranks,
                        family_fires, is_catastrophic, artifact_fires_eval, CATASTROPHE_MULTIPLE)
from loader import (Inputs, load_all, summarize, note_missing)


# ---- exact binomial (Wilson score) interval, hand-implemented --------------------------------

def wilson(k, n, z=1.96):
    """Wilson score 95% interval for an observed binomial count k of n. Returns (lo, hi).
    Degenerate n -> (None, None); clamps lie in [0, 1]. No continuity correction (interval is
    already conservative at the extremes by construction of the score interval)."""
    if n is None or n <= 0:
        return None, None
    p = k / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * sqrt((p * (1.0 - p)) / n + z2 / (4.0 * n * n)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return lo, hi


def wilson_rate(k, n, z=1.96):
    """(point estimate, lo, hi)."""
    if n is None or n <= 0:
        return None, None, None
    lo, hi = wilson(k, n, z)
    return k / float(n), lo, hi


def _fmt(r, width=9):
    if r is None:
        return "unavailable"
    return f"{r:.4f}".rjust(width)


# ---- feature aggregation --------------------------------------------------------------------

def family_rows_by_artifact(fam_rows):
    by = {}
    for r in fam_rows:
        by.setdefault(r["artifact"], []).append(r)
    return by


def total_rows_by_artifact(total_rows):
    return {t["artifact"]: t for t in total_rows}


def family_fires_c(flag, row, c):
    """family_fires under an explicit primary cut c (secondary feature keeps its own threshold)."""
    d = FLAG_DEFS[flag]
    feat = d["feature"]
    v = row.get(feat)
    if v is None:
        return None
    over = v > c
    if d.get("secondary"):
        s = row.get(d["secondary"])
        over = over or (s is not None and s > d["secondary_threshold"])
    return over


def artifact_fires_c(flag, fam_rows, total_row, c):
    """artifact-level firing under an explicit primary cut c; total fallback uses the flag's
    own total_threshold (quant) to keep the aggregate rule intact."""
    d = FLAG_DEFS[flag]
    any_fire = None
    for r in fam_rows:
        f = family_fires_c(flag, r, c)
        if f is True:
            any_fire = True
        elif f is False and any_fire is None:
            any_fire = False
    if any_fire is False and d.get("total_threshold") is not None and total_row is not None:
        t = total_row.get(d["feature"])
        if t is not None and t > d["total_threshold"]:
            any_fire = True
    if any_fire is None and total_row is not None:
        t = total_row.get(d["feature"])
        if t is not None and all(r.get(d["feature"]) is None for r in fam_rows):
            any_fire = t > c
    return any_fire


# ---- labelled rows: explicit labels frame, else derived from ledger cells --------------------

FLAG_BY_OP = {d["op"]: f for f, d in FLAG_DEFS.items()}


def cells_to_labels(cells):
    """Flatten ledger kind=cell records to label rows. predicted/unavailable cells stay labelled
    with their status so tallying code can apply I8 (never count a non-measured cell)."""
    by_id = {c["id"]: c for c in cells}
    out = []
    for c in cells:
        flag = FLAG_BY_OP.get(c.get("op"))
        if flag is None:
            continue
        verdict = c.get("verdict")
        outcome = "unavailable"
        if verdict in ("pass", "fail"):
            outcome = verdict
        m = c.get("metrics") or {}
        dkey = FLAG_DEFS[flag]["damage"]
        damage = m.get(dkey)
        if damage is None and dkey == "export_damage_ratio" and "ppl" in m and "ppl_reference" in m:
            try:
                damage = m["ppl"] / m["ppl_reference"]
            except (ZeroDivisionError, TypeError):
                damage = None
        dref = None
        ref = by_id.get(c.get("reference_cell"))
        if ref is not None:
            dref = (ref.get("metrics") or {}).get(dkey)
        out.append({"artifact": c.get("subject"), "flag": flag, "outcome": outcome,
                    "status": c.get("status") or "measured",
                    "damage": damage, "damage_ref": dref, "src": "cell:" + str(c.get("id"))})
    return out


def all_labels(frames):
    labels = list(frames.get("labels") or [])
    return labels + cells_to_labels(frames.get("cells") or [])


def measured_labels(rows):
    """Rows that are real measured evidence: a verdict plus a measured (or unspecified) status.
    predicted/unavailable rows are never tallied (I8)."""
    out = []
    for r in rows:
        if r.get("outcome") not in ("pass", "fail"):
            continue
        st = (r.get("status") or "measured")
        if st not in ("measured", "MEASURED", None):
            continue
        out.append(dict(r))
    return out


# ---- prevalence -----------------------------------------------------------------------------

def prevalence_by_family(fam_rows, flag, families, thr=None):
    d = FLAG_DEFS[flag]
    feat = d["feature"]
    c = thr if thr is not None else d["threshold"]
    res = {}
    for fam in families:
        rows = [r for r in fam_rows if r.get("family") == fam and r.get(feat) is not None]
        if not rows:
            res[fam] = {"n": 0, "count": 0, "rate": None, "lo": None, "hi": None}
            continue
        count = sum(1 for r in rows if family_fires_c(flag, r, c) is True)
        n = len(rows)
        rate, lo, hi = wilson_rate(count, n)
        res[fam] = {"n": n, "count": count, "rate": rate, "lo": lo, "hi": hi}
    return res


def prevalence_artifact(fam_rows, total_rows, flag, thr=None):
    """P(artifact-level flag fires). Covers artifacts with per-family features AND artifacts that
    only carry a total row (whole-artifact proxy, evaluated through artifact_fires_c's total
    fallback). Rows with no feature at all are left out (cannot be called safe, I8)."""
    d = FLAG_DEFS[flag]
    c = thr if thr is not None else d["threshold"]
    total_by = total_rows_by_artifact(total_rows)
    fam_by = family_rows_by_artifact(fam_rows)
    artids = sorted(set(fam_by) | set(total_by))
    n, count = 0, 0
    for artid in artids:
        fr = fam_by.get(artid) or []
        tr = total_by.get(artid)
        if fr and any(r.get(d["feature"]) is not None for r in fr):
            fired = artifact_fires_c(flag, fr, tr, c)
        elif tr is not None and tr.get(d["feature"]) is not None:
            fired = artifact_fires_c(flag, [], tr, c)    # total proxy for whole-artifact feature
        else:
            continue                               # no evidence at all (I8: never call safe)
        if fired is None:
            continue
        n += 1
        count += 1 if fired else 0
    rate, lo, hi = wilson_rate(count, n)
    return {"n": n, "count": count, "rate": rate, "lo": lo, "hi": hi}


def prevalence_by_arch(fam_rows, total_rows, flag, thr=None):
    """P(artifact-level flag fires) grouped by architecture family. An artifact's arch comes from
    its scan row (`arch`), defaulting to "unknown" when the freeze did not record it. Different
    architectures are never pooled into one rate (I3: cells under different arch conditions are
    not silently comparable); each arch gets its own prevalence + Wilson 95% interval."""
    arch_of = {}
    for r in fam_rows:
        arch_of.setdefault(r["artifact"], r.get("arch") or "unknown")
    for r in total_rows:
        arch_of.setdefault(r["artifact"], r.get("arch") or "unknown")
    out = {}
    for arch in sorted({a for a in arch_of.values()}):
        arts = {a for a, av in arch_of.items() if av == arch}
        fr = [r for r in fam_rows if r["artifact"] in arts]
        tr = [r for r in total_rows if r["artifact"] in arts]
        out[arch] = prevalence_artifact(fr, tr, flag, thr=thr)
    return out


# ---- confusion ------------------------------------------------------------------------------

def _confusion(predicted, outcome_pos):
    """predicted: list of bool; outcome_pos: parallel list of bool (True=fail). Returns mat."""
    tp = fp = tn = fn = 0
    for p, o in zip(predicted, outcome_pos):
        tp += p and o
        fp += p and not o
        tn += (not p) and (not o)
        fn += (not p) and o
    sens = None
    if tp + fn > 0:
        sens = tp / float(tp + fn)
    spec = None
    if tn + fp > 0:
        spec = tn / float(tn + fp)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sens": sens, "spec": spec,
            "sens_lo": wilson(tp, tp + fn)[0], "sens_hi": wilson(tp, tp + fn)[1],
            "spec_lo": wilson(tn, tn + fp)[0], "spec_hi": wilson(tn, tn + fp)[1]}


def build_confusion(flag, fam_rows, total_rows, labels, thr=None):
    """Per-family and artifact-level confusion for labelled artifacts whose verdict we can
    predict from static features. Skips rows whose predicted is None (no evidence). Thr is an
    explicit primary cut (contract override); None uses the referenced contract threshold."""
    d = FLAG_DEFS[flag]
    c = thr if thr is not None else d["threshold"]
    total_by = total_rows_by_artifact(total_rows)
    fam_by = family_rows_by_artifact(fam_rows)
    meas = measured_labels([r for r in labels if r.get("flag") == flag])
    if not meas:
        return None, 0

    per_fam = {}
    for fam in FAMILIES:
        predicted, outcome = [], []
        for r in meas:
            art = r.get("artifact")
            frs = fam_by.get(art) or []
            row = next((x for x in frs if x.get("family") == fam), None)
            if row is None or row.get(d["feature"]) is None:
                continue
            f = family_fires_c(flag, row, c)
            if f is None:
                continue
            predicted.append(f)
            outcome.append(r["outcome"] == "fail")
        per_fam[fam] = _confusion(predicted, outcome)

    predicted, outcome = [], []
    for r in meas:
        art = r.get("artifact")
        frs = fam_by.get(art) or []
        fired = None
        if frs and any(x.get(d["feature"]) is not None for x in frs):
            fired = artifact_fires_c(flag, frs, total_by.get(art), c)
        elif total_by.get(art) is not None and total_by[art].get(d["feature"]) is not None:
            fired = artifact_fires_c(flag, [], total_by[art], c)
        if fired is None:
            continue
        predicted.append(fired)
        outcome.append(r["outcome"] == "fail")
    artifact = _confusion(predicted, outcome)
    return {"families": per_fam, "artifact": artifact}, len(meas)


# ---- catastrophic divergence ---------------------------------------------------------------

def catastrophe(flag, labels):
    """P(damage >= CATASTROPHE_MULTIPLE x reference | measured outcome with a defined ratio).
    Rows whose reference is near zero make the ratio undefined (None), never False (I8)."""
    rows = measured_labels([r for r in labels if r.get("flag") == flag])
    defined, cat, fails_def, cat_fails = [], 0, 0, 0
    for r in rows:
        c = is_catastrophic(r.get("damage"), r.get("damage_ref"))
        if c is None:
            continue
        defined.append(r)
        cat += 1 if c else 0
        if r["outcome"] == "fail":
            fails_def += 1
            cat_fails += 1 if c else 0
    out = {"defined": len(defined), "catastrophic": cat,
           "raw_data": [(r.get("artifact"), r.get("outcome"), r.get("damage"),
                         r.get("damage_ref")) for r in defined]}
    r, lo, hi = wilson_rate(cat, len(defined))
    out.update(rate=r, lo=lo, hi=hi)
    r2, lo2, hi2 = wilson_rate(cat_fails, fails_def)
    out["among_fails"] = {"fails": fails_def, "catastrophic": cat_fails,
                          "rate": r2, "lo": lo2, "hi": hi2}
    return out


# ---- compute + report ----------------------------------------------------------------------

def compute(inputs, thr_overrides=None):
    """Base rates under the active contract. ``thr_overrides`` maps a flag to an explicit
    primary cut used by threshold-fitting flows and tests."""
    frames = load_all(inputs)
    scans, labels = frames["scans"], all_labels(frames)
    fam_rows, total_rows = scans["family"], scans["total"]
    result = {"n_scanned_artifacts": len(set(r["artifact"] for r in fam_rows)),
              "n_measured_labels": len(measured_labels(labels)),
              "threshold_overrides": dict(thr_overrides) if thr_overrides else {},
              "flags": {}}
    for flag in flag_ranks():
        d = FLAG_DEFS[flag]
        c = thr_overrides.get(flag) if thr_overrides else None
        conf, n_meas = build_confusion(flag, fam_rows, total_rows, labels, thr=c)
        result["flags"][flag] = {
            "contract": {"version": CONTRACT_VERSION, "threshold": d["threshold"],
                         "feature": d["feature"],
                         "override_primary_cut": c},
            "prevalence": {
                "families": prevalence_by_family(fam_rows, flag, FAMILIES, thr=c),
                "artifact": prevalence_artifact(fam_rows, total_rows, flag, thr=c),
                "arch": prevalence_by_arch(fam_rows, total_rows, flag, thr=c),
            },
            "confusion": conf,
            "n_measured": n_meas,
            "catastrophic": catastrophe(flag, labels),
        }
    return result


def format_report(res):
    lines = []
    lines.append(f"scanned artifacts: {res['n_scanned_artifacts']}; "
                 f"measured (pass|fail) labels: {res['n_measured_labels']}")
    for flag in flag_ranks():
        d = res["flags"][flag]
        fdef = FLAG_DEFS[flag]
        lines.append("")
        lines.append(f"== {flag}  (feature {fdef['feature']}, contract v{d['contract']['version']} "
                     f"threshold {d['contract']['threshold']:g}) ==")
        pr = d["prevalence"]
        lines.append("  prevalence (static flag base rate, 95% Wilson):")
        places = list(FAMILIES) + ["artifact"]
        if any(pr["families"][f]["n"] for f in FAMILIES):
            for f in FAMILIES:
                p = pr["families"][f]
                if p["n"] == 0:
                    lines.append(f"    {f:10s} n=0")
                else:
                    lines.append(f"    {f:10s} n={p['n']:4d}  "
                                 f"{p['rate']:.4f} [{p['lo']:.4f},{p['hi']:.4f}]")
        for arch in sorted(pr["arch"]):
            p = pr["arch"][arch]
            if p["n"] == 0 or p["rate"] is None:
                lines.append(f"    {'arch=' + arch:12s} n=0  unavailable")
            else:
                lines.append(f"    {'arch=' + arch:12s} n={p['n']:4d}  "
                             f"{p['rate']:.4f} [{p['lo']:.4f},{p['hi']:.4f}]")
        n_meas = d["n_measured"]
        if n_meas == 0 or d["confusion"] is None:
            lines.append(f"  outcomes: unavailable (no measured {fdef['op']} cells/pass-fail "
                         f"labels)")
        else:
            conf = d["confusion"]
            lines.append(f"  confusion (predicted flag vs measured verdict; "
                         f"active contract v{CONTRACT_VERSION}; n={n_meas} artifacts):")
            lines.append(f"    artifact-level: TP={conf['artifact']['tp']} "
                         f"FP={conf['artifact']['fp']} TN={conf['artifact']['tn']} "
                         f"FN={conf['artifact']['fn']}  "
                         f"sens={conf['artifact']['sens']:.3f} "
                         f"[{conf['artifact']['sens_lo']:.3f},{conf['artifact']['sens_hi']:.3f}]  "
                         f"spec={conf['artifact']['spec']:.3f} "
                         f"[{conf['artifact']['spec_lo']:.3f},{conf['artifact']['spec_hi']:.3f}]")
            for f in FAMILIES:
                m = conf["families"][f]
                if m["tp"] or m["fp"] or m["tn"] or m["fn"]:
                    lines.append(f"    {f:10s} TP={m['tp']} FP={m['fp']} TN={m['tn']} "
                                 f"FN={m['fn']}  sens={(m['sens'] if m['sens'] is not None else -1):.3f} "
                                 f"spec={(m['spec'] if m['spec'] is not None else -1):.3f}")
        c = d["catastrophic"]
        if c["defined"]:
            lines.append(f"  catastrophic divergence (damage >= {CATASTROPHE_MULTIPLE:g}x "
                         f"reference; stated in risk_flags): {c['catastrophic']}/{c['defined']} "
                         f"= {c['rate']:.4f} [{c['lo']:.4f},{c['hi']:.4f}]; "
                         f"among fails {c['among_fails']['catastrophic']}/"
                         f"{c['among_fails']['fails']}")
            for art, out, dam, ref in c["raw_data"]:
                lines.append(f"      {art:16s} {out:4s} damage={dam} ref={ref}")
        else:
            lines.append(f"  catastrophic divergence: unavailable (no defined "
                         f"damage/reference pairs)")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="base rates + thresholds-worthy confusion from "
                                            "scans and measured outcomes")
    ap.add_argument("--root", default="analysis/data")
    ap.add_argument("--scans", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--edges", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--json", default=None, help="write the machine-readable result to PATH")
    args = ap.parse_args(argv)

    inputs = Inputs(root=args.root, scans=args.scans, labels=args.labels,
                    manifest=args.manifest, edges=args.edges, ledger=args.ledger)
    res = compute(inputs)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"wrote {args.json}")
    print("\n".join(format_report(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
