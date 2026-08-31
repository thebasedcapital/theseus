# pairs.py — the matched-pair finder that makes claim K-8 testable from published lineage. From
# harvest lineage (parent->child edges and shared-parent siblings) + L0 static features + measured
# outcomes, it finds pairs whose static features agree within a tolerance t yet whose measured
# outcomes diverge. Tolerance is a parameter swept over >=3 values. THE NULL IS MANDATORY:
# outcome labels are shuffled >=200x (default 2000) and we report how often an equal-or-more-
# divergent pair set arises by chance; empirical p per tolerance. No null test, no claim.
# Owned by BaseRates.

import argparse
import json
import sys

import numpy as np

from risk_flags import FLAG_DEFS, flag_ranks
from loader import Inputs, load_all
from baserates import (family_rows_by_artifact, total_rows_by_artifact,
                       all_labels, measured_labels)

MATCH_FEATURES = ("q4_block_mse", "dyn_range_log10",
                  "row_energy_imbalance", "frac_below_f16_normal")
MATCH_FLOOR = 1e-12          # relative-distance denominator floor (features are >= 0)
DIVERGENCE_MULTIPLE = 3.0    # stated multiple of the reference damage for a "divergent" pair
TOLERANCES = (0.05, 0.1, 0.2, 0.4, 0.8)   # >= 3-value sensitivity sweep of the match tolerance
N_NULL_DEFAULT = 2000
N_NULL_MIN = 200


# ---- inputs -----------------------------------------------------------------------------------

def artifact_features(fam_rows, total_rows):
    """One feature dict per artifact: worst-family value per matched feature, falling back to the
    total row. 'worst' = max, matching the risk-flag convention (row_energy_imbalance etc)."""
    total_by = total_rows_by_artifact(total_rows)
    fam_by = family_rows_by_artifact(fam_rows)
    out = {}
    for art in sorted(set(fam_by) | set(total_by)):
        f = {}
        for key in MATCH_FEATURES:
            vals = [r.get(key) for r in fam_by.get(art, []) if r.get(key) is not None]
            if vals:
                f[key] = max(vals)
            else:
                t = total_by.get(art)
                if t is not None and t.get(key) is not None:
                    f[key] = t[key]
        out[art] = f
    return out


def outcome_map(labels):
    """artifact -> flag -> {verdict, damage, damage_ref} for measured labels only."""
    out = {}
    for r in measured_labels(labels):
        out.setdefault(r.get("artifact"), {})[r.get("flag")] = {
            "verdict": r["outcome"], "damage": r.get("damage"),
            "damage_ref": r.get("damage_ref")}
    return out


def relations(edges):
    """[(a, b)] of parent->child edges and sibling pairs (a,b) sharing a parent."""
    parent_children, seen, rel = {}, set(), []
    for e in edges:
        p = e.get("parent") or e.get("from")
        c = e.get("child") or e.get("to")
        if not p or not c or p == c:
            continue
        key = tuple(sorted((p, c)))
        if key in seen:
            continue
        seen.add(key)
        parent_children.setdefault(p, set()).add(c)
        rel.append((p, c))
    for ch in parent_children.values():
        ch = sorted(ch)
        for i in range(len(ch)):
            for j in range(i + 1, len(ch)):
                rel.append((ch[i], ch[j]))
    return rel


# ---- matching + divergence --------------------------------------------------------------------

def match_within(pair, feats, tol):
    """All MATCH_FEATURES present on both sides and each within relative tolerance tol."""
    a, b = pair
    fa, fb = feats.get(a), feats.get(b)
    if not fa or not fb:
        return False
    for key in MATCH_FEATURES:
        va, vb = fa.get(key), fb.get(key)
        if va is None or vb is None:
            return False
        denom = max(abs(va), abs(vb), MATCH_FLOOR)
        if abs(va - vb) / denom > tol:
            return False
    return True


def pair_score(flag, oa, ob):
    """Divergence score for one flag between two outcome records. Returns (diverges: bool|None,
    score: float). None = not evaluable (missing measured outcome on either side)."""
    if not oa or not ob or "verdict" not in oa or "verdict" not in ob:
        return None, None
    if oa["verdict"] != ob["verdict"]:
        return True, float("inf")
    ra, rb = oa.get("damage_ref"), ob.get("damage_ref")
    da, db = oa.get("damage"), ob.get("damage")
    if da is None or db is None or ra is None or rb is None:
        return None, None
    denom = max(abs(ra), abs(rb), 1e-9)
    score = abs(da - db) / denom
    return score >= DIVERGENCE_MULTIPLE, score


def score_pairs(pairs, feats, outcomes):
    """For a list of matched pairs, count divergent pairs and max divergence under the CURRENT
    outcome assignment. Returns dict."""
    n_div, max_score, evaled = 0, 0.0, 0
    examples = []
    for (a, b) in pairs:
        best = None
        for flag in flag_ranks():
            diverges, score = pair_score(flag, outcomes.get(a, {}).get(flag),
                                         outcomes.get(b, {}).get(flag))
            if score is None:
                continue
            if diverges is True:
                best = (flag, True, score)
                break
            if best is None or score > best[2]:
                best = (flag, False, score)
        if best is None:
            continue
        evaled += 1
        flag, diverges, score = best
        max_score = max(max_score, score)
        if diverges:
            n_div += 1
            examples.append((a, b, flag, score))
    return {"n_divergent": n_div, "max_score": max_score, "n_evaluable": evaled,
            "examples": examples}


# ---- the null ---------------------------------------------------------------------------------

def null_outcomes(outcomes, flags, rng):
    """Permute each flag's measured (verdict, damage, damage_ref) triples among the artifacts
    that hold a measured label for that flag. Lineage + features stay fixed: this isolates
    whether divergence beyond chance requires the observed outcome assignment or would arise by
    random relabelling."""
    shuffled = {}
    for flag in flags:
        holder = [(a, o[flag]) for a, o in outcomes.items() if flag in o]
        arts = [a for a, _ in holder]
        tris = [(o["verdict"], o.get("damage"), o.get("damage_ref")) for _, o in holder]
        tris = [tris[i] for i in rng.permutation(len(tris))]
        for a, t in zip(arts, tris):
            shuffled.setdefault(a, {})[flag] = {"verdict": t[0], "damage": t[1],
                                                "damage_ref": t[2]}
    return shuffled


def sweep_and_null(fam_rows, total_rows, labels, edges, n_null=N_NULL_DEFAULT, seed=7):
    """Returns report dict with per-tolerance matched/divergent counts + empirical p."""
    feats = artifact_features(fam_rows, total_rows)
    outcomes = outcome_map(labels)
    rel = relations(edges)
    pairs_by_tol, rel_by_tol = {}, {}
    for t in TOLERANCES:
        pairs_by_tol[t] = [p for p in rel if match_within(p, feats, t)]
        rel_by_tol[t] = len(rel)
    base = {t: score_pairs(pairs_by_tol[t], feats, outcomes) for t in TOLERANCES}

    rng = np.random.RandomState(seed)
    null_counts = {t: 0 for t in TOLERANCES}
    null_iters = max(n_null, N_NULL_MIN)
    for _ in range(null_iters):
        shuf = null_outcomes(outcomes, flag_ranks(), rng)
        for t in TOLERANCES:
            s = score_pairs(pairs_by_tol[t], feats, shuf)
            if s["n_divergent"] >= base[t]["n_divergent"]:
                null_counts[t] += 1

    report = {}
    for t in TOLERANCES:
        p = null_counts[t] / float(null_iters)
        report[t] = {
            "tolerance": t,
            "n_related_pairs": rel_by_tol[t],
            "n_matched": len(pairs_by_tol[t]),
            "n_evaluable": base[t]["n_evaluable"],
            "n_divergent": base[t]["n_divergent"],
            "max_divergence": base[t]["max_score"],
            "null_iters": null_iters,
            "empirical_p": p,
            "claim": base[t]["n_divergent"] > 0 and p < 0.05,
            "examples": base[t]["examples"],
        }
    return {"tolerances": report, "n_permutations": null_iters,
            "divergence_multiple": DIVERGENCE_MULTIPLE,
            "match_features": list(MATCH_FEATURES)}


def format_report(res):
    lines = []
    lines.append("matched-pair finder for K-8 "
                 f"(features {res['match_features']}; divergence = verdict mismatch or "
                 f"|d_a - d_b| >= {res['divergence_multiple']:g}x the reference damage)")
    lines.append(f"null: outcome labels permuted {res['n_permutations']}x; "
                 f"empirical p = P(null pairs >= observed pairs)")
    for t in sorted(res["tolerances"]):
        r = res["tolerances"][t]
        lines.append(f"  tol={r['tolerance']:<5} matched {r['n_matched']:3d} of "
                     f"{r['n_related_pairs']} lineage pairs, evaluable {r['n_evaluable']:3d}, "
                     f"divergent {r['n_divergent']:2d}, max divergence "
                     f"{r['max_divergence']:.1f}, emp. p={r['empirical_p']:.4f} "
                     f"-> {'K-8 candidate' if r['claim'] else 'not distinguishable from shuffled lineage'}")
        for (a, b, flag, score) in r["examples"][:6]:
            lines.append(f"      pair {a} ~ {b} on {flag} (divergence {score:.1f}x)")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="matched-pair finder for K-8 with mandatory null")
    ap.add_argument("--root", default="analysis/data")
    ap.add_argument("--scans", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--edges", default=None)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--null", type=int, default=N_NULL_DEFAULT)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    inputs = Inputs(root=args.root, scans=args.scans, labels=args.labels,
                    manifest=args.manifest, edges=args.edges, ledger=args.ledger)
    frames = load_all(inputs)
    labels = all_labels(frames)
    h = frames["harvest"]
    fam_rows, total_rows = frames["scans"]["family"], frames["scans"]["total"]

    if not h["edges"]:
        print("pairs: unavailable (no lineage edges; needs the harvest slice)")
        return 0
    if not measured_labels(labels):
        print("pairs: unavailable (no measured outcomes; needs labels or ledger cells)")
        return 0
    res = sweep_and_null(fam_rows, total_rows, labels, h["edges"],
                         n_null=args.null, seed=args.seed)
    print("\n".join(format_report(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
