# fixtures.py — synthetic labelled data with KNOWN ground truth. Injects an effect into ONE
# family's frac_below_f16_normal (q_proj) plus matched quant/adapt effects and a known capture
# collapse; writes real Inspector-v1 JSON scans, labels.jsonl, and harvest edges.jsonl so
# baserates/thresholds/pairs consume them exactly like real records. Also produces pure-noise
# variants where the null must stay silent. Every function returns its ground-truth dict; tests
# assert the machinery recovers it. Owned by BaseRates.

import json
import numpy as np
from pathlib import Path

from risk_flags import FAMILIES

# ---- constants shared with the effect design ------------------------------------------------

N_ARTIFACTS = 24
N_COHORT = 8
TRUE_CUTS = {"export.f16": 0.08, "quant.q8_0": 0.018, "quant.q4_k_m": 0.018,
             "adapt.lora.r16": 12.5}      # planted decision boundaries (feature scale)
DAMAGE_REFS = {"export.f16": 1.0004, "quant.q8_0": -0.000989, "quant.q4_k_m": 0.021945,
               "adapt.lora.r16": 2.9598}
FAM_BASES = {"q_proj": 0.0113, "k_proj": 0.0118, "v_proj": 0.0128, "o_proj": 0.0102,
             "gate_proj": 0.0108, "up_proj": 0.0107, "down_proj": 0.0111}


def _rng(rng):
    """Accept an int seed or an existing RandomState and return a RandomState."""
    if isinstance(rng, np.random.RandomState):
        return rng
    return np.random.RandomState(rng)


def _fn(rng, base, sd, lo=None, hi=None):
    v = rng.normal(base, sd)
    if v < 0:
        v = -v
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return float(v)


def write_inspector_json(scan_dir, art, feats_by_family, total, arch="Qwen2ForCausalLM"):
    """One Inspector schema v1 document (family features + total + skipped + verdicts)."""
    doc = {"path": f"/data/scans/{art}.safetensors", "id": art, "arch": arch,
           "families": {fam: dict(row) for fam, row in feats_by_family.items()},
           "total": dict(total), "skipped": [], "verdicts": [], "preflight": []}
    p = Path(scan_dir) / f"{art}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return p


# ---- effect data ----------------------------------------------------------------------------

def effect_data(root, rng, n_artifacts=N_ARTIFACTS, n_cohort=N_COHORT):
    """Healthy Qwen2-like population + one planted cohort: q_proj frac_below_f16_normal,
    q_proj q4_block_mse and q_proj dyn_range_log10 are all elevated, and every cohort artifact
    has a measured FAIL + catastrophic damage on quant/adapt (export collapses but stays under
    100x -> fail, not catastrophic). Returns the ground-truth dict."""
    root = Path(root)
    scan_dir = root / "scans"
    scan_dir.mkdir(parents=True, exist_ok=True)
    cohort = [f"art{i:02d}" for i in range(n_cohort)]
    healthy = [f"art{i:02d}" for i in range(n_cohort, n_artifacts)]
    rng = _rng(rng)
    label_rows = []
    for art in sorted(set(cohort) | set(healthy)):
        in_cohort = art in cohort
        fam_feats = {}
        for fam in FAMILIES:
            e = (fam == "q_proj" and in_cohort)      # effect confined to one family: q_proj
            frac = _fn(rng, 0.30 if e else 0.035, 0.04 if e else 0.004, lo=0.005)
            j = _fn(rng, 0.028 if e else FAM_BASES[fam], 0.003, lo=0.005)
            dyn = _fn(rng, 14.5 if e else 8.8, 0.5 if e else 0.4, lo=8.0, hi=16.0)
            fam_feats[fam] = {"q4_block_mse": j, "q4_block_mse_pooled": None,
                              "dyn_range_log10": dyn, "row_energy_imbalance": 2.0e4,
                              "amax_over_rms": 60.0, "frac_below_f16_normal": frac,
                              "weights": 3.5e7}
        total = {"q4_block_mse": max(r["q4_block_mse"] for r in fam_feats.values()),
                 "q4_block_mse_pooled": None,
                 "dyn_range_log10": max(r["dyn_range_log10"] for r in fam_feats.values()),
                 "row_energy_imbalance": 2.0e4, "amax_over_rms": 60.0,
                 "frac_below_f16_normal": max(r["frac_below_f16_normal"]
                                              for r in fam_feats.values()),
                 "weights": 3.5e7}
        write_inspector_json(scan_dir, art, fam_feats, total)
        for flag in TRUE_CUTS:
            outcome = "fail" if in_cohort else "pass"
            ref = DAMAGE_REFS[flag]
            if flag == "export.f16":
                dam = (_fn(rng, 15.0, 3.0, lo=8.0, hi=30.0) if in_cohort
                       else _fn(rng, 1.0, 0.005, lo=0.99, hi=1.005))
            elif flag == "quant.q8_0":
                dam = (_fn(rng, 3.0, 1.0, lo=0.5, hi=10.0) if in_cohort
                       else _fn(rng, 0.002, 0.002, lo=-0.005, hi=0.01))
            elif flag == "quant.q4_k_m":
                dam = (_fn(rng, 300.0, 200.0, lo=100.0, hi=2000.0) if in_cohort
                       else ref * _fn(rng, 1.0, 0.1, lo=0.8, hi=1.2))
            else:  # adapt.lora.r16: the capture collapse
                dam = (ref * _fn(rng, 500.0, 150.0, lo=300.0, hi=1000.0) if in_cohort
                       else ref * _fn(rng, 1.0, 0.1, lo=0.8, hi=1.2))
            label_rows.append({"artifact": art, "flag": flag, "outcome": outcome,
                               "status": "measured", "damage": dam, "damage_ref": ref,
                               "src": "fixture.effect"})
    with open(root / "labels.jsonl", "w") as f:
        for r in label_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return {
        "n_artifacts": n_artifacts, "n_cohort": n_cohort, "cohort": sorted(cohort),
        "healthy": sorted(healthy), "true_cuts": TRUE_CUTS,
        "damage_refs": DAMAGE_REFS, "effect_family": "q_proj",
        "clean_confusion": {"tp": n_cohort, "fp": 0, "tn": n_artifacts - n_cohort, "fn": 0},
        "v2_export_fp": n_artifacts - n_cohort,    # healthy frac 0.035 > v2 0.02, yet pass
        "catastrophic_flags": ["quant.q8_0", "quant.q4_k_m", "adapt.lora.r16"],
    }


def noise_data(root, rng, n_artifacts=N_ARTIFACTS, fail_prob=0.4):
    """Same scan shapes as effect_data, but outcomes are assigned at random independent of the
    features: the null must stay silent (no contract, no claim). Ground truth dict annotated with
    noise_fail_prob; nothing is recoverable by design."""
    rng = _rng(rng)
    gt = effect_data(root, rng, n_artifacts=n_artifacts, n_cohort=0)
    arts = [f"art{i:02d}" for i in range(n_artifacts)]
    rows = []
    for art in arts:
        for flag in TRUE_CUTS:
            outcome = "fail" if rng.random() < fail_prob else "pass"
            ref = DAMAGE_REFS[flag]
            if flag == "export.f16":
                dam = (_fn(rng, 15.0, 3.0, lo=8.0, hi=30.0) if outcome == "fail"
                       else _fn(rng, 1.0, 0.005, lo=0.99, hi=1.005))
            elif flag == "quant.q8_0":
                dam = (_fn(rng, 3.0, 1.0, lo=0.5, hi=10.0) if outcome == "fail"
                       else _fn(rng, 0.002, 0.002, lo=-0.005, hi=0.01))
            elif flag == "quant.q4_k_m":
                dam = (_fn(rng, 300.0, 200.0, lo=100.0, hi=2000.0) if outcome == "fail"
                       else ref * _fn(rng, 1.0, 0.1, lo=0.8, hi=1.2))
            else:
                dam = (ref * _fn(rng, 500.0, 150.0, lo=300.0, hi=1000.0) if outcome == "fail"
                       else ref * _fn(rng, 1.0, 0.1, lo=0.8, hi=1.2))
            rows.append({"artifact": art, "flag": flag, "outcome": outcome, "status": "measured",
                         "damage": dam, "damage_ref": ref, "src": "fixture.noise"})
    with open(root / "labels.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    gt["noise_fail_prob"] = fail_prob
    return gt


# ---- lineage data (K-8 machinery) -----------------------------------------------------------

def _lineage(root, rng, n_groups=10, seed_families=True, pad=60):
    """n_groups 'families', each with a parent family<N> and two children. Children of the first
    n_match groups have features within ~3% (match from tol=0.05); the rest differ by ~50%
    (never match). In effect mode, 1 fail per matched group is anchored to ONE child (towards
    its feature-matched sibling); in noise mode the same count is scattered uniformly over the
    whole pool. `pad` unparented healthy artifacts widen the permutation pool so the null
    reflects random outcome assignment over the artifact universe, not just the children."""
    root = Path(root)
    (root / "harvest").mkdir(parents=True, exist_ok=True)
    n_match = max(3, n_groups // 2)
    rng = _rng(rng)
    scan_rows, edges, labels, cats = [], [], [], []
    for k in range(pad):
        art = f"pin{k:03d}"
        scan_rows.append({"artifact": art, "level": "family", "family": "q_proj",
                          "q4_block_mse": 0.0113 * (5.0 + rng.uniform(0, 15)),
                          "dyn_range_log10": 8.8 * (5.0 + rng.uniform(0, 15)),
                          "row_energy_imbalance": 2.0e4 * (5.0 + rng.uniform(0, 15)),
                          "frac_below_f16_normal": 0.0028 * (5.0 + rng.uniform(0, 15)),
                          "source": "fixture.lineage.pad"})
        for flag, ref in DAMAGE_REFS.items():
            labels.append({"artifact": art, "flag": flag, "outcome": "pass",
                           "status": "measured", "damage": ref, "damage_ref": ref,
                           "src": "fixture.lineage.pad"})
    for g in range(n_groups):
        parent = f"family{g:02d}"
        a, b = f"{parent}_a", f"{parent}_b"
        edges.append({"parent": parent, "child": a, "op": "finetune", "source": "fixture"})
        edges.append({"parent": parent, "child": b, "op": "finetune", "source": "fixture"})
        matched = g < n_match
        for art in (a, b):
            if matched:
                shift = 1.0 + ((0.02 if art.endswith("_a") else -0.02) + rng.uniform(-0.01, 0.01))
            else:
                shift = 1.0 + rng.uniform(-0.5 if art.endswith("_a") else 0.5,
                                          0.5 if art.endswith("_a") else 1.0)
            scan_rows.append({"artifact": art, "level": "family", "family": "q_proj",
                              "q4_block_mse": 0.0113 * shift,
                              "dyn_range_log10": 8.8 * shift,
                              "row_energy_imbalance": 2.0e4 * shift,
                              "frac_below_f16_normal": 0.0028 * shift,
                              "source": "fixture.lineage"})
            for flag, ref in DAMAGE_REFS.items():
                labels.append({"artifact": art, "flag": flag, "outcome": "pass",
                               "status": "measured", "damage": ref, "damage_ref": ref,
                               "src": "fixture.lineage"})
        if matched and seed_families:
            cats.append((a, b))
    if seed_families:
        for (a, b) in cats:
            target = a if rng.random() < 0.5 else b
            for flag in ("adapt.lora.r16", "quant.q4_k_m"):
                ref = DAMAGE_REFS[flag]
                for lbl in labels:
                    if lbl["artifact"] == target and lbl["flag"] == flag:
                        lbl["outcome"] = "fail"
                        lbl["damage"] = ref * 500.0
    else:
        n_fail = len(cats) * 2            # same count as the effect mode (1 fail x 2 flags)
        for lbl in labels:
            lbl["outcome"] = "pass"
            lbl["damage"] = DAMAGE_REFS[lbl["flag"]]
        pool = [l for l in labels if l["flag"] in ("adapt.lora.r16", "quant.q4_k_m")]
        for i in rng.choice(len(pool), size=min(n_fail, len(pool)), replace=False):
            lbl = pool[i]
            lbl["outcome"] = "fail"
            lbl["damage"] = DAMAGE_REFS[lbl["flag"]] * rng.uniform(300.0, 1000.0)

    with open(root / "scans.jsonl", "w") as f:
        for r in scan_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(root / "harvest" / "edges.jsonl", "w") as f:
        for e in edges:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    with open(root / "labels.jsonl", "w") as f:
        for r in labels:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    return {"n_groups": n_groups, "n_match": n_match, "catastrophic_groups": len(cats),
            "injected_divergent_pairs": len(cats)}


def lineage_effect(root, rng=11, n_groups=10):
    return _lineage(root, rng, n_groups=n_groups, seed_families=True)


def lineage_noise(root, rng=23, n_groups=10):
    """Same lineage graph + features; the catastrophic labels are scattered at random, so matched
    pairs must NOT show divergence beyond chance."""
    return _lineage(root, rng, n_groups=n_groups, seed_families=False)
