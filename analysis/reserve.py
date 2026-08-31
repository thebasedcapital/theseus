#!/usr/bin/env python3
"""`analysis/reserve.py` — turn recorded pass/fail cells into the quantitative reserve of math.md §4.

Weakness #7 was that the formalism promised `R_o(s; B_o) = max(0, 1 - J*_o(s)/B_o)` while the code
emitted booleans. This closes that gap using only cells that already exist: no new surgery, no GPU.

Every limit is read out of the cell that was judged against it, never restated here:
  quantization  `pass_contract.rel_dppl_slack` / `.kl_mean_slack`, reference-relative (amended
                2026-08-30 after base calibration, before any variant was measured)
  adaptation    the cell's own `capture_threshold` and `capture_ref`
  merge         the alpha ladder the cell actually swept

Reserve is a VECTOR (K-7). Nothing here collapses coordinates into one score; where two statistics
disagree on the same artifact - as they do for `g1_haar`, neutral on KLD and failing on relative
ΔPPL - both are reported and the disagreement is the finding.

Bits-per-weight is computed from the measured artifact size against the metered parameter count,
not from llama.cpp's nominal labels.
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "m1" / "work"
OUT = ROOT / "analysis" / "data" / "reserve"

# Measured by theseus-scan on the pristine artifact: 357,826,560 weights.
N_PARAMS = 357_826_560
SCHEMES = ["q8_0", "q5_k_m", "q4_k_m"]          # ladder from least to most aggressive
CONVENTIONS = {
    "reserve": "R_o = clip(1 - excess/allowed_slack, 0, 1); 1.0 = at or better than the reference, "
               "0.0 = the operation is lost. Margin in the statistic's own units, not parameter "
               "distance (math.md §5 notes Euclidean distance is invalid under gauge freedom).",
    "bpw": "measured artifact bytes / metered parameter count * 8, not llama.cpp's nominal label",
    "deep_bit": "most aggressive scheme on the recorded ladder whose OWN cells both pass",
    "merge": "largest alpha that passes / largest alpha swept; the specialist fraction the "
             "coordinates can absorb",
    "collapse": "no scalar is produced (K-7); each operation is reported separately",
}


def clip01(x):
    return max(0.0, min(1.0, x))


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def quant_reserve(cell: dict, base_cell: dict) -> dict:
    """Per-scheme margin against the contract the cell was itself judged under."""
    pc = cell.get("pass_contract") or {}
    slack_d, slack_k = pc.get("rel_dppl_slack"), pc.get("kl_mean_slack")
    ref = (base_cell or {}).get("results", {})
    out = {"schemes": {}, "bpw": {}}
    for s in SCHEMES:
        v = (cell.get("results") or {}).get(s)
        r = ref.get(s) if isinstance(ref, dict) else None
        if not isinstance(v, dict):
            out["schemes"][s] = {"status": "UNAVAILABLE"}
            continue
        if v.get("status") != "OK":
            out["schemes"][s] = {"status": "UNAVAILABLE", "reason": "probe did not produce a number"}
            continue
        entry = {"status": "MEASURED", "pass": v.get("pass")}
        if isinstance(v.get("size_mb"), (int, float)):
            entry["bpw"] = round(v["size_mb"] * 1e6 / N_PARAMS * 8, 3)
            out["bpw"][s] = entry["bpw"]
        for key, slack, base_v in (("rel_dppl", slack_d, (r or {}).get("rel_dppl")),
                                   ("kl_mean", slack_k, (r or {}).get("kl_mean"))):
            got, bs = v.get(key), base_v
            if not isinstance(got, (int, float)) or not isinstance(bs, (int, float)) \
                    or not isinstance(slack, (int, float)) or slack <= 0:
                entry[f"R_{key}"] = None
                continue
            excess = max(0.0, got - bs)              # reference-relative: only the overshoot costs
            entry[f"R_{key}"] = round(clip01(1 - excess / slack), 4)
            entry[f"excess_{key}"] = round(excess, 6)
        out["schemes"][s] = entry
    passing = [s for s in SCHEMES
               if out["schemes"][s].get("R_rel_dppl") not in (None, 0.0)
               and out["schemes"][s].get("R_kl_mean") not in (None, 0.0)]
    out["deepest_passing"] = passing[-1] if passing else None
    return out


def adapt_reserve(cell: dict) -> dict:
    """Adaptation reserve = the BINDING of the contract's two terms.

    The first revision of this function used only `capture`, and so reported bad_all_exact as
    R=0.836 while its own cell records pass=False: that artifact is caught by the collateral term
    (protected_dppl 1.900 against an allowed 1.224 + 0.020), not by capture (0.9455 > 0.7395). A
    reserve that disagrees with the recorded verdict is worse than a boolean, so the two are tied
    together by test_reserve.py across every real cell.
    """
    rec = (cell.get("results") or {}).get("variant")
    if not isinstance(rec, dict) or "capture" not in rec:
        return {"status": "UNAVAILABLE", "reason": "no adaptation record (see #18 shape rule)"}
    thr, ref, cap = rec.get("capture_threshold"), rec.get("capture_ref"), rec.get("capture")
    if not all(isinstance(x, (int, float)) for x in (thr, ref, cap)):
        return {"status": "UNAVAILABLE", "reason": "cell predates the threshold fields"}
    r_cap = clip01((cap - thr) / ((ref - thr) or 1e-12))

    dppl, dppl_ref = rec.get("protected_dppl"), rec.get("protected_dppl_ref")
    # Real cells carry pass_contract inside results.variant; accept either location, but never
    # assume a slack value when neither supplies one - silently falling back to the capture term
    # alone is what overstated bad_all_exact.
    pc = str(rec.get("pass_contract") or cell.get("pass_contract") or "")
    m = re.search(r"protected_dppl_ref\s*\+\s*([0-9.eE+-]+)", pc)
    if not isinstance(dppl, (int, float)) or not isinstance(dppl_ref, (int, float)) or not m:
        return {"status": "UNAVAILABLE", "reason": "collateral term not recoverable from this cell",
                "R_capture_only": round(r_cap, 4)}
    slack = float(m.group(1))
    r_coll = clip01(1 - max(0.0, dppl - (dppl_ref + slack)) / (slack or 1e-12))
    return {"status": "MEASURED", "capture": round(cap, 4), "threshold": round(thr, 4),
            "reference": round(ref, 4), "R_capture": round(r_cap, 4),
            "protected_dppl": round(dppl, 4), "dppl_allowed": round(dppl_ref + slack, 4),
            "R_collateral": round(r_coll, 4), "binding": "capture" if r_cap <= r_coll else "collateral",
            "R_adapt": round(min(r_cap, r_coll), 4), "recorded_pass": rec.get("pass"),
            "contract_version": rec.get("contract_version")}


def merge_reserve(cell: dict) -> dict:
    out = {}
    for op in ("linear", "ties"):
        block = (cell.get("results") or {}).get(op)
        matrix = (block or {}).get("matrix") if isinstance(block, dict) else None
        if not isinstance(matrix, list) or not matrix:
            out[op] = {"status": "UNAVAILABLE"}
            continue
        alphas = [m["alpha"] for m in matrix if isinstance(m, dict) and "alpha" in m]
        ok = [m["alpha"] for m in matrix if isinstance(m, dict) and m.get("pass") is True]
        top = max(alphas) if alphas else None
        out[op] = {"status": "MEASURED", "alphas_swept": sorted(alphas),
                   "largest_passing_alpha": (max(ok) if ok else None),
                   "R_merge": (round(max(ok) / top, 4) if ok and top else 0.0)}
    return out


def build(work: Path = WORK) -> dict:
    base_g = load(work / "ops" / "base.gguf.json")
    artifacts: dict = {}

    for p in sorted((work / "ops").glob("*.gguf.json")):
        name = p.name[: -len(".gguf.json")]
        cell = load(p)
        if not isinstance(cell, dict):
            continue
        # base is the reference, so it is measured against itself: excess 0, reserve 1 by
        # construction. That is exactly what "reference-relative" asserts, and is worth stating
        # numerically rather than leaving the row blank.
        artifacts.setdefault(name, {})["quantize"] = quant_reserve(cell, base_g)

    for p in sorted((work / "ops").glob("*.adapt.json")):
        name = p.name[: -len(".adapt.json")]
        artifacts.setdefault(name, {})["adapt.lora.r16"] = adapt_reserve(load(p) or {})

    for p in sorted((work / "ops").glob("*.merge.json")):
        name = p.name[: -len(".merge.json")]
        artifacts.setdefault(name, {})["merge"] = merge_reserve(load(p) or {})

    return {"conventions": CONVENTIONS, "n_params_metered": N_PARAMS,
            "ladder": SCHEMES, "artifacts": artifacts,
            "note": "UNAVAILABLE is never tallied as 0 (I8); absence of evidence is not evidence."}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="reserve", description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--write", action="store_true", help="persist analysis/data/reserve/reserve.json")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = build(Path(a.work))
    if a.write:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "reserve.json").write_text(json.dumps(rep, indent=2) + "\n")
        print(f"wrote {OUT / 'reserve.json'}")
    if a.json:
        print(json.dumps(rep, indent=2))
        return 0
    print(f"reserve vectors for {len(rep['artifacts'])} artifacts "
          f"(bpw measured: {rep['conventions']['bpw']})\n")
    for name in sorted(rep["artifacts"]):
        q = rep["artifacts"][name].get("quantize", {})
        ad = rep["artifacts"][name].get("adapt.lora.r16", {})
        mg = rep["artifacts"][name].get("merge", {})
        qs = " ".join(f"{s.replace('_k_m','').replace('_0','')}="
                      f"{(q.get('schemes', {}).get(s) or {}).get('R_rel_dppl')}"
                      for s in SCHEMES)
        ad_r = ad.get("R_adapt", "n/a")
        lin = (mg.get("linear") or {}).get("R_merge", "n/a")
        tie = (mg.get("ties") or {}).get("R_merge", "n/a")
        print(f"  {name:18} quant[R_dPPL] {qs} | deep={q.get('deepest_passing')} "
              f"| adapt={ad_r} | merge lin={lin} ties={tie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
