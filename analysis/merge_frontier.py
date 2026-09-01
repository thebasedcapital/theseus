#!/usr/bin/env python3
"""`analysis/merge_frontier.py` — re-grade merge cells against a reference-relative frontier.

Why (incident #21). merge_probe's contract pairs a reference-relative perplexity term with an
ABSOLUTE retention term, `rule_loss_ratio <= 0.75`, where the denominator is the specialist's own
rule loss. On Qwen3-0.6B-Base the recalibrated specialist reaches rule loss 0.0194, so the term
demands merged loss under 0.0146 - better than the specialist itself. The pristine base tops out at
0.958 (linear) and 0.981 (TIES) and asymptotes toward 1.0, so the criterion is unattainable rather
than merely strict, and no verdict could be issued. That is incident #4's defect - a cap the
reference checkpoint cannot pass - one contract field later.

Fix, same shape as the ppl slack: derive the retention ceiling from what the base candidate
actually achieves on that architecture, times a declared slack. This script does NOT re-run
surgery; it re-reads committed cells, so a re-grade is auditable and cheap, and the original
absolute-threshold cells are untouched.

Both ceilings are reported side by side (retention, and perplexity under its recorded absolute
1.05 versus a frontier-derived alternative) because the choice moves one verdict, and hiding that
would be the same sin as the absolute term was.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETENTION_SLACK = 1.03      # ceiling = base frontier x 1.03; mirrors the ppl slack convention
PPL_SLACK = 1.03


def best(rows, key):
    vals = [r for r in rows if isinstance(r, dict) and isinstance(r.get(key), (int, float))]
    return min(vals, key=lambda r: r[key]) if vals else None


def grade(work: Path, candidates=("base", "g3_pow2", "g3_pow2_rep")):
    cells = {}
    for c in candidates:
        p = work / f"{c}.merge.json"
        if not p.exists():
            continue
        try:
            cells[c] = json.loads(p.read_text()).get("results")
        except Exception:
            continue
    if "base" not in cells or not cells["base"]:
        return {"status": "UNAVAILABLE", "reason": "no base merge cell to derive a frontier from"}

    out = {"frontier_slack": {"retention": RETENTION_SLACK, "ppl": PPL_SLACK}, "ops": {}}
    for op in ("linear", "ties"):
        brows = ((cells["base"] or {}).get(op) or {}).get("matrix") or []
        fb, fp = best(brows, "rule_loss_ratio"), best(brows, "ppl_ratio")
        if not fb or not fp:
            out["ops"][op] = {"status": "UNAVAILABLE"}
            continue
        ret_ceiling = round(fb["rule_loss_ratio"] * RETENTION_SLACK, 4)
        ppl_ceiling_abs = ((cells["base"].get("contract") or {}).get("ppl_ratio_max")) or 1.05
        ppl_ceiling_rel = round(fp["ppl_ratio"] * PPL_SLACK, 4)
        rows_out = {}
        for name, res in cells.items():
            rws = ((res or {}).get(op) or {}).get("matrix") or []
            br, bp = best(rws, "rule_loss_ratio"), best(rws, "ppl_ratio")
            if not br or not bp:
                rows_out[name] = {"status": "UNAVAILABLE"}
                continue
            # A verdict must hold at ONE alpha. Minimising rule ratio and ppl ratio separately can
            # combine a pass on each from different alphas and manufacture a verdict that no single
            # merge realises, so both conditions are evaluated row by row.
            ok_rows = [r for r in rws if isinstance(r, dict)
                       and isinstance(r.get("rule_loss_ratio"), (int, float))
                       and isinstance(r.get("ppl_ratio"), (int, float))
                       and r["rule_loss_ratio"] <= ret_ceiling
                       and r["ppl_ratio"] <= ppl_ceiling_rel]
            ok_abs = [r for r in rws if isinstance(r, dict)
                      and isinstance(r.get("rule_loss_ratio"), (int, float))
                      and isinstance(r.get("ppl_ratio"), (int, float))
                      and r["rule_loss_ratio"] <= ret_ceiling
                      and r["ppl_ratio"] <= ppl_ceiling_abs]
            rows_out[name] = {
                "status": "MEASURED",
                "best_rule_loss_ratio": br["rule_loss_ratio"], "at_alpha": br["alpha"],
                "best_ppl_ratio": bp["ppl_ratio"],
                "passing_alphas_frontier": [r["alpha"] for r in ok_rows],
                "passing_alphas_ppl_abs": [r["alpha"] for r in ok_abs],
                "verdict_frontier_relative": bool(ok_rows),
                "verdict_mixed_ppl_absolute": bool(ok_abs),
                "absolute_retention_verdict_old": bool(
                    br["rule_loss_ratio"] <= ((res.get("contract") or {}).get("rule_loss_ratio_max", 0.75))),
            }
        out["ops"][op] = {
            "base_rule_frontier": fb["rule_loss_ratio"], "rule_ceiling": ret_ceiling,
            "base_ppl_frontier": fp["ppl_ratio"],
            "ppl_ceiling_rel": ppl_ceiling_rel, "ppl_ceiling_abs": ppl_ceiling_abs,
            "candidates": rows_out}
    out["status"] = "OK"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="merge_frontier")
    ap.add_argument("--work", default=str(ROOT / "m1" / "work-qwen3"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", default="")
    a = ap.parse_args(argv)
    rep = grade(Path(a.work))
    if a.write:
        Path(a.write).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write).write_text(json.dumps(rep, indent=2) + "\n")
    if a.json:
        print(json.dumps(rep, indent=2)); return 0
    print(f"work={a.work}  {rep['status']}")
    for op, d in (rep.get("ops") or {}).items():
        if d.get("status") != "OK" and "candidates" not in d:
            print(f"  {op}: {d.get('status')}"); continue
        print(f"  {op}: rule ceiling {d['rule_ceiling']} (base frontier {d['base_rule_frontier']} x"
              f"{RETENTION_SLACK}) | ppl ceiling {d['ppl_ceiling_rel']} rel vs {d['ppl_ceiling_abs']} abs")
        for name, c in d["candidates"].items():
            if c.get("status") != "MEASURED":
                print(f"    {name:12} {c.get('status')}"); continue
            print(f"    {name:12} rule={c['best_rule_loss_ratio']:.3f} "
                  f"ppl={c['best_ppl_ratio']:.3f} @a={c['at_alpha']} | "
                  f"verdict(frontier)={'pass' if c['verdict_frontier_relative'] else 'fail'} "
                  f"verdict(ppl-abs)={'pass' if c['verdict_mixed_ppl_absolute'] else 'fail'} "
                  f"old-absolute={'pass' if c['absolute_retention_verdict_old'] else 'fail'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
