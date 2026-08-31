#!/usr/bin/env python3
"""M1 analysis: does a cheap static diagnostic predict surgery damage?

This is the seed of ROADMAP A6/B4 (predictive optionality) built from M1's own ledger. No LLM,
no training: rank correlation between the static block-amax conditioning proxy
(`canonicalize.quant_condition`, computed from the artifact alone) and the measured damage of a
real operation, plus the per-family gauge-debt numbers that make the `prepare` claim concrete.

    <venv python> m1/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SUM = common.WORK / "m1_summary.json"
EQ = common.WORK / "equiv"


def rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> tuple[float, int]:
    if len(x) < 3:
        return float("nan"), len(x)
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return (num / (dx * dy) if dx and dy else float("nan")), n


def jtotal(d: dict | None) -> float | None:
    if not d:
        return None
    v = [x for x in d.values() if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None


def prediction_check(rows):
    """Score the pre-registered static predictions against measured Q4_K_M damage.

    Registered direction (m1/predict.py): debt > 1e-3 -> "Q4 damage expected", else neutral.
    Measured direction: this variant's q4_k_m mean KLD exceeds base's by > 10 % (KLD is the
    monotone statistic; rel_dppl is not, see M1_RESULTS §4). A prediction is only scored when
    both sides exist, so partial runs report coverage instead of guessing.
    """
    pred_file = None
    for cand in (common.WORK / "PREDICTIONS_new.json", common.WORK / "PREDICTIONS.json"):
        if cand.exists():
            pred_file = cand
            break
    if pred_file is None:
        return {"error": "no PREDICTIONS.json snapshot found"}
    preds = json.loads(pred_file.read_text()).get("variants", {})
    base_kld = (rows.get("base", {}) or {}).get("q4_kl_mean")
    out = {"source": str(pred_file), "base_q4_kld": base_kld, "checked": {}, "skipped": {}}
    if not isinstance(base_kld, (int, float)):
        out["skipped"]["all"] = "base q4 KLD not measured yet"
        return out
    for name, r in rows.items():
        p = preds.get(name)
        if p is None:
            out["skipped"][name] = "not in the pre-registration snapshot"
            continue
        kld = r.get("q4_kl_mean")
        if not isinstance(kld, (int, float)):
            out["skipped"][name] = "no measured q4 KLD yet"
            continue
        damaged = kld > base_kld * 1.10
        predicted_damaged = p["debt"] > 1e-3
        out["checked"][name] = {"debt": p["debt"], "predicted": "damage" if predicted_damaged
                                else "neutral", "measured_kld": kld,
                                "ratio_to_base": round(kld / base_kld, 3),
                                "measured": "damage" if damaged else "neutral",
                                "held": predicted_damaged == damaged}
    held = [k for k, v in out["checked"].items() if v["held"]]
    out["summary"] = {"checked": len(out["checked"]), "held": len(held),
                      "broken": [k for k in out["checked"] if not out["checked"][k]["held"]]}
    return out


def main():
    if not SUM.exists():
        raise SystemExit(f"run m1/report.py first (missing {SUM})")
    summ = json.loads(SUM.read_text())
    rows, pairs_kld, pairs_dppl, pairs_lora = [], [], [], []
    for name, v in summ.items():
        eq = common.rjson(EQ / f"{name}.json") if (EQ / f"{name}.json").exists() else None
        j_var = jtotal((eq or {}).get("cond_b"))
        j_base = jtotal((eq or {}).get("cond_a"))
        num = v.get("num", {})
        row = {"variant": name, "equiv": v.get("equiv"), "J_base": j_base, "J_var": j_var,
               "gauge_debt": (None if not j_var or not j_base else round(j_var - j_base, 4)),
               "q4_kl_mean": num.get("q4_k_m_kl_mean"), "q4_rel_dppl": num.get("q4_k_m_rel_dppl"),
               "q4_agree": num.get("q4_k_m_tokagree"), "q8_kl_mean": num.get("q8_0_kl_mean"),
               "lora_capture": num.get("lora_capture"), "lora_pass": num.get("lora_pass"),
               "lora_dppl": num.get("lora_protected_dppl"),
               "merge_linear_pass": num.get("linear_pass"), "merge_ties_pass": num.get("ties_pass")}
        rows.append(row)
        if row["J_var"] is not None:
            if isinstance(row["q4_kl_mean"], (int, float)):
                pairs_kld.append((row["J_var"], row["q4_kl_mean"]))
            if isinstance(row["q4_rel_dppl"], (int, float)):
                pairs_dppl.append((row["J_var"], row["q4_rel_dppl"]))
            if isinstance(row["lora_capture"], (int, float)):
                pairs_lora.append((row["J_var"], row["lora_capture"]))
    rho_kld, n_kld = spearman([a for a, _ in pairs_kld], [b for _, b in pairs_kld])
    rho_dppl, n_dppl = spearman([a for a, _ in pairs_dppl], [b for _, b in pairs_dppl])
    rho_lora, n_lora = spearman([a for a, _ in pairs_lora], [b for _, b in pairs_lora])

    hdr = ("variant            equiv        J_base  J_var   debt     Q4 KLD   Q4 rel%  "
           "Q8 KLD   LoRA    merge")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        f = lambda x, spec: (format(x, spec) if isinstance(x, (int, float)) else "  -")
        out.append(f"{r['variant']:<18} {str(r['equiv'])[:10]:<10} "
                   f"{f(r['J_base'], '7.2f')} {f(r['J_var'], '7.2f')} {f(r['gauge_debt'], '+7.3f')} "
                   f"{f(r['q4_kl_mean'], '8.5f')} {f((r['q4_rel_dppl'] or 0) * 100 if isinstance(r['q4_rel_dppl'], (int, float)) else None, '7.2f')} "
                   f"{f(r['q8_kl_mean'], '8.5f')} "
                   f"{f(r['lora_capture'], '6.3f')}{'' if r['lora_pass'] is None else ('P' if r['lora_pass'] else 'F')} "
                   f"{'L' + ('P' if r['merge_linear_pass'] else 'F') if r['merge_linear_pass'] is not None else 'L-'}"
                   f"{'T' + ('P' if r['merge_ties_pass'] else 'F') if r['merge_ties_pass'] is not None else 'T-'}")
    out += ["",
            f"Spearman rho(static conditioning J -> measured Q4_K_M mean KLD) = {rho_kld:.3f} (n={n_kld})",
            f"Spearman rho(J -> Q4_K_M relative dPPL)                          = {rho_dppl:.3f} (n={n_dppl})",
            f"Spearman rho(J -> LoRA capture)                                  = {rho_lora:.3f} (n={n_lora})",
            "Interpretation: a nonzero |rho| with n>=6 is a hint, not a result. M1's job is to"
            " produce the labelled surgery ledger that M6 fits a calibrated predictor on; the"
            " predictor may not be called predictive until it is validated out of sample."]
    text = "\n".join(out)
    print(text)
    pc = prediction_check({r["variant"]: r for r in rows})
    print("\npre-registered prediction check (static debt -> measured Q4_K_M damage):")
    for k, v in sorted(pc.get("checked", {}).items()):
        print(f"  {k:16s} debt {v['debt']:+.5f} predicted {v['predicted']:7s} measured "
              f"{v['measured']:7s} KLD {v['measured_kld']:.5f} ({v['ratio_to_base']:.2f}x base) "
              f"{'HELD' if v['held'] else 'BROKEN'}")
    if pc.get("summary"):
        print(f"  coverage: {pc['summary']['checked']} checked, {pc['summary']['held']} held, "
              f"broken={pc['summary']['broken']}")
    common.wjson(common.WORK / "m1_prediction_check.json", pc)
    common.wjson(common.WORK / "m1_analysis.json",
                 {"rows": rows, "spearman": {"J_to_q4_kld": [rho_kld, n_kld],
                                             "J_to_q4_dppl": [rho_dppl, n_dppl],
                                             "J_to_lora_capture": [rho_lora, n_lora]}})
    (common.REPO / "M1_ANALYSIS.md").write_text(
        "# M1 analysis — static conditioning vs measured surgery damage\n\n"
        "Generated by `m1/analyze.py`. `J` = mean block-max-abs quantization MSE proxy over the "
        "seven weight families (artifact-only, no surgery); `debt` = J(variant) - J(base), i.e. "
        "avoidable lifecycle debt created by the representation rather than the function.\n\n"
        "```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
