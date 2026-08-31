#!/usr/bin/env python3
"""M1 report: fold the equivalence gate + surgery probes into the reserve table.

    <venv python> m1/report.py [--out M1_TABLE.md]

Reads only persisted JSON under m1/work/ (registry, equivalence gate, per-op probe results),
so it can be re-run at any point of the sweep. Missing evidence stays visibly missing: an
operation that never ran prints UNAVAILABLE and is excluded from the pass count rather than
counted as failure — and never counted as a PASS either (ROADMAP B8).

m1/work/m1_summary.json carries both the display strings and a "num" block of raw numbers,
which is what m1/plot_m1.py draws from.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

EQ, OPS = common.WORK / "equiv", common.WORK / "ops"
GGUF_TAGS = ("f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "iq4_xs")
ORDER = ["base", "prep_base",
         "g1_haar", "g1_haar_rep", "g1_svd", "g1_svd_rep",
         "g2_rand", "g2_rand_rep", "g3_rand", "g3_rand_rep", "g3_smooth", "g3_smooth_rep",
         "g7_rand", "g7_rand_rep", "g7_few",
         "g4_perm", "g6_perm", "g5_c8", "g5_c8_rep", "g5_c8_eps", "g5_c8_eps_rep",
         "bad_all", "bad_all_rep", "bad_all_s2", "bad_all_s2_rep", "bad_all_s3", "bad_all_s3_rep"]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def op_json(variant: str, op: str) -> dict | None:
    d = load(OPS / f"{variant}.{op}.json")
    if not d or d.get("error") or d.get("status") in ("UNAVAILABLE", "BAD_JSON"):
        return None
    return d


def gguf_cells(variant: str) -> tuple[dict, dict]:
    d, disp, num = op_json(variant, "gguf"), {t: "UNAVAILABLE" for t in GGUF_TAGS}, {}
    if not d:
        return disp, num
    for t, v in (d.get("results") or {}).items():
        if not isinstance(v, dict):
            continue
        if v.get("status") not in (None, "OK"):
            disp[t] = str(v.get("status"))
            continue
        bits = []
        for key, label in (("ppl", "ppl"), ("dppl", "dppl"), ("rel_dppl", "rel"),
                           ("kl_mean", "KLD"), ("kl_p999", "p99.9"), ("tokagree", "agree"),
                           ("size_mb", "MB")):
            val = v.get(key)
            if not isinstance(val, (int, float)):
                continue
            shown = val * 100 if label == "rel" else val
            bits.append(f"{label} {shown:.5g}{'%' if label == 'rel' else ''}")
            num[f"{t}_{key}"] = val
            if label == "rel":
                num[f"{t}_rel_dppl_pct"] = shown
        if v.get("pass") is not None or v.get("passes") is not None:
            ok = v.get("pass", v.get("passes"))
            bits.append("✅" if ok else "❌")
            num[f"{t}_pass"] = bool(ok)
        elif v.get("role") == "reference_calibration":
            # this run defined the threshold, so by construction it is the passing case
            bits.append("(reference)")
            num[f"{t}_pass"] = True
        disp[t] = " · ".join(bits) or "no fields"
    return disp, num


def adapt_cells(variant: str) -> tuple[str, dict]:
    d, num = op_json(variant, "adapt"), {}
    if not d:
        return "UNAVAILABLE", num
    rows = []
    res = d.get("results") or {}
    v = res.get("variant") or res.get("sanity_reference")
    if isinstance(v, dict) and "capture" in v:
        s = f"capture {v['capture']:.3f}"
        if isinstance(v.get("protected_dppl"), (int, float)):
            s += f", protΔppl {v['protected_dppl']:+.3f}"
        if v.get("pass") is not None:
            s += " ✅" if v["pass"] else " ❌"
            num["lora_pass"] = bool(v["pass"])
        num["lora_capture"] = v["capture"]
        if isinstance(v.get("protected_dppl"), (int, float)):
            num["lora_protected_dppl"] = v["protected_dppl"]
        rows.append(s)
    return " · ".join(rows) or "UNAVAILABLE", num


def merge_cells(variant: str) -> tuple[str, dict]:
    d, num = op_json(variant, "merge"), {}
    if not d:
        return "UNAVAILABLE", num
    rows = []
    res = d.get("results") or {}
    # merge_probe emits {"linear": {"matrix": [{alpha, eval_ppl, specialist_rule_loss, pass}],
    #                              "smallest_passing_alpha": a|None}, "ties": {...}}
    for k in ("linear", "ties"):
        v = res.get(k)
        if not isinstance(v, dict):
            continue
        best = v.get("smallest_passing_alpha")
        ok = best is not None if "smallest_passing_alpha" in v else bool(v.get("pass"))
        mark = "✅" if ok else "❌"
        rows.append(f"{k}: " + (f"pass@a={best:.2f} {mark}" if ok and best is not None
                                else (f"pass {mark}" if ok else f"fail {mark}")))
        num[f"merge_{k}_pass"] = bool(ok)
        if best is not None:
            num[f"merge_{k}_best_alpha"] = best
    for k, v in res.items():                      # flat {op: {pass: bool}} fallback
        if isinstance(v, dict) and "pass" in v and k not in ("linear", "ties"):
            rows.append(f"{k}: {'pass' if v['pass'] else 'fail'}")
            num[f"merge_{k}_pass"] = bool(v["pass"])
    if isinstance(res.get("candidate_ppl"), (int, float)):
        num["merge_candidate_ppl"] = res["candidate_ppl"]
    return " · ".join(rows) or "UNAVAILABLE", num


def mark_count(s: str) -> tuple[int, int]:
    return s.count("✅"), s.count("❌")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.REPO / "M1_TABLE.md"))
    a = ap.parse_args()
    reg = load(common.WORK / "VARIANTS.json") or {}
    # a checkpoint may have probe/equivalence data even if the registry file was rewritten by a
    # later build pass, so the row set is the union of registry, table order and on-disk evidence
    seen = set(reg) | {"base"} | {f.split(".")[0] for f in os.listdir(OPS) if f.endswith(".json")} \
        | {f[:-5] for f in os.listdir(EQ) if f.endswith(".json")} if OPS.exists() and EQ.exists() else set(reg)
    names = [n for n in ORDER if n in seen] + sorted(n for n in seen if n not in ORDER)
    lines = ["# M1 reserve table", "", "Generated by `m1/report.py` from `m1/work/` — do not hand-edit.",
             "",
             "| checkpoint | equivalence | PPL | " + " | ".join(GGUF_TAGS) + " | LoRA r16 | merge | pass/measured |",
             "|---|---|---|" + "---|" * len(GGUF_TAGS) + "---|---|---|"]
    summary = {}
    for n in names:
        eq = load(EQ / f"{n}.json")
        if n == "base":
            verdict, ppl = "reference", (eq or {}).get("metrics", {}).get("ppl_a")
        elif eq:
            m = eq.get("metrics", {})
            verdict = eq.get("verdict", "?")
            ppl = m.get("ppl_b")
            if verdict != "EQUIVALENT":
                verdict += f" (KLD {m.get('kl_mean_nats', float('nan')):.1e}, top1 {m.get('top1_agree', 0):.4f})"
        else:
            verdict, ppl = "NOT VERIFIED", None
        g, gnum = gguf_cells(n)
        ad, anum = adapt_cells(n)
        mg, mnum = merge_cells(n)
        nums = {**gnum, **anum, **mnum}
        verdicts = [v for k, v in nums.items() if k.endswith("_pass") and isinstance(v, bool)]
        p = sum(verdicts)
        f = len(verdicts) - p
        summary[n] = {"note": reg.get(n, {}).get("note", ""), "spec": reg.get(n, {}).get("spec"),
                      "canonicalize": reg.get(n, {}).get("canonicalize"), "equiv": verdict,
                      "ppl": ppl, "gguf": g, "adapt": ad, "merge": mg,
                      "passed": p, "failed": f, "num": nums}
        lines.append(f"| `{n}` | {verdict} | " + (f"{ppl:.3f}" if ppl else "-") + " | "
                     + " | ".join(g[t] for t in GGUF_TAGS) + f" | {ad} | {mg} | {p}/{p + f} |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    common.wjson(common.WORK / "m1_summary.json", summary)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
