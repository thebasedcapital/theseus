#!/usr/bin/env python3
"""M1 ledger: static features (Rust inspector) joined to measured surgery outcomes.

This is the dataset M6 will fit a calibrated predictor on, and the evidence for the claim that
surgery damage is predictable from artifact bytes. Features come from `theseus-inspect`
(zero-dependency safetensors reader, cross-validated against the Python implementation to ~1e-9);
outcomes come from the probe JSONs in m1/work/ops/.

    <venv python> m1/ledger.py [--rebuild] [--bins theseus-inspect path]

`--rebuild` regenerates variant dirs that the panel already freed (each ~15 s, one at a time,
deleted again afterwards) so the ledger can cover every measured cell.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

INSPECT = common.REPO / "inspect" / "target" / "release" / "theseus-inspect"
FEATS = ("q4_block_mse", "dyn_range_log10", "row_energy_imbalance", "amax_over_rms",
         "frac_below_f16_normal")


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None, len(x)
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return (num / (dx * dy) if dx and dy else None), n


def inspect_dir(d: Path, tag: str) -> dict:
    out = common.WORK / f"inspect.{tag}.json"
    r = subprocess.run([str(INSPECT), str(d / "model.safetensors"), "--json", str(out)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode or not out.exists():
        raise RuntimeError(f"inspect failed for {tag}: {r.stderr[-300:]}")
    j = json.loads(out.read_text())
    out.unlink(missing_ok=True)
    tot = j["total"]
    fam = {k: v["q4_block_mse"] for k, v in j["families"].items()}
    return {**{f"tot_{k}": tot.get(k) for k in FEATS},
            "worst_dyn_range": max(v["dyn_range_log10"] for v in j["families"].values()),
            "worst_below_f16": max(v["frac_below_f16_normal"] for v in j["families"].values()),
            "per_family_q4": fam}


def outcomes(tag: str, base_out: dict) -> dict:
    o = {}
    g = common.WORK / "ops" / f"{tag}.gguf.json"
    if g.exists():
        r = json.loads(g.read_text()).get("results") or {}
        for t in ("q8_0", "q5_k_m", "q4_k_m"):
            v = r.get(t) or {}
            if isinstance(v.get("kl_mean"), (int, float)):
                o[f"{t}_kld"] = v["kl_mean"]
                if base_out.get(f"{t}_kld"):
                    o[f"{t}_kld_x_base"] = v["kl_mean"] / base_out[f"{t}_kld"]
            if isinstance(v.get("rel_dppl"), (int, float)):
                o[f"{t}_rel_dppl"] = v["rel_dppl"]
            if isinstance(v.get("prefix_agree"), (int, float)):
                o[f"{t}_prefix_agree"] = v["prefix_agree"]
            if v.get("pass") is not None:
                o[f"{t}_pass"] = bool(v["pass"])
    a = common.WORK / "ops" / f"{tag}.adapt.json"
    if a.exists():
        v = (json.loads(a.read_text()).get("results") or {}).get("variant") or {}
        for k in ("capture", "protected_dppl", "pass"):
            if isinstance(v.get(k), (int, float, bool)):
                o[f"lora_{k}"] = v[k]
    m = common.WORK / "ops" / f"{tag}.merge.json"
    if m.exists():
        r = json.loads(m.read_text()).get("results") or {}
        for k in ("linear", "ties"):
            if k in r:
                o[f"merge_{k}_pass"] = r[k].get("smallest_passing_alpha") is not None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", default=str(common.WORK / "m1_ledger.json"))
    a = ap.parse_args()
    reg = common.rjson(common.WORK / "VARIANTS.json") if (common.WORK / "VARIANTS.json").exists() else {}
    tags = ([t.strip() for t in a.tags.split(",") if t.strip()] or sorted(reg))
    ledger = {}
    base_out = outcomes("base", {})
    for tag in tags:
        d = common.REF_MODEL if tag == "base" else common.WORK / tag
        built = False
        if not (d / "model.safetensors").exists():
            if not a.rebuild or tag == "base":
                log(f"{tag:16s} skipped (no artifact; use --rebuild)")
                continue
            subprocess.run([sys.executable, str(common.M1 / "make_variants.py"), "--only", tag,
                            "--out", str(common.WORK)], check=False, capture_output=True)
            built = True
        if not (d / "model.safetensors").exists():
            log(f"{tag:16s} skipped (build failed)")
            continue
        try:
            feats = inspect_dir(d, tag)
        except Exception as e:                              # noqa: BLE001
            log(f"{tag:16s} inspect error {e}")
            continue
        out = outcomes(tag, base_out)
        ledger[tag] = {"features": feats, "outcomes": out,
                       "equiv": (common.rjson(common.WORK / "equiv" / f"{tag}.json")
                                 if (common.WORK / "equiv" / f"{tag}.json").exists() else {})
                       .get("verdict")}
        log(f"{tag:16s} J={feats['tot_q4_block_mse']:.5f} rng={feats['tot_dyn_range_log10']:.2f} "
            f"below_f16={feats['tot_frac_below_f16_normal']:.5f} "
            f"q4x={(out.get('q4_k_m_kld_x_base') or float('nan')):.2f} "
            f"lora={out.get('lora_capture')}")
        if built and tag != "base":
            shutil.rmtree(d, ignore_errors=True)
    common.wjson(Path(a.out), ledger)

    # correlations: each measured outcome vs each static feature, across the ledger
    corr = {}
    pairs = [("q4_k_m_kld_x_base", "Q4_K_M KLD / base"), ("q4_k_m_rel_dppl", "Q4_K_M rel dPPL"),
             ("q8_0_kld_x_base", "Q8_0 KLD / base"), ("lora_capture", "LoRA capture"),
             ("lora_protected_dppl", "LoRA collateral dPPL")]
    for feat in FEATS:
        key = f"tot_{feat}"
        for okey, label in pairs:
            xs = [v["features"][key] for v in ledger.values() if okey in v["outcomes"]]
            ys = [v["outcomes"][okey] for v in ledger.values() if okey in v["outcomes"]]
            rho, n = spearman(xs, ys)
            if rho is not None:
                corr.setdefault(label, {})[feat] = [round(rho, 3), n]
    print("\nSpearman rho(static feature -> measured outcome):")
    labels = list(next(iter(corr.values())).keys()) if corr else []
    if corr:
        print(f"{'outcome':22s}" + "".join(f"{l:>14s}" for l in labels))
        for okey, label in pairs:
            if okey.replace("_x_base", "") in ("",):
                continue
            row = corr.get(label, {})
            if not row:
                continue
            print(f"{label:22s}" + "".join(f"{(str(row[l][0])+' n'+str(row[l][1])):>14s}"
                                           if l in row else f"{'-':>14s}" for l in labels))
    common.wjson(common.WORK / "m1_ledger_corr.json", corr)


if __name__ == "__main__":
    main()
