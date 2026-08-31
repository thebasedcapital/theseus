#!/usr/bin/env python3
"""M1 -> passport: one machine-readable lifecycle record per checkpoint (ROADMAP B7 + B8).

Every claim in the file carries a status, and status is not optional:

    MEASURED     a probe ran on this artifact and produced numbers
    PREDICTED    a static statistic implies something, no surgery was run for it
    UNAVAILABLE  no evidence; never reported as pass or fail

Provenance is recorded from the build manifests (gauge family, mode, seed, whether the
embedding tie was broken, whether rms_norm_eps was rewritten), so a reader can reconstruct the
artifact from `base` + this file. Nothing here asserts lineage it did not observe.

    <venv python> m1/passport.py [--variants a,b] [--out-dir m1/work/passport]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

UNCLAIMED = [
    "no natural post-training history was run for this artifact (ROADMAP A5/M3)",
    "one model, one scale, base (non-instruct) weights",
    "reserve thresholds are reference-relative and calibrated on this checkpoint family only",
    "static risk flags in the inspector are provisional (n=2 measured contrast per rule)",
]


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:                                     # noqa: BLE001
        return None


def reserve_entry(status: str, **kw) -> dict:
    return {"status": status, **kw}


def build(tag: str, reg: dict, base_eq: dict | None) -> dict:
    entry = reg.get(tag, {})
    ops_dir, eq_dir = common.WORK / "ops", common.WORK / "equiv"
    eq = load(eq_dir / f"{tag}.json")

    ident = {"name": tag, "dir": entry.get("dir"), "sha256_16": entry.get("sha256"),
             "bytes": entry.get("bytes"), "untied": entry.get("untied"),
             "built_from": "Qwen/Qwen2.5-0.5B (bf16, tied)" if tag != "base" else "reference"}

    prov = {"spec": entry.get("spec"), "canonicalize": entry.get("canonicalize"),
            "note": entry.get("note")}
    g = entry.get("gauge")
    if g:
        prov["transforms"] = g.get("transforms")
        prov["config_patch"] = g.get("config_patch")
    c = entry.get("canon")
    if c:
        prov["repairs"] = c.get("canonicalize")

    current = {"status": "MEASURED" if eq else "UNAVAILABLE"}
    if eq:
        m = eq.get("metrics", {})
        current.update({"vs_base": {"max_abs_dlogit": m.get("max_dlogit"),
                                    "kl_mean_nats": m.get("kl_mean_nats"),
                                    "top1_agree": m.get("top1_agree"),
                                    "ppl_base": m.get("ppl_a"), "ppl_this": m.get("ppl_b"),
                                    "gate": eq.get("gate"), "verdict": eq.get("verdict"),
                                    "flags": eq.get("flags", [])}})

    static = {"status": "PREDICTED"}
    if eq and eq.get("cond_a") and eq.get("cond_b"):
        mean = lambda d: sum(d.values()) / len(d)
        static.update({"J_total": round(mean(eq["cond_b"]), 6),
                       "J_reference": round(mean(eq["cond_a"]), 6),
                       "gauge_debt": round(mean(eq["cond_b"]) - mean(eq["cond_a"]), 6),
                       "per_family_J": {k: round(v, 6) for k, v in eq["cond_b"].items()}})
    led = load(common.WORK / "m1_ledger.json")
    if led and tag in led:
        static["inspector"] = led[tag]["features"]
        static["inspector_status"] = "MEASURED" if eq else "PREDICTED"

    reserve = {}
    gg = load(ops_dir / f"{tag}.gguf.json")
    if gg and gg.get("results"):
        r = gg["results"]
        for t, v in r.items():
            if not isinstance(v, dict) or t == "f16":
                continue
            if v.get("pass") is None and v.get("role") != "reference_calibration":
                reserve[f"quantize.gguf.{t}"] = reserve_entry("UNAVAILABLE",
                                                              reason=v.get("reason", "no verdict"))
                continue
            reserve[f"quantize.gguf.{t}"] = reserve_entry(
                "MEASURED",
                ok=(True if v.get("pass") is None else bool(v.get("pass"))),
                ppl=v.get("ppl"), ppl_reference=v.get("ppl_f16"),
                rel_dppl=v.get("rel_dppl"), kl_mean_nats=v.get("kl_mean"),
                kl_p999=v.get("kl_p999"), size_mb=v.get("size_mb"),
                greedy_prefix_agree=v.get("prefix_agree"),
                measured_against=gg.get("quant_ref"), export=gg.get("export"))
    ad = load(ops_dir / f"{tag}.adapt.json")
    if ad and (ad.get("results") or {}).get("variant"):
        v = ad["results"]["variant"]
        reserve["adapt.lora.r16"] = reserve_entry(
            "MEASURED", ok=bool(v.get("pass")), capture=v.get("capture"),
            capture_reference=v.get("capture_ref"), task_loss=[v.get("task_loss_before"),
                                                               v.get("task_loss_after")],
            protected_dppl=v.get("protected_dppl"),
            protected_dppl_reference=v.get("protected_dppl_ref"), selected_lr=v.get("selected_lr"),
            contract=ad["results"].get("pass_contract"), runtime_s=v.get("runtime_s"))
    mg = load(ops_dir / f"{tag}.merge.json")
    if mg and (mg.get("results") or {}):
        r = mg["results"]
        for k in ("linear", "ties"):
            if k in r:
                v = r[k]
                ok = v.get("smallest_passing_alpha") is not None
                reserve[f"merge.{k}"] = reserve_entry("MEASURED", ok=ok,
                                                      best_alpha=v.get("smallest_passing_alpha"),
                                                      matrix=v.get("matrix"))
    measured = [k for k, v in reserve.items() if v["status"] == "MEASURED"]
    omega0 = (sum(1 for k in measured if reserve[k].get("ok")) / len(measured)) if measured else None

    return {"schema": "theseus.passport/0.1", "written_utc":
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "identity": ident, "provenance": prov, "current_behavior": current,
            "static_features": static, "reserve": reserve,
            "coverage": {"measured_ops": sorted(measured),
                         "unavailable_ops": sorted(k for k, v in reserve.items()
                                                   if v["status"] == "UNAVAILABLE"),
                         "optionality_fraction": omega0},
            "environment": {"llama_cpp": (gg or {}).get("versions", {}).get("llama_cpp"),
                            "torch": (gg or {}).get("torch"), "git_head": common.REPO and _head()},
            "not_claimed": UNCLAIMED}


def _head() -> str:
    import subprocess
    return subprocess.run(["git", "-C", str(common.REPO), "rev-parse", "--short", "HEAD"],
                          text=True, capture_output=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="")
    ap.add_argument("--out-dir", default=str(common.WORK / "passport"))
    a = ap.parse_args()
    reg = common.rjson(common.WORK / "VARIANTS.json") if (common.WORK / "VARIANTS.json").exists() else {}
    tags = [t.strip() for t in a.variants.split(",") if t.strip()] or sorted(reg)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_eq = load(common.WORK / "equiv" / "base.json")
    index = {}
    for tag in tags:
        p = build(tag, reg, base_eq)
        (out_dir / f"{tag}.passport.json").write_text(json.dumps(p, indent=2, sort_keys=True) + "\n")
        index[tag] = {"measured": len(p["coverage"]["measured_ops"]),
                      "ok": sum(1 for k in p["coverage"]["measured_ops"]
                                if p["reserve"][k].get("ok")),
                      "unavailable": len(p["coverage"]["unavailable_ops"]),
                      "optionality": p["coverage"]["optionality_fraction"]}
        print(f"{tag:16s} measured {index[tag]['measured']:2d} "
              f"ok {index[tag]['ok']:2d} unavailable {index[tag]['unavailable']:2d} "
              f"Omega0={index[tag]['optionality'] if index[tag]['optionality'] is None else round(index[tag]['optionality'],3)}")
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"passports -> {out_dir}")


if __name__ == "__main__":
    main()
