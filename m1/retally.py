#!/usr/bin/env python3
"""Rebuild m1/work/M1_OPS.json tallies from the persisted per-op probe JSONs.

The driver records its own pass/fail tally while it runs, which goes stale whenever a probe is
re-run or repaired afterwards (e.g. a `pass: null` reference row once tallied as a failure).
The per-op JSONs are the source of truth, so the summary is derived from them on demand.

    <venv python> m1/retally.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OPS, EQ = common.WORK / "ops", common.WORK / "equiv"


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:                                     # noqa: BLE001
        return None


def main():
    tags = sorted({f.name.split(".")[0] for f in OPS.glob("*.json")}) if OPS.exists() else []
    out = {}
    for tag in tags:
        row = {"variant": tag, "passes": [], "fails": [], "unavailable": [], "flags": [],
               "equiv": None, "ops": {}}
        eq = load(EQ / f"{tag}.json")
        if tag == "base":
            row["equiv"] = "reference"
        elif eq:
            row["equiv"] = eq.get("verdict")
            row["flags"] = eq.get("flags", [])
            row["max_dlogit"] = (eq.get("metrics") or {}).get("max_dlogit")
            row["kl"] = (eq.get("metrics") or {}).get("kl_mean_nats")
            row["top1"] = (eq.get("metrics") or {}).get("top1_agree")
        else:
            row["equiv"] = "NOT VERIFIED"
        for op in ("gguf", "adapt", "merge"):
            d = load(OPS / f"{tag}.{op}.json")
            if not d or d.get("error") or d.get("status") in ("FAILED", "UNAVAILABLE", "BAD_JSON"):
                row["unavailable"].append(op)
                row["ops"][op] = {"status": (d or {}).get("status", "MISSING"),
                                  "error": str((d or {}).get("error", ""))[:200]}
                continue
            res = d.get("results") or {}
            verdicts = {}
            for k, v in res.items():
                if not isinstance(v, dict):
                    continue
                if v.get("pass") is True or (v.get("pass") is None
                                             and v.get("role") == "reference_calibration"):
                    verdicts[k] = True
                elif v.get("pass") is False:
                    verdicts[k] = False
                elif v.get("smallest_passing_alpha") is not None:
                    verdicts[k] = True
                elif "smallest_passing_alpha" in v:
                    verdicts[k] = False
            row["ops"][op] = {"results": res, "pass_contract": d.get("pass_contract")
                              or d.get("results", {}).get("pass_contract")}
            for k, ok in verdicts.items():
                (row["passes"] if ok else row["fails"]).append(f"{op}:{k}")
            if not verdicts:
                row["unavailable"].append(f"{op}(no verdicts)")
        n = len(row["passes"]) + len(row["fails"])
        row["omega0"] = len(row["passes"]) / n if n else 0.0
        out[tag] = row
    common.wjson(common.WORK / "M1_OPS.json", out)
    for k, v in sorted(out.items()):
        print(f"{k:16s} {str(v['equiv']):20s} pass {len(v['passes'])} fail {len(v['fails'])} "
              f"unavail {v['unavailable']} Ω0={v['omega0']:.2f}")


if __name__ == "__main__":
    main()
