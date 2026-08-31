#!/usr/bin/env python3
"""Pre-registration snapshot of the artifact-only conditioning predictions.

Writes a JSON that maps each *verified* variant to its static quantization-conditioning
numbers (mean J over the seven weight families, and the debt J(variant) - J(base)) plus the
equivalence verdict. This file is the prediction side of the M1 experiment: it is produced
from the checkpoint bytes only, before any surgery result exists, so when the GGUF/LoRA/merge
numbers land, agreement or disagreement cannot be back-fitted.

    <venv python> m1/predict.py [--out m1/work/PREDICTIONS.json]

Do not overwrite a committed snapshot; write a new file and keep the old one as the record.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.WORK / "PREDICTIONS_new.json"))
    a = ap.parse_args()
    snap = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_head": __import__("subprocess").run(
                ["git", "-C", str(common.REPO), "rev-parse", "HEAD"],
                text=True, capture_output=True).stdout.strip(),
            "definition": "J = mean over weight families of sum_blocks(amax^2*n)/(12*(2^(b-1)-1)^2*sum w^2), b=4, block=32",
            "variants": {}}
    for f in sorted((common.WORK / "equiv").glob("*.json")):
        d = json.loads(f.read_text())
        ca, cb = d.get("cond_a") or {}, d.get("cond_b") or {}
        if not ca or not cb:
            continue
        mean = lambda x: sum(x.values()) / len(x)
        snap["variants"][f.stem] = {
            "equiv": d.get("verdict"), "max_dlogit": d.get("metrics", {}).get("max_dlogit"),
            "kl_mean_nats": d.get("metrics", {}).get("kl_mean_nats"),
            "top1_agree": d.get("metrics", {}).get("top1_agree"),
            "J_base": round(mean(ca), 6), "J_var": round(mean(cb), 6),
            "debt": round(mean(cb) - mean(ca), 6),
            "per_tensor": {k: round(v, 6) for k, v in cb.items()},
            "prediction": ("Q4 damage expected" if mean(cb) - mean(ca) > 1e-3
                           else "quantization-neutral expected"),
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
    for k, v in sorted(snap["variants"].items(), key=lambda kv: -kv[1]["debt"]):
        print(f"{k:16s} debt {v['debt']:+.5f}  {v['prediction']:28s} {v['equiv']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
