#!/usr/bin/env python3
"""M1 phase 2 driver: run real surgery operations on function-equivalent checkpoints.

Pipeline per variant (all of them already passed the equivalence gate in phase 1):

    build artifact -> prove equivalence -> Q4/Q5/Q6/Q8 via llama.cpp
                                  -> bounded LoRA capture
                                  -> linear / TIES merge capture
                                  -> reserve vector -> free the disk

Fail-closed rules:
  * a variant whose equivalence JSON is missing or NOT_EQUIVALENT is never probed —
    a gauge that changed the function is not a control, it is a different model;
  * an operation that could not run is recorded UNAVAILABLE and never counted as PASS;
  * thresholds come from each probe script's own frozen PASS_CONTRACT, echoed in its JSON.

    <venv python> m1/run_m1.py --variants g3_rand,g3_rand_rep --ops gguf,adapt,merge
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import make_variants as mv  # noqa: E402
from common import log  # noqa: E402

PY = sys.executable
OPS_DIR = common.WORK / "ops"
EQ_DIR = common.WORK / "equiv"

# op name -> (script, extra args, json path inside result["results"] that holds pass/bool)
OPS = {
    "gguf": ("gguf_probe.py",
             ["--tags", os.environ.get("TSX_QUANT_TAGS", "q8_0,q5_k_m,q4_k_m")], None),
    "adapt": ("adapt_probe.py", [], None),
    "merge": ("merge_probe.py", [], None),
}


def run_probe(op: str, model_dir: Path, variant: str, extra: list[str]) -> dict:
    script, default_args, _ = OPS[op]
    out = OPS_DIR / f"{variant}.{op}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(common.M1 / script), "--model-dir", str(model_dir), "--out", str(out)]
    if op == "gguf":                       # only the GGUF probe takes an explicit tag
        cmd += ["--tag", variant]
    cmd += default_args + extra
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    t0 = time.time()
    log(f"  [{op}] {' '.join(cmd[-6:])}")
    p = subprocess.run(cmd, cwd=str(common.REPO), env=env, capture_output=True, text=True)
    if out.exists():
        try:
            res = common.rjson(out)
        except Exception as e:                                  # noqa: BLE001
            return {"op": op, "status": "BAD_JSON", "error": str(e), "rc": p.returncode,
                    "stderr": p.stderr[-800:]}
        res["_rc"] = p.returncode
        res["_wall_s"] = round(time.time() - t0, 1)
        return res
    return {"op": op, "status": "UNAVAILABLE", "rc": p.returncode,
            "error": (p.stderr or p.stdout)[-800:], "_wall_s": round(time.time() - t0, 1)}


def equivalence(variant: str) -> dict | None:
    f = EQ_DIR / f"{variant}.json"
    return common.rjson(f) if f.exists() else None


def summarize(variant: str, eq: dict | None, probes: dict) -> dict:
    """Binary optionality vector z(s) from math.md section 3 + measured detail."""
    row: dict = {"variant": variant, "equiv": (eq or {}).get("verdict", "MISSING"),
                 "max_dlogit": (eq or {}).get("metrics", {}).get("max_dlogit"),
                 "flags": (eq or {}).get("flags", []),
                 "kl": (eq or {}).get("metrics", {}).get("kl_mean_nats"),
                 "top1": (eq or {}).get("metrics", {}).get("top1_agree"),
                 "ppl": (eq or {}).get("metrics", {}).get("ppl_b"),
                 "cond_J": (eq or {}).get("cond_b"), "ops": {}, "passes": [], "fails": [],
                 "unavailable": []}
    for op, res in probes.items():
        if res.get("status") in ("UNAVAILABLE", "BAD_JSON") or res.get("error"):
            row["unavailable"].append(op)
            row["ops"][op] = {"status": res.get("status", "ERROR"),
                              "error": str(res.get("error", ""))[:400]}
            continue
        verdicts = {}
        for k, v in (res.get("results") or {}).items():
            if not isinstance(v, dict):
                continue
            if v.get("pass") is not None:      # None means "this run defined the reference",
                verdicts[k] = bool(v["pass"])   # never tally it as a failure
            elif "smallest_passing_alpha" in v:          # merge matrix shape
                verdicts[k] = v["smallest_passing_alpha"] is not None
            elif isinstance(v.get("matrix"), list):
                verdicts[k] = any(r.get("pass") for r in v["matrix"] if isinstance(r, dict))
        row["ops"][op] = {"results": res.get("results"), "pass_contract": res.get("pass_contract"),
                          "cmds": res.get("cmds")}
        if not verdicts:
            row["unavailable"].append(f"{op}(no pass fields)")
        for k, ok in verdicts.items():
            (row["passes"] if ok else row["fails"]).append(f"{op}:{k}")
    row["omega0"] = len(row["passes"]) / max(1, len(row["passes"]) + len(row["fails"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="", help="comma list; empty = registry order")
    ap.add_argument("--ops", default="gguf,adapt,merge")
    ap.add_argument("--keep", action="store_true", help="keep variant dirs on disk")
    ap.add_argument("--out", default=str(common.WORK / "M1_OPS.json"))
    a = ap.parse_args()
    names = ([x.strip() for x in a.variants.split(",") if x.strip()] or list(mv.VARIANTS))
    ops = [x.strip() for x in a.ops.split(",") if x.strip()]
    bad = [o for o in ops if o not in OPS]
    if bad:
        raise SystemExit(f"unknown ops {bad}")
    results = common.rjson(Path(a.out)) if Path(a.out).exists() else {}
    arch = common.read_arch(common.REF_MODEL)
    for v in names:
        # Admission is derived from what exists on disk, not from a run log: a variant is only
        # skipped when every requested cell already has a usable result. Re-running after a
        # partial crash therefore fills holes instead of re-doing work or skipping a variant
        # because a stale tally said it was done (RUNBOOK triage: "summary disagrees with cells").
        pending_ops = []
        for op in ops:
            f = OPS_DIR / f"{v}.{op}.json"
            if f.exists():
                try:
                    d0 = json.loads(f.read_text())
                except Exception:                                     # noqa: BLE001
                    pending_ops.append(op)
                    continue
                if d0.get("results") and not d0.get("error"):
                    continue
                pending_ops.append(op)
            else:
                pending_ops.append(op)
        if not pending_ops:
            log(f"skip {v} (all {len(ops)} cells present)")
            continue
        ops_for_v = pending_ops
        eq = equivalence(v)
        # A missing equivalence record is a prerequisite the driver can satisfy itself, not a
        # reason to skip: verify_equiv costs ~2 min of CPU and every downstream verdict depends
        # on it existing (I5's obligation, discharged automatically).
        d0 = common.REF_MODEL if v == "base" else common.WORK / v
        if eq is None and v != "base" and (d0 / "model.safetensors").exists():
            log(f"  [{v}] no equivalence record -> verifying before probing")
            subprocess.run([PY, str(common.M1 / "verify_equiv.py"), "--b", str(d0),
                            "--ntokens", os.environ.get("TSX_EQ_TOKENS", "4096"),
                            "--out", str(EQ_DIR / f"{v}.json")], cwd=str(common.REPO), check=False)
            eq = equivalence(v)
        if v != "base" and (eq is None or not eq.get("distributional_pass",
                                                     eq.get("verdict") == "EQUIVALENT")):
            log(f"SKIP {v}: equivalence gate not passed "
                f"({'missing json' if eq is None else eq.get('verdict')})")
            results[v] = {"variant": v, "equiv": "MISSING" if eq is None else eq["verdict"],
                          "ops": {}, "passes": [], "fails": [], "unavailable": ops_for_v,
                          "omega0": 0.0}
            common.wjson(Path(a.out), results)
            continue
        d = common.REF_MODEL if v == "base" else common.WORK / v
        if v != "base" and not (d / "model.safetensors").exists():
            subprocess.run([PY, str(common.M1 / "make_variants.py"), "--only", v],
                           cwd=str(common.REPO), check=True)
        probes = {op: run_probe(op, d, v, []) for op in ops_for_v}
        # keep any previously recorded cells for this variant
        prior = (results.get(v) or {}).get("ops") or {}
        probes = {**{k: {"status": "REUSED"} for k in prior if k not in probes}, **probes}
        results[v] = summarize(v, eq or {"verdict": "EQUIVALENT", "metrics": {}, "cond_b": {}},
                               probes)
        common.wjson(Path(a.out), results)
        log(f"  {v}: pass={len(results[v]['passes'])} fail={len(results[v]['fails'])} "
            f"unavailable={results[v]['unavailable']} omega0={results[v]['omega0']:.2f}")
        if not a.keep and v != "base":
            import shutil
            shutil.rmtree(d, ignore_errors=True)
    print(json.dumps({k: {"omega0": r.get("omega0"), "passes": r.get("passes"),
                          "fails": r.get("fails")} for k, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
