#!/usr/bin/env python3
"""Multi-seed replication of the adaptation probe — error bars for the capture gaps.

M1's exit condition asks for multi-seed replication, and the panel varies the *stress* seed but
holds the *training* seed fixed (1729). A 0.8 pp capture difference between two function-equivalent
checkpoints is only a result if it is larger than the run-to-run spread. This driver measures that
spread by re-running the same probe code across training seeds, without editing the probe file
while the panel is reading it: it imports the module and overrides its seed constant in-process.

    <venv python> m1/seed_replicate.py --variants base,g1_haar,g2_rand,g4_perm --seeds 1729,23,44

Writes m1/work/seed_replicate.json with per (variant, seed) capture for every lr, and a summary of
within-variant spread vs between-variant gap.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import adapt_probe as AP  # noqa: E402
from common import log  # noqa: E402
CONTRACT = {"version": "adapt-v2-true-lora-base-frozen", "base_frozen": True,
            "gap_threshold_sd": 3}


def _head() -> str:
    """Commit the seed actually ran under. Seed records were previously anonymous, which let a
    panel average measurements taken under different code: the base-reference drift logged as
    incident #20 moves capture by points, far past the 2e-3 equivalence slack, so a mean across
    commits is not an estimate of anything. Stamping it makes the mixture detectable."""
    import subprocess
    return subprocess.check_output(["git", "-C", str(common.REPO), "rev-parse", "--short", "HEAD"],
                                   text=True).strip()


def mixed_provenance(out: dict, variants) -> list:
    """Variants whose seeds were recorded under more than one commit.

    Extracted from main() so it can be tested: the first version of this guard lived inline in the
    summary loop, which meant the only way to exercise it was a real multi-hour GPU run, and an
    untested guard is decoration. An unstamped seed is its own provenance class rather than a
    wildcard - it predates the field, so treating None as compatible with everything would let
    exactly the mixture this exists to catch pass silently.
    """
    mixed = []
    for v in variants:
        seeds = (out.get(v) or {}).get("seeds") or {}
        heads = {(s.get("git_head") or "unstamped") for s in seeds.values() if isinstance(s, dict)}
        if len(heads) > 1:
            mixed.append({"variant": v, "commits": sorted(heads), "n_seeds": len(seeds)})
    return mixed


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--variants", default="base,g1_haar,g1_haar_rep,g2_rand,g4_perm")
    ap_.add_argument("--seeds", default="1729,23,44")
    ap_.add_argument("--out", default=str(common.WORK / "seed_replicate.json"))
    a = ap_.parse_args()
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    dev = common.pick_device(2.4)
    out = common.rjson(Path(a.out)) if Path(a.out).exists() else {}
    if out.get("_contract") != CONTRACT:
        out = {"_contract": CONTRACT}
    tok = common.load_tokenizer()
    train = AP.make_data(tok, AP.examples(AP.TRAIN_N, AP.RULE_SEED))
    held = AP.make_data(tok, AP.examples(AP.HELDOUT_N, AP.RULE_SEED, offset=AP.TRAIN_N))

    with common.lock("gpu"):
        for v in variants:
            d = common.REF_MODEL if v == "base" else common.WORK / v
            built = False
            if not (d / "model.safetensors").exists():
                import subprocess
                subprocess.run([sys.executable, str(common.M1 / "make_variants.py"), "--only", v,
                                "--out", str(common.WORK)], check=False, capture_output=True)
                built = True
            if not (d / "model.safetensors").exists():
                log(f"{v:14s} skipped (no artifact)")
                continue
            entry = out.setdefault(v, {"device": dev, "seeds": {}, "capture_ref_seed": AP.SEED})
            before, ppl_before = AP.base_metrics(d, tok, train, held, dev)
            entry["task_loss_before"] = before
            for sd in seeds:
                if str(sd) in entry["seeds"]:
                    log(f"{v:14s} seed {sd:5d} cached under {CONTRACT['version']}")
                    continue
                AP.SEED = sd                      # in-process override; module untouched
                rows = []
                for lr in AP.LR_GRID:
                    after, el = AP.train_once(d, tok, train, held, lr, dev)
                    rows.append({"lr": lr, "task_loss_after": after,
                                 "capture": (before - after) / before, "runtime_s": round(el, 1)})
                best = min(rows, key=lambda r: r["task_loss_after"])
                entry["seeds"][str(sd)] = {"grid": rows, "capture": best["capture"],
                                           "selected_lr": best["lr"], "git_head": _head()}
                log(f"{v:14s} seed {sd:5d} capture {best['capture']:.4f} (lr {best['lr']})")
            AP.SEED = seeds[0]
            caps = [entry["seeds"][s]["capture"] for s in entry["seeds"]]
            entry["capture_mean"] = round(st.mean(caps), 5)
            entry["capture_spread"] = (round(min(caps), 5), round(max(caps), 5))
            entry["capture_stdev"] = round(st.pstdev(caps), 5) if len(caps) > 1 else None
            common.wjson(Path(a.out), out)
            if built and v != "base":
                import shutil
                shutil.rmtree(d, ignore_errors=True)

    base = out.get("base", {}).get("capture_mean")
    measured = sorted(v for v, e in out.items()
                      if not v.startswith("_") and isinstance(e, dict) and "capture_mean" in e)
    print("\nseed replication summary (capture mean [min,max]):")
    for v in measured:
        e = out[v]
        gap = None if base is None else e["capture_mean"] - base
        suffix = f" gap vs base={gap:+.4f}" if gap is not None else ""
        print(f"  {v:14s} {e['capture_mean']:.4f} {e['capture_spread']} "
              f"sd={e['capture_stdev']}{suffix}")
    # I3 at seed granularity: a variant whose seeds were recorded under different commits cannot
    # contribute a comparable mean. Flag it, and keep it out of the cross-variant spread that
    # defines the 3-sigma bar, rather than silently averaging it.
    mixed = mixed_provenance(out, measured)
    if mixed:
        print("\nMIXED-PROVENANCE VARIANTS (seeds span multiple commits; means are not comparable):")
        for m in mixed:
            print(f"  {m['variant']:14} commits {m['commits']} over {m['n_seeds']} seeds")
        out["_summary_mixed_warning"] = mixed
    clean = {m["variant"] for m in mixed}
    measured_homog = [v for v in measured if v not in clean]
    gaps = [(v, out[v]["capture_mean"] - base, out[v]["capture_stdev"])
            for v in measured_homog if base is not None]
    spread = max([s for _, _, s in gaps if s] or [0.0])
    real = [(v, g) for v, g, _ in gaps if abs(g) > 3 * max(spread, 1e-4)]
    print(f"\nlargest within-variant sd = {spread:.4f}; "
          f"gaps exceeding 3 sd: {real if real else 'none'}")
    out["_summary"] = {"base_capture": base, "max_within_variant_sd": spread,
                       "threshold_sd": 3,
                       "gaps_beyond_3sd": [[v, round(g, 5)] for v, g in real]}
    out["_contract"] = CONTRACT
    common.wjson(Path(a.out), out)


if __name__ == "__main__":
    main()
