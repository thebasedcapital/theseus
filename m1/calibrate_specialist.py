#!/usr/bin/env python3
"""Search for a specialist calibration that clears merge_probe's gate on a new architecture.

Why this exists: on Qwen3-0.6B-Base the Qwen2-calibrated specialist (600 steps, rank 32, lr 3e-4)
learns the key:value rule cleanly (rule loss 0.0717 against a 0.962 ceiling) but degrades collateral
perplexity from 12.004 to 45.46, so merge_probe correctly refuses and K-9 cannot be tested there.
The refusal is right; what is missing is a calibration that satisfies BOTH gate conditions.

The sweep drives merge_probe's module globals instead of forking its training loop. That is
deliberate: a second copy of the trainer is exactly how incident #19 happened, where a duplicate
silently inherited AdamW's default weight decay and changed the experiment.

Data composition mirrors merge_probe.main() line for line so a passing configuration is the same
measurement the panel will later make.
"""
from __future__ import annotations

import argparse, json, shutil, sys, time

import torch
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common            # noqa: E402
import merge_probe as MP  # noqa: E402

# Each entry: steps, rank, alpha, lr. Ordered from mildest deviation from the Qwen2 recipe.
GRID = [(150, 16, 16, 3e-4), (300, 16, 16, 3e-4), (150, 8, 8, 3e-4), (300, 8, 8, 3e-4),
        (600, 8, 8, 1e-4), (150, 32, 32, 1e-4), (600, 16, 16, 1e-4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(common.WORK / "specialist_calibration.json"))
    ap.add_argument("--stop-on-first-pass", action="store_true", default=True)
    ap.add_argument("--full-grid", dest="stop_on_first_pass", action="store_false")
    a = ap.parse_args()

    rows = []
    with common.lock("gpu"):
        while common.pick_device(3.2) != "cuda":
            common.log("waiting for gpu"); time.sleep(30)
        device = "cuda"
        tok = common.load_tokenizer(common.REF_MODEL)
        ex = MP.examples(MP.TRAIN_N + MP.HELDOUT_N)
        train = MP.make_data(tok, ex[:MP.TRAIN_N])
        held = MP.make_data(tok, ex[MP.TRAIN_N:])
        MP.HELD_DATA = held
        bm = common.load_model(common.REF_MODEL, dtype=torch.bfloat16, device=device)
        base_rule_loss, base_ppl = MP.model_metrics(bm, held, device)
        del bm
        common.release(device)
        ceiling = 1.5 * base_ppl
        print(f"base rule_loss={base_rule_loss:.4f} (need <{0.5*base_rule_loss:.4f}) "
              f"ppl={base_ppl:.4f} (need <= {ceiling:.4f})", flush=True)

        for steps, rank, alpha, lr in GRID:
            # Wipe any cached specialist so each config trains fresh and disk stays bounded.
            shutil.rmtree(MP.SPECIALIST_DIR, ignore_errors=True)
            MP.STEPS, MP.RANK, MP.ALPHA, MP.LR = steps, rank, alpha, lr
            t0 = time.perf_counter()
            try:
                quality = MP.train_specialist(tok, train, held, device, base_rule_loss, base_ppl)
                q, p = quality["rule_loss"], quality.get("eval_ppl")
                ok = (q < 0.5 * base_rule_loss) and (p is not None and p <= ceiling)
                rows.append({"steps": steps, "rank": rank, "alpha": alpha, "lr": lr,
                             "rule_loss": q, "eval_ppl": p, "gate_pass": bool(ok),
                             "wall_s": round(time.perf_counter() - t0, 1)})
                print(f"  steps={steps:4} rank={rank:3} lr={lr:g}  rule={q:.4f} ppl={p:.2f}  "
                      f"{'PASS' if ok else 'FAIL'}  ({rows[-1]['wall_s']}s)", flush=True)
                if ok and a.stop_on_first_pass:
                    break
            except RuntimeError as e:
                msg = str(e)
                pl = msg.split("eval_ppl=")[-1].split()[0] if "eval_ppl=" in msg else None
                rows.append({"steps": steps, "rank": rank, "alpha": alpha, "lr": lr,
                             "refused": msg[:160], "eval_ppl": float(pl) if pl else None,
                             "gate_pass": False, "wall_s": round(time.perf_counter() - t0, 1)})
                print(f"  steps={steps:4} rank={rank:3} lr={lr:g}  REFUSED ppl={pl}"
                      f"  ({rows[-1]['wall_s']}s)", flush=True)
            finally:
                shutil.rmtree(MP.SPECIALIST_DIR, ignore_errors=True)

    passing = [r for r in rows if r["gate_pass"]]
    out = {"architecture": common.REF_MODEL.name, "base_rule_loss": base_rule_loss,
           "base_ppl": base_ppl, "ppl_ceiling": ceiling,
           "gate": "rule_loss < 0.5*base_rule_loss AND eval_ppl <= 1.5*base_ppl",
           "configs": rows, "passing": passing,
           "chosen": passing[0] if passing else None,
           "note": "trains via merge_probe.train_specialist with module globals overridden; no "
                   "forked trainer (incident #19). Specialist directory removed after each trial."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {a.out}")
    print("PASSING CONFIGS:", [(r['steps'], r['rank'], r['lr']) for r in passing] or "none")
    return 0 if passing else 1


if __name__ == "__main__":
    sys.exit(main())
