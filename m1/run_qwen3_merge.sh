#!/usr/bin/env bash
# Qwen3 merge-reserve panel (K-9). Serial and self-cleaning: 6.1 GB free on a shared host cannot
# hold the specialist plus two 1.2 GB variants at once, so each candidate is built, probed and
# deleted in turn. merge_probe trains the specialist once and caches it under $W/specialist with
# its verified marker, exactly as the Qwen2 panel does.
set -uo pipefail
cd /home/admin/theseus
REF=/home/admin/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
W="$PWD/m1/work-qwen3"
PY=/home/admin/counterpoint/.venv/bin/python
export THESEUS_REF_MODEL="$REF" THESEUS_WORK="$W"

for tag in base g3_pow2 g3_pow2_rep; do
  echo "=== $tag ==="
  dir="$REF"; [ "$tag" != base ] && dir="$W/$tag"
  if [ "$tag" != base ] && [ ! -f "$dir/model.safetensors" ]; then
    timeout 600 "$PY" m1/make_variants.py --only "$tag" --out "$W" >/tmp/mq_$tag.log 2>&1 || {
      echo "  build failed: $(tail -1 /tmp/mq_$tag.log | cut -c1-90)"; continue; }
  fi
  # Calibration found by m1/calibrate_specialist.py: Qwen2's 600 steps at lr 3e-4 breaches the
  # collateral gate on Qwen3 (rule-holdout ppl 45.46 vs 42.44 allowed). Adapter geometry is kept
  # identical to Qwen2 (rank 32, alpha/rank = 1) because rank changes merge arithmetic; only the
  # optimisation budget is softened. rule 0.0194, ppl 29.17 against a 42.44 ceiling.
  timeout 3000 "$PY" m1/merge_probe.py --model-dir "$dir" --out "$W/$tag.merge.json" \
      --steps 150 --rank 32 --alpha 32 --lr 1e-4 >/tmp/mp_$tag.log 2>&1
  rc=$?
  "$PY" - "$tag" "$W/$tag.merge.json" "$rc" <<'PYIN'
import json, sys
tag, out, rc = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    d = json.load(open(out)); r = d.get("results", {})
    if "error" in d and d["error"]:
        print(f"  ERROR {d['error'].get('type')}: {str(d['error'].get('message'))[:110]}"); sys.exit()
    for op in ("linear", "ties"):
        blk = r.get(op) or {}
        rows = blk.get("matrix") or []
        ok = [m["alpha"] for m in rows if m.get("pass")]
        worst = max((m["ppl_ratio"] for m in rows), default=None)
        print(f"  {op:7} passing alphas={ok or 'none'} smallest_passing={blk.get('smallest_passing_alpha')}"
              f" worst ppl_ratio={round(worst,2) if worst else None}")
    print(f"  specialist_provenance={r.get('specialist_provenance')} rc={rc}")
except FileNotFoundError:
    print(f"  no cell written (rc={rc})")
PYIN
  [ "$tag" != base ] && rm -rf "$dir"
done
echo; du -sh "$W"; df -h /home/admin | tail -1 | awk '{print "disk free:",$4}'
