#!/usr/bin/env bash
# Re-run any (variant, op) pair that has no probe JSON. Probes are per-(variant,op) files, so a
# crash mid-panel — including ones I cause by editing a script while its next invocation starts —
# costs one op, not the run. Idempotent: existing JSONs are left alone unless --force.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "queue_final.sh|queue_extra.sh|queue_fix.sh" >/dev/null; do sleep 30; done
REF=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987
VARS=$($PY -c "
import sys; sys.path.insert(0,'m1'); import json, os
reg=json.load(open('m1/work/VARIANTS.json')) if os.path.exists('m1/work/VARIANTS.json') else {}
eq={f[:-5] for f in os.listdir('m1/work/equiv') if f.endswith('.json')}
ok=[v for v in sorted(eq|set(reg)) if (json.load(open(f'm1/work/equiv/{v}.json')).get('distributional_pass', json.load(open(f'm1/work/equiv/{v}.json')).get('verdict')=='EQUIVALENT'))]
print(','.join(ok))")
echo "[backfill] gated variants: $VARS"
IFS=',' read -ra A <<< "$VARS"
for v in "${A[@]}"; do
  d=$([ "$v" = base ] && echo "$REF" || echo "m1/work/$v")
  [ -f "$d/model.safetensors" ] || $PY m1/make_variants.py --only "$v" >/dev/null 2>&1
  for op in gguf adapt merge; do
    out="m1/work/ops/$v.$op.json"
    [ -s "$out" ] && $PY -c "
import json,sys; d=json.load(open('$out')); sys.exit(0 if d.get('results') and not d.get('error') else 1)" && continue
    echo "[backfill] $v / $op"
    case $op in
      gguf)  $PY m1/gguf_probe.py --model-dir "$d" --tag "$v" --out "$out" >/dev/null 2>&1 || echo "   FAILED $v gguf";;
      adapt) $PY m1/adapt_probe.py --model-dir "$d" --out "$out" >/dev/null 2>&1 || echo "   FAILED $v adapt";;
      merge) $PY m1/merge_probe.py --model-dir "$d" --out "$out" >/dev/null 2>&1 || echo "   FAILED $v merge";;
    esac
  done
  [ "$v" != base ] && rm -rf "m1/work/$v"
done
$PY m1/report.py --out M1_TABLE.md >/dev/null 2>&1
$PY m1/analyze.py >/dev/null 2>&1
$PY m1/plot_m1.py --out m1/work/m1_optionality.svg | tail -1
echo "[backfill] done $(date +%T)"
