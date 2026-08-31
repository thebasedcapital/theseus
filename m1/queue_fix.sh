#!/usr/bin/env bash
# Re-run the base merge probe with the fixed density trim, then refresh artifacts.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "queue_final.sh|queue_extra.sh" >/dev/null; do sleep 30; done
rm -f m1/work/ops/base.merge.json
$PY m1/merge_probe.py --model-dir ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 \
   --out m1/work/ops/base.merge.json 2>&1 | tail -3
$PY m1/report.py --out M1_TABLE.md >/dev/null 2>&1
$PY m1/analyze.py >/dev/null 2>&1
$PY m1/plot_m1.py --out m1/work/m1_optionality.svg | tail -1
echo "[queue_fix] done $(date +%T)"
