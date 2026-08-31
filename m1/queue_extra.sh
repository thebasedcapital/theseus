#!/usr/bin/env bash
# M1 phase 2b: the lossless-prepare comparison, after the main panel.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "queue_final.sh" >/dev/null; do sleep 30; done
TSX_CPU=1 ./m1/phase1.sh bad_all_exact prep_base_exact
$PY m1/run_m1.py --ops gguf,adapt,merge --variants bad_all_exact,prep_base_exact 2>&1 | tail -12
$PY m1/report.py --out M1_TABLE.md >/dev/null && $PY m1/analyze.py >/dev/null 2>&1
$PY m1/plot_m1.py --out m1/work/m1_optionality.svg | tail -1
echo "[queue_extra] done $(date +%T)"
