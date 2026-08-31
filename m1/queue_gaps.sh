#!/usr/bin/env bash
# Fill the remaining headline cells (quant for g7_rand / bad_all / bad_all_rep), then rebuild
# every derived artifact. Runs after the merge pass so the GPU is not double-booked.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TSX_THREADS=4 OMP_NUM_THREADS=4
while pgrep -f "queue_merge.sh|merge_probe.py|backfill.sh" >/dev/null; do sleep 30; done
$PY m1/run_m1.py --ops gguf,adapt,merge --variants g7_rand,bad_all,bad_all_rep 2>&1 | tail -12
$PY m1/retally.py >/dev/null 2>&1
$PY m1/report.py --out M1_TABLE.md >/dev/null 2>&1
$PY m1/analyze.py >/dev/null 2>&1
$PY m1/passport.py >/dev/null 2>&1
$PY m1/plot_m1.py --out m1/work/m1_optionality.svg | tail -1
echo "[gaps] done $(date +%T)"
