#!/usr/bin/env bash
# M1 phase 2c: task-vector merges over the core checkpoints, with the repaired specialist.
# Runs only when no other GPU user is alive: one 8 GB card, and the probes serialize on
# m1/work/gpu.lock, so overlapping drivers just stall each other mid-cell.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CORE="base,g1_haar,g1_haar_rep,g2_rand,g3_pow2,g3_pow2_rep,g7_rand,g7_rand_rep,g4_perm,g5_c8,bad_all,bad_all_rep"
while pgrep -f "queue_panel.sh|backfill.sh|queue_gaps.sh|queue_exact.sh|merge_probe.py|gguf_probe.py" >/dev/null; do
  sleep 30
done
echo "[merge] gated core: $CORE"
$PY m1/run_m1.py --ops merge --variants "$CORE" 2>&1 | tail -24
$PY m1/retally.py >/dev/null 2>&1
$PY m1/report.py --out M1_TABLE.md >/dev/null 2>&1
echo "[merge] done $(date +%T)"
