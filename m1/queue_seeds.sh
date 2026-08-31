#!/usr/bin/env bash
# M1 phase 2f: error bars for the adaptation gaps, after every other GPU user is done.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TSX_THREADS=4 OMP_NUM_THREADS=4
while pgrep -f "queue_panel.sh|backfill.sh|queue_exact.sh|queue_ledger.sh|merge_probe.py|gguf_probe.py" >/dev/null; do sleep 30; done
$PY m1/seed_replicate.py --variants base,g1_haar,g1_haar_rep,g2_rand,g4_perm,g5_c8 2>&1 | tail -20
echo "[seeds] done $(date +%T)"
