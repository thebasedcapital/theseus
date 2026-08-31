#!/usr/bin/env bash
# M1 driver — ONE scheduler process, sequential phases.
#
# Replaces the queue_panel/queue_merge/queue_gaps/queue_exact chain: four "drivers" whose
# wait-predicates named each other produced a two-cycle deadlock (merge waited for gaps, gaps
# waited for merge) and idled the GPU for six hours. That is incident #13 in
# m1/PIPELINE_FAILURES.md, and SYSTEM.md I6's answer is structural: one scheduler, no predicates,
# admission derived from on-disk cell state.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root, no hardcoded path
PY="${THESEUS_PY:-/home/admin/counterpoint/.venv/bin/python}"
export TSX_THREADS=4 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CORE="base,g1_haar,g1_haar_rep,g2_rand,g3_pow2,g3_pow2_rep,g7_rand,g7_rand_rep,g4_perm,g5_c8,bad_all,bad_all_rep,bad_all_exact,prep_base_exact"
LOG=m1/work/drive.log
echo "[drive] start $(date +%F_%T)"

echo "[drive] phase 1: fill every missing cell (equivalence-gated)"
$PY m1/run_m1.py --ops gguf,adapt,merge --variants "$CORE" >> $LOG 2>&1
$PY m1/retally.py >> $LOG 2>&1
$PY m1/report.py --out M1_TABLE.md >/dev/null 2>&1
$PY m1/passport.py >> $LOG 2>&1

echo "[drive] phase 2: probe-seed replication for the adaptation gaps"
$PY m1/seed_replicate.py --variants base,g1_haar,g1_haar_rep,g2_rand,g4_perm,g5_c8 >> $LOG 2>&1

echo "[drive] phase 3: meter every artifact (static features -> outcome ledger)"
$PY m1/ledger.py --rebuild >> $LOG 2>&1

echo "[drive] phase 4: final renders"
$PY m1/retally.py >> $LOG 2>&1
$PY m1/report.py --out M1_TABLE.md >/dev/null 2>&1
$PY m1/analyze.py > /dev/null 2>&1
$PY m1/passport.py >/dev/null 2>&1
$PY m1/plot_m1.py --out m1/work/m1_optionality.svg >> $LOG 2>&1
echo "[drive] done $(date +%F_%T)"
