#!/usr/bin/env bash
# M1 phase 2 queue: equivalence sweep -> gate -> real surgeries -> table.
# run_m1 refuses to probe any variant that did not pass m1/verify_equiv.py, so this can never
# measure a model that is not function-equivalent to base.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4
export TSX_QUANT_TAGS="q8_0,q5_k_m,q4_k_m"

echo "[queue] $(date +%T) waiting for phase1 equivalence sweep"
while pgrep -f "phase1.sh" >/dev/null; do sleep 20; done

echo "[queue] $(date +%T) verifying the bf16-exact G3 variants"
./m1/phase1.sh g3_pow2 g3_pow2_rep

CORE="base,g1_haar,g1_haar_rep,g2_rand,g3_pow2,g3_pow2_rep,g7_rand,g7_rand_rep,g4_perm,g5_c8,bad_all,bad_all_rep"
EXTRA="bad_all_s2,bad_all_s2_rep,bad_all_s3,bad_all_s3_rep,g2_rand_rep,g5_c8_rep,g6_perm,g3_smooth,g3_smooth_rep,prep_base,g1_svd,g1_svd_rep"

echo "[queue] $(date +%T) full surgery panel: $CORE"
$PY m1/run_m1.py --ops gguf,adapt,merge --variants "$CORE" 2>&1 | tail -30
echo "[queue] $(date +%T) quantization-only replication: $EXTRA"
$PY m1/run_m1.py --ops gguf --variants "$EXTRA" 2>&1 | tail -30

echo "[queue] $(date +%T) report"
$PY m1/report.py --out M1_TABLE.md >/dev/null && echo "[queue] M1_TABLE.md written"
$PY m1/analyze.py >/dev/null 2>&1 && echo "[queue] M1_ANALYSIS.md written"
$PY m1/plot_m1.py --out m1/work/m1_optionality.svg 2>&1 | tail -2
echo "[queue] $(date +%T) done"
