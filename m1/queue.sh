#!/usr/bin/env bash
# M1 phase 2 queue: wait for the equivalence sweep, gate on it, then run real surgeries.
# run_m1 refuses to probe a variant that did not pass verify_equiv, so this cannot measure a
# model that is not actually function-equivalent to base.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4
while pgrep -f "phase1.sh" >/dev/null; do sleep 20; done
echo "[queue] phase1 done; verifying the bf16-exact G3 variants"
./m1/phase1.sh g3_pow2 g3_pow2_rep
echo "[queue] surgery queue"
$PY m1/run_m1.py --ops gguf,adapt,merge --variants \
base,g1_haar,g1_haar_rep,g2_rand,g2_rand_rep,g3_pow2,g3_pow2_rep,g7_rand,g7_rand_rep,g4_perm,g6_perm,g5_c8,g5_c8_rep,bad_all,bad_all_rep,prep_base \
--keep 2>&1 | tail -40
echo "[queue] table"
$PY m1/report.py --out M1_TABLE.md >/dev/null && echo "[queue] M1_TABLE.md written"
