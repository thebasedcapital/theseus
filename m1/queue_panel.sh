#!/usr/bin/env bash
# M1 phase 2 (resumed): quantization + bounded LoRA over the certified checkpoints.
# Merge is deliberately excluded and driven separately once m1/merge_probe.py's specialist
# passes its own sanity gate (see M1_RESULTS.md; the first specialist was broken).
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "merge_probe.py" >/dev/null; do echo "[panel] waiting for merge fix $(date +%T)"; sleep 20; done
CORE="base,g1_haar,g1_haar_rep,g2_rand,g3_pow2,g3_pow2_rep,g7_rand,g7_rand_rep,g4_perm,g5_c8,bad_all,bad_all_rep"
echo "[panel] starting quant+LoRA panel: $CORE"
$PY m1/run_m1.py --ops gguf,adapt --variants "$CORE" 2>&1 | tail -40
$PY m1/retally.py >/dev/null 2>&1
$PY m1/report.py --out M1_TABLE.md >/dev/null && $PY m1/analyze.py >/dev/null 2>&1
$PY m1/plot_m1.py --out m1/work/m1_optionality.svg | tail -1
echo "[panel] quant+LoRA done $(date +%T)"
