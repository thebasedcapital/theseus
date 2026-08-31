#!/usr/bin/env bash
# M1 phase 2d: the lattice-exact `prepare` comparison, after every other driver is done.
# bad_all_exact / prep_base_exact use only bf16-lossless repair families (G5, G3, G7 on the
# exponent lattice), so unlike the full canonicalizer their output must stay bit-identical in
# behaviour -- which is the claim `theseus prepare` has to be able to make.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "queue_panel.sh|backfill.sh|merge_probe.py" >/dev/null; do sleep 30; done
TSX_CPU=1 ./m1/phase1.sh bad_all_exact prep_base_exact
$PY m1/run_m1.py --ops gguf,adapt,merge --variants bad_all_exact,prep_base_exact 2>&1 | tail -10
$PY m1/retally.py >/dev/null 2>&1
$PY m1/report.py --out M1_TABLE.md >/dev/null && $PY m1/analyze.py >/dev/null 2>&1
$PY m1/plot_m1.py --out m1/work/m1_optionality.svg | tail -1
echo "[exact] done $(date +%T)"
