#!/usr/bin/env bash
# After all surgery drivers finish: rebuild each measured artifact once, meter it with the Rust
# inspector, and join features to outcomes (the M6 dataset). One variant at a time: disk.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4
while pgrep -f "queue_panel.sh|backfill.sh|queue_exact.sh|queue_merge.sh|queue_gaps.sh|merge_probe.py|gguf_probe.py" >/dev/null; do sleep 30; done
$PY m1/ledger.py --rebuild 2>&1 | tail -30
$PY m1/report.py --out M1_TABLE.md >/dev/null 2>&1
$PY m1/analyze.py >/dev/null 2>&1
$PY m1/plot_m1.py --out m1/work/m1_optionality.svg >/dev/null 2>&1
echo "[ledger] done $(date +%T)"
