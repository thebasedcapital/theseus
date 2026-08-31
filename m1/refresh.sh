#!/usr/bin/env bash
# Regenerate the M1 artifacts every 10 min so partial results are always on disk and committable.
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
while pgrep -f "queue_final.sh|queue_extra.sh" >/dev/null; do
  $PY m1/report.py --out M1_TABLE.md >/dev/null 2>&1
  $PY m1/analyze.py >/dev/null 2>&1
  $PY m1/plot_m1.py --out m1/work/m1_optionality.svg >/dev/null 2>&1
  sleep 600
done
$PY m1/report.py --out M1_TABLE.md >/dev/null 2>&1
$PY m1/analyze.py >/dev/null 2>&1
echo "[refresh] final artifacts regenerated $(date +%T)"
