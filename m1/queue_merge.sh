#!/usr/bin/env bash
# M1 phase 2c: task-vector merges over every gated checkpoint, using the repaired specialist.
# Run only after m1/merge_probe.py passes its own specialist gate (base numbers must be sane)
# and after the quant+LoRA panel is finished, so the single GPU is not oversubscribed.
set -uo pipefail
cd /home/admin/theseus
PY=/home/admin/counterpoint/.venv/bin/python
export TSX_THREADS=4 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VARS=$($PY -c "
import json,sys,os; sys.path.insert(0,'m1')
ok=[]
for f in sorted(os.listdir('m1/work/equiv')):
    if not f.endswith('.json'): continue
    d=json.load(open('m1/work/equiv/'+f))
    if d.get('distributional_pass', d.get('verdict')=='EQUIVALENT'): ok.append(f[:-5])
print(','.join(v for v in ok if v in {'base','g1_haar','g1_haar_rep','g2_rand','g3_pow2','g3_pow2_rep','g7_rand','g7_rand_rep','g4_perm','g5_c8','bad_all','bad_all_rep','prep_base_exact','bad_all_exact'}))")
echo "[merge] gated: $VARS"
$PY m1/run_m1.py --ops merge --variants "$VARS" 2>&1 | tail -24
$PY m1/retally.py >/dev/null 2>&1
$PY m1/report.py --out M1_TABLE.md >/dev/null && $PY m1/analyze.py >/dev/null 2>&1
echo "[merge] done $(date +%T)"
