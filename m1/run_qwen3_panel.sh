#!/usr/bin/env bash
# First MEASURED Qwen3 panel: build one gauge at a time, certify equivalence, delete.
#
# Disk is the binding constraint here (6.1 GB free, shared host), and four 1.2 GB variants would
# not fit even before GGUF exports, so the loop is deliberately serial and self-cleaning.
# G2 is included on purpose: it must now REFUSE on a QK-norm architecture rather than silently
# produce a false EXACT cell - that was incident #2's whole point.
set -uo pipefail
cd /home/admin/theseus
REF=/home/admin/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd
W="$PWD/m1/work-qwen3"
# project-local interpreter; override with THESEUS_PY
PY="${THESEUS_PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="${THESEUS_PY:-python3}"
export THESEUS_REF_MODEL="$REF" THESEUS_WORK="$W"
mkdir -p "$W/equiv"
pass=0; refused=0; failed=0

for v in g1_haar g3_pow2 g5_c8 g7_rand g2_rand; do
  echo "--- $v ---"
  if ! timeout 900 "$PY" m1/make_variants.py --only "$v" --out "$W" >/tmp/mv_$v.log 2>&1; then
    if grep -qiE 'G2 is not exact here|QK-norm' /tmp/mv_$v.log; then
      echo "  REFUSED as designed: $(grep -oiE 'G2 is not exact here[^\"]*' /tmp/mv_$v.log | head -1 | cut -c1-72)"
      refused=$((refused+1))
    else
      echo "  BUILD FAILED (unexpected): $(tail -2 /tmp/mv_$v.log | tr '\n' ' ' | cut -c1-100)"
      failed=$((failed+1))
    fi
    rm -rf "$W/$v"; continue
  fi
  [ -f "$W/$v/model.safetensors" ] || { echo "  no artifact produced"; failed=$((failed+1)); continue; }
  timeout 1200 "$PY" m1/verify_equiv.py --a "$REF" --b "$W/$v" --ntokens 2048 --seqlen 512 \
      --out "$W/equiv/$v.json" >/dev/null 2>&1
  rc=$?
  "$PY" - "$v" "$W/equiv/$v.json" "$rc" <<'PYIN'
import json,sys
v,out,rc=sys.argv[1],sys.argv[2],int(sys.argv[3])
try:
    d=json.load(open(out)); m=d["metrics"]
    print(f"  verdict={d.get('verdict')} max|dlogit|={m['max_dlogit']:.3e} KL={m['kl_mean_nats']:.3e} "
          f"top1={m['top1_agree']:.5f} rc={rc}")
except Exception as e:
    print(f"  no cell written ({type(e).__name__}); rc={rc}")
PYIN
  rm -rf "$W/$v"
done

echo
echo "kept artifact dirs: $(ls -d "$W"/*/ 2>/dev/null | wc -l) (expect 0)"
df -h /home/admin | tail -1 | awk '{print "disk free:",$4}'
