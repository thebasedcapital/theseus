#!/usr/bin/env bash
# M1 phase 1: exactness gate on the real Qwen2.5-0.5B.
# build -> verify vs base -> record J -> free the disk (variants are rebuildable in ~30 s).
set -uo pipefail
cd /home/admin/theseus
PY="${THESEUS_PY:-/home/admin/counterpoint/.venv/bin/python}"
export TSX_THREADS="${TSX_THREADS:-4}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TSX_CPU=1   # the surgery probes own the GPU; verification is CPU-only
for v in "$@"; do
  [ "$v" = "base" ] && { echo "== control: base vs itself (harness floor)"; \
      $PY m1/verify_equiv.py --ntokens "${NTOK:-4096}" --out m1/work/equiv/base.json >/dev/null 2>&1; \
      echo "base rc=$?"; continue; }
  if [ ! -f "m1/work/$v/model.safetensors" ]; then
    $PY m1/make_variants.py --only "$v" >/dev/null 2>&1 || { echo "BUILD FAIL $v"; continue; }
  fi
  if [ -f "m1/work/equiv/$v.json" ]; then echo "skip $v"; continue; fi
  echo "== verify $v"
  $PY m1/verify_equiv.py --b "m1/work/$v" --ntokens "${NTOK:-4096}" \
      --out "m1/work/equiv/$v.json" >/dev/null 2>"m1/work/equiv/$v.err"
  rc=$?
  $PY - "$v" "$rc" <<'PY'
import json, sys
v, rc = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(f"m1/work/equiv/{v}.json"))
    m = d["metrics"]
    print(f"{v:16s} rc={rc} dlogit={m['max_dlogit']:.2e} kl={m['kl_mean_nats']:.2e} "
          f"top1={m['top1_agree']:.5f} ppl={m['ppl_b']:.4f} (base {m['ppl_a']:.4f}) "
          f"dev={m['device']} {d['verdict']}")
except Exception as e:
    print(f"{v:16s} rc={rc} NO JSON: {e}", open(f"m1/work/equiv/{v}.err").read()[-400:])
PY
  [ "${KEEP:-}" = "1" ] || rm -rf "m1/work/$v"
done
