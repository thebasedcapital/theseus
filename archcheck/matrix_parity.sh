#!/usr/bin/env bash
# Theseus ships two static binaries. If they ever disagree about what is known, one of them is
# lying - the same class of drift as incident #19 (a duplicate implementation that changed the
# experiment). Compare their preflight verdicts operation by operation.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOB="${1:-$(ls "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/blobs/*88c14255* 2>/dev/null | head -1)}"
[ -n "$BLOB" ] || { echo "no artifact to compare; pass one as \$1"; exit 2; }
for m in scan inspect; do
  bin="$ROOT/$m/target/release/theseus-$m"
  [ -x "$bin" ] || { echo "build first: cargo build --release --manifest-path $m/Cargo.toml"; exit 2; }
done
"$ROOT/scan/target/release/theseus-scan" preflight "$BLOB" 2>/dev/null \
  | grep -E '^(quantize|export|adapt|merge)' | awk '{print $1, $2}' | sort > /tmp/_parity_scan.txt
"$ROOT/inspect/target/release/theseus-inspect" preflight "$BLOB" 2>/dev/null \
  | grep -E '^(quantize|export|adapt|merge)' | awk '{print $1, $2}' | sort > /tmp/_parity_inspect.txt
if diff -u /tmp/_parity_scan.txt /tmp/_parity_inspect.txt; then
  echo "MATRIX PARITY OK ($(wc -l < /tmp/_parity_scan.txt) operations agree)"
  exit 0
fi
echo "MATRIX PARITY FAILED: the two binaries disagree about a verdict"; exit 1
