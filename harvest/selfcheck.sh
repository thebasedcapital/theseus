#!/bin/bash
# HarvestLineage (harvest/selfcheck.sh): idempotency + integrity check for the harvest cache.
# Wraps lineage.py --selfcheck and additionally proves a re-run changes nothing (no dup lines).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/usr/bin/python3}"
cd "$HERE" || exit 1

fail=0

# 1. integrity: every line parses, ids unique, kind enum, edges resolve-or-dangle, cache <= 1 GiB
"$PY" lineage.py --selfcheck || fail=1

# 2. idempotent re-run of the builder (must be offline thanks to the cache) produces the same
#    manifest/edges byte-for-byte — proves no duplicate lines accrue.
h1_m=$(sha256sum cache/manifest.jsonl | cut -d' ' -f1)
h1_e=$(sha256sum cache/edges.jsonl  | cut -d' ' -f1) 2>/dev/null || h1_e=-
"$PY" lineage.py list >/dev/null 2>&1 || { echo "FAIL: re-run of lineage.py list errored"; fail=1; }
h2_m=$(sha256sum cache/manifest.jsonl | cut -d' ' -f1)
h2_e=$(sha256sum cache/edges.jsonl  | cut -d' ' -f1) 2>/dev/null || h2_e=-
if [ "$h1_m" != "$h2_m" ] || [ "$h1_e" != "$h2_e" ]; then
    echo "FAIL: manifest/edges changed across an idempotent re-run (h1=$h1_m h2=$h2_m)"
    fail=1
else
    echo "idempotency OK: manifest/edges byte-identical across re-run"
fi

# 3. hard cap: cache directory must stay under 1 GiB
sz=$(du -sb cache | awk '{print $1}')
budget=$(( 1 << 30 ))
if [ "$sz" -gt "$budget" ]; then
    echo "FAIL: cache $sz bytes > 1 GiB"
    fail=1
else
    echo "cache size OK: $(( sz / 1048576 )) MB (< 1 GiB)"
fi

# 4. nothing lives outside cache/ and the two scripts
extra=$(find . -mindepth 1 -not -path './cache*' -not -path './__pycache__*' \
            -not -name 'lineage.py' -not -name 'selfcheck.sh' \
            -not -name 'README.md' -not -name '.gitignore' | head -20)
if [ -n "$extra" ]; then
    echo "FAIL: files outside the owned layout:"; echo "$extra"
    fail=1
fi

exit "$fail"
