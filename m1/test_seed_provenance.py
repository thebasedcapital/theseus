#!/usr/bin/env python3
"""Guard tests for seed-panel provenance (incident #20 follow-up).

A cross-commit mean is not an estimate of anything: the base-reference drift moved capture by
points, orders of magnitude past the 2e-3 equivalence slack. These assert the detector fires on a
mixture and stays quiet on a clean panel, without needing a GPU run to find out.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_replicate import mixed_provenance  # noqa: E402

fails = []


def check(name, got, want):
    if got != want:
        fails.append(f"{name}: got {got!r} want {want!r}")


# 1. mixture of commits must be reported
out = {"g3_pow2": {"seeds": {"1": {"capture": 0.04, "git_head": "aaaaaaa"},
                             "2": {"capture": 0.21, "git_head": "bbbbbbb"}}}}
check("detects two commits", [m["variant"] for m in mixed_provenance(out, ["g3_pow2"])],
      ["g3_pow2"])

# 2. homogeneous panel stays quiet
out2 = {"base": {"seeds": {str(i): {"capture": 0.9, "git_head": "aaaaaaa"} for i in range(3)}}}
check("clean panel", mixed_provenance(out2, ["base"]), [])

# 3. unstamped seeds are a provenance class, not a wildcard - this is the case that would
#    silently pass if None were treated as compatible with everything
out3 = {"g7_rand": {"seeds": {"1": {"capture": 0.19},
                              "2": {"capture": 0.21, "git_head": "aaaaaaa"}}}}
check("unstamped vs stamped is a mixture",
      [m["variant"] for m in mixed_provenance(out3, ["g7_rand"])], ["g7_rand"])

# 4. all-unstamped is internally consistent (one class), so no false alarm
out4 = {"v": {"seeds": {"1": {"capture": 0.1}, "2": {"capture": 0.2}}}}
check("all unstamped is quiet", mixed_provenance(out4, ["v"]), [])

# 5. malformed / missing records must not raise
check("missing variant", mixed_provenance({}, ["absent"]), [])
check("non-dict seed", mixed_provenance({"v": {"seeds": {"1": "junk"}}}, ["v"]), [])

# 6. commit list and seed count are reported for the audit trail
m = mixed_provenance(out, ["g3_pow2"])[0]
check("commits listed", m["commits"], ["aaaaaaa", "bbbbbbb"])
check("seed count", m["n_seeds"], 2)

print("\n".join(f"FAIL {f}" for f in fails) if fails else
      "seed provenance guard: 7/7 checks pass")
sys.exit(1 if fails else 0)
