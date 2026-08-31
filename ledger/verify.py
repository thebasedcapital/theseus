#!/usr/bin/env python3
"""`ledger/verify.py` — does every persisted cell name a generator that could have written it?

Motivation (incidents #18, #19). `m3/results.json` recorded a natural-history result whose generator
had never executed: `train_lora_state` called `opt.step()` with no optimizer, and the stored `adapt`
dicts were missing the `capture` / `task_loss_before` / `task_loss_after` keys that function always
returns. Nothing in the repo could have told me that, because nothing checked it. The rule I wrote
for #15 - "acceptance follows executed edges, not labels" - was still being satisfied by reading code
by hand, which is how it slipped twice.

Two independent checks, both cheap and both mechanical:

1. COMMIT AND FILE. A cell's recorded `git_head` must resolve to a real commit, and its recorded
   `script` must exist in that commit's tree. A cell naming a file that was not there when it
   allegedly ran is a fabrication, whatever its numbers look like.

2. SHAPE. Each operation kind has fields it cannot be reported without. An adaptation cell without
   `capture` is the #18 signature exactly: the writer returned a dict with different keys, so it
   was not this writer.

Quarantined records are expected failures listed in QUARANTINE and reported as such, so a clean run
means "nothing new is broken", not "nothing is broken". Violations exit non-zero.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The rule follows the generator named in the cell, which is the claim being verified. A shape
# heuristic over dict contents is fragile: an early revision classified the gguf layout audit as a
# quantization cell and reported a false violation because `status` lives on each per-scheme entry,
# not on the container.
KIND_BY_SCRIPT = {
    "gguf_probe.py": "quantization",
    "adapt_probe.py": "adaptation",
    "seed_replicate.py": "adaptation-summary",
    "verify_equiv.py": "equivalence",
    "export_damage.py": "export",
    "merge_probe.py": "merge",
    "fidelity.py": "fidelity",
    "rescue.py": "repair",
    "make_variants.py": "construction",
}

# Known-bad records, kept deliberately, reported as quarantined rather than as new violations.
QUARANTINE = {
    "m3/results.json": "incident #18 - generator could not execute; retained as a failed-attempt record",
    "m1/work/invalidated/full_model_training": "invalidated first adaptation panel (full-model training)",
    "m1/work/invalidated/unversioned_true_lora_v1": "pre-contract true-LoRA cells",
    "m1/work/M1_OPS.old-f16.json": "pre-amendment f16-export quantization cells (incident #10)",
}

_TREES: dict[str, set[str]] = {}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("git", "-C", str(ROOT)) + args, capture_output=True, text=True, check=False)


def tree_of(sha: str) -> set[str]:
    if sha not in _TREES:
        r = _git("ls-tree", "-r", "--name-only", sha)
        _TREES[sha] = set(r.stdout.split()) if r.returncode == 0 else set()
    return _TREES[sha]


def commit_exists(sha: str) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def _rel(p: Path) -> str:
    """Repo-relative when possible. A cell store under --work outside the repo must still report
    an absolute path rather than raising: relative_to() is not total."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def script_in_tree(script: str, sha: str) -> bool:
    """True if the generator named by a cell existed in the commit the cell says produced it."""
    names = tree_of(sha)
    return bool(script) and any(p == script or p.endswith("/" + script) for p in names)


def iter_cells(work: Path):
    """Yield (relpath, kind, payload) for every persisted cell that claims a generator."""
    for p in sorted(work.rglob("*.json")):
        rel = _rel(p)
        if any(q in rel for q in QUARANTINE):
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for kind, payload in _cells_in(data):
            yield rel, kind, payload


def _cells_in(data, depth=0):
    """Find cell-shaped dicts anywhere in a document, incl. nested per-variant/per-seed maps."""
    out = []
    if depth > 6 or not isinstance(data, dict):
        return out
    if "script" in data and "git_head" in data:
        out.append((KIND_BY_SCRIPT.get(Path(str(data["script"])).name, "cell"), data))
    for v in data.values():
        if isinstance(v, dict):
            out.extend(_cells_in(v, depth + 1))
    return out


def check_quarantined(m3_file: Path) -> list:
    """The #18 signature, asserted rather than remembered: an adapt dict with no capture."""
    bad = []
    try:
        d = json.loads(m3_file.read_text())
    except Exception:
        return bad
    hc = d.get("history_construction") or {}
    for order, cell in hc.items():
        adapt = (cell or {}).get("adapt") if isinstance(cell, dict) else None
        if isinstance(adapt, dict) and "capture" not in adapt:
            bad.append({"file": _rel(m3_file), "subject": f"{order}.adapt",
                        "problem": "adaptation record without 'capture': not written by the "
                                   "current train_lora_state", "status": "QUARANTINE (#18)"})
    return bad


def _need(*fields):
    def check(payload):
        return [f for f in fields if f not in payload]
    return check


def _per_scheme(payload):
    """Quantization/export cells carry one entry per scheme, each with its own status."""
    res = payload.get("results")
    if not isinstance(res, dict) or not res:
        return ["results (per-scheme map)"]
    return [f"results.{k}.status" for k, v in res.items()
            if not (isinstance(v, dict) and "status" in v)]


# An adaptation cell without `capture` is the #18 signature: that key is unconditional in
# train_lora_state's return, so its absence means a different writer produced the record.
VALIDATORS = {
    "quantization": _per_scheme,
    "export": _per_scheme,
    "equivalence": _need("metrics", "gate", "checks"),
    "merge": _need("results"),
}
# 'adaptation' is handled by _check_adaptation because its numbers are nested per record.

CURRENT_ADAPT_CONTRACT = "adapt-v2-true-lora-base-frozen"


def _adapt_records(payload):
    """Adaptation numbers live either at top level (m3 style) or under results.<name> (m1 style)."""
    recs = []
    if "capture" in payload or "task_loss_after" in payload:
        recs.append(("", payload))
    res = payload.get("results")
    if isinstance(res, dict):
        recs += [(k, v) for k, v in res.items() if isinstance(v, dict)]
    return recs


def _check_adaptation(rel, sha, script, payload):
    """Returns (violations, unversioned). capture is unconditional in the v2 writer's return, so a
    v2-tagged record without it means some other code wrote it: the #18 signature."""
    viol, unv = [], []
    for name, rec in _adapt_records(payload):
        where = f"{rel} results.{name}" if name else rel
        cv = rec.get("contract_version")
        if cv is None:
            unv.append({"file": where, "git_head": sha[:12], "script": script,
                        "problem": "adaptation record carries no contract_version, so its "
                                   "comparability to adapt-v2 cells cannot be established",
                        "status": "UNVERSIONED"})
            continue
        if cv != CURRENT_ADAPT_CONTRACT:
            viol.append({"file": where, "git_head": sha[:12], "script": script,
                         "problem": f"contract_version {cv!r} is not the current {CURRENT_ADAPT_CONTRACT!r}",
                         "status": "VIOLATION"})
            continue
        missing = [f for f in ("capture", "task_loss_before", "task_loss_after", "base_frozen")
                   if f not in rec]
        if missing:
            viol.append({"file": where, "git_head": sha[:12], "script": script,
                         "problem": f"v2 adaptation record missing {missing} (#18 signature: a "
                                    "different writer produced this)", "status": "VIOLATION"})
    return viol, unv

def verify(work: Path) -> dict:
    violations, unversioned, seen, checked = [], [], {}, 0
    for rel, kind, payload in iter_cells(work):
        checked += 1
        sha = str(payload.get("git_head") or "")
        script = str(payload.get("script") or "")
        key = (rel, sha, script)
        if key in seen:
            continue
        seen[key] = True
        if not sha or not commit_exists(sha):
            violations.append({"file": rel, "git_head": sha or "(absent)",
                               "problem": "recorded commit does not resolve in this repository",
                               "status": "VIOLATION"})
            continue
        if not script_in_tree(script, sha):
            violations.append({"file": rel, "git_head": sha[:12], "script": script,
                               "problem": "named script is absent from the recorded commit's tree",
                               "status": "VIOLATION"})
            continue
        if kind == "adaptation":
            v, u = _check_adaptation(rel, sha, script, payload)
            violations += v
            unversioned += u
            continue
        missing = VALIDATORS.get(kind, lambda p: [])(payload)
        if missing:
            violations.append({"file": rel, "git_head": sha[:12], "script": script, "kind": kind,
                               "problem": f"missing required fields {missing}", "status": "VIOLATION"})

    m3 = ROOT / "m3" / "results.json"
    quarantined = check_quarantined(m3) if m3.exists() else []
    return {"cells_checked": checked, "unique_generators": len(seen),
            "violations": violations, "unversioned": unversioned, "quarantined": quarantined,
            "verdict": "FAIL" if violations else ("PASS WITH WARNINGS" if unversioned else "PASS")}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="verify-provenance", description=__doc__.splitlines()[0])
    ap.add_argument("--work", default=str(ROOT / "m1" / "work"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rep = verify(Path(a.work))
    if a.json:
        print(json.dumps(rep, indent=2))
        return 1 if rep["violations"] else 0
    print(f"cells checked: {rep['cells_checked']} | distinct generator claims: "
          f"{rep['unique_generators']}")
    for v in rep["violations"]:
        print(f"  VIOLATION {v['file']}: {v['problem']}")
    for u in rep["unversioned"]:
        print(f"  WARNING {u['file']}: {u['problem']}")
    for q in rep["quarantined"]:
        print(f"  quarantined (known, expected): {q['file']} {q['subject']}")
    print("verdict:", rep["verdict"])
    return 1 if rep["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
