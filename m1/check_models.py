#!/usr/bin/env python3
"""Assert the external weights behind this repo's evidence are present and byte-identical.

A HuggingFace cache is shared, project-external state: on 2026-08-31 it was deleted wholesale by
another process mid-panel, so three variants measured and seven died on a missing config.json.
Nothing in the repo declared that dependency, so the failure looked like a bug in the pipeline.
This is the declaration and the guard.

  python m1/check_models.py            # verify; exit 1 if anything is missing or substituted
  python m1/check_models.py --fix      # restore pinned snapshots, then verify
  python m1/check_models.py --print-hash Qwen/Qwen3-0.6B-Base
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "data" / "MODELS.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_dir(repo: str, revision: str) -> Path:
    org, name = repo.split("/")
    return (Path.home() / ".cache" / "huggingface" / "hub"
            / f"models--{org}--{name}" / "snapshots" / revision)


def check(name: str, spec: dict) -> list[str]:
    """Return problems; empty list means this model is usable as an evidence source."""
    problems = []
    d = snapshot_dir(spec["repo"], spec["revision"])
    weights = d / spec["weight_file"]
    if not d.is_dir():
        problems.append(f"{name}: snapshot absent at {d} (cache wiped?)")
        return problems
    for required in (spec["weight_file"], "config.json"):
        if not (d / required).is_file():
            problems.append(f"{name}: {required} missing from the pinned snapshot")
    if problems:
        return problems
    expected = spec.get("sha256")
    got = sha256(weights)
    if expected and got != expected:
        problems.append(f"{name}: WEIGHT SUBSTITUTION - {weights.name} is {got}, "
                        f"manifest says {expected}. Every cell derived from it is suspect.")
    elif not expected:
        print(f"  {name:10} sha256={got}  (unpinned in MODELS.json)")
    return problems


def restore(specs: dict) -> None:
    from huggingface_hub import snapshot_download
    for name, spec in specs.items():
        print(f"fetching {name}: {spec['repo']}@{spec['revision'][:12]} ...", flush=True)
        snapshot_download(spec["repo"], revision=spec["revision"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="download the pinned snapshots first")
    ap.add_argument("--print-hash", metavar="REPO", help="print a repo's weight sha256 and exit")
    a = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    specs = manifest["models"]

    if a.print_hash:
        spec = next((s for s in specs.values() if s["repo"] == a.print_hash), None)
        d = snapshot_dir(spec["repo"], spec["revision"]) if spec else None
        if not d or not (d / (spec["weight_file"] if spec else "")).is_file():
            print(f"{a.print_hash} not in cache at a pinned revision; run --fix first", file=sys.stderr)
            return 2
        print(sha256(d / spec["weight_file"]))
        return 0

    if a.fix:
        restore(specs)

    bad = []
    for name, spec in specs.items():
        p = check(name, spec)
        if p:
            bad.extend(p)
        else:
            print(f"  {name:10} OK  {spec['repo']}@{spec['revision'][:12]}")
    if bad:
        print("\nREFUSING TO MEASURE:")
        for line in bad:
            print("  " + line)
        print(f"\nRestore with: {MANIFEST.parent.parent.name}/check_models.py --fix")
        return 1
    print("model identity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
