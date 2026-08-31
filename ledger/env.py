"""`ledger/env.py` — environment digest (I3), part of a cell's identity.

Digest is computed over the *conditions* a measurement was taken under:
code snapshot, tool versions (llama.cpp / torch / transformers), export outtype (never guessed),
compute dtype, corpus byte-range + hash, seqlen, chunk policy, contract version, threshold
version. Geometry from SYSTEM.md §3 I3 and the schema `cell.environment` example.

Two cells with different digests are NOT comparable: any join that spans differing digests must
REFUSE and name the ids (`--allow-mixed-env` stamps the output instead).

`environment.unknown: true` (imported legacy numbers whose conditions cannot be recovered) gets
digest `None` and is excluded from every comparison.
"""

from __future__ import annotations

import hashlib

from .store import canonical, content_id, LedgerError

# Keys that participate in the digest (in schema order). Absent keys are dropped and do not
# contribute: a null condition is simply not part of what we compare on.
ENV_CORE_KEYS = (
    "code_snapshot",
    "llama_cpp",
    "torch",
    "transformers",
    "export_outtype",
    "compute_dtype",
    "corpus",  # {"file","byte_range","sha256_16"}
    "seqlen",
    "chunk_policy",
    "contract_version",
    "threshold_version",
)

DIGEST_LEN = 12


def env_core(env: dict) -> dict:
    if not isinstance(env, dict):
        raise LedgerError(f"environment must be a dict, got {type(env).__name__}")
    core = {}
    for k in ENV_CORE_KEYS:
        v = env.get(k)
        if v is not None:
            core[k] = v
    return core


def env_digest(env: dict) -> str | None:
    """12-hex digest of the conditions. `None` when conditions are unknown (I8: never guess)."""
    if env.get("unknown"):
        return None
    core = env_core(env)
    if not core:
        raise LedgerError("environment has no comparable fields and is not marked unknown")
    return hashlib.sha256(canonical(core)).hexdigest()[:DIGEST_LEN]


def environments_comparable(env_a: dict, env_b: dict) -> bool:
    a = env_digest(env_a)
    b = env_digest(env_b)
    if a is None or b is None:
        return False
    return a == b


def mixed_environments(cells, *, allow_mixed_env: bool = False, kind: str = "cells"):
    """Given a list of cell records, return (ok, digest_set, offenders).

    `ok` is False when more than one distinct comparable digest is present or any env is
    unknown. The refusal names the cell ids. `allow_mixed_env` returns a stamped result instead.
    """
    seen = {}
    for c in cells:
        env = c.get("environment") or {}
        d = env_digest(env)
        if d is None:
            seen.setdefault(None, []).append(c.get("id"))
        else:
            seen.setdefault(d, []).append(c.get("id"))
    if len(seen) <= 1 and None not in seen:
        return True, list(seen), []
    digests = sorted((k if k else "UNKNOWN") for k in seen)
    offenders = sorted(i for ids in seen.values() for i in ids if i)
    if allow_mixed_env:
        return True, digests, offenders
    raise LedgerError(
        f"I3 mixed-environment refusal: joining {kind} spans {len(seen)} distinct environment "
        f"digests {digests}; cells are not comparable. Cell ids: {offenders}. "
        f"Re-run with --allow-mixed-env to stamp the output as mixed-env."
    )
