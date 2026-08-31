"""`ledger/store.py` — write-once content-addressed store (LedgerSpine slice 3).

Invariant I1: a record id never changes; re-adding identical content is a no-op; a conflicting
add (caller-supplied id that does not match the content hash, or a duplicate `key` with a
different body) is a HARD error naming the existing id.

Directory layout (SCHEMA.md §5):
    <root>/ledger/{artifact,cell,claim,incident}/<id>.json
Records carry `"kind"` inside the body; `id = sha256(canonical_json(body minus id))[:12]`
(sorted keys, compact separators, no trailing whitespace).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

KINDS = ("artifact", "cell", "claim", "incident")
ID_LEN = 12


class LedgerError(Exception):
    """Base class for ledger-domain errors (I1 violations are the dominant use)."""


class IdConflictError(LedgerError):
    """Adding a record whose id/key collides with an existing one."""


def canonical(obj) -> bytes:
    """Deterministic UTF-8 JSON: sorted keys, `,:` separators, no whitespace."""
    if not isinstance(obj, (dict, list)):
        obj = {"value": obj}
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_id(body, *, id_len: int = ID_LEN) -> str:
    """Content address of a record body, computed over all keys except `id`."""
    if not isinstance(body, dict):
        raise LedgerError(f"record body must be a dict, got {type(body).__name__}")
    stripped = {k: v for k, v in body.items() if k != "id"}
    return hashlib.sha256(canonical(stripped)).hexdigest()[:id_len]


def load_record(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise LedgerError(f"unreadable record {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise LedgerError(f"record {path} is not a JSON object")
    return data


class Ledger:
    """Write-once store rooted at `<root>/ledger/`.

    All mutation goes through `add`, which is atomic (O_CREAT|O_EXCL) and idempotent:
    identical content returns `("noop", id)`, a genuine conflict raises IdConflictError.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.dir = {k: self.root / "ledger" / k for k in KINDS}
        self._lock = threading.Lock()
        self._index_cache = {}  # (kind, key_field) -> {key: {"id":..., "body":...}} or None

    # -- path / IO ---------------------------------------------------------

    def path(self, kind: str, rid: str) -> Path:
        if kind not in KINDS:
            raise LedgerError(f"unknown kind {kind!r}; expected one of {KINDS}")
        return self.dir[kind] / f"{rid}.json"

    def get(self, kind: str, rid: str, *, _default=None):
        p = self.path(kind, rid)
        return load_record(p) if p.exists() else _default

    def all(self, kind: str):
        d = self.dir[kind]
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.glob("*.json")):
            rec = load_record(p)
            if not isinstance(rec, dict):
                continue
            rec["_file"] = str(p)  # path is not part of identity
            out.append(rec)
        return out

    def counts(self) -> dict:
        return {k: len(list(self.dir[k].glob("*.json"))) if self.dir[k].is_dir() else 0
                for k in KINDS}

    # -- write-once add ----------------------------------------------------

    def add(self, kind: str, body, *, id: str | None = None, key: str | None = None) -> tuple:
        """Add a record. Returns (id, action) where action in {"added","noop"}.

        `id` (optional) is a caller-supplied id: it MUST equal the content hash or the add is a
        HARD conflict error. `key` (optional) is a human label (claims: "K-3", incidents: "PF-3")
        kept in a per-kind key registry; a second distinct body under the same key is a HARD
        conflict error naming the existing id (I1).
        """
        if body.get("kind") != kind:
            raise LedgerError(f"body kind {body.get('kind')!r} != store dir kind {kind!r}")
        cid = content_id(body)
        if id is not None and id != cid:
            raise IdConflictError(
                f"{kind} add refused: caller-supplied id {id!r} does not match content hash "
                f"{cid!r}; an id other than the content hash would violate write-once (I1). "
                f"Existing id for this content: {cid}"
            )
        record = dict(body)
        record["id"] = cid
        if key is not None:
            record["key"] = key

        with self._lock:
            if key is not None:
                self._guard_key(kind, key, cid)
            action = self._write_once(self.dir[kind], cid, record)
            self._index_cache = {}
            return cid, action

    def _guard_key(self, kind: str, key: str, cid: str) -> None:
        idx = self._index(kind, "key")
        prev = idx.get(key)
        if prev is not None and prev["id"] != cid:
            raise IdConflictError(
                f"{kind} key {key!r} is already claimed by id {prev['id']} (.json at "
                f"{self.path(kind, prev['id'])}); correction must be a new record with a "
                f"different key plus an `invalidates` edge, never an overwrite (I1)."
            )

    def _write_once(self, d: Path, cid: str, record: dict) -> str:
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{cid}.json"
        payload = canonical(record)
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            existing = load_record(p)
            if existing == record:
                return "noop"
            raise IdConflictError(
                f"{p} already exists with id {existing.get('id')} and DIFFERENT content; "
                f"write-once (I1) forbids overwriting an established meaning: existing id "
                f"{existing.get('id')}"
            ) from None
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        return "added"

    # -- lookup ------------------------------------------------------------

    def _index(self, kind: str, key_field: str):
        cache_key = (kind, key_field)
        cached = self._index_cache.get(cache_key)
        if cached is not None:
            return cached
        idx = {}
        for rec in self.all(kind):
            k = rec.get(key_field)
            if k is not None:
                idx[str(k)] = {"id": rec.get("id"), "body": rec}
        self._index_cache[cache_key] = idx
        return idx

    def resolve(self, kind: str, key: str, key_field: str = "key"):
        """Map a human label (e.g. claim "K-3") to the stored record (by key registry)."""
        idx = self._index(kind, key_field)
        hit = idx.get(str(key))
        if hit is None:
            return None
        return hit["body"]

    def find(self, kind: str, **fields):
        """Return records whose top-level fields match all given `fields` (exact equality)."""
        out = []
        for rec in self.all(kind):
            if all(rec.get(k) == v for k, v in fields.items()):
                out.append(rec)
        return out

    # -- integrity ---------------------------------------------------------

    def verify(self) -> list:
        """Re-read every record and recompute its content hash. Returns [(path, error)]."""
        problems = []
        for kind in KINDS:
            for p in sorted(self.dir[kind].glob("*.json")):
                try:
                    rec = load_record(p)
                    if rec.get("kind") != kind:
                        problems.append((str(p), f"kind mismatch: {rec.get('kind')!r}"))
                        continue
                    cid = content_id(rec)
                    if p.name != f"{cid}.json":
                        problems.append(
                            (str(p), f"filename {p.name} != content id {cid}.json")
                        )
                    if rec.get("id") != cid:
                        problems.append(
                            (str(p), f"record id {rec.get('id')!r} != content id {cid}")
                        )
                except LedgerError as exc:
                    problems.append((str(p), str(exc)))
        return problems

    def toplevel(self, name: str) -> Path:
        return self.root / name
