"""Theseus ledger spine — the write-once record store behind the `theseus` CLI.

Owned by LedgerSpine (slice 3). Implements the L0-L4 tower of SYSTEM.md:
`.theseus/ledger/{artifact,cell,claim,incident}/<id>.json` records, content-addressed by
sha256(canonical body minus id)[:12]. Runtime contract: Python stdlib only, and this package
never imports `m1/` (the live pipeline). Reading `m1/work/*.json` is fine.
"""

__version__ = "0.1.0"
