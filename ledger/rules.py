"""`ledger/rules.py` — mechanical enforcement of the invariants (I4, I5, I7, I8, K-7).

Closed vocabulary (I7): verdicts pass/fail/unavailable/stale/invalidated, statuses
measured/predicted/unavailable, claim states unsupported|preliminary|controlled|confirmed|
refuted plus the operational qualifiers `blocked`/`partial` (SCHEMA §3 extension, used because
`explain K-9` must honestly report BLOCKED).

Admission checks:
  * I4  — a non-reference cell in verdict pass/fail requires a well-formed `reference_cell`
          whose subject is the calibration artifact and whose environment digest matches.
  * I3  — that reference must share this cell's environment digest; a mismatch REFUSES and
          names both ids.
  * I5  — `explain`-time: a claim with a required-but-missing control is capped at PRELIMINARY
          and the promoting (control) cell is named. Claim capping lives here too.
  * I8  — status:predicted cells carry a basis and are never tallied; status:unavailable
          requires a reason; `null`/UNAVAILABLE is never cast to False or 0.
  * K-7 — a scalar "health" is rejected by the schema with a message pointing at claim K-7.
"""

from __future__ import annotations

from .store import LedgerError
from .env import env_digest

VERDICTS = {"pass", "fail", "unavailable", "stale", "invalidated"}
DECISION_STATUSES = {"measured", "predicted", "unavailable"}
CLAIM_STATES = {
    "unsupported", "preliminary", "controlled", "confirmed", "refuted",
    "blocked",          # operational: required cells cannot exist yet (I8-honest), scope K-9
    "partial",          # operational: some obligations met, a declared partial is the result
}
# I7: a closed claim-state cap for obligations that can never enter the evidence ladder alone.
CONTROLS = ("identity_roundtrip", "null_gauge", "permutation")
HEALTH_SCALAR_KEYS = ("health", "health_score", "reserve_health")
REFERENCE_REQUIRED_PREFIXES = ("quantize.", "adapt.", "merge.")


def _invalidates_of(cell: dict):
    inv = cell.get("invalidates")
    return [inv] if isinstance(inv, str) and inv else []


def superseded_ids(cells) -> set:
    """Ids of cells superseded by a `invalidates` edge (I1: correction = new record + edge).

    Any cell named by another cell's `invalidates` is superseded: it keeps its meaning under
    its own conditions but is excluded from every verdict tally (I8 — an invalidated number is
    not a measured pass or fail).
    """
    out = set()
    for c in cells:
        out.update(_invalidates_of(c))
    return out


def check_cell(cell: dict, ledger=None) -> list:
    """Structural + invariant checks on a `cell` record. Returns a list of violation strings."""
    problems = []
    if cell.get("kind") != "cell":
        problems.append("kind must be 'cell'")
    op = cell.get("op") or {}
    if not isinstance(op, dict) or not op.get("name"):
        problems.append("missing op.name")
    result = cell.get("result") or {}
    status = result.get("status")
    if status not in DECISION_STATUSES:
        problems.append(f"result.status {status!r} not in {sorted(DECISION_STATUSES)}")
    verdict = result.get("verdict")
    if verdict is not None and verdict not in VERDICTS:
        problems.append(f"result.verdict {verdict!r} not in {sorted(VERDICTS)}")

    # K-7: reserve is a vector — a scalar health score is rejected by schema.
    for key in HEALTH_SCALAR_KEYS:
        if key in cell or key in result or any(key in (m or {}) for m in result.get("metrics") or {}):
            problems.append(
                f"scalar health score {key!r} is rejected by schema: reserve is a vector "
                f"(claim K-7); render a typed map or a weighted scalar that names its weights."
            )

    # I8: unavailable must carry a reason; predicted must carry a basis and is never tallied.
    if status == "unavailable" and not result.get("reason"):
        problems.append("status:unavailable requires a result.reason (I8: never omit failure)")
    if status == "predicted":
        basis = cell.get("basis") or result.get("basis")
        if not isinstance(basis, dict) or not basis.get("claim") or not basis.get("static_feature"):
            problems.append(
                "status:predicted requires basis {claim, static_feature}; predicted cells are "
                "never counted in a verdict tally (I8)."
            )

    # I1 invalidates edge: a correction names the cell it supersedes, and that cell must exist.
    inv = cell.get("invalidates")
    if isinstance(inv, str) and inv:
        if ledger is not None and ledger.get("cell", inv) is None:
            problems.append(
                f"invalidates names cell {inv} which does not exist; a correction must supersede "
                f"a real measured cell — the new record is the only change, never an overwrite (I1)."
            )
    elif inv is not None:
        problems.append(f"invalidates must be a cell id or null, got {inv!r}")

    # I4 + I3 apply to surgery operations with external calibration references.
    # Equivalence verification has its gate inside op.spec and does not use reference_cell.
    needs_reference = str(op.get("name", "")).startswith(REFERENCE_REQUIRED_PREFIXES)
    if verdict in ("pass", "fail") and needs_reference:
        ref_id = op.get("reference_cell")
        is_calibration = op.get("role") == "reference_calibration"
        if not ref_id and not is_calibration:
            problems.append(
                f"I4: verdict {verdict} requires op.reference_cell (the calibration it is "
                f"judged against); no cell named — schedule the calibration first."
            )
        elif ref_id is not None and ledger is not None:
            ref = ledger.get("cell", ref_id)
            if ref is None:
                problems.append(
                    f"I4: op{'.'+op['name']} reference_cell {ref_id} does not exist; "
                    f"no non-reference cell may be scheduled before its calibration (I4)."
                )
            else:
                a, b = cell.get("environment", {}), ref.get("environment", {})
                da, db = env_digest(a), env_digest(b)
                if da is None or db is None or da != db:
                    problems.append(
                        f"I3: cell {cell.get('id')} and its reference {ref_id} have differing "
                        f"environment digests ({da} vs {db}); they are not comparable — "
                        f"refuse the join, re-measure under one frozen environment."
                    )
    return problems


def check_artifact(rec: dict, ledger=None) -> list:
    problems = []
    if rec.get("kind") != "artifact":
        problems.append("kind must be 'artifact'")
    features = rec.get("features") or {}
    conv = (features.get("total") or {}).get("convention")
    if conv is None:
        problems.append(
            "artifact.features.total.convention is mandatory: pooled and per-tensor means of "
            "the same blocks differ by up to 5.7% (PIPELINE_FAILURES #11); a number without a "
            "convention is not comparable."
        )
    def _scan(container):
        # scan a feature dict one level deep (features.total / per_family) — the schema has
        # no scalar-health field anywhere, so a nested `health` is just as much a violation.
        for key in HEALTH_SCALAR_KEYS:
            if key in container:
                yield f"scalar health score {key!r} is rejected by schema (claim K-7)."
    for key in HEALTH_SCALAR_KEYS:
        if key in rec:
            problems.append(f"scalar health score {key!r} is rejected by schema (claim K-7).")
    for sub in [features] + [v for v in features.values() if isinstance(v, dict)]:
        problems.extend(_scan(sub))
    return problems


def tally_cells(cells, *, kind: str = "cell") -> dict:
    """Verdict tally over cells. I8: denominators are made of MEASURED cells only; `predicted`
    and `unavailable` are never counted in either direction; a cell superseded by an
    `invalidates` correction is never tallied either (I1/I8). `null` never means False."""
    t = {"pass": 0, "fail": 0, "pass_measured": 0, "fail_measured": 0,
         "predicted": 0, "unavailable": 0, "invalidated": 0, "measured": 0, "total": 0}
    superseded = superseded_ids(cells)
    for c in cells:
        r = c.get("result") or {}
        status = r.get("status")
        verdict = r.get("verdict")
        t["total"] += 1
        if c.get("id") in superseded:
            t["invalidated"] += 1        # superseded edge: recorded, excluded in both directions
            continue
        if status == "measured":
            t["measured"] += 1
            if verdict == "pass":
                t["pass"] += 1
                t["pass_measured"] += 1
            elif verdict == "fail":
                t["fail"] += 1
                t["fail_measured"] += 1
        elif status == "predicted":
            t["predicted"] += 1          # recorded but excluded from the score (I8)
        elif status == "unavailable":
            t["unavailable"] += 1        # recorded, never cast to False (I8)
    t["denominator_measured"] = t["measured"]
    t["numerator_pass"] = t["pass_measured"]
    t["pass_rate_measured"] = (t["pass_measured"] / t["measured"]) if t["measured"] else None
    return t


def claim_cap(claim, obligations) -> dict:
    """I5: derive a claim's capping evidence from its declared obligations.

    Returns {state, missing: [obligation names], capped_by_control: bool,
             promoting_cells: [ids], blocked_by: [ids/descriptions]}.
    Any *required* obligation with no evidence caps the claim at PRELIMINARY (or lower); if the
    claim declares `blocked: true` because cells cannot exist, state is BLOCKED.
    """
    declared = claim.get("state", "unsupported").lower()
    missing = [o.name for o in obligations if o.required and not o.satisfied]
    # Missing control cells are the I5 promoting-cells the explainer must name.
    promoting = sorted(
        pid for o in obligations for pid in (o.promoting_ids or [])
        if o.required and not o.satisfied
    )
    blocked_by = []
    for o in obligations:
        if o.blocked_reason and o.required and not o.satisfied:
            blocked_by.append((o.name, o.blocked_reason))
    if blocked_by or claim.get("blocked"):
        return {
            "state": "blocked",
            "missing": missing,
            "capped_by_control": False,
            "promoting_cells": promoting,
            "blocked_by": blocked_by or (claim.get("blocked") and [("(declared)", "K-9: merge cells running/incomplete; inspector answers merge.linear: UNAVAILABLE — that blank is load-bearing")] or []),
        }
    if missing:
        capped_by_control = any(o.required and not o.satisfied
                                and (o.kind == "control" or o.name.startswith("controls."))
                                for o in obligations)
        any_evidence = any(o.evidence_ids for o in obligations
                           if o.kind not in ("failed_attempt", "refuter"))
        return {
            "state": _cap_from_missing(declared, missing, any_evidence),
            "missing": missing,
            "capped_by_control": capped_by_control,
            "promoting_cells": promoting,
            "blocked_by": [],
            "any_evidence": any_evidence,
        }
    return {"state": declared, "missing": [], "capped_by_control": False,
            "promoting_cells": [], "blocked_by": [], "any_evidence": True}


def _cap_from_missing(declared: str, missing: list, any_evidence: bool) -> str:
    # The evidence ladder: nothing is allowed to assert a status the ledger has not earned.
    if declared == "refuted":
        return "refuted"
    if declared == "unsupported" and not any_evidence:
        return "unsupported"   # no evidence at all: stay UNSUPPORTED, never promote
    return "preliminary"       # required-evidence missing always caps at PRELIMINARY (I5)
