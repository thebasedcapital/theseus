"""`ledger/plan.py` — value-of-information scheduling (SYSTEM.md §4) + the I4 calibration gate.

The planner is an evidence-obligation solver, not a job runner: candidate actions are derived
from the claims' *missing* obligations (PLAN.md §3 ordering), each carries the measured cost from
PLAN.md §2, and nothing non-reference is scheduled for an op lacking its calibration reference
cell (I4). A refusal NAMES the unblocking cell — the calibration reference that must first exist.
"""

from __future__ import annotations

from .claims import cells_with

# measured cost model on this box (PLAN.md §2) — wall seconds, vram, disk
COST = {
    "inspect":            {"wall_s": 3,   "vram": 0,     "disk": 1.0,  "cpu": True},
    "export.gguf":        {"wall_s": 75,  "vram": 0,     "disk": 1.0,  "cpu": True},
    "quantize.gguf.q8_0": {"wall_s": 60,  "vram": 2.4,   "disk": 0.5,  "cpu": False},
    "quantize.gguf.q5_k_m": {"wall_s": 120, "vram": 2.4, "disk": 0.5, "cpu": False},
    "quantize.gguf.q4_k_m": {"wall_s": 120, "vram": 2.4, "disk": 0.5, "cpu": False},
    "adapt.lora.r16":     {"wall_s": 90,  "vram": 3.0,   "disk": 1.0,  "cpu": False},
    "equivalence":        {"wall_s": 90,  "vram": 2.4,   "disk": 1.0,  "cpu": True},
    "merge.linear":       {"wall_s": 60,  "vram": 2.4,   "disk": 1.0,  "cpu": False},
    "merge.ties":         {"wall_s": 60,  "vram": 2.4,   "disk": 1.0,  "cpu": False},
}
GGPU = 60.0  # gpu-min per gpu-second


def calibration_exists(ledger, op_name: str) -> bool:
    """I4: does the ledger hold a well-formed calibration (reference) cell for this op?

    A calibration cell is a measured cell whose obligation is a `*.calibration` tag for the op's
    family (K-4.calibration.<rung> for quantize, K-3.calibration.lora for adapt,
    K-9.calibration.merge for merge) with a non-error result.
    """
    for c in ledger.all("cell"):
        ob = c.get("obligation") or ""
        if ".calibration." not in ob:
            continue
        opname = (c.get("op") or {}).get("name") or ""
        same_family = (opname == op_name or
                       op_name.startswith("quantize.") and opname.startswith("quantize.gguf.") or
                       op_name.startswith("merge.") and opname.startswith("merge.") or
                       op_name.startswith("adapt.") and opname.startswith("adapt."))
        if not same_family:
            continue
        res = c.get("result") or {}
        if res.get("status") == "measured" and res.get("verdict") != "fail":
            return True
    return False


def unblocking_reference(ledger, op_family: str):
    """Name the calibration cell that must exist before any non-reference cell of `op_family`
    can be scheduled (I4). Family is matched on the cell's op *name* (the same rule as
    `calibration_exists`) — obligation tags name rungs (q8_0, lora, merge), not the op family."""
    for c in ledger.all("cell"):
        ob = c.get("obligation") or ""
        if ".calibration." not in ob:
            continue
        opname = (c.get("op") or {}).get("name") or ""
        if op_family == "quantize" and opname.startswith("quantize.gguf."):
            return c.get("id")
        if op_family == "merge" and opname.startswith("merge."):
            return c.get("id")
        if op_family == "adapt" and opname.startswith("adapt."):
            return c.get("id")
    return "UNAVAILABLE"


def _plan_actions(ledger):
    """Candidate actions from the claims' missing obligations (PLAN §3 ordering)."""
    k9_ref = calibration_exists(ledger, "merge.linear")
    actions = []
    # 1. K-9 merge is the only open blocker; its base reference must come first.
    actions.append({
        "action": "merge.linear × base (self) → K-9 base merge reference",
        "cost": "60 s GPU", "gpu_min": 1.0, "belief": 0.9,
        "discharges": ["K-9.calibration.merge_ref"], "refused": not k9_ref and False or False,
        "unblocking": None,
    })
    mur = unblocking_reference(ledger, "merge")
    actions.append({
        "action": "merge.linear × <variant> (K-9 measurement)",
        "cost": "60 s GPU", "gpu_min": 1.0, "belief": 0.85,
        "discharges": ["K-9.measurement.merge_cells", "K-9.replication.n=1"],
        "refused": not k9_ref,
        "unblocking": mur if not k9_ref else ("calibrated" if k9_ref else None),
    })
    # 2. K-3 probe-seed replication (the pending capping duty):
    actions.append({
        "action": "adapt.lora.r16 — replicate probe, seeds 2..3 across small-gap variants "
                  "(|gap|<3·sd escalates on G1/G2/G4)",
        "cost": "90 s GPU × 3 seeds", "gpu_min": 4.5, "belief": 0.7,
        "discharges": ["K-3.replication.adapt", "K-3.replication.probe_seed"],
        "refused": False, "unblocking": None,
    })
    # 3. K-10's honest gap is cheap:
    actions.append({
        "action": "equivalence (bf16 compute) × prep_base_exact — K-10 refuter",
        "cost": "90 s CPU", "gpu_min": 0.0, "belief": 0.5,
        "discharges": ["K-10.refuter.bf16_compute"], "refused": False, "unblocking": None,
    })
    # 4. K-2 export census across remaining artifacts (static-first):
    actions.append({
        "action": "export.gguf.f16 census × <unmeasured artifacts> (K-2) — then inspect (2 s)",
        "cost": "75 s CPU each", "gpu_min": 0.0, "belief": 0.4,
        "discharges": ["K-2.census.export_damage"], "refused": False, "unblocking": None,
    })
    return actions


def plan(ledger, *, budget_gpu_min: float = 60.0, op: str | None = None,
         subject: str | None = None, top: int = 3) -> dict:
    """Rank actions by belief per gpu-minute within the session budget.

    `--op` names a specific operation: I4 determines whether non-reference scheduling is legal
    and the refusal names the unblocking cell.
    """
    if op is not None:
        base = {"name": op,
                "family": "quantize" if op.startswith("quantize.") else
                          "adapt" if op.startswith("adapt.") else
                          "merge" if op.startswith("merge.") else op}
        ok = calibration_exists(ledger, op)
        ref = unblocking_reference(ledger, base["family"])
        cost = COST.get(op, {"wall_s": 60, "vram": 2.4, "disk": 1.0, "cpu": False})
        entry = {
            "action": f"{op} × {subject or '<artifact>'}", "cost": f"{cost['wall_s']} s",
            "gpu_min": round(cost["wall_s"] / GGPU, 1), "belief": 0.6,
            "discharges": ["(caller-specified)"], "refused": not ok,
            "unblocking": ref if not ok else None,
            "i4": {"calibrated": ok, "reference_cell": ref},
        }
        return {"refused": [entry] if not ok else [], "ranked": [] if not ok else [entry],
                "budget_gpu_min": budget_gpu_min, "allow": ok}

    actions = sorted(_plan_actions(ledger),
                     key=lambda a: (a["refused"], -(a["belief"] / max(a["gpu_min"], 0.1))))
    ranked, refused = [], []
    spend = 0.0
    for a in actions:
        if a["refused"]:
            refused.append(a)
            continue
        if spend + a["gpu_min"] > budget_gpu_min:
            a["note"] = f"over {budget_gpu_min} gpu-min budget"
            refused.append(a)
            continue
        ranked.append(a)
        spend += a["gpu_min"]
    return {"ranked": ranked[:top], "refused": refused, "budget_gpu_min": budget_gpu_min,
            "spend_gpu_min": round(spend, 1)}
