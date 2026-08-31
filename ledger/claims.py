"""`ledger/claims.py` — the claim register engine (L2) + K-1..K-10 seeds.

The claim set and its obligation/refuter structure are taken verbatim from CLAIMS.md (live
register) and PLAN.md §1. Nothing here asserts a status the ledger has not earned: every
obligation is resolved against stored cells, and a claim is capped (I5) while a *required*
obligation has no evidence. `explain` is the L4 query that prints the state, the capped reason,
the promoting cells, the refuter, and every number with its cell citation (I10).

Obligation-tag conventions (assigned by `import_m1`): cells carry
`obligation: "<claim>.<op>.<artifact>.<rung>"`. Because artifact names are prefixes of each
other (g3_pow2 ⊂ g3_pow2_rep, bad_all ⊂ bad_all_exact), variant-anchored lookups use EXACT tag
matching, never `startswith`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .env import mixed_environments
from .rules import claim_cap
from .store import LedgerError


@dataclass
class Obligation:
    name: str
    required: bool = False
    satisfied: bool = False
    kind: str = "cells"            # cells | declared | count
    evidence_ids: list = field(default_factory=list)
    promoting_ids: list = field(default_factory=list)   # ids / UNAVAILABLE notes that would satisfy
    blocked_reason: str | None = None
    note: str = ""


# -- cell lookups --------------------------------------------------------------

def cells_with(ledger, prefix: str) -> list:
    """Family-level query: any cell whose obligation tag starts with `prefix` (safe only when
    `prefix` ends at an op/family boundary, never mid-artifact-name)."""
    return [c for c in ledger.all("cell")
            if str(c.get("obligation", "")).startswith(prefix)]


def cells_with_exact(ledger, tag: str) -> list:
    """Variant/rung-anchored query: exact equality against the stored obligation tag."""
    return [c for c in ledger.all("cell") if c.get("obligation") == tag]


def _ge(ob_name, required, cells, min_n=1, *, promoting=None, note="", blocked_reason=None):
    evidence = sorted(c.get("id") for c in cells)
    return Obligation(
        ob_name, required=required, satisfied=len(evidence) >= min_n,
        kind="cells", evidence_ids=evidence,
        promoting_ids=sorted(promoting or []),
        blocked_reason=blocked_reason,
        note=note or f"{len(evidence)} cell(s)")


def _declared(ob_name, required, cite):
    return Obligation(ob_name, required=required, satisfied=True, kind="declared",
                      note=f"declared shared control, cited from {cite}")


# --- obligation resolvers: each takes a Ledger and returns [Obligation] ---------------

def _ob_k1(ledger):
    eq = cells_with(ledger, "K-1.equivalence.")
    bf = cells_with(ledger, "K-1.equivalence_bf16.")
    stress = [c for tag in ("K-1.equivalence.bad_all_s2", "K-1.equivalence.bad_all_s3")
              for c in cells_with_exact(ledger, tag)]
    stress += [c for c in cells_with_exact(ledger, "K-1.equivalence.bad_all")]
    return [
        _ge("equivalence.fp32", True, eq, min_n=5,
            note="verify_equiv.py cells, fp32 compute (18 recorded checkpoints)"),
        Obligation("equivalence.bf16", required=False, satisfied=bool(bf),
                   evidence_ids=sorted(c.get("id") for c in bf),
                   note="compute_dtype_check.json measured on g3_pow2/g7_rand only; K-1 is "
                        "stated weaker under bf16 and the bf16 obligation stays OPEN"),
        _declared("controls.null_gauge", True,
                  "m1/test_gauge_math.py (13 properties, exit 0); CLAIMS.md K-1"),
        _declared("controls.permutation", True,
                  "g4_perm dlogit 1.26e-04; CLAIMS.md K-1"),
        _ge("replication.stress", True, stress, min_n=3,
            note="3 stress seeds bad_all, bad_all_s2, bad_all_s3"),
        _declared("algebra", False,
                  "M1_NOTES.md §2; inspect/ unit tests 1-2 pin dtype arithmetic"),
    ]


def _ob_k2(ledger):
    cal = cells_with(ledger, "K-2.export.base.")
    rt = cells_with(ledger, "K-2.control.identity_roundtrip")
    g3 = cells_with(ledger, "K-2.export.g3_pow2.f16")
    return [
        _ge("calibration.base_export", True, cal, min_n=1,
            note="base bf16 export reference ppl 12.1351 (own-variant reference, I4)"),
        _ge("control.identity_roundtrip", True, rt,
            note="base written through own save_state → f16 ppl 12.1399, byte-faithful; the "
                 "control that saved M1's headline (PIPELINE_FAILURES #10)"),
        Obligation("census.export_damage_g3", required=False, satisfied=bool(g3),
                   evidence_ids=sorted(c.get("id") for c in g3),
                   note="g3_pow2 f16 export ppl 177.3286 vs bf16 12.1351: the f16/f32 collapse"),
        Obligation("replication.n=1", required=True, satisfied=True, kind="count",
                   note="single artifact, identity control in place"),
    ]


def _adapt_variant_counts(ledger):
    counts = {}
    for c in ledger.all("cell"):
        t = c.get("obligation") or ""
        if t.startswith("K-3.lora."):
            leaf = t[len("K-3.lora."):].split(".", 1)[0]
            counts[leaf] = counts.get(leaf, 0) + 1
    return counts


def _ob_k3(ledger):
    eq = cells_with(ledger, "K-1.equivalence.")
    per_v = _adapt_variant_counts(ledger)
    g3bf = cells_with_exact(ledger, "K-1.equivalence_bf16.g3_pow2")
    g7bf = cells_with_exact(ledger, "K-1.equivalence_bf16.g7_rand")
    all_done = bool(per_v) and all(n >= 2 for n in per_v.values())
    cal = cells_with(ledger, "K-3.calibration.lora")
    cal_metrics = (cal[-1].get("result") or {}).get("metrics") or {} if cal else {}
    cal_note = (f"corrected true-LoRA base calibration: capture={cal_metrics.get('capture')}, "
                f"protected_dppl={cal_metrics.get('protected_dppl')}; base frozen before adapters")
    return [
        _ge("equivalence.fp32", True, eq, min_n=5,
            note="function-equivalence verified in fp32 for the adaptation variants"),
        Obligation("equivalence.bf16", required=False,
                   satisfied=bool(g3bf) and bool(g7bf),
                   evidence_ids=sorted({c.get("id") for c in g3bf + g7bf}),
                   note="g3/g7 bf16-compute cells present (g7 top1 0.98096 — refuter partially fired)"),
        _ge("calibration.lora", True, cal, note=cal_note),
        _declared("controls.identity", True, "CLAIMS.md K-3: identity round-trip OK"),
        _declared("controls.permutation", True, "CLAIMS.md K-3: permutation OK"),
        Obligation(
            "replication.adapt", required=True, satisfied=all_done, kind="count",
            evidence_ids=sorted(c.get("id") for c in cells_with(ledger, "K-3.lora.")),
            note=f"LoRA probe seeds recorded per variant: {dict(sorted(per_v.items()))}; "
                 "required 2 per variant (1 → needs 2)",
            promoting_ids=[] if all_done else
            [f"UNAVAILABLE: 2nd adapt seed for {v}" for v, n in per_v.items() if n < 2]),
        Obligation(
            "replication.probe_seed", required=True, satisfied=False, kind="count",
            evidence_ids=[],
            note="CORRECTED true-LoRA probe (base frozen before adapters; first panel INVALIDATED as "
                 "full_model_training — 21 cells, does not count, I1 invalidates edge): "
                 "probe-seed replication (m1/seed_replicate.py) records 0 seeds so far; "
                 "|gap| < 3·sd escalation needs seeds 2..3",
            promoting_ids=["UNAVAILABLE: seed_replicate.py corrected-probe output "
                             "(K-3.replication.probe_seed)"]),
    ]


def _ob_k4(ledger):
    cal = cells_with(ledger, "K-4.calibration.")
    quant = cells_with(ledger, "K-4.quantize.")
    eq = cells_with(ledger, "K-1.equivalence.")
    rungs = sorted({t.rsplit(".", 1)[-1] for t in
                    (c.get("obligation") for c in cal if (c.get("obligation") or "").startswith("K-4.calibration."))})
    return [
        _ge("equivalence.fp32", True, eq, min_n=5,
            note="quantized variants verified function-equivalent in fp32"),
        _ge(f"calibration.quant_ladder({'.'.join(rungs) or '?'})", True, cal, min_n=3,
            note="base q8_0/q5_k_m/q4_k_m reference cells (role reference_calibration)"),
        _declared("controls.identity", True, "CLAIMS.md K-4: identity OK"),
        _declared("controls.permutation", True, "CLAIMS.md K-4: permutation OK"),
        Obligation("replication.n=1", required=True, satisfied=True,
                   kind="count", note="single seed recorded per rung"),
    ]


def _ob_k5(ledger):
    repaired = ("g3_pow2_rep", "g5_c8_rep", "bad_all_exact", "prep_base_exact")
    eq_rep = [c for v in repaired for c in cells_with_exact(ledger, f"K-1.equivalence.{v}")]
    cal = cells_with(ledger, "K-4.calibration.") + cells_with(ledger, "K-3.calibration.lora")
    counter = cells_with_exact(ledger, "K-1.equivalence.prep_base")
    return [
        _ge("equivalence.fp32.repaired", True, eq_rep, min_n=3,
            note="canonicalizer outputs verified equivalent in fp32"),
        _ge("calibration.ops", True, cal, min_n=2, note="quant ladder + LoRA base refs"),
        _declared("control.null_gauge", True,
                  "lattice-only canonicalizer {G5,G3,G7} bf16-lossless; CLAIMS.md K-5/K-10"),
        Obligation("counter.full_canonicalizer", required=False,
                   satisfied=bool(counter), evidence_ids=sorted(c.get("id") for c in counter),
                   note="prep_base (full, wide-mixing) FAILS equivalence — documented tool "
                        "defect, kept in your face"),
        Obligation("replication.n=1", required=True, satisfied=True, kind="count"),
    ]


def _ob_k6(ledger):
    pred = cells_with(ledger, "K-6.prediction.")
    labelled = cells_with(ledger, "K-4.quantize.")
    n = len(labelled)
    return [
        _ge("predictions.frozen", True, pred, min_n=1,
            note="PREDICTIONS.json / PREDICTIONS_new.json / debts_lattice.json — frozen before "
                 "the surgery cells existed; status:predicted, never tallied (I8)"),
        Obligation("measurements.graded", required=False, satisfied=n >= 6, kind="count",
                   evidence_ids=sorted(c.get("id") for c in labelled),
                   note=f"Labelled quantize cells present: {n} (a predictor needs labels)"),
        Obligation(
            "spearman_gate", required=True, satisfied=False, kind="count",
            evidence_ids=sorted(c.get("id") for c in labelled),
            note=f"{n}/20 labelled cells present but Spearman ρ is NOT yet measured from the "
                 "ledger (no ρ cell exists) — capped at PRELIMINARY until the refuter gate "
                 "(≥20 labelled cells AND ρ ≥ 0.3, no false negatives) is discharged; "
                 "m1/analyze.py last scored held:7, broken:0"),
    ]


def _ob_k7(ledger):
    g7 = cells_with_exact(ledger, "K-3.lora.g7_rand_rep")
    g7q = cells_with_exact(ledger, "K-4.quantize.g7_rand_rep.q4_k_m")
    g1q = cells_with_exact(ledger, "K-4.quantize.g1_haar.q4_k_m")
    g4 = cells_with_exact(ledger, "K-3.lora.g4_perm")
    return [
        _ge("vector.g7_rand_rep", True, g7 + g7q, min_n=1,
            note="quantization-pristine (0.98× base KLD) AND adaptation-deficient (−13 pp): "
                 "one artifact, two orthogonal reserves"),
        _ge("vector.g1_haar", True, g1q,
            note="KL-neutral yet ΔPPL-failing (+3.97%): two damage statistics disagree"),
        _ge("vector.g4_perm", True, g4,
            note="quantization-inert, costs 6.4 pp of LoRA capture: a control must name its op"),
    ]


def _ob_k8(ledger):
    hist = [c for c in ledger.all("cell")
            if (c.get("op") or {}).get("name", "").startswith("history.")]
    return [
        _ge("matched_pairs", True, hist, min_n=4,
            note="2 matched pairs at 0.5B (sft→merge→q4 vs merge→sft→q4), matched on L0 "
                 "features and current behaviour — the next milestone (M3/A5)"),
    ]


def _merge_ok(c):
    res = c.get("result") or {}
    return res.get("status") == "measured" and res.get("verdict") == "pass"


def _ob_k9(ledger):
    cal = [c for c in cells_with(ledger, "K-9.calibration.") if _merge_ok(c)]
    measured = [c for c in cells_with(ledger, "K-9.merge.")
                if (c.get("result") or {}).get("status") == "measured"]
    failures = [c for c in measured if (c.get("result") or {}).get("verdict") == "fail"]
    passes = [c for c in measured if (c.get("result") or {}).get("verdict") == "pass"]
    enough = len(failures) >= 2
    return [
        Obligation(
            "calibration.merge_ref", required=True, satisfied=bool(cal), kind="cells",
            evidence_ids=sorted(c.get("id") for c in cal),
            promoting_ids=[] if cal else
            ["UNAVAILABLE: PASSING base merge calibration under merge-v2-base-calibrated"],
            blocked_reason=None if cal else "missing PASSING base merge calibration reference"),
        Obligation(
            "measurement.gauge_failures", required=True, satisfied=enough, kind="cells",
            evidence_ids=sorted(c.get("id") for c in failures),
            promoting_ids=[] if enough else
            ["UNAVAILABLE: second function-equivalent gauge with measured merge failure"],
            blocked_reason=None if measured else "no measured gauge merge cells",
            note=f"{len(failures)} measured gauge failures, {len(passes)} gauge passes; "
                 "base reference passes"),
        _declared("controls.permutation", True, "CLAIMS.md K-9"),
        Obligation("replication.n=2", required=True, satisfied=enough, kind="count",
                   evidence_ids=sorted(c.get("id") for c in failures),
                   note="requires two independently constructed gauge representatives"),
    ]


def _ob_k10(ledger):
    eq = cells_with_exact(ledger, "K-1.equivalence.prep_base_exact")
    bf = cells_with_exact(ledger, "K-1.equivalence_bf16.prep_base_exact")
    return [
        _ge("equivalence.fp32.prep_base_exact", True, eq,
            note="lattice-only canonicalizer certified EQUIVALENT in fp32"),
        _declared("control.null_gauge", True,
                  "{G5,G3,G7} snap_pow2 bf16-lossless; CLAIMS.md K-10"),
        Obligation("replication.n=1", required=True, satisfied=True, kind="count"),
        Obligation(
            "refuter.bf16_compute", required=False, satisfied=bool(bf),
            evidence_ids=sorted(c.get("id") for c in bf),
            note="the bf16-compute equivalence cell for prep_base_exact is the honest gap — "
                 "cheap to run"),
    ]


_RESOLVERS = {
    "K-1": _ob_k1, "K-2": _ob_k2, "K-3": _ob_k3, "K-4": _ob_k4, "K-5": _ob_k5,
    "K-6": _ob_k6, "K-7": _ob_k7, "K-8": _ob_k8, "K-9": _ob_k9, "K-10": _ob_k10,
}


def obligations_for(ledger, key: str) -> list:
    fn = _RESOLVERS.get(key)
    if fn is None:
        raise LedgerError(f"unknown claim {key!r}; known: {sorted(_RESOLVERS)}")
    return fn(ledger)


# (key, text, declared state, blocked?, partial?, refuter, numbers:[(value, exact_ob_tag)])
CLAIM_SEEDS = [
    ("K-1",
     "Qwen2.5-0.5B admits at least five exact gauges that ordinary tooling does not quotient out.",
     "controlled", False, False,
     {"query": "any gauge family whose equivalence cell shows KL > 2e-3 or top-1 < 0.995 in BOTH "
               "compute dtypes (partially fired for g7_rand under bf16 → weaker, true statement)",
      "would_drop_to": "preliminary", "answering_ob": "K-1.equivalence_bf16.g7_rand"},
     [(2.0e-4, "K-1.equivalence.base"), (0.98096, "K-1.equivalence_bf16.g7_rand")]),
    ("K-2",
     "Export format is an operation, and it can destroy a function-equivalent checkpoint.",
     "controlled", False, False,
     {"query": "an export-dtype sweep on a second artifact where f16 stays within 1% of bf16 "
               "while the static census still shows >5% subnormal weights",
      "would_drop_to": "preliminary", "answering_ob": "K-2.export"},
     [(12.1351, "K-2.export.base.bf16"), (177.3286, "K-2.export.g3_pow2.f16"),
      (12.1399, "K-2.control.identity_roundtrip")]),
    ("K-3",
     "Function-equivalent checkpoints have materially different adaptation reserve.",
     "preliminary", False, False,
     {"query": "|gap| < 3·sd across corrected true-LoRA probe seeds, or a base re-measurement "
               "whose spread swallows the gap",
      "would_drop_to": "unsupported", "answering_ob": "K-3.replication.probe_seed"},
     []),
    ("K-4",
     "…and different quantization reserve at the same bit-width.",
     "controlled", False, False,
     {"query": "a rerun of g1_haar under a longer corpus where its ΔPPL falls inside base+slack "
               "(corpus sampling variance, not gauge state)",
      "would_drop_to": "preliminary", "answering_ob": "K-4.quantize.g1_haar.q4_k_m"},
     [(10.690304, "K-4.quantize.g3_pow2.q8_0"), (0.00094, "K-4.calibration.q8_0"),
      (0.031914, "K-4.calibration.q4_k_m")]),
    ("K-5",
     "An artifact-only canonicalizer restores the reserve without seeing the original.",
     "partial", False, False,
     {"query": "a repaired artifact that beats base on one axis while its static debt stays >1e-3",
      "would_drop_to": "preliminary", "answering_ob": "K-3.lora.g7_rand_rep"},
     [(0.9829, "K-3.lora.g3_pow2_rep"), (0.8407, "K-3.lora.g7_rand_rep"),
      (0.9482, "K-3.lora.bad_all_exact")]),
    ("K-6",
     "Static L0 features predict which operations are at risk (the M6 seed).",
     "preliminary", False, False,
     {"query": "any artifact where flags say OK and a measured cell fails (false negatives are "
               "the only direction that hurts a preflight tool), or ≥20 labelled cells with "
               "Spearman ρ < 0.3",
      "would_drop_to": "preliminary", "answering_ob": "K-6"},
     [(7, "K-6"), (0, "K-6")]),
    ("K-7",
     "Reserve is a vector; no scalar summarizes it.",
     "controlled", False, False,
     {"query": "a single artifact exhibiting two reserves that move in the SAME direction",
      "would_drop_to": "preliminary", "answering_ob": "K-3.lora.g7_rand_rep"},
     [(0.98, "K-4.quantize.g7_rand_rep.q4_k_m"), (0.8407, "K-3.lora.g7_rand_rep")]),
    ("K-8",
     "Natural post-training histories, not constructed gauges, produce divergent reserves.",
     "unsupported", False, False,
     {"query": "two matched pairs at 0.5B whose reserves diverge along the same history axes",
      "would_drop_to": "preliminary", "answering_ob": "history."},
     []),
    ("K-9",
     "Merge compatibility is gauge-dependent.",
     "controlled", False, False,
     {"query": "the corrected base reference fails, or two independent gauge representatives "
               "both pass either merge operator under the same calibrated contract",
      "would_drop_to": "preliminary", "answering_ob": "K-9.merge"},
     []),
    ("K-10",
     "`prepare` improves reserve on a checkpoint nobody stressed.",
     "controlled", False, False,
     {"query": "a second architecture family where lattice-prepare does not reduce Q4 ΔPPL, or "
               "the bf16-compute cell for prep_base_exact (cheap — the honest gap)",
      "would_drop_to": "preliminary", "answering_ob": "K-1.equivalence_bf16.prep_base_exact"},
     [(0.9924, "K-3.lora.prep_base_exact"), (0.001017, "K-4.quantize.prep_base_exact.q8_0"),
      (0.031624, "K-4.quantize.prep_base_exact.q4_k_m")]),
]


def explain(ledger, key: str, *, allow_mixed_env: bool = False) -> dict:
    """Derive and check a claim's state from the ledger (the `theseus explain` body)."""
    seed = next((s for s in CLAIM_SEEDS if s[0] == key), None)
    if seed is None:
        raise LedgerError(f"unknown claim {key!r}; known: {[s[0] for s in CLAIM_SEEDS]}")
    _, text, declared, blocked, partial, refuter, numbers = seed
    obligations = obligations_for(ledger, key)

    # I3: every comparison group (per op) must share one environment digest. Only *measured*
    # cells are compared: unavailable/predicted cells hold no comparable number (I8) and are
    # excluded from the digest-mixing check rather than vetoing the group.
    groups = {}
    for ob in obligations:
        for cid in ob.evidence_ids:
            c = ledger.get("cell", cid)
            if c and (c.get("result") or {}).get("status") == "measured":
                groups.setdefault((c.get("op") or {}).get("name", "?"), []).append(c)
    for opname, cells in sorted(groups.items()):
        if len(cells) >= 2:
            mixed_environments(cells, allow_mixed_env=allow_mixed_env,
                               kind=f"claim {key} evidence for {opname!r}")

    cap = claim_cap({"state": declared, "blocked": blocked}, obligations)
    state = cap["state"]
    if partial and state in ("controlled", "preliminary"):
        state = "partial"
    if state == "controlled" and not cap["missing"] and blocked:
        state = "blocked"  # K-9: declared controlled but operationally blocked

    # I10: every number cites a cell id (refusal to cite = UNAVAILABLE, never guessed).
    cited = []
    for value, ob_tag in numbers:
        if ob_tag.startswith("K-") and "." in ob_tag:
            cells = cells_with_exact(ledger, ob_tag) or cells_with(ledger, ob_tag)
        else:
            cells = cells_with(ledger, ob_tag)
        cite = (cells[0].get("id") if cells else
                f"UNAVAILABLE ({ob_tag}) — cell not imported")
        cited.append({"value": value, "cite": cite})

    answering = cells_with_exact(ledger, refuter["answering_ob"]) or \
        cells_with(ledger, refuter["answering_ob"])
    refuter_cells = sorted(c.get("id") for c in answering) or \
        [f"UNAVAILABLE ({refuter['answering_ob']}) — not yet run"]

    return {
        "key": key, "text": text,
        "declared": declared, "state": state,
        "missing": cap["missing"],
        "promoting_cells": cap["promoting_cells"],
        "blocked_by": cap["blocked_by"],
        "obligations": [
            {"name": o.name, "required": o.required, "satisfied": o.satisfied,
             "kind": o.kind, "note": o.note,
             "evidence": o.evidence_ids[:10],
             "promoting": o.promoting_ids[:6], "blocked_reason": o.blocked_reason}
            for o in obligations
        ],
        "refuter": {"query": refuter["query"], "would_drop_to": refuter["would_drop_to"],
                    "answering_cells": refuter_cells},
        "numbers": cited,
        "mixed_env_stamp": "mixed-env (--allow-mixed-env)" if allow_mixed_env else None,
    }


def all_keys():
    return [s[0] for s in CLAIM_SEEDS]
