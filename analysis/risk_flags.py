# risk_flags.py — analysis/build_contract: the single definition of what a "risk flag" means to
# the base-rate/threshold pipeline. Owned by BaseRates. Conventions match inspect/src/main.rs
# (F16_NORMAL_MIN = 2^-14, BLOCK = 32, QBITS = 7) and SCHEMA.md (features.total.convention is
# mandatory: mean-of-per-tensor-ratios vs ratio-of-sums differ by up to 5.7% and are two numbers).

FAMILIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
F16_NORMAL_MIN = 6.103515625e-5          # 2^-14, the smallest f16 normal (not subnormal) value
BLOCK = 32
QBITS = 7.0                              # 2^(4-1) - 1, symmetric 4-bit max-abs block quantizer

# ---- The one stated multiple ----
# "Catastrophic divergence" = a measured damage metric whose magnitude is at least this many
# times the magnitude of the operation's reference (calibration) measurement. Reference = the
# pristine artifact's own measured value for the same op: base q4 rel_dppl +2.2%, q8 -0.1%,
# adaptation +2.96 protected dppl, f16 export ratio 1.0004. With abs(reference) below a floor the
# ratio is degenerate -> catastrophic = None (undefined), never silently False (I8).
CATASTROPHE_MULTIPLE = 100.0
REF_ABS_MIN = 1e-9    # below this the damage/reference ratio is meaningless; catastrophic undefined

# ---- Contract v2: the provisional thresholds currently compiled into inspect/src/main.rs ----
# Calibrated on the n = 2 measured contrast (base vs g3_pow2) and printed as provisional with
# their n in the binary's own output. Thresholds are chosen by thresholds.py and never changed
# in place: a new contract is emitted as a new version, prior verdicts it invalidates are listed.
CONTRACT_V2 = 2
FLAG_DEFS = {
    "export.f16": {
        "op": "export.gguf.f16",
        "feature": "frac_below_f16_normal",       # per family; share of weights below 2^-14
        "threshold": 0.02,                        # provisional; K-2 refuter is the 1% export ratio
        "aggregate": "worst_family",              # artifact fires iff any family exceeds threshold
        "damage": "export_damage_ratio",          # measured ppl(f16)/ppl(bf16); >1.01 => fail
        "reference_note": "pristine base export ratio 1.0004 (f16 12.1399 vs bf16 12.1351)",
    },
    "quant.q8_0": {
        "op": "quantize.gguf.q8_0",
        "feature": "q4_block_mse",                # mean of per-tensor 32-block ratios
        "threshold": 0.0165,                      # 1.5x measured dense Qwen2.5 family (0.011)
        "total_threshold": 0.0168,                # 1.5x measured base total J (0.01123)
        "aggregate": "worst_family_or_total",
        "damage": "rel_dppl",                     # fraction, not percent (base q8 = -0.000989)
        "reference_note": "base q8_0 rel_dppl -0.000989",
    },
    "quant.q4_k_m": {
        "op": "quantize.gguf.q4_k_m",
        "feature": "q4_block_mse",
        "threshold": 0.0165,
        "total_threshold": 0.0168,
        "aggregate": "worst_family_or_total",
        "damage": "rel_dppl",
        "reference_note": "base q4_k_m rel_dppl 0.021945",
    },
    "adapt.lora.r16": {
        "op": "adapt.lora.r16",
        "feature": "dyn_range_log10",             # primary; worst family
        "threshold": 12.0,                        # provisional
        "secondary": "row_energy_imbalance",
        "secondary_threshold": 2.0e5,
        "aggregate": "worst_family",
        "damage": "protected_dppl",               # collateral general-PPL move after LoRA
        "reference_note": "base protected_dppl 2.9598",
    },
}

FEATURE_KEYS = ("q4_block_mse", "q4_block_mse_pooled", "dyn_range_log10",
                "row_energy_imbalance", "amax_over_rms", "frac_below_f16_normal")


def family_fires(flag: str, row: dict):
    """Does this single family's features exceed the flag's threshold?
    Returns True/False, or None when the feature is absent (no evidence, I8)."""
    d = FLAG_DEFS[flag]
    v = row.get(d["feature"])
    if v is None:
        return None
    over = v > d["threshold"]
    if d.get("secondary"):
        s = row.get(d["secondary"])
        over = over or (s is not None and s > d["secondary_threshold"])
    return over
def artifact_fires(flag: str, fam_rows, total_row=None):
    """Does the artifact-level flag fire? Uses the flag's aggregate rule over per-family rows
    (plus the total row for the quant OR). Returns True/False; None if no family row carries the
    feature (we cannot say the artifact is safe)."""
    d = FLAG_DEFS[flag]
    any_fire = None
    for r in fam_rows:
        f = family_fires(flag, r)
        if f is True:
            any_fire = True
        elif f is False and any_fire is None:
            any_fire = False
    if any_fire is False and d.get("total_threshold") is not None and total_row is not None:
        t = total_row.get(d["feature"])
        if t is not None and t > d["total_threshold"]:
            any_fire = True
    return any_fire


def artifact_fires_eval(flag: str, fam_rows, total_row=None):
    """Same as artifact_fires, but when no family row carries the primary feature and a total row
    does, falls back to the total value as a whole-artifact proxy (marked `proxy=total` in
    callers' output). Keeps a safe verdict possible when only pooled features were recorded."""
    v = artifact_fires(flag, fam_rows, total_row)
    if v is not None:
        return v
    d = FLAG_DEFS[flag]
    if total_row is not None:
        t = total_row.get(d["feature"])
        if t is not None:
            return t > d["threshold"]
    return None


def is_catastrophic(damage, damage_ref):
    """Stated definition: abs(damage) >= CATASTROPHE_MULTIPLE * abs(damage_ref), with a
    degenerate near-zero reference meaning *undefined*, never False."""
    if damage is None or damage_ref is None:
        return None
    r = abs(damage_ref)
    if r < REF_ABS_MIN:
        return None
    return abs(damage) >= CATASTROPHE_MULTIPLE * r


def flag_ranks():
    """Print ordering used everywhere (CLI tables)."""
    return ("export.f16", "quant.q8_0", "quant.q4_k_m", "adapt.lora.r16")
