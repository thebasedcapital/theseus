#!/usr/bin/env python3
"""Theseus rescue: equivalence-gated, artifact-only `prepare` that turns static flags into
treatment. Owned by RescueMode (Theseus m1 / RescueMode slice).

   meter pre/post with the FROZEN inspector binary (read-only subprocess, nice -n 19),
   apply the lattice-exact canonicalizer via m1/canonicalize.py (read-only import),
   prove the function did not change with an m1/verify_equiv.py SUBPROCESS,
   and REFUSE (never emit) the repaired artifact if the equivalence cell fails.

WRITES ONLY UNDER m1/rescue_out/ plus the --out json (default there too). Never touches
m1/work/ (drive.live scratch), never rebuilds inspect/, never modifies the input artifact,
CPU-only, 2 threads, nice -n 19. Any numbers carry `convention` (I7/I8: the mean-of-tensors
and pooled q4 ratios are distinct metrics, and UNAVAILABLE/null is never a pass).

    <venv python> m1/rescue.py --artifact <hf dir> \
                     --for quantize,adapt,merge,export-f16 \
                     --out m1/rescue_out/rescue-<name>.json
    <venv python> m1/rescue.py --artifact <hf dir> --full   # full non-lattice canon (REFUSED demo)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402
import canonicalize as canon  # noqa: E402

PY = os.environ.get("THESEUS_PY", "/home/admin/counterpoint/.venv/bin/python")
INSPECT = common.REPO / "inspect" / "target" / "release" / "theseus-inspect"
SCRATCH = common.M1 / "rescue_out"          # this slice's owned scratch (never m1/work/)
DISK_MIN_GB = 2.0                           # refuse any weight write below this much free

LATTICE = ("G5", "G3", "G7")                # bf16-lossless: tie-witness + exponent-lattice G3/G7
FULL = ("G5", "G3", "G2", "G7", "G1")       # + value-subspace Hadamard + RoPE rotations (not lossless)
FAMILIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DIFF_KEYS = ("q4_block_mse", "q4_block_mse_pooled", "dyn_range_log10",
             "row_energy_imbalance", "amax_over_rms", "frac_below_f16_normal")
CONVENTION = ("q4_block_mse = mean over tensors of sum_blocks(amax^2*32)/(12*49*sum w^2); "
              "q4_block_mse_pooled = ratio of sums; deltas are after-minus-before "
              "(negative = improvement); missing evidence is null and never counts as a pass")

PURPOSE_OPS = {  # `--for <purpose>` -> the preflight operations on the inspector's surface
    "quantize": ["quantize.gguf.q8_0", "quantize.gguf.q5_k_m", "quantize.gguf.q4_k_m", "quantize.awlora"],
    "adapt": ["adapt.lora.r16"],
    "merge": ["merge.linear", "merge.ties"],
    "export-f16": ["export.gguf.f16"],
}


def disk_free_gb() -> float:
    return shutil.disk_usage(str(common.REPO)).free / 1e9


def check_disk(what: str) -> None:
    free = disk_free_gb()
    if free < DISK_MIN_GB:
        raise SystemExit(
            f"disk guard: {free:.1f} GB free < {DISK_MIN_GB:.1f} GB — cannot {what} "
            f"(each artifact ~1 GB). Free space by deleting scratch you own.")


# --- inspector subprocesses (frozen binary, read-only) ------------------------------------

def _run_inspect(saf: Path, tag: str, preflight: bool) -> dict:
    out_json = SCRATCH / (f"_meter-{tag}-pre.json" if preflight else f"_meter-{tag}.json")
    argv = ["nice", "-n", "19", str(INSPECT)]
    if preflight:
        argv.append("preflight")
    argv += [str(saf), "--json", str(out_json)]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True)
    if p.returncode not in (0, 1):          # preflight exits 1 when any op is AT_RISK (by design)
        raise RuntimeError(f"inspector failed rc={p.returncode}: {p.stderr[-600:]}")
    if not preflight:
        rep = common.rjson(out_json)
    else:
        # Current inspector emits a complete JSON object. Keep stdout parsing as a compatibility
        # fallback for older local binaries that truncated preflight JSON before the closing brace.
        try:
            rep = common.rjson(out_json)
        except (OSError, ValueError, json.JSONDecodeError):
            rep = {"preflight": _preflight_from_stdout(p.stdout)}
    rep["meter_wall_s"] = round(time.time() - t0, 2)
    rep["inspector_rc"] = p.returncode
    return rep


def _preflight_from_stdout(stdout: str) -> list:
    """Parse the frozen inspector's PREFLIGHT table:
    '<op>  <AT_RISK|OK|UNAVAILABLE>  <reason>'."""
    pat = re.compile(r"^(\S+)\s+(AT_RISK|OK|UNAVAILABLE)\s+(.*)$")
    ops = []
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if m:
            ops.append([m.group(1), m.group(2), m.group(3).strip()])
    if not ops:
        raise RuntimeError(f"preflight table not found in inspector stdout:\n{stdout[-800:]}")
    return ops


def meter(saf: Path, tag: str) -> dict:
    """Combined report: per-family features + verdicts (inspect) AND the per-operation
    preflight matrix, so `--for` purposes and family flags both have before/after."""
    rep = _run_inspect(saf, tag, preflight=False)
    pre = _run_inspect(saf, tag, preflight=True)
    rep["preflight"] = pre.get("preflight") or []
    rep["preflight_rc"] = pre.get("inspector_rc")
    return rep


def count_flags(rep: dict) -> int:
    return len(rep.get("verdicts") or [])


def preflight_ops(rep: dict) -> dict:
    return {o: v for o, v, _r in (rep.get("preflight") or [])}


# --- changed-tensor census + power-of-two lattice assertion --------------------------------

def _family_of(name: str) -> str:
    for f in FAMILIES:
        if f in name:
            return f
    if "layernorm.weight" in name:
        return "norm"
    if "embed_tokens" in name or "lm_head" in name:
        return "embed"
    if name.endswith(".bias"):
        return "bias"
    return "other"


def changed_tensors_census(before: dict, after: dict, lattice_path: bool) -> dict:
    """Census changed entries and prove every changed nonzero ratio is a positive 2^k.

    Zeroing/overflow and sign changes are violations on the lattice path. Unchanged entries are
    excluded from the ratio check; otherwise log2(1) would dominate the proof mechanically.
    """
    per, fams, other = {}, {f: {"tensors_changed": 0, "entries_changed": 0} for f in FAMILIES}, {}
    viol, zeroed, ents = {}, {}, 0
    for k in after:
        av, bv = before[k].to(torch.float64), after[k].to(torch.float64)
        diff = av != bv
        n = int(diff.sum().item())
        ents += n
        fam = _family_of(k)
        bucket = fams[fam] if fam in fams else other.setdefault(
            fam, {"tensors_changed": 0, "entries_changed": 0})
        if n:
            bucket["tensors_changed"] += 1
        bucket["entries_changed"] += n
        rec = {"family": fam, "entries_changed": n,
               "pow2_ok": None, "log2_min": None, "log2_max": None}
        if n:
            changed_nonzero = diff & (av != 0) & (bv != 0)
            ratio = bv[changed_nonzero] / av[changed_nonzero]
            positive = ratio > 0
            nv = int((~positive).sum().item())
            logs = torch.log2(ratio[positive])
            if logs.numel():
                rec["log2_min"], rec["log2_max"] = float(logs.min()), float(logs.max())
                nv += int(((logs - logs.round()).abs() > 1e-3).sum().item())
            z = int(((av != 0) & (bv == 0) & diff).sum().item())
            z += int((diff & ~torch.isfinite(bv)).sum().item())
            if z:
                zeroed[k] = z
                nv += z
            rec["pow2_ok"] = nv == 0
            if nv:
                viol[k] = nv
        per[k] = rec
    return {
        "tensors_total": len(after),
        "tensors_changed": sum(1 for r in per.values() if r["entries_changed"]),
        "tensors_unchanged": sum(1 for r in per.values() if not r["entries_changed"]),
        "entries_changed": ents,
        "bytes_changed": 2 * ents,
        "lattice_path": lattice_path,
        "pow2_asserted": lattice_path,
        "pow2_violations": sum(viol.values()),
        "pow2_violation_tensors": list(viol),
        "zeroed_or_overflow_entries": zeroed,
        "per_family": {**fams, **other},
        "per_tensor": {k: v for k, v in per.items() if v["entries_changed"]},
    }


def diff_reports(before: dict, after: dict) -> dict:
    out = {"convention": CONVENTION}
    for scope, keys in (("families", FAMILIES), ("total", ("total",))):
        body = {}
        for f in keys:
            src = before.get("total", {}) if scope == "total" else (before.get("families") or {}).get(f, {})
            dst = after.get("total", {}) if scope == "total" else (after.get("families") or {}).get(f, {})
            body[f] = {k: (round(dst[k] - src[k], 8)
                           if src.get(k) is not None and dst.get(k) is not None else None)
                       for k in DIFF_KEYS}
        out[scope] = body
    bfl, afl = count_flags(before), count_flags(after)
    cleared = [f for f in (before.get("verdicts") or []) if f not in (after.get("verdicts") or [])]
    new = [f for f in (after.get("verdicts") or []) if f not in (before.get("verdicts") or [])]
    out["flags"] = {"before": bfl, "after": afl, "cleared": cleared, "new": new,
                    "cleared_frac": (bfl - afl) / bfl if bfl else None}
    out["preflight"] = {"before": preflight_ops(before), "after": preflight_ops(after)}
    out["at_risk"] = {
        "before": sum(1 for v in out["preflight"]["before"].values() if v == "AT_RISK"),
        "after": sum(1 for v in out["preflight"]["after"].values() if v == "AT_RISK")}
    return out


# --- equivalence subprocess -----------------------------------------------------------------

def verify_equiv(a: Path, b: Path, name: str, ntokens: int) -> dict:
    """m1/verify_equiv.py as a SUBPROCESS (frozen gate), outputs under m1/rescue_out/,
    CPU-only (TSX_CPU=1), 2 threads, nice -n 19."""
    out_json = SCRATCH / "verify" / f"{name}.equivalence.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TSX_CPU="1", TSX_THREADS="2", OMP_NUM_THREADS="2")
    argv = ["nice", "-n", "19", PY, str(common.M1 / "verify_equiv.py"),
            "--a", str(a), "--b", str(b), "--ntokens", str(ntokens),
            "--seqlen", "512", "--out", str(out_json)]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True, env=env)
    if not out_json.exists():
        raise RuntimeError(f"verify_equiv produced no output rc={p.returncode}: {p.stderr[-800:]}")
    rep = common.rjson(out_json)
    rep["carrier_rc"] = p.returncode
    rep["carrier_wall_s"] = round(time.time() - t0, 1)
    return rep


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def equip_metrics(eq: dict) -> dict:
    return {k: eq.get("metrics", {}).get(k) for k in
            ("max_dlogit", "kl_mean_nats", "top1_agree", "ppl_a", "ppl_b")}


def run_artifact(artifact: Path, purposes, out_json: Path, full: bool, verify_ntokens: int) -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    safetensors = sorted(artifact.glob("*.safetensors"))
    if len(safetensors) != 1:
        raise SystemExit(f"rescue requires exactly one safetensors file; found {len(safetensors)} in {artifact}")
    saf = safetensors[0]
    name = out_json.stem
    fams = FULL if full else LATTICE
    lattice = not full

    log(f"[rescue {name}] meter BEFORE ({artifact.name})")
    before = meter(saf, f"{name}-before")

    log(f"[rescue {name}] load artifact (bf16) + canonicalize fams={fams} snaps=(G3,G7)")
    t0 = time.time()
    sd = common.load_state(artifact)
    arch = common.read_arch(artifact)
    sd_r, man = canon.run(sd, arch, fams, g3_snap=True, g7_snap=True)
    canon_wall = round(time.time() - t0, 2)
    census = changed_tensors_census(sd, sd_r, lattice)
    log(f"[rescue {name}] changed {census['tensors_changed']}/{census['tensors_total']} "
        f"tensors, {census['entries_changed']} entries, pow2_violations="
        f"{census['pow2_violations']} (lattice_path={lattice})")

    # --- the only weight write in this slice, guarded --------------------------------
    check_disk(f"write repaired artifact (~1 GB) under {SCRATCH.name}/")
    repaired = SCRATCH / f"{name}.repaired"
    if repaired.exists():                       # my scratch only: deterministic re-runs
        shutil.rmtree(repaired)
    common.save_state(sd_r, repaired)
    sha16_out = sha16(repaired / "model.safetensors")

    eq = verify_equiv(artifact, repaired, name, verify_ntokens)
    verifier_passed = eq.get("verdict") in ("EQUIVALENT", "EQUIVALENT_FLAGGED")
    shippable = lattice and verifier_passed and census["pow2_violations"] == 0
    result = {
        "script": "rescue.py", "artifact_input": str(artifact.resolve()),
        "safetensors": str(saf.resolve()), "purpose": purposes,
        "canonicalizer": {"mode": "lattice" if lattice else "full(non-lattice, lossy)",
                          "fams": list(fams), "g3_snap": True, "g7_snap": True,
                          "call": "canonicalize.run(sd, arch, fams, g3_snap=True, g7_snap=True)"},
        "disk": {"free_gb_on_entry": round(disk_free_gb(), 2)},
        "canon_wall_s": canon_wall,
        "changed_tensors": census,
        "canon_manifest": man,
        "convention": CONVENTION,
        "verify": {"script": "verify_equiv.py (subprocess)", "verdict": eq.get("verdict"),
                   "distributional_pass": eq.get("distributional_pass"),
                   "metrics": equip_metrics(eq), "rel_ppl": eq.get("rel_ppl"),
                   "flags": list(eq.get("flags") or []),
                   "cond_before": eq.get("cond_a"), "cond_after": eq.get("cond_b"),
                   "out": str((SCRATCH / "verify" / f"{name}.equivalence.json").resolve()),
                   "carrier_rc": eq.get("carrier_rc"),
                   "carrier_wall_s": eq.get("carrier_wall_s")},
        "git_head": eq.get("git_head"), "torch": eq.get("torch"),
    }
    if not shippable:
        # A finite verifier cannot certify a non-lattice transform as function-identical.
        # `prepare` ships only the exact lattice path and only when both gates pass.
        shutil.rmtree(repaired)
        result["status"] = "REFUSED"
        if not lattice:
            result["refused_reason"] = {
                "policy": "non-lattice transformations are diagnostic-only and never shippable",
                "verifier_passed": verifier_passed,
                "metrics": equip_metrics(eq),
            }
        elif census["pow2_violations"]:
            result["refused_reason"] = {
                "policy": "lattice proof failed",
                "pow2_violations": census["pow2_violations"],
            }
        else:
            result["refused_reason"] = equip_metrics(eq)
        result["before"] = before
        result["repaired_dir"] = None
        common.wjson(out_json, result)
        log(f"[rescue {name}] REFUSED — nothing shipped")
        return result

    log(f"[rescue {name}] meter AFTER")
    after = meter(repaired / "model.safetensors", f"{name}-after")
    result.update({
        "status": "RESCUED", "before": before, "after": after,
        "diff": diff_reports(before, after),
        "repaired_dir": str(repaired.resolve()),
        "repaired_sha256_16": sha16_out,
    })
    common.wjson(out_json, result)
    return result


def main():
    ap = argparse.ArgumentParser(
        prog="rescue.py",
        description="Theseus rescue: static flags -> treatment. Meter with the frozen "
                    "inspector binary, apply the lattice-exact canonicalizer, prove "
                    "equivalence with m1/verify_equiv.py (subprocess), REFUSE if the cell "
                    "fails. Writes only under m1/rescue_out/. CPU-only, 2 threads, nice -n 19.")
    ap.add_argument("--artifact", required=True, help="HF dir of the artifact to rescue (read-only)")
    ap.add_argument("--for", dest="for_", default="quantize,adapt,merge,export-f16",
                    help="purposes to highlight (comma list: quantize,adapt,merge,export-f16)")
    ap.add_argument("--out", default="",
                    help="JSON to write (default m1/rescue_out/rescue-<name>.json)")
    ap.add_argument("--full", action="store_true",
                    help="apply the FULL non-lattice canonicalizer (G5,G3,G2,G7,G1, G3/G7 "
                         "snapped). Deliberate REFUSED demonstration: the value-subspace "
                         "Hadamard and RoPE rotations re-round every entry, so the gate fails")
    ap.add_argument("--verify-ntokens", type=int, default=4096,
                    help="tokens for the equivalence subprocess (frozen gate default)")
    a = ap.parse_args()

    artifact = Path(a.artifact).resolve()
    if not artifact.is_dir():
        raise SystemExit(f"--artifact {artifact} is not a directory")
    purposes = [p.strip() for p in a.for_.split(",") if p.strip()]
    unknown = [p for p in purposes if p not in PURPOSE_OPS]
    if unknown:
        ap.error(f"unknown purpose(s) {unknown}; choose from {sorted(PURPOSE_OPS)}")
    out = Path(a.out).resolve() if a.out else SCRATCH / f"rescue-{artifact.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    log(f"[rescue] purpose={purposes} full={a.full} verify_ntokens={a.verify_ntokens}")
    res = run_artifact(artifact, purposes, out, a.full, a.verify_ntokens)
    print(json.dumps({
        "status": res["status"], "artifact": res["artifact_input"], "out": str(out),
        "tensors_changed": res["changed_tensors"]["tensors_changed"],
        "pow2_violations": res["changed_tensors"]["pow2_violations"],
        "equiv_verdict": res["verify"].get("verdict"),
    }, indent=2))


if __name__ == "__main__":
    main()
