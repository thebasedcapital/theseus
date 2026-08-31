#!/usr/bin/env python3
"""M1 variant registry: build function-equivalent checkpoints from the pinned base.

    <venv python> m1/make_variants.py --all          # writes m1/work/<name>/ + VARIANTS.json
    <venv python> m1/make_variants.py --only bad_all

`VARIANTS` is the single source of truth for names/specs; run_m1.py and M1.md read it.
Stress parameters are simple declared draws (random Haar / random angles / log-uniform
scales / random permutations), deliberately NOT optimized against any objective, so a
damage result cannot be an artifact of adversarial construction. The adversarial end of
each orbit is included separately and labelled as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import shutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import canonicalize as canon  # noqa: E402
import gauge  # noqa: E402
from common import log  # noqa: E402

# name -> (spec, canonicalize-or-None, note)
#   spec None            -> the untouched reference checkpoint
#   canon None           -> no repair applied
#   "all"                -> canonicalize every family (G5,G3,G2,G1)
def _v(name, spec, cmethod, note):
    VARIANTS[name] = (spec, cmethod, note)


VARIANTS: dict[str, tuple[str | None, str | None, str]] = {}
# --- reference ---------------------------------------------------------------------------
_v("base", None, None, "pristine Qwen2.5-0.5B (bf16, tied)")

# --- exact gauges: each family gets a STRESS artifact and a REPAIRED artifact -------------
# Stress parameters are simple declared draws (random Haar / random angles / log-uniform
# scales / random permutations), deliberately NOT optimized against any objective, so a
# damage result cannot be an artifact of adversarial construction. The adversarial end of
# each orbit is included separately and labelled as such.
_v("g1_haar",      "G1:haar:1",  None,  "random O(64) basis change per kv-group (V/O)")
_v("g1_haar_rep",  "G1:haar:1",  "g1",  "same stress, then artifact-only value-subspace canonicalizer")
_v("g1_svd",       "G1:svd:1",   None,  "ADVERSARIAL: value block into eigenbasis (energy concentrated)")
_v("g1_svd_rep",   "G1:svd:1",   "g1",  "adversarial stress + repair")
_v("g2_rand",      "G2:random:1", None, "random RoPE-pair rotations of q/k (maximal exact q/k gauge)")
_v("g2_rand_rep",  "G2:random:1", "g2", "stress + paired-row energy equalization")
_v("g3_rand",      "G3:random:1", None, "RMSNorm scale absorption, log-uniform +-3 decades per column")
_v("g3_pow2",      "G3:pow2:1",   None, "same family with bf16-EXACT scales (d = 2^k, k in [-512,512]) so the gauge itself costs no representation noise")
_v("g3_pow2_rep",  "G3:pow2:1",   "g3", "exact-storage stress + consumer column-energy equalization")
_v("g3_rand_rep",  "G3:random:1", "g3", "stress + consumer column-energy equalization")
_v("g3_smooth",    "G3:smooth:1", None, "same family, block-constant d (a 32-block quantizer can absorb it)")
_v("g3_smooth_rep", "G3:smooth:1", "g3", "control stress + repair")
_v("g4_perm",      "G4::1",      None,  "CONTROL: head/group permutation — exact and quantization-neutral")
_v("g5_c8",        "G5:c8",      None,  "residual stream x8; needs the tie broken; RMSNorm-eps floor left in place")
_v("g5_c8_eps",    "G5:c8:eps",  None,  "same gauge, made EXACT by rewriting rms_norm_eps -> eps/c^2 in config")
_v("g5_c8_rep",    "G5:c8",      "g5",  "stress + tie-witness canonicalizer (recovers c, re-ties)")
_v("g5_c8_eps_rep", "G5:c8:eps",  "g5",  "exact-stress + tie-witness repair (eps returned to the tied point)")
_v("g6_perm",      "G6::1",      None,  "CONTROL: SwiGLU neuron permutation (the only exact MLP gauge)")

# --- combined multi-family stress, 3 seeds (M1 requires multi-seed replication) ----------
_v("g7_rand",      "G7:random:1", None, "SwiGLU up-branch diagonal +-3 decades per neuron (V0\'s mechanism)")
_v("g7_few",       "G7:few:1",    None, "same family, extreme scale on 0.1% of neurons (outlier-neuron model)")
_v("g7_rand_rep",  "G7:random:1", "g7", "stress + V0 closed-form balance")
_v("bad_all",       "G3:random:1+G1:haar:1+G2:random:1+G7:random:1", None,  "V0-style multi-family stress, seed 1")
_v("bad_all_rep",   "G3:random:1+G1:haar:1+G2:random:1", "all", "seed 1 stress + full canonicalization")
_v("bad_all_s2",    "G3:random:2+G1:haar:2+G2:random:2+G7:random:2", None,  "replication seed 2")
_v("bad_all_s2_rep", "G3:random:2+G1:haar:2+G2:random:2+G7:random:2", "all", "replication seed 2 + repair")
_v("bad_all_s3",    "G3:random:3+G1:haar:3+G2:random:3+G7:random:3", None,  "replication seed 3")
_v("bad_all_s3_rep", "G3:random:3+G1:haar:3+G2:random:3+G7:random:3", "all", "replication seed 3 + repair")

# --- `prepare` on a model nobody stressed (does canonicalization buy reserve on its own?) --
_v("prep_base",    None, "all",  "canonicalizer applied to the pristine base")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


# canonicalize -> the exact-family repairs to run
CANS = {"g1": ("G1",), "g2": ("G2",), "g3": ("G3",), "g5": ("G5",), "g7": ("G7",),
        "all": ("G5", "G3", "G2", "G7", "G1")}


def build(name: str, arch: common.Arch, base_sd: dict, out_root: Path,
          eps0: float | None = None) -> dict:
    spec, cmethod, note = VARIANTS[name]
    t0 = time.time()
    sd = {k: v.clone() for k, v in base_sd.items()}
    man: dict = {"name": name, "spec": spec, "canonicalize": cmethod, "note": note}
    cfg_patch: dict = {}
    if spec:
        sd, m = gauge.apply_spec(sd, arch, spec, cfg_eps=eps0)
        man["gauge"] = m
        cfg_patch.update(m.get("config_patch") or {})
    if cmethod:
        sd, ms = canon.run(sd, arch, CANS[cmethod])
        man["canon"] = ms
        if "G5" in CANS[cmethod] and "rms_norm_eps" in cfg_patch:
            # the tied representative is the c=1 point, where the original eps belongs
            cfg_patch["rms_norm_eps"] = eps0
    untied = bool(spec) and "G5" in spec
    d = out_root / name
    if untied:
        cfg_patch["tie_word_embeddings"] = False
    else:
        sd.pop("lm_head.weight", None)          # keep tied artifacts tie-shaped
    common.save_state(sd, d, common.REF_MODEL, config_patch=cfg_patch or None)
    man.update(dir=str(d), untied=untied,
               bytes=(d / "model.safetensors").stat().st_size,
               sha256=sha(d / "model.safetensors"),
               build_s=round(time.time() - t0, 1))
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(common.WORK))
    a = ap.parse_args()
    names = list(VARIANTS) if a.all or not a.only else [x.strip() for x in a.only.split(",") if x.strip()]
    unknown = [x for x in names if x not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}")
    common.WORK.mkdir(parents=True, exist_ok=True)
    reg_path = Path(a.out) / "VARIANTS.json"
    reg = common.rjson(reg_path) if reg_path.exists() else {}
    arch = common.read_arch(common.REF_MODEL)
    eps0 = float(json.loads((common.REF_MODEL / "config.json").read_text())["rms_norm_eps"])
    base_sd = None
    for n in names:
        if n == "base":                      # the reference checkpoint needs no copy
            reg[n] = {"name": n, "spec": None, "canonicalize": None, "note": VARIANTS[n][2],
                      "dir": str(common.REF_MODEL), "untied": False,
                      "bytes": sum(f.stat().st_size for f in common.REF_MODEL.glob("*.safetensors")),
                      "sha256": "reference", "build_s": 0.0}
            continue
        free = shutil.disk_usage(str(common.WORK)).free / 1e9
        if free < 3.5:
            raise SystemExit(f"disk guard: {free:.1f} GB free — delete m1/work/<variant> first")
        if base_sd is None:
            base_sd = common.load_state(common.REF_MODEL)
            log(f"arch hidden={arch.hidden} heads={arch.n_q}/{arch.n_kv} hd={arch.head_dim} "
                f"inter={arch.intermediate} layers={arch.layers} tie={arch.tie}")
        tgt = Path(a.out) / n / "model.safetensors"
        if n in reg and tgt.exists() and reg[n].get("sha256") == sha(tgt):
            log(f"skip {n} (already built)")
            continue
        m = build(n, arch, base_sd, Path(a.out), eps0=eps0)
        reg[n] = m
        common.wjson(reg_path, reg)
        log(f"built {n:16s} {m['bytes'] / 1e6:.0f} MB sha={m['sha256']} {m['build_s']}s")
    common.wjson(reg_path, reg)
    log(f"registry -> {reg_path}")


if __name__ == "__main__":
    main()
