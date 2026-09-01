#!/usr/bin/env python3
"""Weakness #8: put a size on the effect relative to the fixes a local user already has.

Everything measured so far says a function-equivalent artifact can lose reserve. The unanswered
question is the one a practitioner asks: how much does that cost to buy back with ordinary levers,
and is there a lever that costs nothing?

Two remedy arms, measured against the same contract the main panel uses (reference-relative:
rel_dPPL <= base + 0.01, mean KLD <= base + 0.005, thresholds read from the recorded cell rather
than restated here):

  arm 1  bits ladder - quantize at q2_K..q8_0 and find the LEAST lossy type that passes. This is
         J*_o from math.md 4 expressed in the unit people actually spend: bits, and therefore MB.
  arm 2  tensor-type override - Q4_K_M but with the five gauge-touched families forced to q8_0,
         which is the standard surgical fix. Same contract, different cost shape.

Arm 3 already exists and needs no re-measurement: the lattice repair (m1/rescue.py) restores Q4
for zero extra bytes, which is the whole argument for Theseus. Comparing arms is the point.

Disk discipline: one bf16 export per artifact, each quantized file measured and deleted before the
next is written, because this host has ~4 GB free and seven quantisations of one model do not fit.
"""
from __future__ import annotations

import argparse, json, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

LADDER = ["q2_K", "q3_K_M", "q4_0", "q4_K_M", "q5_K_M", "q6_K", "q8_0"]
# families the norm-diagonal / up-branch gauges touch; o_proj and down_proj deliberately excluded
# because g3_pow2 raises no flags on them (M1_RESULTS appendix)
OVERRIDE_FAMS = ["attn_q", "attn_k", "attn_v", "ffn_gate", "ffn_up"]
Quant = common.llama_bin("llama-quantize")
PPL = common.llama_bin("llama-perplexity")


def run(cmd, timeout=1800):
    p = subprocess.run([str(x) for x in cmd], capture_output=True, text=True,
                       timeout=timeout, check=False)
    return p.returncode, p.stdout + p.stderr


def measure(model: Path, corpus: Path, base_gguf: Path, logit_ref: Path):
    """rel dPPL vs this artifact's own bf16 export, plus mean KLD against saved base logits."""
    rc, out = run([PPL, "-m", model, "-f", corpus, "-c", "512", "--temp", "0", "--seed", "0",
                   "-ngl", "0", "--chunks", "4"])
    ppl = None
    v = re.findall(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)", out)
    if v:
        ppl = float(v[-1])
    rc2, out2 = run([PPL, "-m", model, "-f", corpus, "-c", "512", "--temp", "0", "--seed", "0",
                     "-ngl", "0", "--chunks", "1", "--kl-divergence",
                     "--kl-divergence-base", logit_ref])
    kl = None
    k = re.findall(r"Mean\s+KLD:\s*([0-9.eE+-]+)", out2)
    if k:
        kl = float(k[-1])
    return ppl, kl


def artifact_arm(name: Path, work: Path, corpus: Path, contract: dict, converter_dir: Path):
    """Returns the bits ladder result plus the tensor-type override for one artifact."""
    bf16 = work / f"{name}.bf16.gguf"
    rc, out = run([sys.executable, common.converter(), "--outfile", bf16, "--outtype", "bf16",
                   converter_dir], timeout=1800)
    if rc or not bf16.exists():
        return {"artifact": str(name), "status": "UNAVAILABLE",
                "reason": f"export failed rc={rc}: {out[-200:]}"}
    ref_ppl_rc, ref_out = run([PPL, "-m", bf16, "-f", corpus, "-c", "512", "--temp", "0",
                               "--seed", "0", "-ngl", "0", "--chunks", "4"])
    rv = re.findall(r"PPL\s*=\s*([0-9.]+)", ref_out)
    ref_ppl = float(rv[-1]) if rv else None
    logits = work / "base.logits.bin"
    run([PPL, "-m", bf16, "-f", corpus, "-c", "512", "--temp", "0", "--seed", "0", "-ngl", "0",
         "--chunks", "1", "--save-all-logits", logits], timeout=1800)

    res = {"artifact": str(name), "status": "MEASURED", "bf16_ppl": ref_ppl, "ladder": {}, "size_mb": {}}
    slack_d, slack_k = contract["rel_dppl_slack"], contract["kl_mean_slack"]
    for t in LADDER + ["q4_k_m+fams@q8_0"]:
        q = work / f"q.{t.replace('+', '_').replace('@', 'at')}.gguf"
        if t.endswith("@q8_0"):
            base_type = "q4_k_m"
            extra = sum([["--tensor-type", f"*.{f}.weight=q8_0"] for f in OVERRIDE_FAMS], [])
            cmd = [Quant, bf16, q, base_type, "8"] + [x for pair in extra for x in pair]
        else:
            cmd = [Quant, bf16, q, t, "8"]
        rc, o = run(cmd, timeout=1800)
        if rc or not q.exists():
            res["ladder"][t] = {"status": "UNAVAILABLE", "reason": o[-160:]}
            continue
        size = round(q.stat().st_size / 1e6, 1)
        ppl, kl = measure(q, corpus, bf16, logits)
        rel = None if (ppl is None or not ref_ppl) else (ppl - ref_ppl) / ref_ppl
        entry = {"status": "OK", "ppl": ppl, "rel_dppl": rel, "kl_mean": kl, "size_mb": size}
        if ref_ppl and rel is not None and kl is not None:
            # reference-relative pass test, thresholds from the recorded contract
            entry["pass"] = bool(rel <= contract["base_rel_dppl"] + slack_d
                                 and kl <= contract["base_kl_mean"] + slack_k)
        res["ladder"][t] = entry
        res["size_mb"][t] = size
        q.unlink(missing_ok=True)
        shutil.rmtree(q.parent / "out", ignore_errors=True)
    bf16.unlink(missing_ok=True)
    logits.unlink(missing_ok=True)
    passing = [t for t in LADDER if res["ladder"].get(t, {}).get("pass")]
    res["min_passing_type"] = passing[0] if passing else None
    res["min_passing_mb"] = res["size_mb"].get(passing[0]) if passing else None

    # Arm 2 self-check. llama.cpp b9851 accepted --tensor-type in three name forms (glob, literal
    # index, bare family) at two argument positions, with and without --allow-requantize, and the
    # output was byte-identical to plain q4_K_M every time - 397,807,328 bytes - so the override was
    # never applied. Reporting override_passes=False from that would claim "the standard surgical fix
    # does not work", which was never measured. A remedy arm must demonstrate it changed the artifact
    # before its verdict means anything.
    plain = res["ladder"].get("q4_k_m", {})
    ovr = res["ladder"].get("q4_k_m+fams@q8_0", {})
    changed = (ovr.get("size_mb") is not None and plain.get("size_mb") is not None
               and abs(ovr["size_mb"] - plain["size_mb"]) > 1e-9)
    res["override_arm"] = {
        "status": "MEASURED" if changed else "UNAVAILABLE",
        "size_mb": ovr.get("size_mb"), "plain_size_mb": plain.get("size_mb"),
        "verdict": ovr.get("pass") if changed else None,
        "reason": None if changed else (
            "llama-quantize b9851 --tensor-type produced a byte-identical file in every tried form "
            "(glob/literal/bare name, before/after positionals, with/without --allow-requantize); "
            "the override was not applied, so this arm measured nothing")}
    res["override_passes"] = bool(ovr.get("pass")) if changed else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(common.WORK))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    work = Path(a.work)
    out = Path(a.out or (work / "remedy_baseline.json"))
    scratch = Path("/tmp/theseus-remedy"); shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    # thresholds and the base reference are read from the committed base cell, never restated
    base_cell = json.loads((work / "ops" / "base.gguf.json").read_text())
    contract = dict(base_cell.get("pass_contract") or {})
    bq = (base_cell.get("results") or {}).get("q4_k_m") or {}
    contract.setdefault("base_rel_dppl", bq.get("rel_dppl"))
    contract.setdefault("base_kl_mean", bq.get("kl_mean"))
    contract.setdefault("rel_dppl_slack", 0.010)
    contract.setdefault("kl_mean_slack", 0.005)
    if contract["base_rel_dppl"] is None or contract["base_kl_mean"] is None:
        print("UNAVAILABLE: base reference cell lacks the q4_k_m reference numbers")
        return 1

    corpus = scratch / "eval_32k.txt"
    corpus.write_bytes(common.EVAL_TEXT.read_bytes()[:32768])

    arms = {"contract_used": contract, "ladder": LADDER,
            "override_families": OVERRIDE_FAMS, "artifacts": []}
    targets = [("base", common.REF_MODEL), ("g3_pow2", work / "g3_pow2")]
    for label, d in targets:
        if not (Path(d) / "model.safetensors").exists():
            arms["artifacts"].append({"artifact": label, "status": "UNAVAILABLE",
                                      "reason": "artifact not on disk; run make_variants --only "
                                                f"{label} first"})
            continue
        print(f"--- {label} ---", flush=True)
        r = artifact_arm(Path(label), scratch, corpus, contract, Path(d))
        arms["artifacts"].append(r)
        for t, e in r.get("ladder", {}).items():
            if e.get("status") == "OK":
                print(f"   {t:20} ppl={e['ppl']} rel={None if e['rel_dppl'] is None else round(e['rel_dppl'],5)} "
                      f"kl={e['kl_mean']} {e['size_mb']}MB pass={e.get('pass')}", flush=True)
        print(f"   min passing type={r.get('min_passing_type')} ({r.get('min_passing_mb')} MB) "
              f"override_passes={r.get('override_passes')}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(arms, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
