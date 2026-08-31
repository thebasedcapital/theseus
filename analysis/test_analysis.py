# test_analysis.py — scoped self-check for the analysis slice. Asserts the planted-fixture
# effects are recovered (baserates prevalence/confusion/catastrophe, threshold contract emission
# with its invalidation set, matched-pair K-8 finder) and that the null stays silent on
# pure noise, plus Wilson closed-form cross-checks and the clean no-data behaviour.
# Owned by BaseRates. Run: python -m unittest discover -s analysis

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(__file__)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import loader
import baserates
import thresholds
import pairs
import fixtures
from baserates import wilson, wilson_rate


def make_tmp():
    d = tempfile.mkdtemp(prefix="theseus-an-")
    return Path(d)


def rm(d):
    shutil.rmtree(d, ignore_errors=True)


def capture(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **kw)
    return buf.getvalue()


class TestLoader(unittest.TestCase):
    def test_no_data_is_empty_and_loud(self):
        root = make_tmp()
        try:
            inp = loader.Inputs(root=str(root))
            out = capture(loader.load_all, inp)
            frames = loader.load_all(inp)
            self.assertEqual(frames["scans"]["family"], [])
            self.assertEqual(frames["scans"]["total"], [])
            self.assertEqual(frames["harvest"]["manifest"], [])
            self.assertEqual(frames["harvest"]["edges"], [])
            self.assertEqual(frames["cells"], [])
            self.assertEqual(frames["labels"], [])
            self.assertIn("no data yet:", out)
            self.assertGreater(out.count("no data yet:"), 0)
        finally:
            rm(root)

    def test_inspector_scan_parses(self):
        root = make_tmp()
        try:
            fams = {"q_proj": {"q4_block_mse": 0.0113, "q4_block_mse_pooled": None,
                               "dyn_range_log10": 8.8, "row_energy_imbalance": 2e4,
                               "amax_over_rms": 60.0, "frac_below_f16_normal": 0.003,
                               "weights": 2.5e7}}
            fixtures.write_inspector_json(root / "scans", "aaa", fams,
                                          {"q4_block_mse": 0.0113,
                                           "frac_below_f16_normal": 0.003})
            inp = loader.Inputs(root=str(root))
            scans = loader.load_scans(inp)
            self.assertEqual(len(scans["family"]), 1)
            self.assertEqual(scans["family"][0]["artifact"], "aaa")
            self.assertEqual(scans["family"][0]["family"], "q_proj")
            self.assertAlmostEqual(scans["family"][0]["q4_block_mse"], 0.0113)
            self.assertEqual(len(scans["total"]), 1)
            self.assertEqual(scans["total"][0]["artifact"], "aaa")
            self.assertEqual(scans["context"]["aaa"]["artifact"], "aaa")
        finally:
            rm(root)

    def test_harvest_manifest_synth_edges(self):
        root = make_tmp()
        try:
            (root / "harvest").mkdir(parents=True)
            with open(root / "harvest" / "manifest.jsonl", "w") as f:
                f.write(json.dumps({"id": "child1", "kind": "finetune",
                                    "declared_base": ["baseA"], "arch": "Qwen2ForCausalLM",
                                    "params": 5e8, "dtype": "bf16", "quant": None,
                                    "files": {}, "weights_present": True, "sha256_16": "ab12",
                                    "downloads": 7, "created": "2026-01-01"}) + "\n")
                f.write(json.dumps({"id": "child2", "kind": "merge", "declared_base": None}) + "\n")
            inp = loader.Inputs(root=str(root))
            h = loader.load_harvest(inp)
            self.assertEqual(len(h["manifest"]), 2)
            implied = [e for e in h["edges"] if e["parent"] == "baseA" and e["child"] == "child1"]
            self.assertEqual(len(implied), 1)
        finally:
            rm(root)


class TestWilson(unittest.TestCase):
    """Wilson score interval, hand-implemented; cross-checked against its closed-form endpoints."""

    def test_k0_closed_form(self):
        # k=0: interval is [0, z^2/(n+z^2)] exactly.
        n, z = 10, 1.96
        lo, hi = wilson(0, n, z)
        self.assertAlmostEqual(lo, 0.0, places=12)
        self.assertAlmostEqual(hi, z * z / (n + z * z), places=9)

    def test_kn_closed_form(self):
        # k=n: interval is [n/(n+z^2), 1] exactly.
        n, z = 10, 1.96
        lo, hi = wilson(n, n, z)
        self.assertAlmostEqual(hi, 1.0, places=12)
        self.assertAlmostEqual(lo, n / (n + z * z), places=9)

    def test_point_five_symmetric(self):
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(lo + hi, 1.0, places=9)
        self.assertAlmostEqual((lo + hi) / 2.0, 0.5, places=9)

    def test_zero_n_yields_none(self):
        self.assertEqual(wilson(0, 0), (None, None))
        self.assertEqual(wilson_rate(0, 0), (None, None, None))

    def test_rate_point_estimate(self):
        rate, lo, hi = wilson_rate(8, 24)
        self.assertAlmostEqual(rate, 8.0 / 24.0, places=9)
        self.assertLess(lo, 8.0 / 24.0)
        self.assertGreater(hi, 8.0 / 24.0)

    def test_interval_contains_true_rate(self):
        # planting: 8/24 -> interval must contain the population rate 8/24 (and cover 0.33)
        rate, lo, hi = wilson_rate(8, 24)
        self.assertGreaterEqual(hi, 8.0 / 24.0 - 1e-12)
        self.assertLessEqual(lo, 8.0 / 24.0 + 1e-12)


class TestBaserates(unittest.TestCase):
    def test_prevalence_by_arch_partitions(self):
        """Prevalence is reported per architecture and never pooled across archs (I3)."""
        def row(art, arch, frac, j=0.0113):
            return {"artifact": art, "family": "q_proj", "level": "family", "arch": arch,
                    "frac_below_f16_normal": frac, "q4_block_mse": j,
                    "q4_block_mse_pooled": None, "dyn_range_log10": 8.8,
                    "row_energy_imbalance": 2e4, "amax_over_rms": 60.0}
        # A_ARCH: 2/3 fire export.f16 (thr 0.02); B_ARCH: 1/3 fires
        fam_rows = [row("a1", "A_ARCH", 0.05), row("a2", "A_ARCH", 0.04),
                    row("a3", "A_ARCH", 0.01), row("b1", "B_ARCH", 0.05),
                    row("b2", "B_ARCH", 0.01), row("b3", "B_ARCH", 0.01)]
        by_arch = baserates.prevalence_by_arch(fam_rows, [], "export.f16")
        self.assertEqual(sorted(by_arch), ["A_ARCH", "B_ARCH"])
        self.assertEqual(by_arch["A_ARCH"]["n"], 3)
        self.assertAlmostEqual(by_arch["A_ARCH"]["rate"], 2.0 / 3.0)
        self.assertEqual(by_arch["B_ARCH"]["n"], 3)
        self.assertAlmostEqual(by_arch["B_ARCH"]["rate"], 1.0 / 3.0)

    def test_prevalence_by_arch_defaults_unknown(self):
        """Rows without an arch are grouped under 'unknown', not dropped."""
        row = {"artifact": "z", "family": "q_proj", "level": "family",
               "frac_below_f16_normal": 0.05, "q4_block_mse": 0.0113}
        by_arch = baserates.prevalence_by_arch([row], [], "export.f16")
        self.assertEqual(list(by_arch), ["unknown"])
        self.assertEqual(by_arch["unknown"]["n"], 1)
        self.assertAlmostEqual(by_arch["unknown"]["rate"], 1.0)

    def test_effect_recovered_under_true_cut(self):
        root = make_tmp()
        try:
            gt = fixtures.effect_data(root, rng=1)
            inp = loader.Inputs(root=str(root))
            res = baserates.compute(inp, thr_overrides={"export.f16": gt["true_cuts"]["export.f16"]})
            ef = res["flags"]["export.f16"]
            artp = ef["prevalence"]["artifact"]
            # planted cohort = 8 of 24 -> prevalence 1/3, interval covers it
            self.assertEqual(artp["n"], gt["n_artifacts"])
            self.assertAlmostEqual(artp["rate"], gt["n_cohort"] / gt["n_artifacts"])
            self.assertLessEqual(artp["lo"], artp["rate"])
            self.assertGreaterEqual(artp["hi"], artp["rate"])
            # effect is confined to q_proj: its family prevalence matches the artifact rate,
            # and the other families stay clean
            famp = ef["prevalence"]["families"]
            self.assertAlmostEqual(famp["q_proj"]["rate"], artp["rate"])
            self.assertEqual(famp["gate_proj"]["count"], 0)
            # clean confusion under the true cut
            conf = ef["confusion"]["artifact"]
            self.assertEqual(conf["tp"], gt["clean_confusion"]["tp"])
            self.assertEqual(conf["fp"], 0)
            self.assertEqual(conf["tn"], gt["clean_confusion"]["tn"])
            self.assertEqual(conf["fn"], 0)
            self.assertAlmostEqual(conf["sens"], 1.0)
            self.assertAlmostEqual(conf["spec"], 1.0)
            # export collapse is FAIL but not catastrophic (damage < 100x reference)
            c = ef["catastrophic"]
            self.assertEqual(c["defined"], gt["n_artifacts"])
            self.assertEqual(c["catastrophic"], 0)
            # adaptation capture collapse IS catastrophic
            ca = res["flags"]["adapt.lora.r16"]["catastrophic"]
            self.assertEqual(ca["catastrophic"], gt["n_cohort"])
        finally:
            rm(root)

    def test_v2_overfires_on_healthy_never_misses_cohort(self):
        root = make_tmp()
        try:
            gt = fixtures.effect_data(root, rng=1)
            inp = loader.Inputs(root=str(root))
            res = baserates.compute(inp)   # referenced contract = v2
            conf = res["flags"]["export.f16"]["confusion"]["artifact"]
            self.assertEqual(conf["fn"], 0)                     # cohort never missed (safe)
            self.assertEqual(conf["fp"], gt["v2_export_fp"])    # healthy over-flagged by v2
        finally:
            rm(root)


class TestThresholds(unittest.TestCase):
    def test_no_data_contract_unchanged(self):
        root = make_tmp()
        try:
            res, contracts_dir = thresholds.compute(loader.Inputs(root=str(root)))
            self.assertFalse(res["emitted"])
            self.assertIsNone(thresholds.emit(res, contracts_dir))
            out = "\n".join(thresholds.format_report(res, None))
            self.assertIn("contract unchanged", out)
        finally:
            rm(root)

    def test_effect_emits_contract_v3_with_invalidation_set(self):
        root = make_tmp()
        try:
            gt = fixtures.effect_data(root, rng=1)
            inp = loader.Inputs(root=str(root))
            res, contracts_dir = thresholds.compute(inp)
            path = thresholds.emit(res, contracts_dir)
            self.assertIsNotNone(path)
            self.assertTrue(path.name.startswith("contract-3"))
            doc = json.loads(Path(path).read_text())
            self.assertEqual(doc["version"], 3)
            self.assertEqual(doc["prev_version"], 2)
            ex = doc["flags"]["export.f16"]
            self.assertGreaterEqual(ex["n"], 20)
            self.assertAlmostEqual(ex["quality"] if "quality" in ex else ex["precision"], 1.0,
                                   delta=1e-9)
            self.assertGreaterEqual(ex["recall"] if "recall" in ex else 1.0, 0.95)
            # chosen cut sits in the (healthy-max, cohort-min) gap of q_proj frac_below
            scans = loader.load_scans(inp)
            frac = {r["artifact"]: r["frac_below_f16_normal"]
                    for r in scans["family"] if r["family"] == "q_proj"}
            hi_healthy = max(frac[a] for a in gt["healthy"])
            lo_cohort = min(frac[a] for a in gt["cohort"])
            self.assertGreater(ex["threshold"], 0.8 * hi_healthy)
            self.assertLess(ex["threshold"], lo_cohort)
            # the prior (v2) verdicts it invalidates are exactly the over-flagged healthy artifacts
            inv = [i for i in doc["invalidates"] if i["flag"] == "export.f16"]
            self.assertEqual(sorted({i["artifact"] for i in inv}), sorted(gt["healthy"]))
            self.assertTrue(all((i["old_verdict"], i["new_verdict"]) == ("AT_RISK", "OK")
                                for i in inv))
        finally:
            rm(root)

    def test_noise_does_not_emit(self):
        root = make_tmp()
        try:
            fixtures.noise_data(root, rng=3, fail_prob=0.4)
            inp = loader.Inputs(root=str(root))
            res, contracts_dir = thresholds.compute(inp)
            self.assertFalse(res["emitted"])
            self.assertIsNone(thresholds.emit(res, contracts_dir))
        finally:
            rm(root)

    def test_small_n_does_not_emit(self):
        root = make_tmp()
        try:
            fixtures.effect_data(root, rng=5, n_artifacts=14, n_cohort=4)   # n=14 < 20
            inp = loader.Inputs(root=str(root))
            res, contracts_dir = thresholds.compute(inp)
            self.assertFalse(res["emitted"])
        finally:
            rm(root)

    def test_history_is_never_rewritten(self):
        root = make_tmp()
        try:
            fixtures.effect_data(root, rng=1)
            inp = loader.Inputs(root=str(root))
            res, contracts_dir = thresholds.compute(inp)
            p1 = thresholds.emit(res, contracts_dir)
            self.assertEqual(Path(p1).name, "contract-3.json")
            before = Path(p1).read_text()
            # a second run writes v4; v3 is untouched byte-for-byte
            res2, contracts_dir2 = thresholds.compute(inp)
            p2 = thresholds.emit(res2, contracts_dir2)
            self.assertEqual(Path(p2).name, "contract-4.json")
            self.assertEqual(Path(p1).read_text(), before)
        finally:
            rm(root)


class TestPairs(unittest.TestCase):
    def test_no_data_unavailable(self):
        root = make_tmp()
        try:
            out = capture(pairs.main, ["--root", str(root)])
            self.assertEqual(pairs.main(["--root", str(root)]), 0)
            self.assertIn("pairs: unavailable", out)
        finally:
            rm(root)

    def test_effect_claims_matched_pairs(self):
        root = make_tmp()
        try:
            gt = fixtures.lineage_effect(root, rng=11, n_groups=10)
            self.assertGreaterEqual(gt["injected_divergent_pairs"], 1)
            res = pairs.sweep_and_null(*(_inputs_for(root)), seed=7)
            pvals = [res["tolerances"][t]["empirical_p"] for t in res["tolerances"]]
            claims = [res["tolerances"][t]["claim"] for t in res["tolerances"]]
            self.assertGreaterEqual(res["n_permutations"], 200)
            self.assertLess(min(pvals), 0.05)
            self.assertGreaterEqual(sum(claims), 1)   # at least one tolerance fires
            # the observed divergent pairs are the injected matched siblings
            obs = {frozenset((a, b)) for t in res["tolerances"]
                   for (a, b, _f, _s) in res["tolerances"][t]["examples"]}
            names = sorted({x for pair in obs for x in pair})
            self.assertTrue(all(n.startswith("family") for n in names))
            self.assertLessEqual(len(obs), gt["injected_divergent_pairs"] + 1)
        finally:
            rm(root)

    def test_noise_null_stays_silent(self):
        root = make_tmp()
        try:
            fixtures.lineage_noise(root, rng=23, n_groups=10)
            res = pairs.sweep_and_null(*(_inputs_for(root)), seed=7)
            for t in res["tolerances"]:
                self.assertFalse(res["tolerances"][t]["claim"])
                self.assertGreaterEqual(res["tolerances"][t]["empirical_p"], 0.05)
                self.assertEqual(res["tolerances"][t]["n_divergent"], 0)
        finally:
            rm(root)


def _inputs_for(root):
    inp = loader.Inputs(root=str(root))
    frames = loader.load_all(inp)
    labels = baserates.all_labels(frames)
    fam_rows = frames["scans"]["family"]
    total_rows = frames["scans"]["total"]
    edges = frames["harvest"]["edges"]
    return fam_rows, total_rows, labels, edges


if __name__ == "__main__":
    unittest.main()
