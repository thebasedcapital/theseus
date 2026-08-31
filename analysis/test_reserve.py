"""Tests for analysis/reserve.py.

Fixtures are synthetic so the suite runs on a fresh clone: m1/work/ is gitignored, and a test that
silently no-ops when the evidence directory is absent is worse than no test. The integration test
against real cells is skipped when the cells are not on disk, and says so.
"""
from __future__ import annotations

import json, unittest
from pathlib import Path

import reserve as R

ROOT = Path(__file__).resolve().parents[1]


def cell(results, slack_d=0.01, slack_k=0.005):
    return {"pass_contract": {"rel_dppl_slack": slack_d, "kl_mean_slack": slack_k,
                              "mode": "reference-relative"},
            "results": results}


BASE = cell({"q8_0": {"status": "OK", "rel_dppl": 0.001, "kl_mean": 0.0009, "size_mb": 400.0},
             "q5_k_m": {"status": "OK", "rel_dppl": 0.010, "kl_mean": 0.0100, "size_mb": 262.0},
             "q4_k_m": {"status": "OK", "rel_dppl": 0.022, "kl_mean": 0.0319, "size_mb": 215.0}})


class QuantTests(unittest.TestCase):
    def test_reference_gets_full_reserve_by_definition(self):
        q = R.quant_reserve(BASE, BASE)
        self.assertEqual([1.0, 1.0, 1.0],
                         [q["schemes"][s]["R_rel_dppl"] for s in R.SCHEMES])
        self.assertEqual("q4_k_m", q["deepest_passing"])

    def test_margin_interpolates_instead_of_collapsing_to_boolean(self):
        # q4: rel_dppl 0.027 against reference 0.022 with 0.010 slack -> 1 - 0.005/0.010 = 0.5
        # q4: kl_mean 0.0369 against 0.0319 with 0.005 slack -> excess equals slack -> 0.0
        # q5: rel_dppl 0.012 against 0.010 -> 0.8; kl_mean 0.011 against 0.010 -> 0.8
        v = {"q8_0": {"status": "OK", "rel_dppl": 0.002, "kl_mean": 0.001},
             "q5_k_m": {"status": "OK", "rel_dppl": 0.012, "kl_mean": 0.011},
             "q4_k_m": {"status": "OK", "rel_dppl": 0.027, "kl_mean": 0.0369}}
        q = R.quant_reserve(cell(v), BASE)
        self.assertEqual(0.5, q["schemes"]["q4_k_m"]["R_rel_dppl"])
        self.assertEqual(0.0, q["schemes"]["q4_k_m"]["R_kl_mean"])
        self.assertEqual(0.8, q["schemes"]["q5_k_m"]["R_rel_dppl"])
        self.assertEqual(0.8, q["schemes"]["q5_k_m"]["R_kl_mean"])

    def test_missing_probe_is_unavailable_never_zero(self):
        """I8: absence of evidence must not be tallied as evidence of absence."""
        v = {"q4_k_m": {"status": "FAILED"}}
        q = R.quant_reserve(cell(v), BASE)
        self.assertEqual("UNAVAILABLE", q["schemes"]["q4_k_m"]["status"])
        self.assertNotIn(0.0, [q["schemes"]["q4_k_m"].get("R_rel_dppl")])
        self.assertIsNone(q["deepest_passing"])

    def test_bpw_is_measured_from_bytes_not_labels(self):
        q = R.quant_reserve(BASE, BASE)
        self.assertAlmostEqual(400e6 / R.N_PARAMS * 8, q["schemes"]["q8_0"]["bpw"], places=3)


class AdaptTests(unittest.TestCase):
    CONTRACT = ("adapt-v2: capture >= 0.75*capture_ref AND "
                "protected_dppl <= protected_dppl_ref + 0.02; base globally frozen")

    def cell(self, **over):
        rec = {"capture": 0.9, "capture_ref": 1.0, "capture_threshold": 0.75,
               "protected_dppl": 1.20, "protected_dppl_ref": 1.2238, "pass": True,
               "pass_contract": self.CONTRACT}      # real cells carry it here, not at top level
        rec.update(over)
        return {"results": {"variant": rec}}

    def test_reserve_is_relative_to_the_gate_not_the_reference(self):
        a = R.adapt_reserve(self.cell())
        self.assertEqual(0.6, a["R_adapt"])          # (0.90-0.75)/(1.00-0.75)
        self.assertEqual("capture", a["binding"])

    def test_collateral_term_binds_even_when_capture_is_healthy(self):
        """The bad_all_exact case: capture 0.9455 passes its gate, dppl 1.900 does not."""
        a = R.adapt_reserve(self.cell(protected_dppl=1.90, capture=0.9455))
        self.assertEqual(0.0, a["R_adapt"])
        self.assertEqual("collateral", a["binding"])
        self.assertGreater(a["R_capture"], 0.5)     # capture alone would have looked healthy

    def test_at_threshold_is_zero(self):
        self.assertEqual(0.0, R.adapt_reserve(self.cell(capture=0.75))["R_adapt"])

    def test_pre_threshold_cell_is_unavailable_not_zero(self):
        bad = {"results": {"variant": {"capture": 0.5}}}
        self.assertEqual("UNAVAILABLE", R.adapt_reserve(bad)["status"])

    def test_record_without_capture_is_unavailable(self):
        self.assertEqual("UNAVAILABLE", R.adapt_reserve({"results": {}})["status"])

    def test_missing_collateral_data_is_unavailable_not_a_lie(self):
        """Never fall back to the single-term formula: it silently overstates reserve."""
        cell = {"results": {"variant": {"capture": 0.9, "capture_ref": 1.0,
                                        "capture_threshold": 0.75}}}
        out = R.adapt_reserve(cell)
        self.assertEqual("UNAVAILABLE", out["status"])
        self.assertNotIn("R_adapt", out)

    @unittest.skipUnless((ROOT / "m1" / "work" / "ops" / "base.adapt.json").exists(),
                         "m1/work is gitignored; needs a measured machine")
    def test_reserve_agrees_with_every_recorded_verdict(self):
        """R_adapt > 0 must mean the cell's own pass=True. This is the check that catches a
        reserve built from half a contract."""
        rep = R.build()
        for name, arts in rep["artifacts"].items():
            a = arts.get("adapt.lora.r16") or {}
            if a.get("status") != "MEASURED":
                continue
            self.assertEqual(a["R_adapt"] > 0.0, bool(a["recorded_pass"]),
                             f"{name}: R_adapt={a['R_adapt']} but pass={a['recorded_pass']}")


class MergeTests(unittest.TestCase):
    def test_reserve_is_the_absorbed_specialist_fraction(self):
        m = {"results": {"linear": {"matrix": [{"alpha": a, "pass": a <= 0.4} for a in
                                               (0.3, 0.4, 0.5, 0.6, 0.7)]},
                         "ties": {"matrix": [{"alpha": a, "pass": False} for a in
                                             (0.3, 0.4, 0.5, 0.6, 0.7)]}}}
        r = R.merge_reserve(m)
        self.assertEqual(0.4, r["linear"]["largest_passing_alpha"])
        self.assertAlmostEqual(0.5714, r["linear"]["R_merge"])   # 0.4 / 0.7
        self.assertEqual(0.0, r["ties"]["R_merge"])
        self.assertIsNone(r["ties"]["largest_passing_alpha"])

    def test_unswept_operation_is_unavailable(self):
        self.assertEqual("UNAVAILABLE", R.merge_reserve({})["linear"]["status"])


class BuildIntegration(unittest.TestCase):
    def test_build_over_synthetic_tree(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ops = Path(d) / "ops"
            ops.mkdir()
            (ops / "base.gguf.json").write_text(json.dumps(BASE))
            rep = R.build(Path(d))
        self.assertIn("base", rep["artifacts"])
        self.assertIn("conventions", rep)

    @unittest.skipUnless((ROOT / "m1" / "work" / "ops" / "base.gguf.json").exists(),
                         "m1/work is gitignored; real-cell check needs a measured machine")
    def test_real_cells_reproduce_the_documented_vector_story(self):
        rep = R.build()
        arts = rep["artifacts"]
        # g3_pow2 is the collapse; its repair restores quantization and adaptation reserve.
        self.assertEqual(0.0, arts["g3_pow2"]["quantize"]["schemes"]["q8_0"]["R_rel_dppl"])
        self.assertEqual(0.0, arts["g3_pow2"]["adapt.lora.r16"]["R_adapt"])
        self.assertGreater(arts["g3_pow2_rep"]["adapt.lora.r16"]["R_adapt"], 0.5)
        # K-7: one artifact must improve one operation while losing another.
        prep = arts["prep_base_exact"]
        self.assertGreater(prep["quantize"]["schemes"]["q4_k_m"]["R_rel_dppl"], 0.0)
        self.assertEqual(0.0, prep["merge"]["linear"]["R_merge"])
        self.assertEqual(0.0, prep["merge"]["ties"]["R_merge"])
        # no scalar is emitted anywhere
        self.assertNotIn("health", json.dumps(rep).lower())


if __name__ == "__main__":
    unittest.main()
