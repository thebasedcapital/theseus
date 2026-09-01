"""Tests for analysis/merge_frontier.py.

The regression that matters is single-alpha grading. An earlier revision minimised rule_loss_ratio
and ppl_ratio independently and then combined them, so a candidate could "pass" by taking a good
retention at one alpha and a good perplexity at another - a merge that no single setting realises.
Every verdict test below therefore checks the alpha it is credited to actually satisfies both terms.
"""
from __future__ import annotations

import json, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_frontier as MF  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def cell(cand_rows, contract=None):
    """Build a merge_probe-shaped results blob."""
    c = contract or {"ppl_ratio_max": 1.05, "rule_loss_ratio_max": 0.75}
    def rows(items):
        return {"matrix": [{"alpha": a, "rule_loss_ratio": r, "ppl_ratio": p, "pass": False}
                           for a, r, p in items],
                "smallest_passing_alpha": None}
    return {"results": {
        "contract": c,
        "linear": rows(cand_rows["linear"]),
        "ties": rows(cand_rows["ties"]),
        "specialist": {"rule_loss": 0.0194}, "base_rule_loss": 1.9245}}


BASE_OK = {"linear": [(0.5, 1.227, 1.003), (0.7, 0.958, 1.010)],
           "ties": [(0.5, 2.070, 1.004), (0.7, 0.981, 1.009)]}
DESTROYED = {"linear": [(0.5, 768.268, 696800.4), (0.7, 769.0, 700000.0)],
             "ties": [(0.5, 770.0, 710000.0), (0.7, 772.168, 717216.8)]}
# retention fine at a=0.7 but ppl only ever 1.057 -> must NOT pass
REPAIRED = {"linear": [(0.5, 1.663, 1.084), (0.7, 0.939, 1.057)],
            "ties": [(0.5, 1.9, 1.060), (0.7, 1.012, 1.056)]}


def grade_with(cands):
    with tempfile.TemporaryDirectory() as d:
        w = Path(d)
        for name, rows in cands.items():
            (w / f"{name}.merge.json").write_text(json.dumps(cell(rows)))
        return MF.grade(w)


class FrontierTests(unittest.TestCase):
    def test_base_defines_a_satisfiable_ceiling(self):
        rep = grade_with({"base": BASE_OK})
        lin = rep["ops"]["linear"]
        self.assertAlmostEqual(lin["base_rule_frontier"], 0.958)
        self.assertGreater(lin["rule_ceiling"], 0.958)          # base must be able to pass
        self.assertTrue(lin["candidates"]["base"]["verdict_frontier_relative"])

    def test_destroyed_candidate_fails_every_axis(self):
        rep = grade_with({"base": BASE_OK, "g3_pow2": DESTROYED})
        c = rep["ops"]["linear"]["candidates"]["g3_pow2"]
        self.assertFalse(c["verdict_frontier_relative"])
        self.assertFalse(c["verdict_mixed_ppl_absolute"])
        self.assertEqual(c["passing_alphas_frontier"], [])

    def test_repaired_candidate_cannot_borrow_across_alphas(self):
        """a=0.7 passes retention, a=0.5 has the better ppl, but NO single alpha passes both."""
        rep = grade_with({"base": BASE_OK, "g3_pow2_rep": REPAIRED})
        c = rep["ops"]["linear"]["candidates"]["g3_pow2_rep"]
        self.assertLess(c["best_rule_loss_ratio"], rep["ops"]["linear"]["rule_ceiling"],
                        "retention alone looks fine - that is the trap")
        self.assertEqual(c["passing_alphas_frontier"], [])
        self.assertFalse(c["verdict_frontier_relative"])

    def test_absent_base_is_unavailable_not_zero(self):
        rep = grade_with({"g3_pow2": DESTROYED})
        self.assertEqual(rep["status"], "UNAVAILABLE")

    def test_missing_candidate_is_unavailable_not_failed(self):
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)
            (w / "base.merge.json").write_text(json.dumps(cell(BASE_OK)))
            rep = MF.grade(w, candidates=("base", "ghost"))
        self.assertNotIn("ghost", rep["ops"]["linear"]["candidates"])

    def test_old_absolute_term_rejects_the_base_itself(self):
        """Incident #21 in one assertion: the reference cannot satisfy 0.75, so it is unattainable."""
        rep = grade_with({"base": BASE_OK})
        self.assertFalse(rep["ops"]["linear"]["candidates"]["base"]["absolute_retention_verdict_old"])

    def test_both_ppl_ceilings_are_reported_where_they_disagree(self):
        rep = grade_with({"base": BASE_OK, "g3_pow2_rep": REPAIRED})
        lin = rep["ops"]["linear"]
        self.assertIsNotNone(lin["ppl_ceiling_abs"])
        self.assertIsNotNone(lin["ppl_ceiling_rel"])

    @unittest.skipUnless((ROOT / "m1" / "work-qwen3" / "base.merge.json").exists(),
                         "Qwen3 merge cells are machine-local evidence")
    def test_real_cells_reproduce_the_recorded_verdicts(self):
        rep = MF.grade(ROOT / "m1" / "work-qwen3")
        for op in ("linear", "ties"):
            cands = rep["ops"][op]["candidates"]
            self.assertTrue(cands["base"]["verdict_frontier_relative"], f"{op} base must pass")
            self.assertFalse(cands["g3_pow2"]["verdict_frontier_relative"], f"{op} gauge must fail")
        self.assertEqual(rep["ops"]["linear"]["candidates"]["g3_pow2_rep"]
                         ["passing_alphas_frontier"], [])


if __name__ == "__main__":
    unittest.main()
