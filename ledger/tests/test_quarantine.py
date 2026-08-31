"""Tests for the quarantine manifest and the LoRA base-reference selection.

Both were written to stop a specific recurrence: incident #18 left voided evidence citable because
the quarantine existed only as prose in three markdown files, and incident #19's lesson (a duplicate
implementation silently changed the experiment) applies to calibrating a v2 verdict against an
unversioned cell.
"""
from __future__ import annotations

import json, subprocess, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger import import_m1 as IM  # noqa: E402
from ledger import verify as V  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _repo_manifest():
    return json.loads((ROOT / "ledger" / "quarantine.json").read_text())


class ManifestTests(unittest.TestCase):
    def test_manifest_is_well_formed_and_paths_exist(self):
        entries = _repo_manifest()["quarantined"]
        self.assertGreaterEqual(len(entries), 4)
        for e in entries:
            for field in ("path", "status", "reason", "incident", "may_be_cited_as"):
                self.assertIn(field, e), f"{e.get('path')} missing {field}"
            self.assertTrue((ROOT / e["path"]).exists(),
                            f"quarantine names {e['path']} which is gone - voided records stay")

    def test_hard_statuses_are_skipped_caveated_are_not(self):
        man = {e["path"]: e for e in _repo_manifest()["quarantined"]}
        skipped = V.skipped_paths(man)
        self.assertIn("m3/results.json", skipped)
        self.assertIn("m1/work/invalidated/full_model_training", skipped)
        # the caveated one must still be audited, or its warning disappears
        self.assertNotIn("m1/work/probes/base_adapt.json", skipped)

    def test_missing_manifest_is_reported_not_swallowed(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("__error__", V.load_quarantine(Path(d)))

    def test_stale_manifest_entry_is_a_violation(self):
        """Deleting a quarantine entry quietly is the same failure as deleting the evidence."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "ledger").mkdir()
            (root / "ledger" / "quarantine.json").write_text(json.dumps(
                {"version": 1, "quarantined": [{"path": "gone/results.json", "status": "voided",
                                                "reason": "x", "incident": 99}]}))
            rep = V.verify(root / "nope", repo=root)
            self.assertTrue(any("no longer exists" in v["problem"]
                                for v in rep["violations"]), rep["violations"])

    def test_verifier_still_detects_the_real_18_signature(self):
        bad = V.check_quarantined(ROOT / "m3" / "results.json")
        self.assertEqual({"A.adapt", "B.adapt"}, {x["subject"] for x in bad})


class BaseReferenceTests(unittest.TestCase):
    """`_adapt_reference` must adopt a v2 cell first and label an unversioned one."""

    def _cell(self, contract):
        rec = {"capture": 0.9, "protected_dppl": 1.2, "seed": 1, "lora_rank": 16,
               "steps": 80, "device": "cuda", "seq_len": 128}
        if contract:
            rec["contract_version"] = contract
        return {"results": {"variant": rec}}

    def test_prefers_versioned_over_ordering(self):
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)
            # the unversioned candidate is listed FIRST on purpose: preference is by contract,
            # not by iteration order
            (w / "probes").mkdir()
            (w / "probes" / "base_adapt.json").write_text(json.dumps(self._cell(None)))
            (w / "ref_capture.json").write_text(json.dumps(self._cell(IM.ADAPT_V2)))
            got = IM._adapt_reference(w)
            self.assertEqual("versioned", got["base_reference_status"])
            self.assertEqual(IM.ADAPT_V2, got["contract_version"])

    def test_unversioned_fallback_is_labelled_not_silent(self):
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)
            (w / "probes").mkdir()
            (w / "probes" / "base_adapt.json").write_text(json.dumps(self._cell(None)))
            got = IM._adapt_reference(w)
            self.assertEqual("unversioned-fallback", got["base_reference_status"])
            self.assertEqual("probes/base_adapt.json", got["base_reference_file"])

    def test_no_candidates_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(IM._adapt_reference(Path(d)))

    def test_real_ledger_calibration_cites_a_versioned_source(self):
        """Guards the actual import: the shipped base calibration cell must not rest on the
        unversioned probes record without saying so."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable, "-m", "ledger.cli", "--root", d, "import-m1",
                            "--work", str(ROOT / "m1" / "work")],
                           cwd=ROOT, capture_output=True, text=True, check=True)
            from ledger.store import Ledger
            cal = [c for c in Ledger(d).all("cell")
                   if c.get("obligation") == "K-3.calibration.lora"]
            self.assertEqual(1, len(cal))
            src = cal[0]["provenance"]["source_file"]
            self.assertTrue("adapt-v2" in src or "unversioned" in src,
                            f"base reference not labelled: {src}")


if __name__ == "__main__":
    unittest.main()
