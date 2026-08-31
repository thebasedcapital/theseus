"""Tests for ledger/verify.py.

Every check is paired with a mutation that must trip it. A provenance guard that has only ever
returned PASS is indistinguishable from a guard that cannot fail, which is precisely how incidents
#18 and #19 survived: the harness self-check replaced the unit under test with a lambda.
"""
from __future__ import annotations

import json, subprocess, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger import verify as V  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def head_sha() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def real_script_at(sha: str, needle: str) -> str | None:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", sha],
                         capture_output=True, text=True).stdout.split()
    return next((p for p in out if p.endswith(needle)), None)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.sha = head_sha()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self._orig_root, V.ROOT = V.ROOT, self.root
        # mirror the repo layout the checks resolve against
        (self.root / ".git").symlink_to(ROOT / ".git")
        subprocess.run(["git", "-C", str(self.root), "config", "safe.*", "true"], capture_output=True)

    def tearDown(self):
        V.ROOT = self._orig_root
        self.tmp.cleanup()

    def write(self, name, payload):
        p = self.work / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload))
        return p

    def quant_cell(self, **over):
        d = {"script": real_script_at(self.sha, "gguf_probe.py"), "git_head": self.sha,
             "results": {"q8_0": {"status": "OK"}, "q4_k_m": {"status": "OK"}}}
        d.update(over)
        return d

    def adapt_cell(self, **over):
        rec = {"contract_version": V.CURRENT_ADAPT_CONTRACT, "capture": 0.9,
               "task_loss_before": 2.0, "task_loss_after": 0.5, "base_frozen": True}
        d = {"script": real_script_at(self.sha, "adapt_probe.py"), "git_head": self.sha,
             "results": {"variant": rec}}
        d.update(over)
        return d

    def test_happy_path_is_clean(self):
        self.write("q.json", self.quant_cell())
        self.write("a.json", self.adapt_cell())
        rep = V.verify(self.work)
        self.assertEqual([], rep["violations"], rep)
        self.assertEqual("PASS", rep["verdict"])

    def test_flags_unresolvable_commit(self):
        self.write("q.json", self.quant_cell(git_head="0" * 24))
        rep = V.verify(self.work)
        self.assertEqual(1, len(rep["violations"]))
        self.assertIn("does not resolve", rep["violations"][0]["problem"])

    def test_flags_script_absent_at_recorded_commit(self):
        self.write("q.json", self.quant_cell(script="never_existed_probe.py"))
        rep = V.verify(self.work)
        self.assertEqual(1, len(rep["violations"]))
        self.assertIn("absent from the recorded commit", rep["violations"][0]["problem"])

    def test_flags_v2_adapt_record_missing_capture(self):
        """The #18 signature: the v2 writer returns capture unconditionally."""
        cell = self.adapt_cell()
        for k in ("capture", "task_loss_before", "task_loss_after"):
            cell["results"]["variant"].pop(k, None)
        self.write("a.json", cell)
        rep = V.verify(self.work)
        self.assertEqual(1, len(rep["violations"]))
        self.assertIn("#18 signature", rep["violations"][0]["problem"])

    def test_flags_superseded_contract_version(self):
        cell = self.adapt_cell()
        cell["results"]["variant"]["contract_version"] = "adapt-v1-anything"
        self.write("a.json", cell)
        rep = V.verify(self.work)
        self.assertEqual(1, len(rep["violations"]))
        self.assertIn("not the current", rep["violations"][0]["problem"])

    def test_unversioned_adapt_record_is_warning_not_violation(self):
        cell = self.adapt_cell()
        cell["results"]["variant"].pop("contract_version")
        self.write("a.json", cell)
        rep = V.verify(self.work)
        self.assertEqual([], rep["violations"])
        self.assertEqual(1, len(rep["unversioned"]))
        self.assertEqual("PASS WITH WARNINGS", rep["verdict"])

    def test_quantization_cell_needs_per_scheme_status(self):
        cell = self.quant_cell()
        cell["results"]["q4_k_m"].pop("status")
        self.write("q.json", cell)
        rep = V.verify(self.work)
        self.assertEqual(1, len(rep["violations"]))
        self.assertIn("results.q4_k_m.status", rep["violations"][0]["problem"])

    def test_quarantine_scan_finds_the_real_m3_record(self):
        """The check must catch the actual artefact it was written for, not only fixtures."""
        bad = V.check_quarantined(ROOT / "m3" / "results.json")
        self.assertEqual({"A.adapt", "B.adapt"}, {x["subject"] for x in bad})
        self.assertTrue(all("QUARANTINE" in x["status"] for x in bad))

    def test_quarantine_scan_accepts_a_sound_record(self):
        good = {"history_construction": {"A": {"adapt": {"capture": 0.9}}}}
        p = self.root / "ok.json"
        p.write_text(json.dumps(good))
        self.assertEqual([], V.check_quarantined(p))


if __name__ == "__main__":
    unittest.main()
