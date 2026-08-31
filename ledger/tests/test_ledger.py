"""Tests for the write-once ledger spine (LedgerSpine) and its invariants.

Stdlib only (`unittest`). Each test builds its own temporary Ledger root under
`tempfile.TemporaryDirectory` — nothing is ever written outside /tmp.
Coverage (>=8 tests): deterministic ids, write conflict, env refusal (I3),
I4 calibration gate, I5 control cap, I8 predicted/unavailable never tallied,
K-7 scalar-health rejection, I1 invalidates edges, and a tiny import-m1 fixture.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ledger import rules
from ledger.env import env_digest, mixed_environments, environments_comparable
from ledger.import_m1 import run
from ledger.claims import Obligation
from ledger.store import Ledger, LedgerError, IdConflictError, canonical, content_id

CONVENTION = "mean_of_per_tensor_ratios"


def env(**kw):
    base = {"code_snapshot": "g", "torch": "2.13.0+cu130",
            "corpus": {"file": "w.txt", "byte_range": [0, 32], "sha256_16": "ab"},
            "seqlen": 512, "contract_version": "c1"}
    base.update(kw)
    return base


def artifact_body(name="a1"):
    return {"kind": "artifact",
            "origin": {"hf": f"org/{name}", "revision": "main", "blob_sha256": "hash-" + name},
            "container": {"file": "model.safetensors", "bytes": 100, "sha256": "hash-" + name,
                          "tensors": 2, "weights": 100, "dtype": "BF16", "tied": False},
            "config": {"arch": "A"}, "ancestry": [],
            "features": {"total": {"q4_block_mse": 0.011, "convention": CONVENTION}}}


def cell_body(obligation="T.cell", **kw):
    body = {"kind": "cell",
            "op": {"name": "quantize.gguf.q8_0", "spec": {"tag": "q8_0"}, "contract": "v2",
                   "reference_cell": None},
            "subject": "art", "environment": env(),
            "lease": {"wall_s": 1.0},
            "result": {"status": "measured", "verdict": "pass", "metrics": {"ppl": 1.0}},
            "invalidates": None, "notes": [], "obligation": obligation}
    body.update(kw)
    return body


class TestStore(unittest.TestCase):
    def test_deterministic_ids_and_verify(self):
        """I1: ids are sha256(canonical body minus id)[:12]; re-adding is a noop."""
        with tempfile.TemporaryDirectory() as td:
            led = Ledger(td)
            b = artifact_body("det")
            cid, action = led.add("artifact", b)
            self.assertEqual(action, "added")
            self.assertEqual(len(cid), 12)
            self.assertEqual(cid, content_id(b))
            self.assertTrue(led.path("artifact", cid).exists())
            # byte-identical re-add is a no-op (I1)
            cid2, action2 = led.add("artifact", b)
            self.assertEqual(cid2, cid)
            self.assertEqual(action2, "noop")
            # integrity re-verification is clean
            self.assertEqual(led.verify(), [])
            rec = led.get("artifact", cid)
            self.assertEqual(rec["id"], cid)
            self.assertEqual(rec["kind"], "artifact")

    def test_write_conflict_detection(self):
        """I1: a caller-supplied id must equal the content hash; one key, one meaning."""
        with tempfile.TemporaryDirectory() as td:
            led = Ledger(td)
            b = artifact_body("conf")
            cid, _ = led.add("artifact", b)
            with self.assertRaises(IdConflictError) as ctx:
                led.add("artifact", b, id="0" * 12)
            self.assertIn(cid, str(ctx.exception))  # names the real content hash
            # distinct body under the same human key is a HARD conflict
            k1 = {"kind": "claim", "key": "K-T", "text": "one", "state": "unsupported",
                  "state_history": [], "refuter": {"query": "q", "would_drop_to": "p",
                                                   "answering_ob": "n"}, "numbers": []}
            k2 = dict(k1, text="two")
            led.add("claim", k1, key="K-T")
            with self.assertRaises(IdConflictError) as ctx:
                led.add("claim", k2, key="K-T")
            self.assertIn("K-T", str(ctx.exception))


class TestEnv(unittest.TestCase):
    def test_mixed_environment_refusal(self):
        """I3: joining cells across distinct digests refuses and names ids; allow stamps."""
        a = dict(env(), torch="2.13")
        b = dict(env(), torch="2.14")
        c1 = {"id": "aaaaaaaaaaaa", "environment": a,
              "result": {"status": "measured", "verdict": "pass"}}
        c2 = {"id": "bbbbbbbbbbbb", "environment": b,
              "result": {"status": "measured", "verdict": "pass"}}
        self.assertFalse(environments_comparable(a, b))
        with self.assertRaises(LedgerError) as ctx:
            mixed_environments([c1, c2])
        self.assertIn("aaaaaaaaaaaa", str(ctx.exception))
        self.assertIn("bbbbbbbbbbbb", str(ctx.exception))
        ok, digests, _ = mixed_environments([c1, c2], allow_mixed_env=True)
        self.assertTrue(ok)
        self.assertEqual(len(digests), 2)
        # unknown environments are never comparable (I8: never guess conditions)
        c3 = {"id": "cccccccccccc", "environment": {"unknown": True, "digest": None},
              "result": {"status": "unavailable", "verdict": None}}
        self.assertIsNone(env_digest(c3["environment"]))
        with self.assertRaises(LedgerError):
            mixed_environments([c1, c3])
        # single digest group passes
        ok, digests, _ = mixed_environments([c1])
        self.assertTrue(ok)
        self.assertEqual(len(digests), 1)


class TestCellInvariants(unittest.TestCase):
    def _ledger_with_reference(self):
        led = Ledger(tempfile.mkdtemp())
        ref = cell_body(obligation="K-4.calibration.q8_0")
        ref["op"]["reference_cell"] = None
        ref["result"] = {"status": "measured", "verdict": None, "metrics": {"ppl": 12.1351}}
        ref_id, _ = led.add("cell", ref)
        return led, ref_id, ref["environment"]

    def test_i4_calibration_gate(self):
        """I4: a pass/fail verdict requires an existing calibration reference."""
        led, ref_id, ref_env = self._ledger_with_reference()
        # no reference at all
        bad = cell_body(obligation="K-4.quantize.a.q8_0")
        bad["op"]["reference_cell"] = None
        probs = rules.check_cell(bad, ledger=led)
        self.assertTrue(any("I4" in p for p in probs))
        # reference names a cell that does not exist
        bad2 = cell_body(obligation="K-4.quantize.a.q8_0")
        bad2["op"]["reference_cell"] = "000000000000"
        probs = rules.check_cell(bad2, ledger=led)
        self.assertTrue(any("000000000000" in p for p in probs))
        # valid reference with the same environment passes
        good = cell_body(obligation="K-4.quantize.a.q8_0")
        good["op"]["reference_cell"] = ref_id
        good["environment"] = ref_env
        self.assertEqual(rules.check_cell(good, ledger=led), [])

    def test_i3_env_mismatch_refusal(self):
        """I3: a cell and its reference with differing digests are refused, naming both."""
        led, ref_id, ref_env = self._ledger_with_reference()
        bad = cell_body(obligation="K-4.quantize.a.q8_0")
        bad["op"]["reference_cell"] = ref_id
        bad["environment"] = dict(ref_env, corpus={
            "file": "other.txt", "byte_range": [0, 32], "sha256_16": "cd"})
        probs = rules.check_cell(bad, ledger=led)
        self.assertTrue(any("I3" in p and ref_id in p for p in probs))

    def test_i5_control_cap(self):
        """I5: a missing required control caps a claim at PRELIMINARY and names the promoter."""
        obs = [
            Obligation("equivalence", required=True, satisfied=True, kind="cells",
                       evidence_ids=["c1"]),
            Obligation("controls.identity_roundtrip", required=True, satisfied=False,
                       kind="cells", promoting_ids=["ctl_id_roundtrip"]),
        ]
        cap = rules.claim_cap({"state": "controlled"}, obs)
        self.assertEqual(cap["state"], "preliminary")
        self.assertTrue(cap["capped_by_control"])
        self.assertIn("ctl_id_roundtrip", cap["promoting_cells"])
        # no state is asserted the ledger has not earned:
        # even a CONFIRMED-declared claim caps at PRELIMINARY while a required obligation is bare.
        cap2 = rules.claim_cap({"state": "confirmed"},
                               [Obligation("equivalence", required=True, satisfied=False)])
        self.assertEqual(cap2["state"], "preliminary")

        cap3 = rules.claim_cap(
            {"state": "unsupported"},
            [Obligation("attempt", required=False, satisfied=True, kind="failed_attempt",
                        evidence_ids=["attempt-cell"]),
             Obligation("matched_pair", required=True, satisfied=False, kind="cells")])
        self.assertEqual(cap3["state"], "unsupported")
    def test_i8_predicted_unavailable_not_tallied(self):
        """I8: predicted/unavailable are recorded, never counted; basis/reason are mandatory."""
        with tempfile.TemporaryDirectory() as td:
            led = Ledger(td)
            # predicted without a basis is refused
            badp = cell_body(obligation="K-6.prediction.a")
            badp["result"] = {"status": "predicted", "verdict": "pass", "metrics": {}}
            probs = rules.check_cell(badp, ledger=led)
            self.assertTrue(any("basis" in p and "I8" in p for p in probs))
            # unavailable without a reason is refused
            badu = cell_body(obligation="T")
            badu["result"] = {"status": "unavailable", "verdict": None, "metrics": {}}
            probs = rules.check_cell(badu, ledger=led)
            self.assertTrue(any("I8" in p for p in probs))
            # tally: denominators are measured cells only
            cells = [
                {"id": "m1", "result": {"status": "measured", "verdict": "pass"}},
                {"id": "m2", "result": {"status": "measured", "verdict": "pass"}},
                {"id": "m3", "result": {"status": "measured", "verdict": "fail"}},
                {"id": "p1", "result": {"status": "predicted", "verdict": "pass"}},
                {"id": "u1", "result": {"status": "unavailable", "verdict": None}},
            ]
            t = rules.tally_cells(cells)
            self.assertEqual(t["pass_measured"], 2)
            self.assertEqual(t["fail_measured"], 1)
            self.assertEqual(t["predicted"], 1)
            self.assertEqual(t["unavailable"], 1)
            self.assertEqual(t["denominator_measured"], 3)
            self.assertAlmostEqual(t["pass_rate_measured"], 2 / 3)

    def test_scalar_health_rejected(self):
        """K-7: the schema has no scalar-health field; `render`/admission rejects one."""
        bad = cell_body(obligation="T")
        bad["result"]["metrics"]["health_score"] = 0.5
        probs = rules.check_cell(bad)
        self.assertTrue(any("health_score" in p and "K-7" in p for p in probs))
        art = artifact_body("h")
        art["features"]["total"]["health"] = 1.0
        ap = rules.check_artifact(art)
        self.assertTrue(any("health" in p and "K-7" in p for p in ap))


class TestInvalidatesEdge(unittest.TestCase):
    def test_invalidates_correction_excludes_superseded(self):
        """I1: a correction is a new record with an `invalidates` edge; the old cell stays and
        is never tallied; naming a nonexistent target is refused."""
        with tempfile.TemporaryDirectory() as td:
            led = Ledger(td)
            ref = cell_body(obligation="K-4.calibration.q8_0")
            ref["op"]["reference_cell"] = None
            ref["result"] = {"status": "measured", "verdict": None, "metrics": {}}
            ref_id, _ = led.add("cell", ref)

            old = cell_body(obligation="T.invalidates.old")
            old["op"]["reference_cell"] = ref_id
            old["result"] = {"status": "measured", "verdict": "pass", "metrics": {"ppl": 12.1}}
            old_id, _ = led.add("cell", old)

            corr = cell_body(obligation="T.invalidates.new")
            corr["op"]["reference_cell"] = ref_id
            corr["invalidates"] = old_id
            corr["result"] = {"status": "measured", "verdict": "pass", "metrics": {"ppl": 12.05}}
            self.assertEqual(rules.check_cell(corr, ledger=led), [])
            corr_id, _ = led.add("cell", corr)

            # old cell remains, untouched (write-once)
            self.assertIsNotNone(led.get("cell", old_id))
            stored = led.get("cell", corr_id)
            self.assertEqual(stored["invalidates"], old_id)

            # a correction naming a nonexistent target is refused
            bogus = cell_body(obligation="T.invalidates.bogus")
            bogus["op"]["reference_cell"] = ref_id
            bogus["invalidates"] = "ffffffffffff"
            probs = rules.check_cell(bogus, ledger=led)
            self.assertTrue(any("ffffffffffff" in p for p in probs))

            # tally: superseded old cell excluded, correction counted
            sel = [c for c in led.all("cell") if c.get("obligation", "").startswith("T.invalidates.")]
            t = rules.tally_cells(sel)
            self.assertEqual(t["invalidated"], 1)
            self.assertEqual(t["measured"], 1)
            self.assertEqual(t["denominator_measured"], 1)


FIXTURE_TABLE = (
    "| # | break | symptom | caught by | rule |\n"
    "|---|-------|---------|-----------|------|\n"
    "| 4 | Absolute pass thresholds | pristine base failed its own contract | calibrating on base | Contracts are reference-relative |\n"
    "| 9 | pass: null tallied as failure | driver summary showed base failing | m1/retally.py | null means no verdict, never False |\n"
)


class TestImportM1(unittest.TestCase):
    def _write(self, root: Path, name: str, obj):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)

    def _fixture(self, root: Path):
        """Minimal but real-shaped m1/work tree exercising every PLAN.md §0 mapping."""
        w = root / "work"
        self._write(w, "VARIANTS.json", {
            "base": {"bytes": 100, "untied": False},
            "g3_pow2": {"bytes": 101, "untied": True,
                        "gauge": {"transforms": [{"family": "norm_diag", "mode": "pow2",
                                                  "seed": 1}]}},
        })
        self._write(w, "equiv/g3_pow2.json", {
            "gate": {"kl_mean_nats_max": 0.002, "top1_agree_min": 0.995},
            "verdict": "EQUIVALENT", "torch": "2.13.0+cu130", "duration_s": 1.0,
            "metrics": {"kl_mean_nats": 0.0, "top1_agree": 1.0, "seqlen": 2048},
            "cond_b": {"q_proj": 0.031, "down_proj": 0.010},
        })
        self._write(w, "equiv/g2_rand.json", {   # metadata-only artifact (dir freed)
            "gate": {}, "verdict": "EQUIVALENT", "metrics": {"top1_agree": 0.995},
        })
        self._write(w, "ops/base.gguf.json", {
            "torch": "2.13.0+cu130", "export": {"outtype": "bf16"}, "duration_s": 10.0,
            "results": {"f16": {"ppl": 12.1351, "status": "OK"},
                        "q8_0": {"kl_mean": 0.00094, "rel_dppl": -0.001, "pass": None,
                                 "role": "reference_calibration"}},
        })
        self._write(w, "ops/g3_pow2.gguf.json", {
            "torch": "2.13.0+cu130", "export": {"outtype": "bf16"}, "duration_s": 10.0,
            "results": {"q8_0": {"kl_mean": 10.69, "rel_dppl": 5.0, "pass": False}},
        })
        self._write(w, "ops/base.adapt.json", {
            "torch": "2.13.0+cu130",
            "results": {"variant": {"capture": 0.973, "seed": 1729, "rank": 16, "steps": 80,
                                    "seq_len": 128, "pass": True}},
        })
        self._write(w, "ops/g3_pow2.adapt.json", {
            "torch": "2.13.0+cu130",
            "results": {"variant": {"capture": 0.156, "seed": 1729, "pass": False}},
        })
        merge_contract = {"version": "merge-v2-base-calibrated"}
        self._write(w, "ops/base.merge.json", {
            "torch": "2.13.0+cu130", "duration_s": 5.0,
            "results": {"contract": merge_contract,
                        "linear": {"smallest_passing_alpha": 0.3,
                                   "matrix": [{"alpha": 0.3, "ppl_ratio": 1.03, "pass": True}]},
                        "ties": {"smallest_passing_alpha": 0.4,
                                 "matrix": [{"alpha": 0.4, "ppl_ratio": 1.04, "pass": True}]}}})
        self._write(w, "ops/g3_pow2.merge.json", {
            "torch": "2.13.0+cu130", "duration_s": 5.0,
            "results": {"contract": merge_contract,
                        "linear": {"smallest_passing_alpha": None,
                                   "matrix": [{"alpha": 0.3, "ppl_ratio": 6.0, "pass": False}]},
                        "ties": {"smallest_passing_alpha": None,
                                 "matrix": [{"alpha": 0.4, "ppl_ratio": 9.0, "pass": False}]}}})
        # aggregate row whose export outtype is unrecoverable -> environment.unknown
        self._write(w, "M1_OPS.json", {
            "g2_rand": {"ops": {"gguf": {
                "results": {"q8_0": {"kl_mean": 0.01, "pass": None, "rel_dppl": 0.0}}}},
                "torch": "2.13.0+cu130"},
        })
        self._write(w, "quant_ref.json", {"q8_0": {"tag": "base", "kl_mean": 0.00094}})
        self._write(w, "ref_capture.json", {"capture": 0.973, "seed": 1729, "rank": 16,
                                            "steps": 80})
        self._write(w, "seed_replicate.json", {
            "_contract": {"version": "adapt-v2-true-lora-base-frozen",
                          "base_frozen": True, "gap_threshold_sd": 3},
            "_summary": {"base_capture": 0.97, "max_within_variant_sd": 0.02,
                         "threshold_sd": 3, "gaps_beyond_3sd": [["g3_pow2", -0.87]]},
            "base": {"seeds": {"1729": {"capture": 0.98, "selected_lr": 0.0003,
                                             "grid": [{"runtime_s": 1.0}]}}},
            "g3_pow2": {"seeds": {"1729": {"capture": 0.11, "selected_lr": 0.0003,
                                                "grid": [{"runtime_s": 1.0}]}}},
        })
        self._write(w, "debts_lattice.json", {
            "base": {"J_base": 0.01123, "J_var": 0.01123, "debt": 0.0,
                     "per_tensor": {"q_proj": 0.0113}},
            "g3_pow2": {"J_base": 0.01123, "J_var": 0.03, "debt": 0.0187,
                        "per_tensor": {"q_proj": 0.0295}},
        })
        self._write(w, "PREDICTIONS_new.json", {
            "variants": {"g3_pow2": {"J_base": 0.01123, "J_var": 0.03, "debt": 0.0187}},
        })
        self._write(root, "PIPELINE_FAILURES.md", "## table\n\n" + FIXTURE_TABLE)
        return w

    def test_tiny_import_fixture(self):
        """PLAN §0 mapping on a synthetic m1/work: artifacts, cells, calibration wiring,
        predicted cells, and environment.unknown for unrecoverable conditions."""
        report = __import__("ledger.import_m1", fromlist=["ImportReport"]).ImportReport()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            w = self._fixture(root)
            led = Ledger(root / ".theseus")
            rep = run(led, str(w))
            self.assertEqual(rep.total["artifact"] >= 3, True)   # base + g3_pow2 + g2_rand
            self.assertGreaterEqual(rep.total["cell"], 8)
            self.assertEqual(rep.total["claim"], len(
                __import__("ledger.claims", fromlist=["CLAIM_SEEDS"]).CLAIM_SEEDS))
            self.assertEqual(rep.total["incident"], 2)

            # calibration reference wiring (I4): variant quantize cell points at the base cell
            cal = [c for c in led.all("cell") if c.get("obligation") == "K-4.calibration.q8_0"]
            self.assertEqual(len(cal), 1)
            var = [c for c in led.all("cell") if c.get("obligation") == "K-4.quantize.g3_pow2.q8_0"]
            self.assertEqual(len(var), 1)
            self.assertEqual(var[0]["op"].get("reference_cell"), cal[0]["id"])
            self.assertEqual(var[0].get("subject"),
                             [a for a in led.all("artifact")
                              if "norm_diag" in str(a.get("ancestry"))][0]["id"])
            # merge cells obey the same I4 rule: variant points at the passing base merge cell
            mcal = [c for c in led.all("cell") if c.get("obligation") == "K-9.calibration.merge"]
            mvar = [c for c in led.all("cell") if c.get("obligation") == "K-9.merge.g3_pow2"]
            self.assertEqual(len(mcal), 1)
            self.assertEqual(len(mvar), 1)
            self.assertEqual(mcal[0]["result"]["verdict"], "pass")
            self.assertEqual(mvar[0]["result"]["verdict"], "fail")
            self.assertEqual(mvar[0]["op"]["reference_cell"], mcal[0]["id"])
            self.assertEqual(rules.check_cell(mvar[0], ledger=led), [])

            # unrecoverable conditions -> environment.unknown: true, digest None (never compared)
            unk = [c for c in led.all("cell") if c.get("environment", {}).get("unknown")]
            self.assertEqual(len(unk) >= 1, True)
            self.assertIsNone(env_digest(unk[0]["environment"]))
            self.assertTrue(unk[0]["environment"]["unknown"])
            # predicted cells exist and are status:predicted (I8, never tallied)
            pred = [c for c in led.all("cell")
                    if (c.get("result") or {}).get("status") == "predicted"]
            self.assertGreaterEqual(len(pred), 1)
            for p in pred:
                self.assertTrue((p.get("basis") or {}).get("claim"))

            # run() is deterministic + no reported failures from verify()
            self.assertEqual(led.verify(), [])
            # replicated adaptation rows share the corrected base calibration environment/reference
            seed_rows = [c for c in led.all("cell")
                         if str(c.get("obligation", "")).startswith("K-3.replication.")]
            self.assertGreaterEqual(len(seed_rows), 3)
            lcal = [c for c in led.all("cell") if c.get("obligation") == "K-3.calibration.lora"][0]
            for c in seed_rows:
                self.assertEqual(c["op"]["reference_cell"], lcal["id"])
                self.assertEqual(rules.check_cell(c, ledger=led), [])

    def test_import_deterministic_across_roots(self):
        """Two fresh imports of the same work tree produce identical id sets."""
        with tempfile.TemporaryDirectory() as td:
            root1 = Path(td)
            w1 = self._fixture(root1)
            r1 = Path(td) / "r1"
            r2 = Path(td) / "r2"
            run(Ledger(r1), str(w1))
            run(Ledger(r2), str(w1))
            def id_sets(led):
                return {k: sorted({c["id"] for c in led.all(k)}) for k in
                        ("artifact", "cell", "claim", "incident")}
            self.assertEqual(id_sets(Ledger(r1)), id_sets(Ledger(r2)))

    def test_dry_run_writes_nothing(self):
        """import-m1 --dry-run maps records without writing to disk."""
        report = __import__("ledger.import_m1", fromlist=["ImportReport"]).ImportReport()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            w = self._fixture(root)
            led = Ledger(root / ".theseus")
            rep = run(led, str(w), dry_run=True)
            self.assertGreaterEqual(rep.total["cell"], 8)
            self.assertFalse(led.dir["cell"].exists())


if __name__ == "__main__":
    unittest.main()
