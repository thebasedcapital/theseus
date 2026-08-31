#!/usr/bin/env python3
"""No-GPU checks for the K-8 natural-history harness."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import history_pair as h


def main():
    assert h.CONTRACT["history_pair"]["A"] == ["adapt.lora.rule", "merge.linear", "quantize.q4_k_m"]
    assert h.CONTRACT["history_pair"]["B"] == ["merge.linear", "adapt.lora.rule", "quantize.q4_k_m"]
    assert h.CONTRACT["history_seed"] != h.CONTRACT["specialist_seed"]
    assert h.CONTRACT["future_seed"] not in (h.CONTRACT["history_seed"], h.CONTRACT["specialist_seed"])
    assert h.static_match({"q4_block_mse": 1, "dyn_range_log10": 2,
                           "row_energy_imbalance": 3, "frac_below_f16_normal": 4},
                          {"q4_block_mse": 1.04, "dyn_range_log10": 2.02,
                           "row_energy_imbalance": 3.01, "frac_below_f16_normal": 4.01}, .05)
    assert not h.static_match({}, {"q4_block_mse": 1}, .8)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "missing.json"
        got = h.inspect(Path(d), out)
        assert got["status"] == "UNAVAILABLE"
    r = h.null_result(False, 1, 7)
    assert r["n_shuffles"] >= 200 and r["observed_divergent"] is False
    assert h.null_result(True, 200, 7)["n_shuffles"] == 200
    calls = []
    old_train, old_save, old_inspect, old_run, old_load = h.train_lora_state, h.save_sd, h.inspect, h.run_cmd, h.common.load_state
    try:
        h.common.load_state = lambda path: {}
        h.train_lora_state = lambda sd, tok, seed, device: (sd, {"seed": seed, "capture": 0.5})
        h.save_sd = lambda sd, dst: dst
        h.inspect = lambda path, out: {"status": "OK"}
        def fake_run(cmd, commands, timeout=3600):
            calls.append([str(x) for x in cmd])
            class P: returncode = 0; stdout = "PPL = 10.0"; stderr = ""
            if "convert_hf_to_gguf.py" in str(cmd[1]): Path(cmd[3]).touch()
            if "llama-quantize" in str(cmd[0]): Path(cmd[2]).touch()
            return P()
        h.run_cmd = fake_run
        with tempfile.TemporaryDirectory() as d:
            cells = h.future_cells({"A": Path(d), "B": Path(d)}, Path(d), object(), "cpu", [])
            assert set(cells) == {"A", "B"}
            assert all(c["fresh_true_lora"]["status"] == "OK" for c in cells.values())
            assert all(c["q4_requantization"]["status"] == "OK" for c in cells.values())
            assert all(c["q4_requantization"]["rel_dppl"] == 0.0 for c in cells.values())
        assert len(calls) == 8
        synthetic = {
            "A": {"fresh_true_lora": {"adapt": {"capture": 0.9}},
                  "q4_requantization": {"rel_dppl": 0.01}},
            "B": {"fresh_true_lora": {"adapt": {"capture": 0.7}},
                  "q4_requantization": {"rel_dppl": 0.03}},
        }
        div = h.future_divergence(synthetic)
        assert div["adapt"] and div["q4"] and div["capture_abs_delta"] > 0.19
    finally:
        h.train_lora_state, h.save_sd, h.inspect, h.run_cmd, h.common.load_state = old_train, old_save, old_inspect, old_run, old_load
    print(json.dumps({"status": "PASS", "checks": ["order", "matching", "unavailable", "null>=200", "future-path-mock"]}))


if __name__ == "__main__":
    main()
