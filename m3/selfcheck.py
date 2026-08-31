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

    # trainer-executes: run the REAL train_lora_state body. The environment (model, data,
    # tokenizer) is stubbed but the function itself is not, so an unexecutable trainer - e.g. the
    # committed version that called opt.step() without ever building an optimizer - fails here
    # instead of hiding behind the lambda mock below. Incident #15, second occurrence.
    torch = h.torch
    seen = {}

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _Model:
        def __init__(self):
            self.config = type("C", (), {"use_cache": True})()
            self._p = torch.nn.Parameter(torch.zeros(4, 4))
        def parameters(self):
            return [self._p]
        def train(self):
            return self
        def named_modules(self):
            return iter(())
        def __call__(self, input_ids=None, attention_mask=None):
            # grad_fn required: the trainer really calls loss.backward().
            return _Out(self._p.float().sum() * torch.ones(2, 8, 16))

    stub = _Model()
    losses = iter([2.0, 1.0])                 # task_loss before, after -> capture 0.5
    real_adamw = torch.optim.AdamW
    old = (h.common.state_to_model, h.common.release, h.adapt_probe.task_loss,
           h.adapt_probe.replace_targets, h.rule_examples, h.make_data)
    old_steps = h.CONTRACT["adapt"]["steps"]
    try:
        def spy_adamw(params, **kw):
            seen["lr"] = kw.get("lr")
            seen["n_params"] = len(list(params))
            return real_adamw([torch.nn.Parameter(torch.zeros(2))], lr=kw.get("lr", 1e-4))
        torch.optim.AdamW = spy_adamw
        h.common.state_to_model = lambda sd, ref, dtype=None, device=None: stub
        h.common.release = lambda device: None
        h.adapt_probe.task_loss = lambda model, held, device: next(losses)
        # The trainer freezes every parameter first (history_pair.py:118-119); adapter insertion is
        # what makes anything trainable again, so the stub must re-enable grad or the test would
        # silently train nothing.
        h.adapt_probe.replace_targets = lambda model: model._p.requires_grad_(True)
        h.rule_examples = lambda n, seed, offset=0: list(range(n))
        h.make_data = lambda tok, ids: (torch.zeros(4, 8, dtype=torch.long),
                                        torch.zeros(4, 8, dtype=torch.long),
                                        torch.ones(4, 8, dtype=torch.long))
        h.CONTRACT["adapt"]["steps"] = 2
        sd, rep = h.train_lora_state({"w": torch.zeros(1)}, object(), 1, "cpu")
        for k in ("seed", "steps", "runtime_s", "base_frozen", "targets",
                  "task_loss_before", "task_loss_after", "capture"):
            assert k in rep, f"train_lora_state returned no {k!r}: declared-vs-executed drift"
        assert rep["capture"] == 0.5 and rep["base_frozen"] is True
        assert seen.get("n_params", 0) >= 1, "optimizer built over zero parameters"
        assert seen.get("lr") == h.CONTRACT["adapt"]["lr"], (
            f"CONTRACT.adapt.lr not consumed by the trainer (saw {seen.get('lr')!r})")
    finally:
        torch.optim.AdamW = real_adamw
        (h.common.state_to_model, h.common.release, h.adapt_probe.task_loss,
         h.adapt_probe.replace_targets, h.rule_examples, h.make_data) = old
        h.CONTRACT["adapt"]["steps"] = old_steps
    checks = ["order", "matching", "unavailable", "null>=200", "trainer-executes", "future-path-mock"]
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
    print(json.dumps({"status": "PASS", "checks": checks}))


if __name__ == "__main__":
    main()
