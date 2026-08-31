# M1 pre-registered predictions (static, artifact-only)

`J` = mean over layers of `Σ_blocks amax²·n / (12·(2^(b-1)-1)²·Σ w²)` for b=4, block=32
(`canonicalize.quant_condition`) — computed from checkpoint bytes alone, no surgery.
Numbers in parentheses are the ratio to the pristine checkpoint's per-tensor J.
Snapshot: `m1/work/PREDICTIONS_new.json`; this file is generated, thresholds were frozen
in `m1/predict.py` (debt > 1e-3 predicts damage) before any probe result existed.

| checkpoint | equiv | total J | debt | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `g3_rand` | NOT_EQUIVALE | 0.02976 | +0.01852 | 0.0373 (3.30x) | 0.0374 (3.18x) | 0.0381 (2.97x) | 0.0102 (1.00x) | 0.0371 (3.44x) | 0.0371 (3.47x) | 0.0111 (1.00x) |
| `g7_rand` | EQUIVALENT | 0.01495 | +0.00371 | 0.0113 (1.00x) | 0.0118 (1.00x) | 0.0128 (1.00x) | 0.0102 (1.00x) | 0.0108 (1.00x) | 0.0107 (1.00x) | 0.0371 (3.34x) |
| `g3_rand_rep` | NOT_EQUIVALE | 0.01145 | +0.00021 | 0.0108 (0.95x) | 0.0122 (1.04x) | 0.0143 (1.11x) | 0.0102 (1.00x) | 0.0108 (1.00x) | 0.0108 (1.01x) | 0.0111 (1.00x) |
| `g3_smooth_rep` | NOT_EQUIVALE | 0.01145 | +0.00021 | 0.0108 (0.95x) | 0.0122 (1.04x) | 0.0143 (1.11x) | 0.0102 (1.00x) | 0.0108 (1.00x) | 0.0108 (1.01x) | 0.0111 (1.00x) |
| `g1_svd` | EQUIVALENT | 0.01135 | +0.00012 | 0.0113 (1.00x) | 0.0118 (1.00x) | 0.0135 (1.05x) | 0.0103 (1.01x) | 0.0108 (1.00x) | 0.0107 (1.00x) | 0.0111 (1.00x) |
| `g2_rand` | EQUIVALENT | 0.01125 | +0.00001 | 0.0114 (1.00x) | 0.0118 (1.00x) | 0.0128 (1.00x) | 0.0102 (1.00x) | 0.0108 (1.00x) | 0.0107 (1.00x) | 0.0111 (1.00x) |
| `g2_rand_rep` | EQUIVALENT | 0.01124 | +0.00000 | 0.0113 (1.00x) | 0.0118 (1.00x) | 0.0128 (1.00x) | 0.0102 (1.00x) | 0.0108 (1.00x) | 0.0107 (1.00x) | 0.0111 (1.00x) |
| `g7_rand_rep` | EQUIVALENT | 0.01124 | +0.00000 | 0.0113 (1.00x) | 0.0118 (1.00x) | 0.0128 (1.00x) | 0.0102 (1.00x) | 0.0108 (1.00x) | 0.0107 (1.00x) | 0.0111 (1.00x) |
| `base` | EQUIVALENT | 0.01123 | +0.00000 | 0.0113 (1.00x) | 0.0118 (1.00x) | 0.0128 (1.00x) | 0.0102 (1.00x) | 0.0108 (1.00x) | 0.0107 (1.00x) | 0.0111 (1.00x) |
| `g3_smooth` | EQUIVALENT | 0.01120 | -0.00003 | 0.0113 (1.00x) | 0.0117 (0.99x) | 0.0128 (1.00x) | 0.0102 (1.00x) | 0.0107 (0.99x) | 0.0107 (1.00x) | 0.0111 (1.00x) |
| `g1_haar` | EQUIVALENT | 0.01115 | -0.00009 | 0.0113 (1.00x) | 0.0118 (1.00x) | 0.0127 (0.99x) | 0.0097 (0.95x) | 0.0108 (1.00x) | 0.0107 (1.00x) | 0.0111 (1.00x) |
| `g1_svd_rep` | EQUIVALENT | 0.01114 | -0.00010 | 0.0113 (1.00x) | 0.0118 (1.00x) | 0.0127 (0.99x) | 0.0097 (0.95x) | 0.0108 (1.00x) | 0.0107 (1.00x) | 0.0111 (1.00x) |
| `g1_haar_rep` | EQUIVALENT | 0.01114 | -0.00010 | 0.0113 (1.00x) | 0.0118 (1.00x) | 0.0126 (0.99x) | 0.0097 (0.95x) | 0.0108 (1.00x) | 0.0107 (1.00x) | 0.0111 (1.00x) |

## What this forecasts for the panel

* `g3_rand` / `g3_pow2`: q,k,v,gate,up conditioning rises ~2.6x → **Q4 damage expected**, concentrated in gate/up and q/k because the diagonal acts on their shared input columns.
* `g7_rand`: up/down conditioning up ~1.3x → **mild damage expected**, localized to the MLP.
* `g1_*`, `g2_*`: per-tensor J moves < 1.5% → **quantization-neutral expected**; if these rows still diverge from base, the static proxy is missing something (activation-side or block-alignment effects) and that is a result about the diagnostic, not the gauge.
* `g4_perm` / `g6_perm`: J identical → neutral by construction; they are the harness control.
* every `*_rep`: debt returns to ≈ base → **repair predicted to restore reserve**, which is the `prepare` claim stated before the evidence arrived.
