# M1 merge specialist

The merge probe builds one deterministic specialist from the pristine Qwen2.5-0.5B reference checkpoint.

## Specialist recipe

- Rule: `kv: KEY=VALUE => KEY: VALUE`.
- Data: 512 training examples plus 128 held-out examples, generated in order from `random.Random(2718)`. Each key is 8 draws from `abcdefghij`; each value is 8 draws from `0123456789`.
- Tokenization/packing: the Qwen tokenizer; prompt `kv: {key}={value} => ` followed by target `{key}: {value}`, EOS appended to target; packed/truncated to `SEQ_LEN=128`; batch size `BATCH_SIZE=2`.
- LoRA: hand-written adapters on `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` in every layer; `RANK=32`, `ALPHA=32`, dropout 0. Base weights are frozen bf16; LoRA A/B parameters are fp32.
- Optimizer: AdamW, learning rate `3e-3`, weight decay 0, 600 steps (`STEPS=600`), deterministic batch index `(step * BATCH_SIZE) % TRAIN_N`; `TRAIN_N=512`, `HELDOUT_N=128`.
- Merge alphas: `[0.3, 0.5, 0.7]`; TIES density `DENSITY=0.2`.
- Eval: pinned `m1/data/eval_wikitext.txt`, 2048 tokens, sequence length 128.

## Correct BA merge

For every wrapped linear module, with `A[r,in]`, `B[out,r]`, and `scale=ALPHA/RANK`, write the unwrapped weight key as:

`W_specialist = W_base + scale * (B @ A)`

Compute the delta in fp32 on CPU, add to the bf16 base state tensor, and cast the result back to bf16. Keep candidate and specialist state dictionaries on CPU for linear/TIES arithmetic. Load one candidate model and apply each merged state with `load_state_dict` in place; do not reconstruct a model per alpha.

## Specialist self-gate

The specialist must pass before a merge matrix is emitted:

1. held-out rule loss `< 0.5 * base_rule_loss`; measured pristine-base held-out rule loss was `1.6447990363`, so require `< 0.8223995182`;
2. specialist eval PPL `<= 1.5 * base_eval_ppl`; measured pristine-base sequence-128 PPL was `27.6652565`, so require `<= 41.49788475` (reported as `41.5`).

The marker `m1/work/specialist/specialist.json` must be written immediately after saving specialist weights and before merge evaluation. It records the seed, budget, rule loss, eval PPL, base measurements, gate result, runtime, and peak CUDA allocation. A failed gate writes an error and no merge matrix.
