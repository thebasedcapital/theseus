# Theseus harvest population — GENERATED, do not edit by hand

owner: HarvestLineage (harvest/lineage.py); describes harvest/cache/manifest.jsonl at regeneration time.
regenerated: 2026-08-31T12:05:16Z
records: 390 | resolved edges: 264 | resolved-parent children: 242 | dangling edges: 88 | dropped: 99

## 1. Composition by kind

| kind | count |
|---|---|
| base | 66 |
| finetune | 69 |
| instruct | 30 |
| merge | 31 |
| quant | 129 |
| adapter | 45 |
| mlx | 20 |

## 2. Resolvable parent (what enables matched pairs)

- records declaring a parent: 308 (79.0% of the population)
- of those, at least one declared parent is inside the population: 242 (78.6% of parent-declaring records)
- resolved edge lines: 264; dangling edge lines: 88
- a parent is resolvable when both the child and the declared parent have manifest records; that is the substrate for base->instruct->quant matched pairs without further discovery.

Dangling (declared parent absent from the population), by relation:
- quant_of: 33
- merge_of: 27
- finetune_of: 16
- adapter_on: 12

## 3. Top architectures

| arch | count |
|---|---|
| null | 157 |
| LlamaForCausalLM | 66 |
| Qwen2ForCausalLM | 48 |
| MistralForCausalLM | 35 |
| Qwen3ForCausalLM | 24 |
| Gemma2ForCausalLM | 16 |
| MixtralForCausalLM | 6 |
| Qwen3MoeForCausalLM | 6 |
| Phi3ForCausalLM | 6 |
| OPTForCausalLM | 4 |
| Gemma3ForCausalLM | 3 |
| FalconForCausalLM | 2 |
| SmolLM3ForCausalLM | 2 |
| TadaForCausalLM | 2 |

## 4. Selection bias of this sampling (self-declared; analysis is the next slice)

1. **`sort=downloads` over-represents popularity.** Every probe page is sorted by download count, so the population skews to famous English instruction/chat models derived from a handful of mega-bases (Qwen2.5, Llama-3.1). Long-tail finetunes, non-English models, and small bases appear only via their cluster page. Any base rate computed here is a rate over *popular* hub artifacts, not over all checkpoints.
2. **Quant pool is concentrated on the same popular bases.** GGUF/GPTQ/AWQ publishers re-release the same ~20 popular instruct models under many quant tags; downloads rank those first. `quant`-kind rows therefore describe a few base distributions repeatedly — great for matched chains, weak for a general quant-vs-base claim.
3. **Resolvable-parent fraction is deliberately inflated.** Declared parents are injected into the manifest regardless of popularity, so 78.6% of parent-declaring records resolve in this generated population. That is a design guarantee for the matched-pairs use case, NOT an independent estimate of how often HF publishers declare lineage.
4. **instruct vs finetune is a name heuristic.** HF tags both as `base_model:finetune:`. We label `instruct` only when the repo name matches instruct|chat|it\b; e.g. 'OpenHermes' or 'Zephyr' (instruction-tuned) land in `finetune`. Splits on that boundary carry classification error.
5. **Survivorship.** The population is what the hub returns today. Deleted, renamed, or private parents become dangling edges; historic artifacts that were removed are simply absent.
6. **Kind/arch filtering.** Non-decoder architectures (vision, embedding, video, speech) are dropped; a no-lineage repo is admitted only when a name/tag marker justifies its kind and it passes the decoder gate. The population is therefore scoped to what Theseus's static scanner can actually read.

## 5. Sample drops (top reasons)

| reason | count |
|---|---|
| non-decoder arch | 97 |
| id mismatch | 2 |
