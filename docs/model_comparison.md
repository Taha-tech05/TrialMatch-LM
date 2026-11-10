# Model Comparison Notes

> **DRAFT written by Claude Code for the author to review.** Every number comes from a file in `results/`, `configs/` or the generated `docs/` tables, and the file is named in an HTML comment.

## 1. Candidate base models

| role | model | params (B) | layers | hidden | query / KV heads | fp16 weights (GiB) | status |
|---|---|---|---|---|---|---|---|
| target | Qwen/Qwen3-4B | 4.02 | 36 | 2560 | 32 / 8 | 7.49 | trained (QLoRA) and benchmarked |
| draft | Qwen/Qwen3-0.6B | 0.596 (loaded; `model_facts.md` says 0.75) | 28 | 1024 | 16 / 8 | 1.11 (my arithmetic; `memory_estimate.md` says 1.40) | used untuned as the speculative draft |
| fallback | Qwen/Qwen3-1.7B | 2.03 | 28 | 2048 | 16 / 8 | 3.78 | never trained or evaluated |
| alternative target | meta-llama/Llama-3.2-3B-Instruct | 3.21 (author-supplied) | 28 | 3072 | 24 / 8 | 6.43 (arithmetic) | rejected: gated (401), not measured |
| alternative draft | meta-llama/Llama-3.2-1B-Instruct | 1.24 (author-supplied) | 16 | 2048 | 32 / 8 | 2.48 (arithmetic) | rejected with its target |

<!-- docs/model_facts.md for params, layers, hidden, heads; docs/memory_estimate.md for the 4B fp16 figure and the Llama "Skipped" line; the draft parameter count is `n_params_draft` 596,049,920 in results/benchmarks/speculative_20261108_113154.json (environment) and the target count 4,022,468,096 is `n_params` in the same file -->

**Draft size, corrected.** The generated tables `docs/model_facts.md` and `docs/memory_estimate.md` list the draft as 0.75 B parameters and 1.40 GiB (hub metadata count; for Qwen3-1.7B likewise 2.03 B). The model actually loaded in the speculative benchmark reports 596,049,920 parameters, so I use that: fp16 weights 596,049,920 x 2 bytes = 1.11 GiB, a size ratio of 4.02 / 0.596 = 6.7x, and a draft KV cache of 2 x 28 x 8 x 128 x 2 = 114,688 bytes per token (against 147,456 for the target). Why the two counts differ is not recorded in the repo ; the Qwen3-4B model card (author-supplied) lists 4.0 B parameters of which 3.6 B are non-embedding, and Qwen3-0.6B, 1.7B and 4B use tied embeddings. The generated files were left unchanged. For the memory of both models together I cite the measured peak in the speculative benchmark (about 9.08 GiB speculative, 8.81 GiB plain, at 512 input and 128 new tokens, both models loaded) instead of the estimate. <!-- results/benchmarks/speculative_20261108_113154.json rows peak_mem_bytes_median, my division by 2^30 -->

**Qwen3 architecture facts checked in the code.** `modeling_qwen3.py` (local transformers 5.22.2) shows grouped-query attention (`k_proj`/`v_proj` output `num_key_value_heads x head_dim`), rotary embeddings applied after a per-head RMSNorm of queries and keys (`q_norm`, `k_norm`), RMSNorm layers and `repeat_kv` with `num_key_value_groups = 32 / 8 = 4` for the 4B model. My standalone `reference_attention` and `tiled_attention` do not implement QK-norm and are not used inside the model; the cached decoder calls the Hugging Face model, so the model's own attention (including QK-norm) runs. That is why the tiled-attention study measures the attention operation in isolation. The MLP is gated (`down_proj(act_fn(gate_proj(x)) * up_proj(x))` in `Qwen3MLP`); with `Qwen3Config` defaulting to `hidden_act = "silu"` this is a SwiGLU MLP (I did not open the hub `config.json` for the value). The configured context is 40,960 tokens (`docs/model_facts.md`); the Qwen3-4B model card (author-supplied screenshot) states 32,768 tokens natively and 131,072 with YaRN.

**Rejected candidate: Llama-3.2-3B-Instruct with Llama-3.2-1B-Instruct as draft.** Both repositories were gated and returned 401, so nothing was measured. <!-- docs/model_facts.md --> The architecture figures in the table were supplied by the author from the model cards, not measured here: 3.21 B and 1.24 B parameters, 28 and 16 layers, hidden sizes 3072 and 2048, 24/8 and 32/8 query/KV heads (head dimension 128 and 64), fp16 weights of about 6.43 GiB and 2.48 GiB (parameters x 2 bytes / 2^30), and a size ratio of 3.21 / 1.24 = 2.6. Reasons for rejecting it: (i) access failed and no measurement was possible (in the repo); (ii) the smaller size ratio (2.6 against 6.7 for Qwen3) would leave less room for a speculative speed-up; (iii) a long context gives no benefit because the longest prompt plus target is 618 tokens (`results/metrics/token_stats.json`). Figures from on-device mobile benchmarks are not T4 results and are not used.

My reading: I chose Qwen3-4B because it is the largest of the accessible candidates whose fp16 weights plus the draft (8.89 GiB together) and a 1024-token KV cache (0.25 GiB for both models at batch 1) still fit a 16 GiB T4 with room to spare, at 57% of 16 GiB in the estimate. <!-- docs/memory_estimate.md, row fp16 / batch 1 / seq 1024 --> That table is an estimate (weights plus KV cache only, no activations), not a measurement. The measured peak allocation of the merged 4B model during decoding was about 8.1 GB, see the KV-cache benchmark (`peak_mem_bytes` 8,082,512,384 B at 128 input tokens). <!-- results/benchmarks/kv_cache_20261107_192136.json -->

Tokenizer: Qwen3-4B and Qwen3-0.6B passed the 200-string identical-tokenizer check, which is what makes speculative decoding valid. <!-- docs/decisions.md 2026-11-07, results/metrics/tokenizer_compat.json (`vocab_identical` true, 151,669 entries each, 200 strings, 0 id mismatches, `passed` true) -->

Token lengths (train+val, annotation 883 skipped, n = 808): prompt median 293, p95 about 458, max 591; target 14 to 39 tokens; prompt+target max 618. <!-- results/metrics/token_stats.json --> So `max_seq_len` 768 truncates nothing. <!-- configs/train.yaml -->

## 2. Training setup actually used

QLoRA (NF4 4-bit base, fp16 compute), LoRA r=16, alpha=32, dropout 0.05, on q/k/v/o/gate/up/down; learning rate 2e-4, 3 epochs, per-device batch 4 x 4 accumulation steps (effective 16), cosine schedule, warmup ratio 0.05, seed 42, gradient checkpointing. <!-- configs/train.yaml -->

Loss log: training loss 0.473 at step 1 and 0.018 at step 114; validation loss 0.124, 0.121 and 0.123 after epochs 1, 2 and 3. <!-- results/logs/train_qwen3-4b_qlora_4bit.jsonl --> My reading: validation loss stayed flat after epoch 1 while training loss kept falling, which is consistent with some overfitting on 603 training rows. I did not run an ablation to confirm it.

## 3. Base vs fine-tuned vs trivial baseline

Same prompt, greedy decoding, `max_new_tokens` 64. <!-- configs/train.yaml (generation), results/eval/base_val_summary.json -->

**Validation split (206 rows, 8 patients)**

| system | accuracy | macro-F1 (present = full) | evidence micro-F1 (incl. empty) | evidence micro-F1 (excl. empty) | invalid rate |
|---|---|---|---|---|---|
| baseline (train majority + empty evidence) | 0.621 | 0.215 | 0.000 | 0.000 | 0.000 |
| base Qwen3-4B | 0.626 | 0.312 | 0.359 | 0.519 | 0.000 |
| fine-tuned (4-bit base + adapter) | 0.748 | 0.549 | 0.529 | 0.637 | 0.000 |

<!-- results/metrics/val_comparison.md (Overall table) -->

Merged fp16 model on val: accuracy 0.733 (151 of 206 rows) against 0.748 (154 of 206) for the 4-bit-base run, so 3 rows differ. <!-- results/eval/finetuned_merged_val_summary.json, results/eval/finetuned_val_summary.json; the row counts are my arithmetic: 0.7330 x 206 and 0.7476 x 206 -->

**Test split (206 rows, 8 patients, evaluated once with the merged model)**

| system | accuracy | macro-F1 (present) | macro-F1 (full) | evidence micro-F1 (incl. empty) | evidence micro-F1 (excl. empty) | invalid rate |
|---|---|---|---|---|---|---|
| baseline | 0.626 | 0.272 | 0.226 | 0.000 | 0.000 | 0.000 |
| base Qwen3-4B | 0.665 | 0.523 | 0.435 | 0.376 | 0.550 | 0.000 |
| fine-tuned (merged fp16) | 0.767 | 0.726 | 0.605 | 0.578 | 0.663 | 0.000 |

<!-- results/metrics/test_comparison.md (Overall table) -->

Reading of the numbers:
- Fine-tuning raised accuracy over the base model by 0.122 on val (0.748 vs 0.626) and 0.102 on test (0.767 vs 0.665). <!-- val_comparison.md, test_comparison.md; differences are my subtraction -->
- The majority baseline is strong on accuracy (0.621 val, 0.626 test) because 'not excluded' dominates exclusion criteria (73.4% of exclusion rows overall). <!-- docs/label_distribution.md --> Macro-F1 separates the systems far more than accuracy does.
- On test, macro-F1 over present classes (0.726) and over the full label set (0.605) differ because 'excluded' has zero gold rows in test. <!-- results/metrics/test_comparison.md note -->
- Where the fine-tuned model is weak (test, exclusion): 'not enough information' recall is 0.11 (precision 0.75), the model mostly answers 'not excluded' for criteria the annotators judged NEI. <!-- results/metrics/test_comparison.md per-class table --> See `docs/error_analysis.md`.
- Both sets are 8 patients, so differences of a few rows are within noise. <!-- results/metrics/val_comparison.md and test_comparison.md definitions section -->

## 4. Fine-tuned model per criterion type (test)

| scope | n | accuracy | macro-F1 (present) | evidence micro-F1 (incl. empty) |
|---|---|---|---|---|
| inclusion | 75 | 0.813 | 0.705 | 0.676 |
| exclusion | 131 | 0.740 | 0.564 | 0.314 |

<!-- results/metrics/test_comparison.md (Inclusion only and Exclusion only tables) -->

## 5. Not compared

- Qwen3-1.7B and Llama-3.2-3B were never fine-tuned or evaluated, so there is no measured quality comparison between candidate sizes. The choice of 4B rests on the memory estimate and the decision rule, not on a head-to-head result. <!-- docs/decisions.md 2026-11-07 -->
- One training seed (42) and one hyper-parameter setting.
