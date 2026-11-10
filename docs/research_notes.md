# Research Notes

> **DRAFT written by Claude Code for the author to review.** The sources below are works that the design relies on.
> Six entries (TrialGPT, RoFormer, GQA, QLoRA, speculative sampling, FlashAttention) use BibTeX supplied by the
> author; the entries for the Qwen3 report, LoRA, Attention Is All You Need, the online softmax and Leviathan et al.
> were completed from memory and should be checked. The "what it says" lines are my paraphrase, not quotes.
> Statements about this repo come from files in the repo, named in an HTML comment.

## Sources (minimum 5)

### 1. TrialGPT: Jin et al., "Matching patients to clinical trials with large language models", Nature Communications 15, 9074, 2024 (BibTeX supplied by the author)
- **What it says (paraphrase):** an LLM can judge, for each inclusion/exclusion criterion of a trial, whether a patient meets it, and can cite the patient-note sentences that support the judgement. The criterion-level annotation set released with it is the dataset used here (the repo README calls it the "TrialGPT Criterion Annotations dataset"). <!-- README.md -->
- **Limitation:** the dataset is small for fine-tuning (1,015 rows, 53 patients, 103 trials) <!-- docs/data_inspection.md --> and it ships GPT-4 predictions and explanations next to the expert labels. Those columns are previous model outputs and would leak answers. <!-- docs/data_inspection.md (leakage columns), docs/decisions.md 2026-11-06 -->
- **How it affected my design:** I removed the four GPT-4 columns from the processed files and keep them in a separate "NOT_FOR_MODEL_USE" file, split by patient (37/8/8) so no note appears in two splits, and trained the model to output only `{label, evidence_ids}` with no explanation field, because no gold explanations exist. <!-- docs/decisions.md 2026-11-06 -->

### 2. Qwen3 Technical Report (Yang et al., Qwen Team, arXiv 2505.09388, 2025)
- **What it says (paraphrase):** describes the Qwen3 family of open-weight decoder-only models in several sizes, including 0.6B, 1.7B and 4B, under a shared tokenizer.
- **Limitation:** I did not read the report in this session. The facts I rely on come from the model configs and hub metadata measured by `scripts/model_facts.py`, not from the report. <!-- docs/model_facts.md -->
- **How it affected my design:** Qwen3-4B is the target, Qwen3-0.6B the speculative draft and Qwen3-1.7B the fallback. The shared tokenizer made speculative decoding possible (200-string compatibility check passed). <!-- docs/decisions.md 2026-11-07, results/metrics/tokenizer_compat.json -->

### 3. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., ICLR 2022)
- **What it says (paraphrase):** freeze the pretrained weights and learn small low-rank update matrices for chosen linear layers, so only a tiny fraction of parameters is trained, and the update can be merged into the base weights afterwards.
- **Limitation:** quality depends on rank and on which layers are adapted; the repo ran one setting only (r=16, alpha=32, dropout 0.05) <!-- configs/train.yaml --> and no rank ablation.
- **How it affected my design:** adapters on q/k/v/o and gate/up/down projections only, never `lm_head` or embeddings (Qwen3 ties them). <!-- configs/train.yaml, docs/decisions.md 2026-11-07 --> Merging the adapter (`merge_and_unload`) gives one fp16 model for all benchmarks. <!-- docs/decisions.md 2026-11-07, results/logs/merge_verify.json -->

### 4. QLoRA: Dettmers, Pagnoni, Holtzman, Zettlemoyer, NeurIPS 36, 2023 (BibTeX supplied by the author)
- **What it says (paraphrase):** train LoRA adapters on top of a base model whose weights are stored in 4-bit NormalFloat, to fit large models on small GPUs.
- **Limitation:** the adapter learns against the quantised weights. Merging it into fp16 weights is a slightly different model. In this repo the merged model scored 0.733 on val against 0.748 for the 4-bit-base run. <!-- results/eval/finetuned_merged_val_summary.json, results/eval/finetuned_val_summary.json -->
- **How it affected my design:** training uses `quantization: qlora_4bit` <!-- configs/train.yaml --> to fit the 16 GB T4; benchmarks use the merged fp16 model and I report both val accuracies instead of assuming they are equal.

### 5. Attention Is All You Need (Vaswani et al., NeurIPS 2017)
- **What it says (paraphrase):** defines scaled dot-product attention, softmax(QK^T / sqrt(d)) V, and the Transformer built from it.
- **Limitation:** the standard formulation materialises the full score matrix, which is O(N^2) in memory.
- **How it affected my design:** `reference_attention` in `src/trialmatch/inference/attention/reference.py` follows this formula step by step and is the baseline for the tiled version.

### 6. Online normalizer calculation for softmax (Milakov and Gimelshein, arXiv 1805.02867, 2018) and FlashAttention: Dao, Fu, Ermon, Rudra, Ré, NeurIPS 35, 2022 (BibTeX supplied by the author)
- **What they say (paraphrase):** softmax can be computed in one pass with a running maximum and running normaliser; FlashAttention applies this blockwise so the full N x N matrix is never stored, and fuses the work in one GPU kernel.
- **Limitation:** my `tiled_attention` is plain PyTorch, so it has none of the kernel fusion. It is an educational implementation and was slower than the reference at short lengths in my benchmark. <!-- results/benchmarks/attention_20261107_193238.json -->
- **How it affected my design:** `tiled_attention` in `src/trialmatch/inference/attention/tiled.py` uses the running max / normaliser / output update from its docstring. The FlashAttention backend of torch SDPA was reported unavailable on the T4 ("No available kernel") at every length. <!-- results/benchmarks/attention_20261107_193238.json --> The reason is not recorded in the repo.

### 7. Fast Inference from Transformers via Speculative Decoding (Leviathan, Kalman and Matias, ICML 2023) and Accelerating LLM decoding with speculative sampling: Chen et al., arXiv 2302.01318, 2023 (BibTeX supplied by the author)
- **What they say (paraphrase):** a small draft model proposes several tokens, the large target model verifies them in one forward pass, and the accepted prefix is kept so the output distribution (or, for greedy decoding, the exact tokens) matches the target.
- **Limitation:** the speed-up depends on the draft being cheap and often right. My draft (untuned Qwen3-0.6B, 28 layers against 36) was neither cheap enough nor accurate enough on short answers, see `docs/speculative_notes.md`.
- **How it affected my design:** `speculative_greedy_decode` is a greedy, batch-size-1 version with explicit cache cropping. <!-- src/trialmatch/inference/speculative/speculative_decode.py -->

### 8. GQA: Ainslie et al., EMNLP 2023 (BibTeX supplied by the author)
- **What it says (paraphrase):** let groups of query heads share one key/value head to shrink the KV cache.
- **Limitation:** none specific to this project beyond needing `repeat_kv` when attention is written by hand.
- **How it affected my design:** Qwen3-4B has 32 query heads and 8 KV heads <!-- docs/model_facts.md -->; `repeat_kv` in `reference.py` maps query head h to KV head h // n_rep, and the KV cache is sized by KV heads. <!-- docs/memory_estimate.md -->

### 9. RoFormer: Su et al., Neurocomputing 568, 127063, 2024 (BibTeX supplied by the author)
- **What it says (paraphrase):** encode position by rotating query and key vectors by an angle that depends on the absolute position, so attention scores depend on relative position.
- **Limitation:** not tested beyond the lengths used here (max 4096 in the attention benchmark; the prompts are under 768 tokens). <!-- results/benchmarks/attention_20261107_193238.json, configs/train.yaml -->
- **How it affected my design:** the cached decoder passes explicit `position_ids` equal to the number of tokens already seen, see `src/trialmatch/inference/kv_cache/cached_decode.py` and `docs/kv_cache_notes.md`.

## Candidate Models (minimum 2)
1. **Qwen/Qwen3-4B (chosen target):** 4.02 B parameters, 36 layers, hidden size 2560, 32 query heads, 8 KV heads, head_dim 128, apache-2.0. <!-- docs/model_facts.md --> Estimated fp16 weights 7.49 GiB. <!-- docs/memory_estimate.md -->
2. **Qwen/Qwen3-1.7B (fallback):** 2.03 B parameters (hub-metadata count in `docs/model_facts.md`, may be overstated, see `docs/model_comparison.md`), 28 layers, hidden size 2048. <!-- docs/model_facts.md --> It would be used only if the 3-epoch projection on a T4 exceeded 90 minutes or the 4B model did not fit. <!-- docs/decisions.md 2026-11-07 --> It was never trained here. The actual 4B run was logged from 10:40:44 to 11:13:53 on 2026-11-07, about 33 minutes. <!-- results/logs/train_qwen3-4b_qlora_4bit.jsonl -->
3. **Qwen/Qwen3-0.6B (draft):** 596,049,920 parameters as loaded (the generated `docs/model_facts.md` lists 0.75 B), 28 layers, hidden size 1024. <!-- results/benchmarks/speculative_20261108_113154.json environment; docs/model_facts.md -->
4. **meta-llama/Llama-3.2-3B-Instruct (reference alternative) with Llama-3.2-1B-Instruct as draft:** inaccessible (gated repo, 401), so it has no measured numbers. <!-- docs/model_facts.md --> Figures supplied by the author from the model cards (not measured here): 3.21 B / 1.24 B parameters, 28 / 16 layers, hidden 3072 / 2048, 24 / 8 and 32 / 8 query/KV heads, size ratio 2.6. See `docs/model_comparison.md`.

## Rejected Options
- **Row-level random split:** leaks notes across splits (one note per patient, up to 43 rows per patient). <!-- docs/decisions.md 2026-11-06, docs/data_inspection.md -->
- **Training on GPT-4 explanations:** leakage. <!-- docs/decisions.md 2026-11-06 -->
- **Using the `training` column as a filter or split:** its meaning is unknown and it varies within 44 of 53 patients. <!-- docs/decisions.md 2026-11-06, docs/data_inspection.md -->
- **Collapsing NEI / N/A or sharing one label set:** would change the task. <!-- docs/decisions.md 2026-11-06 -->
- **Truncating long notes or criteria:** not needed, the longest prompt+target is 618 tokens against `max_seq_len` 768. <!-- results/metrics/token_stats.json, configs/train.yaml -->
- **bf16 training:** `docs/decisions.md` says the T4 lacks bf16, but the environment log reports `bf16_supported: true` for the Tesla T4 <!-- results/logs/env_qwen3-4b_qlora_4bit.json -->, so the repo does not settle why bf16 was rejected; training used fp16 either way.
- **Benchmarking on the 4-bit base plus adapter:** would mix dequantisation cost into timings. <!-- docs/decisions.md 2026-11-07 -->
- **Gradio / Streamlit / a React build for the app:** chose FastAPI plus one static page. <!-- docs/decisions.md 2026-11-08 -->

## Observed Difficulties
- **Class imbalance and an empty class in test:** 'excluded' has 10 rows overall, 2 in val and 0 in test. <!-- docs/label_distribution.md --> I report macro-F1 over present classes and over the full label set. <!-- docs/decisions.md 2026-11-06 -->
- **Gated Llama repos** could not be measured. <!-- docs/model_facts.md -->
- **Merged vs 4-bit-base accuracy gap:** 0.733 against 0.748 on val. <!-- results/eval/finetuned_merged_val_summary.json, results/eval/finetuned_val_summary.json -->
- **fp16 near-ties:** the cached and uncached decoders differ at step 103 of one synthetic case, logit gap 0.029 <!-- results/benchmarks/kv_cache_20261107_192136.json -->, and speculative k=1 differs from plain greedy at step 64 with a top1-top2 margin of 0.0 <!-- results/benchmarks/speculative_20261108_113154.json -->.
- **transformers 5.x API difference:** `DynamicCache.crop` takes a negative count; the code checks the resulting length and falls back to slicing. <!-- src/trialmatch/inference/speculative/cache_sync.py -->
- **Speculative decoding gave no speed-up on real val prompts** (median speedup 0.981 at k=1 down to 0.735 at k=8). <!-- results/benchmarks/speculative_20261108_113154.json -->
- **No Hub id, no paper author list in the repo:** these are placeholders, see the report and README.
