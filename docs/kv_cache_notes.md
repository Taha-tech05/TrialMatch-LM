# KV-cache notes

> **DRAFT written by Claude Code in first person for the author to review and rewrite in your own words.** Numbers come from `results/benchmarks/kv_cache_20261107_192136.json` unless another file is named in an HTML comment. Statements about the code come from files I read in `src/trialmatch/inference/kv_cache/`. Statements about the Hugging Face model internals come from reading `transformers/models/qwen3/modeling_qwen3.py` (installed locally as 5.22.2; the benchmark environment recorded 5.22.1, so I assume these lines are unchanged: TODO(verify)); anything not read there is marked `TODO(verify)`.

Code: `src/trialmatch/inference/kv_cache/` (`reference_decode.py`, `cached_decode.py`, `cache_utils.py`, `compare.py`, `decode_common.py`).
Setup of the benchmark: merged fp16 Qwen3-4B on a Tesla T4 (14.56 GiB), torch 2.14.1+cu130, transformers 5.22.1, 1 warm-up and 5 repeats per case. <!-- kv_cache_20261107_192136.json environment and args -->

## Positional handling (RoPE and position offset)

**Why does the new token need position = current cache length, not 0?**
Qwen3 uses rotary position embeddings, which rotate each query and key by an angle that depends on its absolute position (rope_theta 1,000,000 for Qwen3-4B <!-- docs/model_facts.md -->). When I feed only the newest token, the model cannot see where it sits in the sequence, so I must tell it. In `cached_greedy_decode` I pass `position_ids = next_pos`, which starts at the number of real tokens in the prompt (`mask.long().sum(-1)`) and grows by one each step, and `cache_position = [cache_len]` so the new key/value is written at the next slot. If the new token were given position 0, its query would be rotated as if it were the first token, and the relative distances to all cached keys would be wrong.

**What happens to cached keys: re-rotated, or stored already rotated?**
They are stored already rotated. In `Qwen3Attention.forward` the new key goes through `k_norm`, then `apply_rotary_pos_emb(query_states, key_states, cos, sin)`, and only afterwards `past_key_values.update(key_states, value_states, self.layer_idx)`, so the cache holds rotated keys and my decoder never re-applies rotation to old entries. The model also applies an RMSNorm to queries and keys per head (`q_norm`, `k_norm` over `head_dim`) before the rotation; my decoder calls the model, so it needs no code of its own for this. <!-- modeling_qwen3.py lines 244-252 -->

**What did the wrong-`position_ids` test show?**
`test_wrong_position_ids_breaks_output` in `tests/test_cache_equivalence.py` runs the cached decoder with `position_override=lambda n: 0`. It asserts that the correct run equals the uncached reference, that the wrong run does not, and that the logit difference at the first mismatch is above 1e-3. The author ran it on Colab with `-s` (tiny random model; the pytest output was pasted by the author and is not stored in the repo): `FIRST MISMATCH at step 1 (row 0); max |logit diff| at that step = 3.505e+00`. Step 0 comes from the prefill, so the very first decode step already differs when every new token is given position 0, and by a logit difference of 3.5, far above the 1e-3 threshold of the test and above the fp16 near-tie gaps of the real benchmark (0.029).

**How do left-padded batches change the positions?**
`positions_from_mask` in `decode_common.py` computes `(mask.cumsum(-1) - 1).clamp(min=0)`, so padding tokens get no real position and real tokens count from 0. `test_left_padded_batch_matches_single_prompts` checks that a padded batch gives the same tokens as decoding each prompt alone. The benchmarks all use batch size 1, so this path was only tested on the tiny CPU model, not measured on the T4. <!-- docs/decisions.md 2026-11-07 says batch size 1 is the benchmarked case -->

## Masking (causal mask with cache)

**What shape is the mask for prefill vs for a single decode step?**
In prefill the 2-D `attention_mask` is `[B, T]` (all ones when there is no padding) and the model gets T query positions, so it builds a T x T causal pattern from it. In a decode step the query length is 1 and the 2-D mask has `cache_len + 1` columns. The model builds the mask itself: `Qwen3Model.forward` calls `create_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids)` when it is given a 2-D mask, so my decoder only has to pass the growing 2-D mask and the explicit `position_ids`/`cache_position`. I did not read the internals of `create_causal_mask` (in `masking_utils`): TODO(verify). <!-- modeling_qwen3.py lines 384-399 -->

**Why is the 2-D `attention_mask` length `cache_len + 1` at each decode step?**
Because the new token's key/value is added to the cache in this very step, the attention scores have `cache_len + 1` key columns. In the loop I append one column of ones to `mask` before each forward, and the code asserts after each step that the cache grew by exactly one position.

**Why can the single query row attend to every cached key without an explicit triangular mask?**
The new token is the last position, so every cached key is in its past and nothing needs to be hidden. This is the same convention I wrote for the attention code: query i sees key j iff `j <= i + (seq_k - seq_q)`; with `seq_q = 1` that is every key. <!-- src/trialmatch/inference/attention/reference.py docstring -->

## Cache growth

**How many positions are cached after prefill, and after N decode steps (and why the last token is never fed)?**
After prefill the cache holds T positions (the prompt); the code asserts `cache_lengths[-1] == T`. The first generated token is taken from the prefill logits and is not fed. Each decode step feeds the previous token and appends one position, so after generating n tokens the cache holds T + n - 1 positions. The last generated token is never fed back, because nothing after it needs to be computed (the loop runs `max_new_tokens - 1` times).

**Memory per token (2 x layers x kv_heads x head_dim x bytes) and the measured-vs-formula check.**
For Qwen3-4B that is 2 x 36 x 8 x 128 x 2 bytes = 147,456 bytes per token, so 150,994,944 bytes (0.141 GiB) at 1,024 tokens. <!-- docs/memory_estimate.md (0.141 GiB at seq 1024, batch 1), docs/ai_use_log.md (150,994,944); the per-token figure is my multiplication --> `describe_cache` in `cache_utils.py` compares measured bytes with this formula and returns `formula_matches`, and the test `test_cache_bytes_match_formula` exists. I did not read that test, and the real-model benchmark JSON does not record measured cache bytes, so a measured-vs-formula check on the real 4B model is not documented: TODO(verify).

What the benchmark did record is the peak allocation above the starting level (`peak_mem_delta_bytes`). For a 1,024-token input the cached decoder's delta was 258,499,072 bytes (246.5 MiB) at 16, 64 and 128 new tokens alike, while the uncached decoder's delta was 85,875,200 bytes (81.9 MiB) at 16 new tokens. The cached delta does not change with output length, which suggests the peak is reached during prefill and the cache growth during decode does not exceed it. That is my interpretation of the numbers, not something I traced.

**What does the growth mean for the 1,024-token inputs on a 16 GB T4?**
Very little. The KV cache of one 1,024-token sequence is about 0.141 GiB, whereas the merged fp16 weights alone put the measured peak near 8.1 to 8.3 GB (8,082,512,384 bytes at 128 input tokens, 8,311,982,592 bytes at 1,024). The dataset prompts are far shorter (317 to 338 tokens in the 16 val cases). <!-- kv_cache_20261107_192136.json cases with source "val" -->

## GQA (8 KV heads vs 32 query heads)

**Why are the cached tensors `[batch, kv_heads, seq, head_dim]` and not query-head sized?**
Qwen3-4B has 32 query heads but only 8 key/value heads (head_dim 128). <!-- docs/model_facts.md --> The cache helper documents each K/V tensor as `[batch, kv_heads, seq_len, head_dim]`. The key/value projections themselves output only 8 heads: `k_proj` and `v_proj` are `nn.Linear(hidden_size, num_key_value_heads * head_dim)` while `q_proj` and `o_proj` use `num_attention_heads * head_dim`. <!-- modeling_qwen3.py lines 217-228 -->

**How much memory does GQA save compared with full multi-head caching for Qwen3-4B?**
With 32 KV heads the same formula gives 2 x 36 x 32 x 128 x 2 = 589,824 bytes per token, four times the 147,456 above, i.e. about 0.563 GiB instead of 0.141 GiB at 1,024 tokens. This is my arithmetic from the formula, not a measurement.

**Where does the model expand/share the KV heads at attention time?**
In my own attention code the expansion is `repeat_kv` in `src/trialmatch/inference/attention/reference.py`: KV head g serves query heads g x n_rep to g x n_rep + n_rep - 1. A test (`test_repeat_kv_matches_hf_and_head_mapping`) compares it with the Hugging Face helper, and the real model does the same expansion later than the cache: in the eager path `eager_attention_forward` calls `repeat_kv(key, module.num_key_value_groups)` with `num_key_value_groups = num_attention_heads // num_key_value_heads` (4 for Qwen3-4B), and in the SDPA path `sdpa_attention_forward` calls `repeat_kv` unless `use_gqa_in_sdpa(...)` allows `enable_gqa=True`. Either way the cache itself stays at 8 heads. Which attention implementation the benchmark actually used is not recorded in the result file: TODO(verify). <!-- modeling_qwen3.py lines 187-188, 212; integrations/sdpa_attention.py lines 114-118 -->

## Possible numerical differences (fp16, different matmul shapes)

**Why can prefill (T x T matmuls) and single-token steps (1 x T matmuls) round differently in fp16?**
Matrix products of different shapes can be computed with different kernels and a different order of additions, and floating-point addition is not associative, so the two paths can differ in the last bits. In fp16 a last-bit difference is larger than in fp32. I believe this is the reason, as `docs/decisions.md` (2026-11-07) also states, but I did not isolate it with an experiment. If the two highest logits are very close, the tiny difference can flip the argmax.

**Did the real-model benchmark show token mismatches? At which step, and how big was the logit difference?**
Yes, in one cell of 12 synthetic cells: input 256, 128 new tokens. The report says "FIRST MISMATCH at step 103 (row 0); max |logit diff| at that step = 2.930e-02". All other 11 synthetic cells and all 16 val prompts matched, so `all_tokens_equal` is false for the file. <!-- kv_cache_20261107_192136.json, cases[tokens_equal=false] and all_tokens_equal --> The KV-cache report does not record the top1-top2 margin at that step, so I cannot say if it was an exact tie: TODO(verify). (The speculative benchmark does record it, see `docs/speculative_notes.md`.)

**Why is fp32 equality on the tiny CPU model a stronger statement than fp16 equality on the GPU?**
In fp32 rounding noise is much smaller than the gaps between top logits, so token equality there tests the logic: positions, masks, cache indexing. `docs/decisions.md` says equality was checked on a tiny random Qwen3 in fp32 over 6 prompt/output length pairs. The fp16 mismatch is a different kind of statement: the algorithm is the same but the numerics are not bit-identical. I would note that a random tiny model is unlikely to have near-tied logits, so fp32 equality there does not exclude all numerical issues either. <!-- docs/decisions.md 2026-11-07; the last sentence is my reasoning -->

## Measured results (merged fp16 Qwen3-4B, Tesla T4)

Median total time (prefill + decode) over 5 repeats, synthetic inputs, EOS disabled. Speed-up = uncached / cached.

| input tokens | new tokens | cached (s) | uncached (s) | speed-up | cached decode ms/token | uncached decode ms/token |
|---|---|---|---|---|---|---|
| 128 | 16 | 1.573 | 1.470 | 0.93x | 96.1 | 92.7 |
| 128 | 128 | 11.649 | 16.646 | 1.43x | 90.8 | 130.4 |
| 256 | 128 | 11.305 | 25.972 | 2.30x | 87.8 | 203.5 |
| 512 | 128 | 10.125 | 42.983 | 4.25x | 77.7 | 336.5 |
| 1024 | 16 | 1.680 | 9.745 | 5.80x | 77.9 | 616.1 |
| 1024 | 128 | 11.420 | 81.125 | 7.10x | 86.0 | 634.8 |

<!-- kv_cache_20261107_192136.json; speed-up column is my division of the medians -->

Real val prompts (16 prompts, 317 to 338 input tokens, 15 to 38 new tokens): cached median total 1.219 s against uncached 3.025 s; the per-prompt speed-up ranges from 2.20x to 3.28x with median 3.02x. <!-- kv_cache_20261107_192136.json, cases with source "val"; medians computed by me -->

What I read from this: cached decode time per token stays around 78 to 96 ms at every input length, while the uncached decoder grows from about 93 to about 635 ms per token as the input grows from 128 to 1,024 tokens, because it re-runs the whole sequence each step. At 128 input tokens with 16 new tokens the cached decoder was not faster (1.573 s vs 1.470 s). The repeats vary by more than that gap (cached total standard deviation 0.187 s at that cell), so I treat it as noise, not as evidence that caching hurts. <!-- kv_cache_20261107_192136.json cases[0].decoders.cached.total_s.std -->
