# Speculative decoding notes

> **DRAFT written by Claude Code in first person for the author to review and rewrite in your own words.** Numbers come from `results/benchmarks/speculative_20261108_113154.json` unless another file is named in an HTML comment. Code statements come from files I read in `src/trialmatch/inference/speculative/`. `TODO(verify)` marks things the repo does not establish.

Code: `speculative_decode.py` (`speculative_greedy_decode`), `cache_sync.py` (`crop_cache`), `stats.py` (`aggregate`), benchmark in `src/trialmatch/inference/benchmarks/speculative_bench.py`.
Setup: Tesla T4, torch 2.14.1+cu130, transformers 5.22.1, fp16. Target = merged Qwen3-4B (4,022,468,096 parameters), draft = untuned Qwen/Qwen3-0.6B (596,049,920 parameters). k in {1, 2, 4, 8}, synthetic inputs of 128/256/512 tokens with 16/64/128 new tokens, plus 8 real val prompts; 1 warm-up and 3 repeats; batch size 1. 85 result rows (plain baselines included). <!-- speculative_20261108_113154.json environment, args, rows -->

## Draft / target design

- **Which models, and why the tokenizer must be identical:** the draft proposes token ids and the target's argmax is compared with them id by id, so both models must map the same ids to the same tokens. Both configs have a vocabulary size of 151,936 and both tokenizers have 151,669 entries, and the project's tokenizer check passed. <!-- docs/model_facts.md; docs/decisions.md 2026-11-07 for the check -->
- **Layers and size:** the target has 36 layers and the draft 28, so the draft is only 22% shallower, even though its hidden size is 1,024 against 2,560. <!-- results/benchmarks/speculative_20261108_113154.json environment target_layers, draft_layers; docs/model_facts.md --> At batch size 1 on a T4 I expect the cost of a forward to depend a lot on the number of layers (many small kernels), not only on width. I did not time the draft alone, but the benchmark gives an indirect measure: with k = 1 the draft work is a median 42.1% of (draft + verify) time on the val prompts, so one draft token costs about 0.73 of a target verify step (0.421 / 0.579, my arithmetic). That is a lot more than the 0.6B/4B parameter ratio (about 0.15) would suggest.
- **Memory:** both models stay in fp16 on the GPU. The measured peak was about 9.08 GiB for speculative runs and 8.81 GiB for the plain runs at 512 input and 128 new tokens (both models are loaded in both cases). The memory estimate had predicted 9.02 to 9.14 GiB for target + draft at batch 1 and 512 to 1,024 tokens. <!-- speculative_20261108_113154.json rows peak_mem_bytes_median (my division by 2^30); docs/memory_estimate.md --> Memory was not the limiting factor.
- **Batch size 1 only:** `speculative_greedy_decode` asserts `input_ids.shape[0] == 1`. Its docstring says batching needs per-row accepted lengths and ragged caches: each row would accept a different number of tokens per iteration, so the rows' caches would end up at different lengths.

## Acceptance logic

- **What the target verifies in one forward:** the input is `[last token, d_1, ..., d_k]` at positions n to n + k. Logits row i is the target's prediction for the token after input i, so `g_0` is what the target would emit after `last` and should equal `d_1`, `g_1` should equal `d_2`, and so on; `g_k` has no candidate to check and becomes the bonus token. (`speculative_decode.py` docstring and the `g = o.logits[0].argmax(-1)` line.)
- **Longest matching prefix:** `m = (d == g[:k_eff]).long().cumprod(0).sum()` counts matches until the first miss. After a miss, the later rows of `g` were computed with the draft's wrong token in the context, so they are not what the target would produce after its own choice. Those rows cannot be reused, even if some happen to equal later candidates.
- **Why the output equals plain greedy decoding:** every emitted token is the target's argmax given a prefix that is itself made only of target-argmax tokens (the accepted prefix, then the target's own token). The draft only decides how many of those tokens one target forward can produce. What would break the guarantee: accepting a candidate without comparing, a wrong crop, wrong positions, or numerical near-ties in fp16. The last one happened once in the real benchmark, see the end of this file.
- **Tokens per iteration:** between 1 and k + 1; m accepted candidates plus one correction (or one bonus if m = k). Measured medians on the val prompts: mean accepted per iteration 0.97, 1.5, 2.78 and 4.0, and tokens per verify 1.97, 2.33, 3.6 and 4.67 for k = 1, 2, 4, 8. <!-- speculative_20261108_113154.json rows with source "val", medians over the 8 prompts computed by me -->

## Rejection and correction

- **First mismatch:** the target's own token `g_m` is emitted instead of the draft's `d_{m+1}`. This still makes progress because the iteration always yields at least one correct token for the cost of one target forward, which is the same as one plain decoding step (plus the wasted draft work).
- **Candidates after the mismatch:** they are discarded, and both caches are cropped so that they no longer contain them.
- **EOS inside an accepted run; `max_new_tokens`:** the run is cut right after the first EOS token (EOS kept) and at the remaining token budget; `finish_reason` becomes "eos" if EOS fits within the budget. Near the end `k_eff = min(k, remaining - 1)` so no iteration plans past the budget. Tests `test_eos_inside_speculated_run` and `test_max_new_tokens_respected` exist; I did not run them.

## Cache crop and synchronisation

- **What each cache contains after the verify forward and what must be removed:** the target cache holds `[..., last, d_1..d_k]` (n + 1 + k entries) but only `last, d_1..d_m` are real, so I crop it to `len(seq) - 1`. The draft cache (n + k) is cropped to `min(before, len(seq) - 1)`. (code comment, step 4)
- **Target invariant vs draft invariant:** the target cache always holds `seq[:-1]` (the last token is pending). The draft cache holds at most that many. After full acceptance (m = k) the draft cache is one short because `d_k` was proposed but never fed to the draft. At the next iteration the draft first feeds `seq[draft_len:]` (one or two tokens) in a single forward; this is the catch-up.
- **The transformers 5.x `DynamicCache.crop(-n)` convention:** in the installed version (5.22.2 per the module docstring; the benchmark environment records 5.22.1) the count is negative, the number of tokens to remove. `crop_cache` calls it, verifies the resulting length, and otherwise slices the key/value tensors itself. <!-- cache_sync.py -->
- **What goes wrong without the crop:** `test_skipping_the_cache_crop_breaks_decoding` uses the `_skip_crop` switch and asserts that the run either raises or gives different tokens, and that the target cache length differs from `seq_len_after - 1`. I did not run it.

## Why speedup may not appear

Measured result on the 8 real val prompts (speed-up = plain median total time / speculative median total time, per prompt, then median over prompts):

| k | median acceptance rate | tokens per verify | median speed-up | range | prompts faster than plain | draft share of time |
|---|---|---|---|---|---|---|
| 1 | 0.974 | 1.97 | 0.981 | 0.775 to 1.134 | 4 of 8 | 42.1% |
| 2 | 0.750 | 2.33 | 0.923 | 0.692 to 1.062 | 3 of 8 | 58.3% |
| 4 | 0.694 | 3.60 | 0.928 | 0.807 to 1.036 | 3 of 8 | 72.3% |
| 8 | 0.500 | 4.67 | 0.735 | 0.543 to 0.877 | 0 of 8 | 83.5% |

<!-- speculative_20261108_113154.json rows with source "val"; medians, counts and ranges computed by me -->

So on real prompts speculative decoding did not speed things up in my benchmark: the median speed-up was below 1 for every k, and the best single prompt was 1.134x (k = 1, the one with a 38-token answer). The synthetic runs give a similar picture at 128 and 256 input tokens (speed-up 0.40 to 1.145; only 11 of the 36 speculative synthetic rows were faster than plain across all lengths), but at 512 input tokens the acceptance rate was 1.0 for most rows and the speed-up reached 1.288x (k = 8, 128 new tokens). <!-- speculative_20261108_113154.json rows source "synthetic" --> My explanations, in the order I would check them:

- **Python loop overhead:** each iteration reads results back to Python (`g.tolist()`, `int(... .sum())`, and `now(dev)` synchronises CUDA around each timed region) and launches the draft's k - 1 extra one-token forwards from Python. I did not profile how much time this costs: TODO(verify).
- **Small outputs (about 16 tokens; val answers are 15 to 38 tokens here):** the plain val decode takes a median 1.037 s, of which 0.177 s is prefill (and in speculative mode both models are prefilled). So there are only about 15 tokens, i.e. a handful of iterations, to amortise any overhead. <!-- speculative_20261108_113154.json plain val rows: total_s_median, prefill_s_median, n_new_tokens 15 to 38 -->
- **Draft cost (28 vs 36 layers):** see above: at k = 1 one draft token costs about 0.73 of a verify step. A rough check, which is my back-of-envelope and not a measurement: an iteration costs about 0.73 + 1 = 1.73 plain-step equivalents and yields about 1.97 tokens, a ceiling of roughly 1.14x on the decode part before the extra draft prefill, which fits the observed median of 0.98.
- **Acceptance of an untuned draft:** acceptance is high for the first candidate (0.97 at k = 1) and falls with position (0.5 at k = 8), so a larger k adds draft cost faster than it adds accepted tokens: at k = 8 the draft takes 83.5% of the time. A draft fine-tuned on this task's JSON outputs would probably accept more, but the repo did not try it (README: "a fine-tuned draft is not trained in this step").
- **Verify forward of k + 1 tokens vs 1 token on a T4:** I have no measurement that isolates this; the verify time per iteration is included in `verify_s_median`, together with accept/reject and the crop. Whether the larger verify is nearly free here: TODO(verify).
- **Choosing k:** on the val prompts k = 1 had the best median speed-up (0.981) and k = 8 the worst (0.735); on the 512-token synthetic prompts the largest k was best. So the best k depends on the acceptance rate of the text being generated, not only on the model pair.

## Results to discuss

- **Acceptance rate and mean accepted tokens per iteration vs k:** in the table above and in the paragraph on tokens per iteration.
- **Speed-up table:** per-row values are in `results/benchmarks/speculative_20261108_113154.csv`/`.json`. Synthetic speed-ups by input length and k (medians over the three output lengths, computed by me): 128 tokens: 0.88 (k=1), 0.86 (k=2), 0.67 (k=4), 0.46 (k=8); 256 tokens: 0.87, 0.82, 0.61, 0.47; 512 tokens: 0.98, 1.05, 1.12, 1.19.
- **Why acceptance is 1.0 at 512 input tokens:** the synthetic prompt is a real val prompt tiled or cut to length with EOS disabled <!-- src/trialmatch/inference/benchmarks/kv_cache_bench.py docstring, shared input builder (TODO(verify) that the speculative benchmark builds its synthetic input the same way) -->, and I suspect the continuation is repetitive and easy for the draft to copy. I did not inspect the generated tokens: TODO(verify).
- **Token-equality flags and the fp16 mismatch:** 84 of the 85 rows have `tokens_equal` true. The exception is the synthetic row with 256 input tokens, 128 new tokens and k = 1: the speculative output first differs from plain at step 64, the maximum logit gap there is 0.0234, and the plain decoder's top-1 minus top-2 margin at that step is 0.0 (plain token id 279, speculative token id 6454). That means the two highest plain logits were exactly tied in fp16, so which token wins depends on rounding in the particular matmul shapes (one token at a time versus k + 1 tokens in one forward). I read this as a near-tie, not a logic bug, which is consistent with the other 84 rows being identical, but it is a reading and the cause is not isolated. <!-- speculative_20261108_113154.json, row key synthetic|-|in256|out128|k1|Qwen/Qwen3-0.6B, fields first_diff_step, max_logit_gap_at_diff, plain_top1_top2_margin_at_diff --> The KV-cache benchmark shows the same kind of event (step 103, logit gap 0.029), see `docs/kv_cache_notes.md`.
