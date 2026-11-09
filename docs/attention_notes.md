# Attention notes

> **DRAFT written by Claude Code in first person for the author to review and rewrite in your own words.** Numbers come from `results/benchmarks/attention_20261107_193238.json` unless another file is named in an HTML comment. Code statements come from files I read in `src/trialmatch/inference/attention/`. `TODO(verify)` marks things the repo does not establish.

Code: `reference.py` (`reference_attention`, `repeat_kv`), `tiled.py` (`tiled_attention`), `sdpa_backends.py` (`sdpa_attention`, `try_sdpa`, comparison only), benchmark in `src/trialmatch/inference/benchmarks/attention_bench.py`.
Setup: Tesla T4 (14.56 GiB), fp16 inputs, one sequence (batch 1), 32 query heads, 8 KV heads, head_dim 128, causal, sequence lengths 128 to 4096, tiled block sizes 32/64/128/256, 2 warm-ups and 10 repeats, random normal inputs with seed 0. <!-- attention_20261107_193238.json args; the seed is make_inputs(seed=0) in attention_bench.py -->

## Why online softmax is exact

- For one query row I process the keys in blocks and keep three running quantities: the maximum score `m`, the normaliser `l = sum exp(s - m)` and the unnormalised output `o = sum exp(s - m) v`. When a new block raises the maximum to `m_new`, I multiply the old `l` and `o` by `alpha = exp(m - m_new)`, which expresses everything accumulated so far relative to the new maximum, and then add the new block's terms. <!-- tiled.py docstring and loop -->
- After each block `o / l` equals the softmax-weighted sum over the keys seen so far, so after the last block it is the full softmax attention. The only change from the reference is the order in which the same sums are accumulated, which is exact in real arithmetic; in floating point it differs by rounding only.
- The measurement agrees: against an fp32 reference computed from the same inputs, `reference` and every `tiled_*` variant had the identical maximum absolute error at every sequence length (0.000921 at 128, 0.000966 at 4096), and the largest error among all successful implementations was 0.00120 (`sdpa_efficient`). All were inside the 0.005 tolerance and `mismatches` is 0. <!-- attention_20261107_193238.json results[].max_abs_err and tolerance, mismatches --> I read the identical value as the output rounding to fp16 dominating the error, since my code accumulates in fp32 and casts back only at the end (`reference.py` docstring). That interpretation is mine; I did not run a separate test for it.
- Handling of masked entries: a row with no visible key returns zeros instead of NaN, and the tests include a large-logit stress case and a hand-computed 2-query, 3-key example. <!-- tiled.py, tests/test_attention_correctness.py (test names only; I did not run them) -->

## Memory complexity: O(N^2) vs O(N)

- The reference builds the full score tensor, `[1, 32, S, S]` in fp32. At S = 4096 one such tensor is 32 x 4096 x 4096 x 4 bytes = 2 GiB (my arithmetic). The measured peak above the starting allocation was 6,449 MiB, roughly three times that, which I would attribute to several score-sized temporaries (scores, the masked copy, the exponentials) alive at once. That attribution comes from reading `reference_attention`, not from a memory profile.
- Reference peak memory grows about four-fold per doubling of the length at the long end: 1,684.5 MiB at 2048 to 6,449.1 MiB at 4096 (3.83x). This is the O(N^2) behaviour. <!-- attention_20261107_193238.json results, impl "reference" -->
- The tiled version only holds one `[block_q, block_k]` score tile at a time, so the extra memory is O(block^2 + block x D) plus the output tensor, which is linear in S. At 4096 its peak was 35.8 MiB (block 32), 40.0 MiB (64), 50.1 MiB (128) and 80.2 MiB (256), against 6,449.1 MiB for the reference, i.e. between 80x and 180x smaller. The output tensor alone (32 heads x 4096 x 128 x 2 bytes = 32 MiB) accounts for most of the smallest figure (my arithmetic).
- All peaks are measured above the allocation level at the start of each run, so the inputs and the fp32 baseline are not counted. <!-- attention_bench.py begin_run / end_run: peak - start_alloc -->

## Why an educational PyTorch loop may not beat fused kernels

- **Python overhead:** the tiled loop is a Python double loop. For S = 4096 and block 32 there are 128 query blocks, and with causal skipping about 128 x 129 / 2 = 8,256 (query block, key block) pairs (my count), each executing a dozen or so tensor operations from Python. The measured 4,514.7 ms for that case works out to about 0.55 ms per pair. I did not profile it, so how much of that is interpreter overhead versus GPU time is unknown: TODO(verify).
- **Kernel launches:** each pair launches many small kernels (two matmuls, a scale, a maximum, two exponentials, a sum, a few multiply/adds, plus a mask on diagonal blocks). Small blocks mean many launches with little work each. The block-size sweep fits this: for block 32 to 256 the time at 4096 fell from 4,514.7 ms to 113.85 ms, roughly 4x, 3x and 3x per doubling, as the number of iterations drops about 4x per doubling.
- **No SRAM fusion / HBM traffic:** every intermediate (scores, exponentials, running statistics) is a separate PyTorch tensor, written to and read back from GPU global memory between operations. A fused kernel such as FlashAttention keeps the tile in on-chip memory. This is the standard explanation; I did not measure memory traffic here.

## What the measured results show

Median time (ms) and peak memory above the start level (MiB), fp16, batch 1:

| seq | reference | tiled 32 | tiled 256 | sdpa_math | sdpa_efficient | sdpa_flash |
|---|---|---|---|---|---|---|
| 128 | 1.75 ms / 15.0 | 7.48 / 4.8 | 1.77 / 19.1 | 1.61 / 16.1 | 0.83 / 3.0 | unavailable |
| 512 | 4.89 / 132.4 | 77.25 / 7.8 | 5.00 / 52.2 | 4.61 / 113.0 | 1.08 / 12.0 | unavailable |
| 1024 | 16.90 / 457.3 | 309.26 / 11.8 | 13.60 / 56.2 | 14.59 / 372.0 | 2.23 / 24.0 | unavailable |
| 2048 | 50.40 / 1684.5 | 1148.13 / 19.8 | 33.56 / 64.2 | 40.25 / 1328.1 | 2.78 / 48.0 | unavailable |
| 4096 | 197.06 / 6449.1 | 4514.71 / 35.8 | 113.85 / 80.2 | 158.58 / 4992.1 | 8.28 / 96.0 | unavailable |

<!-- attention_20261107_193238.json results[]; times converted from seconds to ms -->

- **Time vs sequence length:** the reference grows from 1.75 ms at 128 to 197.06 ms at 4096; between 2048 and 4096 it rises 3.9x, close to the quadratic 4x. Tiled with block 256 is about equal to or slightly slower than the reference at 128 to 512 (1.77 vs 1.75, 2.60 vs 2.10, 5.00 vs 4.89 ms; up to 1.24x slower) and faster from 1024 on (13.60 vs 16.90 ms, and 113.85 vs 197.06 ms at 4096, 1.7x). Tiled with block 32 is slower than the reference at every length (22.9x slower at 4096). I would guess the long-length gain comes from skipping the key blocks above the causal diagonal and from touching less memory than the full fp32 score matrix; I did not test that.
- **Peak memory vs sequence length:** reference and `sdpa_math` grow quadratically (6,449.1 and 4,992.1 MiB at 4096); tiled and `sdpa_efficient` stay small (80.2 and 96.0 MiB at 4096).
- **Tiled block size effect:** a larger block is faster and uses more memory: at 4096, block 32/64/128/256 took 4,514.7/1,133.3/362.6/113.9 ms and 35.8/40.0/50.1/80.2 MiB. The block-32 timing is noisy (standard deviation 382.6 ms on a 4,514.7 ms median), so its exact ratios should not be over-read.
- **Comparison with SDPA backends:** `sdpa_efficient` (memory-efficient kernel) was the fastest at every length (8.28 ms at 4096, 23.8x faster than my reference). `sdpa_math` was close to my reference (158.58 vs 197.06 ms). `sdpa_flash` was reported unavailable at every length with `RuntimeError: No available kernel. Aborting execution.`; the cause is not recorded in the repo: TODO(verify) (I would suspect GPU architecture support on the T4 but have not checked). The SDPA calls receive fp16 inputs directly, while my code does its arithmetic in fp32 internally (`reference.py`, `tiled.py`), so the comparison is not like for like.
- **Errors vs fp32 baseline:** see the first section: 0.000921 to 0.000966 for reference, all tiled variants and `sdpa_math`; 0.00104 to 0.00120 for `sdpa_efficient`. Mean absolute error was about 3.3e-5 at 128 and 6.9e-6 at 4096.
- **Caveats:** one GPU type, fp16 only, one random input seed, batch 1, random rather than real activations, and one benchmark file. The tiled and reference code are not used inside the model (`sdpa_backends.py` docstring); the model runs its own library attention path. These benchmarks measure the attention operation in isolation, not end-to-end generation. The quick run `attention_20261107_193016` is not used (`configs/app.yaml` comment); I did not inspect the other earlier attention files in `results/benchmarks/`: TODO(verify).
