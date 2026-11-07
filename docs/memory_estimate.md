# Memory estimates (ESTIMATES, not measurements)

Weights = params x bytes/param (fp16 2, int8 1, 4-bit 0.5; quantisation scales ignored). KV cache = 2 x layers x kv_heads x head_dim x seq_len x bytes (fp16 KV, 2 bytes) x batch. Activations, CUDA context and fragmentation are NOT included (a T4 has 16 GB nominal, ~15 GB usable). Fit label: `fits` <= 80% of 16 GiB, `TIGHT` <= 16 GiB, `NO` above. GiB = 2^30 bytes.

## Weights

| model | params (B) | fp16 GiB | int8 GiB | 4-bit GiB |
|---|---|---|---|---|
| Qwen/Qwen3-4B | 4.02 | 7.49 | 3.75 | 1.87 |
| Qwen/Qwen3-0.6B | 0.75 | 1.40 | 0.70 | 0.35 |
| Qwen/Qwen3-1.7B | 2.03 | 3.78 | 1.89 | 0.95 |

## KV cache (fp16)

| model | batch | seq 512 | seq 1024 | seq 2048 | seq 4096 |
|---|---|---|---|---|---|
| Qwen/Qwen3-4B | 1 | 0.070 GiB | 0.141 GiB | 0.281 GiB | 0.562 GiB |
| Qwen/Qwen3-4B | 4 | 0.281 GiB | 0.562 GiB | 1.125 GiB | 2.250 GiB |
| Qwen/Qwen3-0.6B | 1 | 0.055 GiB | 0.109 GiB | 0.219 GiB | 0.438 GiB |
| Qwen/Qwen3-0.6B | 4 | 0.219 GiB | 0.438 GiB | 0.875 GiB | 1.750 GiB |
| Qwen/Qwen3-1.7B | 1 | 0.055 GiB | 0.109 GiB | 0.219 GiB | 0.438 GiB |
| Qwen/Qwen3-1.7B | 4 | 0.219 GiB | 0.438 GiB | 0.875 GiB | 1.750 GiB |

## Target + draft together: Qwen/Qwen3-4B + Qwen/Qwen3-0.6B

Total = weights(target) + weights(draft) + KV(target) + KV(draft), both KV caches fp16, same seq_len.

| weights precision | batch | seq | weights GiB | KV GiB | total GiB | % of 16 GiB | fit |
|---|---|---|---|---|---|---|---|
| fp16 | 1 | 512 | 8.89 | 0.125 | 9.02 | 56% | fits |
| fp16 | 1 | 1024 | 8.89 | 0.250 | 9.14 | 57% | fits |
| fp16 | 1 | 2048 | 8.89 | 0.500 | 9.39 | 59% | fits |
| fp16 | 1 | 4096 | 8.89 | 1.000 | 9.89 | 62% | fits |
| fp16 | 4 | 512 | 8.89 | 0.500 | 9.39 | 59% | fits |
| fp16 | 4 | 1024 | 8.89 | 1.000 | 9.89 | 62% | fits |
| fp16 | 4 | 2048 | 8.89 | 2.000 | 10.89 | 68% | fits |
| fp16 | 4 | 4096 | 8.89 | 4.000 | 12.89 | 81% | TIGHT |
| int8 | 1 | 512 | 4.45 | 0.125 | 4.57 | 29% | fits |
| int8 | 1 | 1024 | 4.45 | 0.250 | 4.70 | 29% | fits |
| int8 | 1 | 2048 | 4.45 | 0.500 | 4.95 | 31% | fits |
| int8 | 1 | 4096 | 4.45 | 1.000 | 5.45 | 34% | fits |
| int8 | 4 | 512 | 4.45 | 0.500 | 4.95 | 31% | fits |
| int8 | 4 | 1024 | 4.45 | 1.000 | 5.45 | 34% | fits |
| int8 | 4 | 2048 | 4.45 | 2.000 | 6.45 | 40% | fits |
| int8 | 4 | 4096 | 4.45 | 4.000 | 8.45 | 53% | fits |
| 4bit | 1 | 512 | 2.22 | 0.125 | 2.35 | 15% | fits |
| 4bit | 1 | 1024 | 2.22 | 0.250 | 2.47 | 15% | fits |
| 4bit | 1 | 2048 | 2.22 | 0.500 | 2.72 | 17% | fits |
| 4bit | 1 | 4096 | 2.22 | 1.000 | 3.22 | 20% | fits |
| 4bit | 4 | 512 | 2.22 | 0.500 | 2.72 | 17% | fits |
| 4bit | 4 | 1024 | 2.22 | 1.000 | 3.22 | 20% | fits |
| 4bit | 4 | 2048 | 2.22 | 2.000 | 4.22 | 26% | fits |
| 4bit | 4 | 4096 | 2.22 | 4.000 | 6.22 | 39% | fits |

## Target + draft together: Qwen/Qwen3-1.7B + Qwen/Qwen3-0.6B

Total = weights(target) + weights(draft) + KV(target) + KV(draft), both KV caches fp16, same seq_len.

| weights precision | batch | seq | weights GiB | KV GiB | total GiB | % of 16 GiB | fit |
|---|---|---|---|---|---|---|---|
| fp16 | 1 | 512 | 5.18 | 0.109 | 5.29 | 33% | fits |
| fp16 | 1 | 1024 | 5.18 | 0.219 | 5.40 | 34% | fits |
| fp16 | 1 | 2048 | 5.18 | 0.438 | 5.62 | 35% | fits |
| fp16 | 1 | 4096 | 5.18 | 0.875 | 6.06 | 38% | fits |
| fp16 | 4 | 512 | 5.18 | 0.438 | 5.62 | 35% | fits |
| fp16 | 4 | 1024 | 5.18 | 0.875 | 6.06 | 38% | fits |
| fp16 | 4 | 2048 | 5.18 | 1.750 | 6.93 | 43% | fits |
| fp16 | 4 | 4096 | 5.18 | 3.500 | 8.68 | 54% | fits |
| int8 | 1 | 512 | 2.59 | 0.109 | 2.70 | 17% | fits |
| int8 | 1 | 1024 | 2.59 | 0.219 | 2.81 | 18% | fits |
| int8 | 1 | 2048 | 2.59 | 0.438 | 3.03 | 19% | fits |
| int8 | 1 | 4096 | 2.59 | 0.875 | 3.47 | 22% | fits |
| int8 | 4 | 512 | 2.59 | 0.438 | 3.03 | 19% | fits |
| int8 | 4 | 1024 | 2.59 | 0.875 | 3.47 | 22% | fits |
| int8 | 4 | 2048 | 2.59 | 1.750 | 4.34 | 27% | fits |
| int8 | 4 | 4096 | 2.59 | 3.500 | 6.09 | 38% | fits |
| 4bit | 1 | 512 | 1.30 | 0.109 | 1.41 | 9% | fits |
| 4bit | 1 | 1024 | 1.30 | 0.219 | 1.51 | 9% | fits |
| 4bit | 1 | 2048 | 1.30 | 0.438 | 1.73 | 11% | fits |
| 4bit | 1 | 4096 | 1.30 | 0.875 | 2.17 | 14% | fits |
| 4bit | 4 | 512 | 1.30 | 0.438 | 1.73 | 11% | fits |
| 4bit | 4 | 1024 | 1.30 | 0.875 | 2.17 | 14% | fits |
| 4bit | 4 | 2048 | 1.30 | 1.750 | 3.05 | 19% | fits |
| 4bit | 4 | 4096 | 1.30 | 3.500 | 4.80 | 30% | fits |

## Target + draft: meta-llama/Llama-3.2-3B-Instruct + meta-llama/Llama-3.2-1B-Instruct

Skipped (model config inaccessible).
