# KV-cache notes (to be written by me)

Code: `src/trialmatch/inference/kv_cache/` (`reference_decode.py`, `cached_decode.py`, `cache_utils.py`, `compare.py`).
Headings and prompts only; the explanations are mine to fill in.

## Positional handling (RoPE and position offset)
- Why does the new token need position = current cache length, not 0?
- What happens to cached keys: are they re-rotated, or stored already rotated?
- What did the wrong-`position_ids` test show (first mismatching step, logit difference)?
- How do left-padded batches change the positions (mask cumsum)?

## Masking (causal mask with cache)
- What shape is the mask for prefill vs for a single decode step?
- Why is the 2-D `attention_mask` length `cache_len + 1` at each decode step?
- Why can the single query row attend to every cached key without an explicit triangular mask?

## Cache growth
- How many positions are cached after prefill, and after N decode steps (and why the last token is never fed)?
- Memory per token (2 x layers x kv_heads x head_dim x bytes) and the measured-vs-formula check.
- What does the growth mean for the 1024-token inputs on a 16 GB T4?

## GQA (8 KV heads vs 32 query heads)
- Why are the cached tensors `[batch, kv_heads, seq, head_dim]` and not query-head sized?
- How much memory does GQA save compared with full multi-head caching for Qwen3-4B?
- Where does the model expand/share the KV heads at attention time?

## Possible numerical differences (fp16, different matmul shapes)
- Why can prefill (T x T matmuls) and single-token steps (1 x T matmuls) round differently in fp16?
- Did the real-model benchmark show token mismatches? At which step, and how big was the logit difference?
- Why is fp32 equality on the tiny CPU model a stronger statement than fp16 equality on the GPU?
