# Speculative decoding notes

(Prompts only - write the explanations yourself.)

## Draft / target design
- Which models: target = merged fp16 Qwen3-4B, draft = Qwen3-0.6B (untuned). Why the tokenizer must be identical.
- Layers and size: target 36 layers vs draft 28 layers (not much shallower: what does that mean for draft cost per token?).
- Memory: both models plus two KV caches on a T4.
- Batch size 1 only: why batching complicates speculation (different accepted lengths per row).

## Acceptance logic
- What the target verifies in one forward: input `[last token, d_1..d_k]`, which logits row checks which candidate.
- Longest matching prefix: why a later match after a mismatch cannot be used.
- Why the output is token-identical to plain greedy decoding (what would break this guarantee; fp16 near-ties).
- Tokens per iteration: between 1 and k+1; the bonus token when everything is accepted.

## Rejection and correction
- First mismatch: why the target's own token is emitted, and why this still makes progress.
- What happens to the candidates after the mismatch.
- EOS inside an accepted run; max_new_tokens truncation.

## Cache crop and synchronisation
- What each cache contains after the verify forward and what must be removed.
- Target cache invariant vs draft cache invariant; why the draft lags by one after full acceptance and how the catch-up forward fixes it.
- The transformers 5.x `DynamicCache.crop(-n)` convention; the manual-slicing fallback.
- What goes wrong without the crop (see the "no crop" test).

## Why speedup may not appear
- Python loop overhead: many small kernel launches, `.item()` syncs per iteration.
- Small outputs (~16 tokens, val answers ~15-38 tokens): prefill dominates, few iterations to amortise anything.
- Draft cost: 28 layers vs 36 layers means each draft token costs a large fraction of a target token.
- Low acceptance of an untuned draft on this task's JSON-style outputs; effect of fine-tuning the draft (later step).
- Verify forward of k+1 tokens vs 1 token on a T4: when is it nearly free (memory-bound) and when not.
- Choosing k: acceptance falls with position, draft cost grows linearly with k.

## Results to discuss
- Acceptance rate and mean accepted tokens per iteration vs k.
- Speedup table (synthetic lengths, val prompts); draft vs verify time split.
- Token-equality flags and any fp16 mismatches (first differing step, logit gap).
