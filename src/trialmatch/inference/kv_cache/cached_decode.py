"""Manual KV-cache greedy decoder: explicit prefill, then one-token decode steps. No model.generate.

Prefill : one forward over the whole prompt (use_cache=True); we keep `past_key_values` ourselves.
Decode  : feed ONLY the newest token; pass the cache back in; give the model
          * position_ids = number of real tokens already in the sequence (= cache length for batch 1, no padding),
            so RoPE rotates the new token by its true absolute position;
          * cache_position = [cache_len] (where the new K/V is written);
          * a 2-D attention_mask of length cache_len + 1 (all previous columns + a 1 for the new token).
            The model builds the causal mask from it: a single query row may attend to every cached key.
Left-padded batches work: positions come from the mask (padding = no position), the mask keeps its zeros for
padding columns and grows by a 1 column per step. Finished rows keep decoding pad tokens (output discarded).
"""
import torch

from trialmatch.inference.kv_cache.cache_utils import cache_seq_length
from trialmatch.inference.kv_cache.decode_common import DecodeResult, normalize_eos, now, positions_from_mask


@torch.no_grad()
def cached_greedy_decode(model, input_ids, attention_mask=None, max_new_tokens: int = 64, eos_token_id=None,
                         pad_token_id: int | None = None, return_logits: bool = False,
                         position_override=None) -> DecodeResult:
    """position_override(cache_len) -> int is a TEST-ONLY hook that replaces the decode position_ids with a
    wrong value (e.g. lambda n: 0) to prove that position handling matters."""
    dev = input_ids.device
    mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
    B, T = input_ids.shape
    eos = normalize_eos(eos_token_id)
    pad = pad_token_id if pad_token_id is not None else (min(eos) if eos else 0)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    res = DecodeResult(sequences=[[] for _ in range(B)])

    def emit(logits):
        nxt = torch.where(finished, torch.full((B,), pad, device=dev), logits.argmax(-1))
        if return_logits:
            res.step_logits.append(logits.float().cpu())
        for b in range(B):
            if not finished[b]:
                res.sequences[b].append(int(nxt[b]))
                if int(nxt[b]) in eos:
                    finished[b] = True
        return nxt

    # ---- prefill: whole prompt, keep the cache explicitly
    t0 = now(dev)
    out = model(input_ids=input_ids, attention_mask=mask, position_ids=positions_from_mask(mask),
                cache_position=torch.arange(T, device=dev), past_key_values=None, use_cache=True,
                logits_to_keep=1)
    past = out.past_key_values
    nxt = emit(out.logits[:, -1, :])
    res.prefill_s = now(dev) - t0
    res.cache_lengths.append(cache_seq_length(past))
    assert res.cache_lengths[-1] == T, f"cache length {res.cache_lengths[-1]} != prompt length {T}"
    next_pos = mask.long().sum(-1, keepdim=True)          # [B, 1] absolute position of the next token

    # ---- decode: newest token only
    for _ in range(max_new_tokens - 1):
        if bool(finished.all()):
            break
        t0 = now(dev)
        cache_len = cache_seq_length(past)
        mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=dev)], dim=1)  # length cache_len + 1
        pos = next_pos if position_override is None else torch.full_like(next_pos, position_override(cache_len))
        out = model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                    cache_position=torch.tensor([cache_len], device=dev), past_key_values=past, use_cache=True)
        past = out.past_key_values                          # same object (updated in place) or a new one
        res.cache_lengths.append(cache_seq_length(past))
        assert res.cache_lengths[-1] == cache_len + 1, "cache did not grow by exactly one position"
        nxt = emit(out.logits[:, -1, :])
        next_pos = next_pos + 1
        res.decode_s += now(dev) - t0
    res.final_cache = past
    return res
