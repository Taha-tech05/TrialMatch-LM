"""Uncached greedy decoder: the correctness reference.

Every step re-runs a full forward pass over the WHOLE sequence so far (prompt + generated tokens) with
use_cache=False, takes the argmax of the last-position logits and appends it. No model.generate.
(logits_to_keep=1 only skips the lm_head on earlier positions; the transformer body still sees everything.)
"""
import torch

from trialmatch.inference.kv_cache.decode_common import DecodeResult, normalize_eos, now, positions_from_mask


@torch.no_grad()
def reference_greedy_decode(model, input_ids, attention_mask=None, max_new_tokens: int = 64, eos_token_id=None,
                            pad_token_id: int | None = None, return_logits: bool = False) -> DecodeResult:
    """input_ids [B, T] (left-padded if B > 1; pass attention_mask). Stops per row on EOS, overall at
    max_new_tokens. The first forward is timed as 'prefill', the rest as 'decode'."""
    dev = input_ids.device
    ids = input_ids
    mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
    B = ids.shape[0]
    eos = normalize_eos(eos_token_id)
    pad = pad_token_id if pad_token_id is not None else (min(eos) if eos else 0)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    res = DecodeResult(sequences=[[] for _ in range(B)])
    for step in range(max_new_tokens):
        t0 = now(dev)
        out = model(input_ids=ids, attention_mask=mask, position_ids=positions_from_mask(mask),
                    use_cache=False, logits_to_keep=1)
        logits = out.logits[:, -1, :]
        nxt = logits.argmax(-1)
        t1 = now(dev)
        if step == 0:
            res.prefill_s += t1 - t0
        else:
            res.decode_s += t1 - t0
        if return_logits:
            res.step_logits.append(logits.float().cpu())
        nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
        for b in range(B):
            if not finished[b]:
                res.sequences[b].append(int(nxt[b]))
                if int(nxt[b]) in eos:
                    finished[b] = True
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        mask = torch.cat([mask, torch.ones(B, 1, dtype=mask.dtype, device=dev)], dim=1)
        if bool(finished.all()):
            break
    return res
