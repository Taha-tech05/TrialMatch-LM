"""CROSS-CHECK ONLY: Hugging Face assisted generation (`assistant_model=`).

This is NOT the speculative decoder of this project (see speculative_decode.py). It exists so tests can confirm
that our hand-written decoder and HF's own implementation agree on the greedy output. Never used for benchmarks.
"""
import torch


@torch.no_grad()
def hf_assisted_generate(target, draft, input_ids, max_new_tokens: int, eos_token_id=None) -> list[int]:
    out = target.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), assistant_model=draft,
                          do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
                          pad_token_id=0, temperature=None, top_p=None, top_k=None)
    return out[0, input_ids.shape[1]:].tolist()
