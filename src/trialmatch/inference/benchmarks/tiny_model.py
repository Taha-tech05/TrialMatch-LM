"""Tiny randomly initialised Qwen3 (GQA: 8 query heads, 2 KV heads) for CPU tests and script smoke runs.
No downloads. Weights are drawn with a large init std so the greedy output is varied, not one repeated token."""


def build_tiny_qwen3(seed: int = 0, dtype=None, vocab_size: int = 256, layers: int = 3, max_pos: int = 1536):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=8, num_key_value_heads=2, head_dim=16, max_position_embeddings=max_pos,
                      initializer_range=0.2, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                      tie_word_embeddings=False)
    model = Qwen3ForCausalLM(cfg).eval()
    return model.to(dtype) if dtype is not None else model
