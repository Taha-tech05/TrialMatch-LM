"""Tiny randomly initialised Qwen3 (GQA: 8 query heads, 2 KV heads) for CPU tests and script smoke runs.
No downloads. Weights are drawn with a large init std so the greedy output is varied, not one repeated token."""


def build_tiny_qwen3(seed: int = 0, dtype=None, vocab_size: int = 256, layers: int = 3, max_pos: int = 1536,
                     hidden: int = 64, intermediate: int = 128, heads: int = 8, kv_heads: int = 2, head_dim: int = 16):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=vocab_size, hidden_size=hidden, intermediate_size=intermediate, num_hidden_layers=layers,
                      num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=head_dim, max_position_embeddings=max_pos,
                      initializer_range=0.2, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                      tie_word_embeddings=False)
    model = Qwen3ForCausalLM(cfg).eval()
    return model.to(dtype) if dtype is not None else model


def build_tiny_draft(seed: int = 123, dtype=None, vocab_size: int = 256, max_pos: int = 1536):
    """A smaller, independently initialised draft (1 layer, hidden 32) with the same vocab as build_tiny_qwen3:
    a 'bad' draft whose proposals mostly disagree with the target (near-0 acceptance)."""
    return build_tiny_qwen3(seed, dtype, vocab_size, layers=1, max_pos=max_pos, hidden=32, intermediate=64,
                            heads=4, kv_heads=2, head_dim=8)


def noisy_copy(model, noise: float, seed: int = 7):
    """Deep copy of `model` with Gaussian noise (relative to each weight's std) added: a 'decent' draft that agrees
    with the target most of the time (acceptance strictly between the extremes)."""
    import copy

    import torch
    m = copy.deepcopy(model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g) * noise * (p.std() if p.numel() > 1 else 1.0))
    return m.eval()
