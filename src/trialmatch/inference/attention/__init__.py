"""Attention implementations: materialised reference, tiled online-softmax, and SDPA comparison wrappers."""
from trialmatch.inference.attention.reference import reference_attention, repeat_kv
from trialmatch.inference.attention.tiled import tiled_attention

__all__ = ["reference_attention", "repeat_kv", "tiled_attention"]
