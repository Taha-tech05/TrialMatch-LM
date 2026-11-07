"""Measurement helpers (no model weights, no training/decoding code).

All memory figures are ESTIMATES from config values; nothing here loads weights.
"""
import math
from typing import Sequence

GIB = 1024 ** 3

# role -> repo id. Llama is gated: callers must tolerate access failure.
CANDIDATES = {
    "target": "Qwen/Qwen3-4B",
    "draft": "Qwen/Qwen3-0.6B",
    "fallback": "Qwen/Qwen3-1.7B",
    "reference_alt": "meta-llama/Llama-3.2-3B-Instruct",
    "reference_alt_draft": "meta-llama/Llama-3.2-1B-Instruct",
}
COMPAT_PAIRS = [("Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B"),
                ("meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct")]

BYTES_PER_PARAM = {"fp16": 2.0, "int8": 1.0, "4bit": 0.5}


def weights_bytes(n_params: float, precision: str) -> float:
    """Estimated weight memory = params x bytes/param (no quantisation scales/zero-points)."""
    return n_params * BYTES_PER_PARAM[precision]


def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, seq_len: int,
                   batch: int = 1, bytes_per_elem: int = 2) -> int:
    """2 (K and V) x layers x kv_heads x head_dim x seq_len x bytes (x batch)."""
    return 2 * layers * kv_heads * head_dim * seq_len * bytes_per_elem * batch


def estimate_params_from_config(cfg: dict) -> int:
    """Rough dense-decoder parameter count from config values (used only if hub info is missing)."""
    h, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or h // nh
    inter = cfg["intermediate_size"]
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
    mlp = 3 * h * inter
    emb = V * h * (1 if cfg.get("tie_word_embeddings") else 2)
    return L * (attn + mlp + 2 * h) + emb + h


def describe(values: Sequence[float]) -> dict:
    """min / mean / median / p95 / max (p95 by linear interpolation)."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return {k: None for k in ("n", "min", "mean", "median", "p95", "max")}
    pos = 0.95 * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    p95 = s[lo] + (s[hi] - s[lo]) * (pos - lo)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"n": n, "min": s[0], "mean": sum(s) / n, "median": med, "p95": p95, "max": s[-1]}
