"""Helpers to inspect a Hugging Face KV cache: length, per-layer shapes, bytes, and the size formula.

Handles the cache containers seen across transformers versions:
  * >= 4.56 / 5.x  DynamicCache with `.layers[i].keys / .values`   (installed here: 5.22.2)
  * 4.4x - 4.55    DynamicCache with `.key_cache / .value_cache` lists
  * legacy         tuple of (key, value) tensors per layer
Each K/V tensor is [batch, kv_heads, seq_len, head_dim].
"""
import logging

from trialmatch.model.measure import kv_cache_bytes

log = logging.getLogger(__name__)


def cache_layers(past) -> list[tuple]:
    """List of (key, value) tensors, one per initialised layer."""
    if past is None:
        return []
    if hasattr(past, "layers"):
        out = []
        for layer in past.layers:
            k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
            if k is not None and v is not None and getattr(k, "numel", lambda: 0)() > 0:
                out.append((k, v))
        return out
    if hasattr(past, "key_cache"):
        return [(k, v) for k, v in zip(past.key_cache, past.value_cache) if getattr(k, "numel", lambda: 0)() > 0]
    if isinstance(past, (tuple, list)):
        return [(k, v) for k, v in past]
    if hasattr(past, "to_legacy_cache"):
        return [(k, v) for k, v in past.to_legacy_cache()]
    raise TypeError(f"unsupported cache type {type(past)!r}")


def cache_seq_length(past) -> int:
    """Number of cached positions (read from the tensors; all layers must agree)."""
    lens = {int(k.shape[-2]) for k, _ in cache_layers(past)}
    if not lens:
        return 0
    if len(lens) > 1:
        raise AssertionError(f"layers disagree on cache length: {sorted(lens)}")
    return lens.pop()


def cache_shapes(past) -> list[tuple[tuple, tuple]]:
    """Per layer: (key shape, value shape) = [batch, kv_heads, seq_len, head_dim]."""
    return [(tuple(k.shape), tuple(v.shape)) for k, v in cache_layers(past)]


def cache_nbytes(past) -> int:
    return sum(t.numel() * t.element_size() for kv in cache_layers(past) for t in kv)


def model_cache_dims(config) -> dict:
    heads = config.num_attention_heads
    return {"layers": config.num_hidden_layers, "kv_heads": config.num_key_value_heads,
            "q_heads": heads, "head_dim": getattr(config, "head_dim", None) or config.hidden_size // heads}


def expected_cache_bytes(config, seq_len: int, batch: int = 1, bytes_per_elem: int = 2) -> int:
    """2 x layers x kv_heads x head_dim x seq_len x bytes (x batch): the formula of the memory estimator."""
    d = model_cache_dims(config)
    return kv_cache_bytes(d["layers"], d["kv_heads"], d["head_dim"], seq_len, batch, bytes_per_elem)


def describe_cache(past, config=None) -> dict:
    """Length, layer count, shape of layer 0, dtype, measured bytes, and (with config) formula bytes + match."""
    layers = cache_layers(past)
    info = {"type": type(past).__name__, "seq_len": cache_seq_length(past), "n_layers": len(layers),
            "bytes_measured": cache_nbytes(past)}
    if layers:
        k = layers[0][0]
        info.update(key_shape_layer0=tuple(k.shape), dtype=str(k.dtype), batch=int(k.shape[0]))
    if config is not None and layers:
        exp = expected_cache_bytes(config, info["seq_len"], info["batch"], layers[0][0].element_size())
        info.update(bytes_formula=exp, formula_matches=exp == info["bytes_measured"])
        log.info("KV cache len=%d measured=%d B formula=%d B match=%s", info["seq_len"],
                 info["bytes_measured"], exp, info["formula_matches"])
    return info
