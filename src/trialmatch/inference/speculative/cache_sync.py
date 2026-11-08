"""Explicit KV-cache cropping, the operation that keeps a speculative decoder correct.

After a verify pass the cache holds keys/values for tokens that were REJECTED. They must be removed, otherwise
the next forward attends to keys of tokens that are not part of the output (and the cache is longer than the
mask / positions the decoder passes). `crop_cache(past, new_len)` cuts every layer back to `new_len` positions.

transformers API notes (installed 5.22.2):
  * DynamicCache.crop(n) takes a NEGATIVE n = number of tokens to remove (a positive value raises). In 4.x
    crop(max_length) took an absolute length and also accepted negative values with the same meaning.
  * We call the official method with a negative count, then VERIFY the resulting length. If the official method
    is missing, raises, or leaves the wrong length, we slice the K/V tensors ourselves
    (`layers[i].keys/values`, or `key_cache/value_cache` lists, or a legacy tuple which is rebuilt and returned).
Always use the return value: it is the same object except for the legacy tuple layout.
"""
from trialmatch.inference.kv_cache.cache_utils import cache_seq_length


def _manual_crop(past, new_len: int):
    if hasattr(past, "layers"):
        for layer in past.layers:
            if getattr(layer, "keys", None) is not None and layer.keys.numel() > 0:
                layer.keys = layer.keys[..., :new_len, :]
                layer.values = layer.values[..., :new_len, :]
        return past
    if hasattr(past, "key_cache"):
        past.key_cache = [k[..., :new_len, :] if k.numel() else k for k in past.key_cache]
        past.value_cache = [v[..., :new_len, :] if v.numel() else v for v in past.value_cache]
        return past
    if isinstance(past, (tuple, list)):
        return tuple((k[..., :new_len, :], v[..., :new_len, :]) for k, v in past)
    raise TypeError(f"cannot crop cache of type {type(past)!r}")


def crop_cache(past, new_len: int):
    """Cut `past` back to exactly `new_len` cached positions (no-op if already that long). Returns the cache."""
    cur = cache_seq_length(past)
    if new_len > cur:
        raise ValueError(f"cannot crop a cache of length {cur} up to {new_len}")
    if new_len == cur:
        return past
    if hasattr(past, "crop"):
        try:
            past.crop(-(cur - new_len))
            if cache_seq_length(past) == new_len:
                return past
        except Exception:  # noqa: BLE001  API differs across versions; fall through to slicing
            pass
    past = _manual_crop(past, new_len)
    got = cache_seq_length(past)
    if got != new_len:
        raise AssertionError(f"cache crop failed: wanted {new_len}, have {got}")
    return past
