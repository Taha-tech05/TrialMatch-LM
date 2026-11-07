"""torch SDPA with a selectable backend. COMPARISON ONLY: never used by reference.py or tiled.py, not wired into the model.

Backends: "math", "efficient" (mem-efficient), "flash", "auto". Unavailable backends do not crash: `try_sdpa` returns
(None, reason). GQA is handled by repeat_kv so all backends see equal head counts. For causal with Sq != Sk the offset
convention is passed as an explicit boolean mask (is_causal would align top-left); the flash backend does not take a
mask, so it reports an error for that case.
"""
import torch
import torch.nn.functional as F

from trialmatch.inference.attention.reference import check_shapes, repeat_kv

BACKENDS = ("math", "efficient", "flash")


def _backend_enum(name: str):
    from torch.nn.attention import SDPBackend
    return {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION}[name]


def sdpa_attention(q, k, v, causal: bool = True, scale: float | None = None, backend: str = "auto") -> torch.Tensor:
    """Raises on an unavailable backend; use try_sdpa to get the error as a value."""
    from torch.nn.attention import sdpa_kernel
    n_rep = check_shapes(q, k, v)
    k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
    sq, sk = q.shape[2], k.shape[2]
    mask, is_causal = None, False
    if causal:
        if sq == sk:
            is_causal = True
        else:
            qi = torch.arange(sq, device=q.device)[:, None]
            mask = torch.arange(sk, device=q.device)[None, :] <= qi + (sk - sq)
    if backend == "auto":
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, scale=scale)
    with sdpa_kernel([_backend_enum(backend)]):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, scale=scale)


def try_sdpa(q, k, v, causal: bool = True, scale: float | None = None, backend: str = "auto"):
    """(output, None) on success, (None, "<ExceptionType>: message") otherwise (including CPU / missing backend)."""
    try:
        return sdpa_attention(q, k, v, causal, scale, backend), None
    except Exception as e:  # noqa: BLE001 - report any backend failure, including OOM
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
