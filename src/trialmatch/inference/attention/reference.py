"""Reference (materialised) scaled dot-product attention, written out step by step.

    scores = Q K^T * scale  ->  additive mask  ->  softmax (row max subtracted)  ->  weights V

Shapes: q [B, H, Sq, D]; k, v [B, Hkv, Sk, D] with H a multiple of Hkv (grouped-query attention).

Conventions (also used by the tiled implementation, see decisions.md):
* Causal mask with offset: query i (0-based within the Sq queries) may attend to key j iff j <= i + (Sk - Sq).
  Sq == Sk is the usual lower-triangular mask; Sq == 1 with a cache sees every key; Sq < Sk is a decode/prefill-with-cache
  step where the queries are the LAST Sq positions. Note: torch SDPA's is_causal=True aligns the mask at the top-left
  instead, so it only agrees with this convention when Sq == Sk.
* Precision: q, k, v are cast to fp32 for the matmuls, mask and softmax; the result is cast back to the input dtype.
  fp16 inputs therefore give an fp32-accurate result rounded once at the end.
* A query with NO visible key (possible only when Sq > Sk and causal) returns an all-zero output row, not NaN.
"""
import math

import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, Hkv, S, D] -> [B, Hkv * n_rep, S, D] by repeating each KV head n_rep times in place.

    Head mapping (identical to Hugging Face `repeat_kv` / `torch.repeat_interleave(x, n_rep, dim=1)`): KV head g serves
    query heads g*n_rep .. g*n_rep + n_rep - 1, i.e. query head h uses KV head h // n_rep.
    """
    if n_rep == 1:
        return x
    b, hkv, s, d = x.shape
    return x[:, :, None, :, :].expand(b, hkv, n_rep, s, d).reshape(b, hkv * n_rep, s, d)


def check_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> int:
    """Validate shapes, return n_rep = H / Hkv."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4-D [batch, heads, seq, head_dim]")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape, got {tuple(k.shape)} and {tuple(v.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError("batch size and head_dim of q and k must match")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(f"query heads ({q.shape[1]}) must be a multiple of kv heads ({k.shape[1]})")
    return q.shape[1] // k.shape[1]


def reference_attention(q, k, v, causal: bool = True, scale: float | None = None) -> torch.Tensor:
    n_rep = check_shapes(q, k, v)
    sq, sk, d = q.shape[2], k.shape[2], q.shape[3]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    out_dtype = q.dtype
    qf = q.float()
    kf = repeat_kv(k, n_rep).float()
    vf = repeat_kv(v, n_rep).float()

    scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale            # [B, H, Sq, Sk], fp32
    if causal:
        qi = torch.arange(sq, device=q.device)[:, None]
        kj = torch.arange(sk, device=q.device)[None, :]
        allowed = kj <= qi + (sk - sq)                                  # offset causal convention
        scores = scores.masked_fill(~allowed, float("-inf"))           # additive -inf mask

    row_max = scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isinf(row_max), torch.zeros_like(row_max), row_max)  # fully masked row: avoid -inf - -inf
    p = torch.exp(scores - row_max)                                     # masked entries -> exactly 0
    denom = p.sum(dim=-1, keepdim=True)
    weights = p / denom.clamp_min(torch.finfo(torch.float32).tiny)      # fully masked row: 0 / tiny = 0
    return torch.matmul(weights, vf).to(out_dtype)
