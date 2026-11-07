"""Tiled attention with online softmax, plain PyTorch (no fused kernel, no SDPA).

For one query row, process the keys in blocks. With s_b = q K_b^T * scale (block scores) and V_b the block's values,
keep a running row max m, normaliser l and unnormalised output o, starting from m = -inf, l = 0, o = 0:

    m_new = max(m, max_j s_b[j])
    alpha = exp(m - m_new)                         (rescales everything accumulated so far)
    l_new = alpha * l + sum_j exp(s_b[j] - m_new)
    o_new = alpha * o + sum_j exp(s_b[j] - m_new) * V_b[j]
    after the last block:  output = o / l

Invariant after each block: m = max of all scores seen, l = sum exp(s - m), o = sum exp(s - m) * v over those keys, so
o / l = softmax(scores) V restricted to the keys seen. After the last block this is the exact full softmax attention.
Only a [block_q, block_k] score tile exists at any time; extra memory is O(block_q * block_k + block_q * D), not O(Sq * Sk).

Same conventions as reference.py: offset causal mask (query i sees key j iff j <= i + Sk - Sq), fp32 accumulation,
output in the input dtype, all-zero output for a query with no visible key.
"""
import math

import torch

from trialmatch.inference.attention.reference import check_shapes, repeat_kv


def tiled_attention(q, k, v, causal: bool = True, scale: float | None = None,
                    block_q: int = 64, block_k: int = 64) -> torch.Tensor:
    n_rep = check_shapes(q, k, v)
    if block_q < 1 or block_k < 1:
        raise ValueError("block_q and block_k must be >= 1")
    b, h, sq, d = q.shape
    sk = k.shape[2]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    offset = sk - sq
    neg_inf = float("-inf")
    out = torch.empty_like(q)

    for qs in range(0, sq, block_q):
        qe = min(qs + block_q, sq)
        qb = q[:, :, qs:qe].float()                                         # [B, H, bq, D]
        m = torch.full((b, h, qe - qs, 1), neg_inf, dtype=torch.float32, device=q.device)
        l = torch.zeros((b, h, qe - qs, 1), dtype=torch.float32, device=q.device)
        o = torch.zeros((b, h, qe - qs, d), dtype=torch.float32, device=q.device)
        q_pos = torch.arange(qs, qe, device=q.device)[:, None] + offset    # last visible key index per query row

        for ks in range(0, sk, block_k):
            # Causal: this block starts after the last key visible to ANY query in the block, so every score would
            # be masked and the block contributes nothing. Keys only increase, so all later blocks are skipped too.
            if causal and ks > qe - 1 + offset:
                break
            ke = min(ks + block_k, sk)
            kb = repeat_kv(k[:, :, ks:ke], n_rep).float()                  # GQA: expand only this block
            vb = repeat_kv(v[:, :, ks:ke], n_rep).float()
            s = torch.matmul(qb, kb.transpose(-1, -2)) * scale             # [B, H, bq, bk]
            if causal and ke - 1 > qs + offset:                            # block straddles the diagonal: partial mask
                k_pos = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(k_pos > q_pos, neg_inf)

            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # Rows with nothing visible yet still have m_new = -inf; exp(-inf - -inf) would be NaN. Use 0 as the
            # reference point for those rows: exp(-inf - 0) = 0, so they stay at l = 0, o = 0.
            m_ref = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
            alpha = torch.exp(m - m_ref)                                   # m = -inf -> 0 (old l, o are 0 anyway)
            p = torch.exp(s - m_ref)
            l = alpha * l + p.sum(dim=-1, keepdim=True)
            o = alpha * o + torch.matmul(p, vb)
            m = m_new

        out[:, :, qs:qe] = (o / l.clamp_min(torch.finfo(torch.float32).tiny)).to(q.dtype)  # l = 0 -> o = 0 -> 0
    return out
