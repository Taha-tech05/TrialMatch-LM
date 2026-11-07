"""CPU tests: tiled online-softmax attention vs the materialised reference (and torch SDPA / HF as independent checks).

Tolerances: fp32 max abs error 1e-5. fp16 inputs: both implementations compute in fp32 and round once to fp16, so they
can differ by at most about one fp16 rounding step of the output (outputs are O(1): ulp ~ 1e-3); we allow 2e-3.

Run `python tests/test_attention_correctness.py` to print the max/mean error table across shapes and block sizes.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trialmatch.inference.attention.reference import reference_attention, repeat_kv
from trialmatch.inference.attention.sdpa_backends import sdpa_attention, try_sdpa
from trialmatch.inference.attention.tiled import tiled_attention

TOL_FP32 = 1e-5
TOL_FP16 = 2e-3
SHAPES = [(1, 2, 8, 16), (2, 4, 37, 32), (1, 8, 128, 64), (1, 4, 200, 128)]   # (batch, heads, seq, head_dim)


def rand_qkv(b, h, sq, sk, d, hkv=None, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    hkv = hkv or h
    q = torch.randn(b, h, sq, d, generator=g)
    k = torch.randn(b, hkv, sk, d, generator=g)
    v = torch.randn(b, hkv, sk, d, generator=g)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def block_sizes(seq):
    return sorted({1, 7, 16, 64, seq})


def errors(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


# ---------------------------------------------------------------- tiled vs reference
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [True, False])
def test_tiled_matches_reference_fp32(shape, causal):
    b, h, s, d = shape
    q, k, v = rand_qkv(b, h, s, s, d)
    ref = reference_attention(q, k, v, causal=causal)
    for bq in block_sizes(s):
        for bk in block_sizes(s):
            if s > 64 and (bq == 1 or bk == 1):
                continue  # block size 1 is covered on the smaller shapes; keeps the CPU run short
            out = tiled_attention(q, k, v, causal=causal, block_q=bq, block_k=bk)
            mx, _ = errors(out, ref)
            assert mx < TOL_FP32, f"shape={shape} causal={causal} bq={bq} bk={bk} max_err={mx:.2e}"


@pytest.mark.parametrize("shape", SHAPES[:3])
@pytest.mark.parametrize("causal", [True, False])
def test_tiled_matches_reference_fp16_inputs(shape, causal):
    b, h, s, d = shape
    q, k, v = rand_qkv(b, h, s, s, d, dtype=torch.float16)
    ref = reference_attention(q, k, v, causal=causal)
    out = tiled_attention(q, k, v, causal=causal, block_q=16, block_k=7)
    assert out.dtype == torch.float16 and ref.dtype == torch.float16
    assert errors(out, ref)[0] < TOL_FP16
    # both stay close to the fp32 computation on the same (fp16-valued) inputs
    ref32 = reference_attention(q.float(), k.float(), v.float(), causal=causal)
    assert errors(ref, ref32)[0] < TOL_FP16


# ---------------------------------------------------------------- seq_q != seq_k with offset convention
@pytest.mark.parametrize("sq", [1, 5])
@pytest.mark.parametrize("causal", [True, False])
def test_decode_style_sq_ne_sk(sq, causal):
    q, k, v = rand_qkv(2, 4, sq, 50, 32)
    ref = reference_attention(q, k, v, causal=causal)
    for bq, bk in [(1, 1), (2, 7), (64, 16), (5, 50), (3, 64)]:
        out = tiled_attention(q, k, v, causal=causal, block_q=bq, block_k=bk)
        assert errors(out, ref)[0] < TOL_FP32


def test_offset_convention_equals_last_rows_of_full_attention():
    """Queries = last 5 positions of a 50-token sequence: must equal the last 5 rows of the full causal attention."""
    qf, k, v = rand_qkv(1, 2, 50, 50, 16)
    full = reference_attention(qf, k, v, causal=True)
    for fn in (lambda *a: reference_attention(*a, causal=True), lambda *a: tiled_attention(*a, causal=True, block_q=2, block_k=9)):
        part = fn(qf[:, :, -5:], k, v)
        assert errors(part, full[:, :, -5:])[0] < TOL_FP32
    # single decode step sees every key: same as non-causal
    q1 = qf[:, :, -1:]
    assert errors(reference_attention(q1, k, v, causal=True), reference_attention(q1, k, v, causal=False))[0] == 0.0


# ---------------------------------------------------------------- GQA
@pytest.mark.parametrize("h,hkv", [(32, 8), (16, 8)])
def test_gqa(h, hkv):
    q, k, v = rand_qkv(1, h, 40, 40, 16, hkv=hkv)
    ref = reference_attention(q, k, v)
    out = tiled_attention(q, k, v, block_q=16, block_k=7)
    assert errors(out, ref)[0] < TOL_FP32
    # reference with repeated KV heads and equal head counts gives the same answer
    explicit = reference_attention(q, repeat_kv(k, h // hkv), repeat_kv(v, h // hkv))
    assert errors(ref, explicit)[0] == 0.0


def test_repeat_kv_matches_hf_and_head_mapping():
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv as hf_repeat_kv
    x = torch.randn(2, 8, 5, 4)
    r = repeat_kv(x, 4)
    assert r.shape == (2, 32, 5, 4)
    assert torch.equal(r, hf_repeat_kv(x, 4)) and torch.equal(r, torch.repeat_interleave(x, 4, dim=1))
    for h in range(32):   # query head h uses KV head h // 4
        assert torch.equal(r[:, h], x[:, h // 4])


def test_bad_head_ratio_rejected():
    q, k, v = rand_qkv(1, 6, 4, 4, 8, hkv=4)
    with pytest.raises(ValueError):
        reference_attention(q, k, v)


# ---------------------------------------------------------------- independent cross-checks
@pytest.mark.parametrize("causal", [True, False])
def test_reference_and_tiled_vs_torch_sdpa(causal):
    q, k, v = rand_qkv(2, 4, 37, 37, 32)
    sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)   # tests only
    assert errors(reference_attention(q, k, v, causal=causal), sdpa)[0] < TOL_FP32
    assert errors(tiled_attention(q, k, v, causal=causal, block_q=8, block_k=16), sdpa)[0] < TOL_FP32


def test_sdpa_wrapper_math_and_unavailable_backend_reporting():
    q, k, v = rand_qkv(1, 8, 5, 30, 16, hkv=2)
    out, err = try_sdpa(q, k, v, causal=True, backend="math")      # offset convention via explicit mask
    assert err is None and errors(out, reference_attention(q, k, v))[0] < TOL_FP32
    assert errors(sdpa_attention(q, k, v, backend="auto"), out)[0] < TOL_FP32
    for name in ("efficient", "flash"):   # may or may not exist on this machine; must never raise
        out, err = try_sdpa(q, k, v, backend=name)
        assert (out is None) != (err is None)


def test_reference_vs_hf_eager_attention_tiny_config():
    """Same q, k, v and the additive causal mask HF builds: HF eager_attention_forward vs ours (GQA 8/2)."""
    from types import SimpleNamespace

    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    b, h, hkv, s, d = 2, 8, 2, 11, 16
    q, k, v = rand_qkv(b, h, s, s, d, hkv=hkv)
    mask = torch.full((s, s), float("-inf")).triu(1)[None, None].expand(b, 1, s, s)
    mod = SimpleNamespace(num_key_value_groups=h // hkv, training=False)
    hf_out, _ = eager_attention_forward(mod, q, k, v, mask, scaling=1 / math.sqrt(d))   # [B, S, H, D]
    assert errors(reference_attention(q, k, v), hf_out.transpose(1, 2))[0] < TOL_FP32


# ---------------------------------------------------------------- numerical stability
def naive_softmax_attention(q, k, v, scale):
    """Deliberately NOT stable (no row-max subtraction); exists only to show why the real code subtracts the max."""
    w = torch.exp(q @ k.transpose(-1, -2) * scale)
    return (w / w.sum(-1, keepdim=True)) @ v


def test_large_logits_stay_finite_and_naive_softmax_does_not():
    d = 64
    g = torch.Generator().manual_seed(1)
    q = 100 * torch.randn(1, 2, 32, d, generator=g)    # |scores| ~ 100*100*sqrt(64)/sqrt(64) = 1e4
    k = 100 * torch.randn(1, 2, 32, d, generator=g)
    v = torch.randn(1, 2, 32, d, generator=g)
    scores = q @ k.transpose(-1, -2) / math.sqrt(d)
    assert scores.abs().max() > 5e3
    naive = naive_softmax_attention(q, k, v, 1 / math.sqrt(d))
    assert not torch.isfinite(naive).all()                      # exp(1e4) = inf -> inf/inf = NaN
    for causal in (True, False):
        ref = reference_attention(q, k, v, causal=causal)
        til = tiled_attention(q, k, v, causal=causal, block_q=8, block_k=5)
        for o in (ref, til):
            assert torch.isfinite(o).all()
            assert o.min() >= v.min() - 1e-4 and o.max() <= v.max() + 1e-4   # convex combination of V rows
        # near-one-hot weights: score rounding (~1e-3 at 1e4) can flip near ties, so compare loosely
        assert errors(ref, til)[0] < 5e-2


# ---------------------------------------------------------------- fully masked rows
def test_fully_masked_rows_give_zeros_not_nan():
    # Sq > Sk with causal: queries 0..2 see no key (offset = Sk - Sq = -3)
    q, k, v = rand_qkv(1, 2, 8, 5, 16)
    ref = reference_attention(q, k, v, causal=True)
    assert torch.isfinite(ref).all() and (ref[:, :, :3] == 0).all()
    for bq, bk in [(1, 1), (3, 2), (8, 5), (4, 64)]:
        til = tiled_attention(q, k, v, causal=True, block_q=bq, block_k=bk)
        assert torch.isfinite(til).all() and (til[:, :, :3] == 0).all()
        assert errors(til, ref)[0] < TOL_FP32
    assert not (ref[:, :, 3:] == 0).all()    # the visible rows are real


# ---------------------------------------------------------------- hand-computed example
def test_hand_computed_two_queries_three_keys():
    """d=1, scale=1. q = [0, 1]; k = [0, 0, ln 2]; v = [1, 2, 4]; causal offset = 3 - 2 = 1.
    causal query 0 sees keys 0,1: scores (0,0)      -> weights 1/2,1/2       -> (1+2)/2 = 1.5
    query 1 sees keys 0,1,2:      scores (0,0,ln 2) -> e^s = 1,1,2 -> 1/4,1/4,1/2 -> 0.25+0.5+2 = 2.75
    non-causal query 0 also sees key 2: scores (0,0,0) -> 1/3 each -> 7/3."""
    ln2 = math.log(2.0)
    q = torch.tensor([0.0, 1.0]).view(1, 1, 2, 1)
    k = torch.tensor([0.0, 0.0, ln2]).view(1, 1, 3, 1)
    v = torch.tensor([1.0, 2.0, 4.0]).view(1, 1, 3, 1)
    expected_causal = torch.tensor([1.5, 2.75]).view(1, 1, 2, 1)
    expected_full = torch.tensor([7.0 / 3.0, 2.75]).view(1, 1, 2, 1)
    for bq, bk in [(2, 3), (1, 1), (2, 2)]:
        assert torch.allclose(reference_attention(q, k, v, True, scale=1.0), expected_causal, atol=1e-6)
        assert torch.allclose(tiled_attention(q, k, v, True, scale=1.0, block_q=bq, block_k=bk), expected_causal, atol=1e-6)
        assert torch.allclose(reference_attention(q, k, v, False, scale=1.0), expected_full, atol=1e-6)
        assert torch.allclose(tiled_attention(q, k, v, False, scale=1.0, block_q=bq, block_k=bk), expected_full, atol=1e-6)


def test_default_scale_is_inverse_sqrt_head_dim():
    q, k, v = rand_qkv(1, 2, 6, 6, 16)
    assert errors(reference_attention(q, k, v), reference_attention(q, k, v, scale=0.25))[0] == 0.0
    assert errors(tiled_attention(q, k, v), tiled_attention(q, k, v, scale=0.25))[0] == 0.0


def test_invalid_block_size():
    q, k, v = rand_qkv(1, 1, 4, 4, 8)
    with pytest.raises(ValueError):
        tiled_attention(q, k, v, block_q=0)


def error_table():
    rows = []
    for (b, h, s, d) in SHAPES:
        q, k, v = rand_qkv(b, h, s, s, d)
        for causal in (True, False):
            ref = reference_attention(q, k, v, causal=causal)
            for bs in block_sizes(s):
                out = tiled_attention(q, k, v, causal=causal, block_q=bs, block_k=bs)
                mx, mean = errors(out, ref)
                rows.append((f"({b},{h},{s},{d})", "causal" if causal else "full", bs, mx, mean))
    return rows


if __name__ == "__main__":
    print(f"{'shape':<16}{'mask':<8}{'block':>6}{'max_abs_err':>14}{'mean_abs_err':>14}")
    for shp, m, bs, mx, mean in error_table():
        print(f"{shp:<16}{m:<8}{bs:>6}{mx:>14.3e}{mean:>14.3e}")
