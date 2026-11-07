"""KV-cache decoder tests on a tiny random Qwen3 (fp32, CPU, no downloads)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from trialmatch.inference.benchmarks.tiny_model import build_tiny_qwen3
from trialmatch.inference.kv_cache.cache_utils import (cache_nbytes, cache_seq_length, cache_shapes,
                                                       describe_cache, expected_cache_bytes, model_cache_dims)
from trialmatch.inference.kv_cache.cached_decode import cached_greedy_decode
from trialmatch.inference.kv_cache.compare import compare_decoders
from trialmatch.inference.kv_cache.reference_decode import reference_greedy_decode

LENGTHS = [(16, 8), (16, 32), (64, 32), (64, 64), (200, 8), (200, 64)]
EQUALITY_TABLE = []


@pytest.fixture(scope="module")
def model():
    return build_tiny_qwen3(seed=0)


def prompt(n, seed=1, vocab=256):
    g = torch.Generator().manual_seed(seed + n)
    return torch.randint(3, vocab, (1, n), generator=g)   # ids 0-2 are pad/bos/eos: never in the prompt


@pytest.mark.parametrize("p,n", LENGTHS)
def test_greedy_equality_cached_vs_uncached(model, p, n):
    ids = prompt(p)
    ref = reference_greedy_decode(model, ids, max_new_tokens=n)
    cac = cached_greedy_decode(model, ids, max_new_tokens=n)
    cmp = compare_decoders(model, ids, max_new_tokens=n)
    EQUALITY_TABLE.append((p, n, len(set(ref.sequences[0])), ref.sequences == cac.sequences))
    assert len(ref.sequences[0]) == n
    assert ref.sequences == cac.sequences, cmp.summary()
    assert cmp.equal and max(cmp.max_logit_diff_per_step) < 1e-4


def test_output_is_not_degenerate(model):
    """Guards the other tests: a single repeated token would make equality and the wrong-position test vacuous."""
    toks = reference_greedy_decode(model, prompt(64), max_new_tokens=32).sequences[0]
    assert len(set(toks)) >= 4


def test_cache_length_and_shapes(model):
    P, N = 24, 7
    d = model_cache_dims(model.config)
    assert d["kv_heads"] == 2 and d["q_heads"] == 8
    res = cached_greedy_decode(model, prompt(P), max_new_tokens=N + 1)   # 1 token from prefill + N decode steps
    assert res.cache_lengths == [P + i for i in range(N + 1)]            # after prefill = P, after N steps = P + N
    assert cache_seq_length(res.final_cache) == P + N
    shapes = cache_shapes(res.final_cache)
    assert len(shapes) == d["layers"]
    for k, v in shapes:
        assert k == v == (1, d["kv_heads"], P + N, d["head_dim"])        # kv_heads (2), not query heads (8)


def test_cache_bytes_match_formula(model, caplog):
    P = 30
    for n_new in (1, 10):
        res = cached_greedy_decode(model, prompt(P), max_new_tokens=n_new)
        L = P + n_new - 1
        assert cache_nbytes(res.final_cache) == expected_cache_bytes(model.config, L, 1, 4)
        with caplog.at_level("INFO"):
            info = describe_cache(res.final_cache, model.config)
        assert info["formula_matches"] and info["seq_len"] == L
    # the formula of the memory estimator, by hand: 2 x 3 layers x 2 kv_heads x 16 x 39 x 4 bytes
    assert expected_cache_bytes(model.config, 39, 1, 4) == 2 * 3 * 2 * 16 * 39 * 4


def test_matches_hf_generate(model):
    ids = prompt(40)
    cac = cached_greedy_decode(model, ids, max_new_tokens=24)
    gen = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, max_new_tokens=24,
                         min_new_tokens=24, eos_token_id=None, pad_token_id=0)
    assert gen[0, 40:].tolist() == cac.sequences[0]


def test_eos_stops_all_decoders_identically(model):
    ids = prompt(20)
    free = reference_greedy_decode(model, ids, max_new_tokens=12).sequences[0]
    eos = free[4]
    k = free.index(eos)                                # first occurrence
    ref = reference_greedy_decode(model, ids, max_new_tokens=12, eos_token_id=eos)
    cac = cached_greedy_decode(model, ids, max_new_tokens=12, eos_token_id=eos)
    assert ref.sequences == cac.sequences == [free[:k + 1]]


def test_wrong_position_ids_breaks_output(model):
    """position_ids = 0 for every decode step must change the tokens; the correct run must not."""
    ids = prompt(64)
    good = compare_decoders(model, ids, max_new_tokens=32)
    bad = compare_decoders(model, ids, max_new_tokens=32, position_override=lambda n: 0)
    assert good.equal
    assert not bad.equal, "tiny model insensitive to positions: test unreliable, change seed/size"
    assert bad.first_diff_step is not None and bad.max_logit_diff_at_first_diff > 1e-3
    print("\n" + bad.summary())


def test_left_padded_batch_matches_single_prompts(model):
    a, b = prompt(30, seed=5)[0], prompt(18, seed=6)[0]
    pad = 0
    ids = torch.stack([a, torch.cat([torch.full((12,), pad), b])])
    mask = torch.stack([torch.ones(30, dtype=torch.long), torch.cat([torch.zeros(12, dtype=torch.long),
                                                                      torch.ones(18, dtype=torch.long)])])
    n = 16
    ref = reference_greedy_decode(model, ids, mask, n, pad_token_id=pad)
    cac = cached_greedy_decode(model, ids, mask, n, pad_token_id=pad)
    singles = [reference_greedy_decode(model, x[None], max_new_tokens=n).sequences[0] for x in (a, b)]
    assert ref.sequences == cac.sequences == singles


def test_mismatch_investigator_reports_step_and_logit_diff(model):
    bad = compare_decoders(model, prompt(32), max_new_tokens=16, position_override=lambda n: 0)
    assert not bad.equal and isinstance(bad.first_diff_step, int)
    assert bad.max_logit_diff_per_step[bad.first_diff_step] == bad.max_logit_diff_at_first_diff
    assert "FIRST MISMATCH at step" in bad.summary()


def teardown_module(_):
    if EQUALITY_TABLE:
        print("\nprompt  new  distinct_tokens  cached==uncached")
        for p, n, u, ok in sorted(EQUALITY_TABLE):
            print(f"{p:6d} {n:4d} {u:10d}        {ok}")
