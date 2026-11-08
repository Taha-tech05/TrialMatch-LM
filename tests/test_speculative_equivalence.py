"""Speculative decoding tests: tiny random Qwen3 target + smaller draft with the same vocab (fp32, CPU, no downloads).

Reference for every equality check = the Task 3 cached greedy decoder (itself checked against the uncached one)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from trialmatch.inference.benchmarks.tiny_model import build_tiny_draft, build_tiny_qwen3, noisy_copy
from trialmatch.inference.kv_cache.cache_utils import cache_seq_length
from trialmatch.inference.kv_cache.cached_decode import cached_greedy_decode
from trialmatch.inference.speculative.cache_sync import crop_cache
from trialmatch.inference.speculative.speculative_decode import speculative_greedy_decode
from trialmatch.inference.speculative.stats import aggregate

COMBOS = [(16, 8), (32, 24), (64, 40), (48, 33)]       # (prompt length, new tokens)
KS = [1, 2, 4, 8]


@pytest.fixture(scope="module")
def target():
    return build_tiny_qwen3(seed=0)


@pytest.fixture(scope="module")
def drafts(target):
    return {"bad": build_tiny_draft(), "same": target, "mid": noisy_copy(target, 0.02)}


def prompt(n, seed=1, vocab=256):
    g = torch.Generator().manual_seed(seed + n)
    return torch.randint(3, vocab, (1, n), generator=g)   # ids 0-2 are pad/bos/eos: never in the prompt


def plain(target, ids, n, eos=None):
    return cached_greedy_decode(target, ids, max_new_tokens=n, eos_token_id=eos).sequences[0]


@pytest.mark.parametrize("kind", ["mid", "bad", "same"])
@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("p,n", COMBOS)
def test_token_equality_vs_plain_greedy(target, drafts, kind, k, p, n):
    ids = prompt(p)
    ref = plain(target, ids, n)
    res = speculative_greedy_decode(target, drafts[kind], ids, max_new_tokens=n, k=k)
    assert len(set(ref)) >= 4, "degenerate reference would make this test vacuous"
    assert res.tokens == ref
    assert res.n_new_tokens == n


def test_extremes_acceptance(target, drafts):
    ids = prompt(32)
    ref = plain(target, ids, 40)
    bad = aggregate(speculative_greedy_decode(target, drafts["bad"], ids, 40, 4))
    same = aggregate(speculative_greedy_decode(target, drafts["same"], ids, 40, 4))
    mid = aggregate(speculative_greedy_decode(target, drafts["mid"], ids, 40, 4))
    assert bad["acceptance_rate"] < 0.1 and bad["mean_tokens_per_verify"] < 1.2        # one token per target forward
    assert same["acceptance_rate"] == 1.0 and same["mean_tokens_per_verify"] > 4       # ~k+1 per forward
    assert 0.0 <= mid["acceptance_rate"] <= 1.0
    print("acceptance bad/mid/same:", bad["acceptance_rate"], mid["acceptance_rate"], same["acceptance_rate"])
    assert speculative_greedy_decode(target, drafts["bad"], ids, 40, 4).tokens == ref
    assert speculative_greedy_decode(target, drafts["same"], ids, 40, 4).tokens == ref


@pytest.mark.parametrize("kind", ["mid", "bad", "same"])
@pytest.mark.parametrize("k", [1, 4, 8])
def test_cache_length_invariants(target, drafts, kind, k):
    """Documented invariant (speculative_decode.py): after every iteration
         target cache = len(seq) - 1 = prompt + emitted - 1        (last token pending)
         draft  cache = len(seq) - 1, or len(seq) - 2 iff ALL k candidates were accepted (d_k never fed)."""
    P = 20
    ids = prompt(P)
    res = speculative_greedy_decode(target, drafts[kind], ids, max_new_tokens=37, k=k)
    emitted = 1                                                   # the prefill token
    for it in res.iterations:
        emitted += it.n_emitted
        assert it.seq_len_after == P + emitted
        assert it.target_cache_after == it.seq_len_after - 1
        if it.k_proposed == 0:      # last token of the budget: draft skipped on purpose, generation ends here
            assert it is res.iterations[-1]
            continue
        lag = 1 if (it.k_proposed > 0 and it.n_accepted == it.k_proposed and not it.truncated) else 0
        assert it.draft_cache_after == it.seq_len_after - 1 - lag
        # the crop removed exactly the rejected candidates from the target cache
        assert it.target_cache_before_crop - it.target_cache_after == it.k_proposed - (it.n_emitted - 1) or it.truncated
    assert emitted == res.n_new_tokens


def test_crop_cache_helper(target):
    ids = prompt(12)
    out = target(input_ids=ids, use_cache=True)
    past = out.past_key_values
    assert cache_seq_length(past) == 12
    past = crop_cache(past, 7)
    assert cache_seq_length(past) == 7
    assert all(k.shape[-2] == 7 for layer in past.layers for k in (layer.keys, layer.values))
    assert crop_cache(past, 7) is past
    with pytest.raises(ValueError):
        crop_cache(past, 9)
    # manual slicing path (what the fallback does) gives the same shapes
    from trialmatch.inference.speculative.cache_sync import _manual_crop
    out = target(input_ids=ids, use_cache=True)
    assert cache_seq_length(_manual_crop(out.past_key_values, 5)) == 5


def test_eos_inside_speculated_run(target, drafts):
    ids = prompt(24)
    ref = plain(target, ids, 40)
    idx = next(i for i in range(3, 8) if ref[i] not in ref[:i])     # first occurrence of this token at index idx
    eos = ref[idx]
    ref_eos = plain(target, ids, 40, eos)
    assert ref_eos == ref[:idx + 1]
    for kind in ("same", "mid", "bad"):
        for k in (1, 2, 4, 8):
            res = speculative_greedy_decode(target, drafts[kind], ids, 40, k, eos_token_id=eos)
            assert res.tokens == ref_eos and res.tokens[-1] == eos and res.finish_reason == "eos"
    # with draft == target and k=8 the first iteration accepts everything, so the EOS lies inside the run
    res = speculative_greedy_decode(target, drafts["same"], ids, 40, 8, eos_token_id=eos)
    assert res.iterations[-1].truncated or idx == 0
    assert res.iterations[-1].n_emitted < res.iterations[-1].k_proposed + 1


def test_eos_as_first_token(target):
    ids = prompt(24)
    first = plain(target, ids, 1)[0]
    res = speculative_greedy_decode(target, target, ids, 20, 4, eos_token_id=first)
    assert res.tokens == [first] and res.iterations == []


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9, 10, 11])
def test_max_new_tokens_respected(target, drafts, n):
    ids = prompt(16)
    ref = plain(target, ids, n)
    for kind in ("same", "bad"):
        for k in (1, 4, 8):
            res = speculative_greedy_decode(target, drafts[kind], ids, n, k)
            assert res.n_new_tokens == n and res.tokens == ref


def test_stats_consistency(target, drafts):
    for kind in ("same", "mid", "bad"):
        for k in KS:
            res = speculative_greedy_decode(target, drafts[kind], prompt(30), 41, k)
            a = aggregate(res)
            assert res.n_new_tokens == 1 + sum(i.n_emitted for i in res.iterations)
            assert 0.0 <= a["acceptance_rate"] <= 1.0
            assert a["tokens_accepted"] == sum(i.n_accepted for i in res.iterations)
            assert sum(a["accepted_run_hist"].values()) == a["n_iterations"]
            assert all(1 <= i.n_emitted <= k + 1 and 0 <= i.n_accepted <= i.k_proposed <= k for i in res.iterations)
            assert a["mean_tokens_per_target_forward"] == pytest.approx(res.n_new_tokens / (a["n_iterations"] + 1))
            assert 0.0 <= a["draft_share"] <= 1.0
            if kind == "same":
                assert a["acceptance_rate"] == 1.0


def test_logits_recorded_match_plain(target, drafts):
    ids = prompt(20)
    ref = cached_greedy_decode(target, ids, max_new_tokens=15, return_logits=True)
    res = speculative_greedy_decode(target, drafts["mid"], ids, 15, 4, return_logits=True)
    assert res.tokens == ref.sequences[0] and len(res.step_logits) == 15
    for a, b in zip(res.step_logits, ref.step_logits):
        assert (a - b[0]).abs().max() < 1e-3


def test_matches_hf_assisted_generation(target, drafts):
    """CROSS-CHECK ONLY: HF assisted generation is not our implementation."""
    from trialmatch.inference.speculative.hf_assisted_reference import hf_assisted_generate
    ids = prompt(24)
    ours = speculative_greedy_decode(target, drafts["mid"], ids, 30, 4).tokens
    try:
        hf = hf_assisted_generate(target, drafts["mid"], ids, 30)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"HF assisted generation unavailable in this transformers version: {e!r}")
    assert ours == hf == plain(target, ids, 30)


def test_batch_size_must_be_one(target):
    with pytest.raises(AssertionError):
        speculative_greedy_decode(target, target, torch.randint(3, 256, (2, 8)), 4, 2)


def test_skipping_the_cache_crop_breaks_decoding(target, drafts):
    """Why cache sync matters: without the crop the caches keep rejected candidates, so the next forward sees
    wrong keys (and a cache longer than the mask). The result is an error or different tokens - never a match."""
    ids = prompt(24)
    ref = plain(target, ids, 30)
    # sanity: the same call with the crop is correct
    assert speculative_greedy_decode(target, drafts["bad"], ids, 30, 4).tokens == ref
    try:
        res = speculative_greedy_decode(target, drafts["bad"], ids, 30, 4, _skip_crop=True)
    except Exception:  # noqa: BLE001  shape / length mismatch is an acceptable demonstration
        return
    assert res.tokens != ref
    # and the cache lengths show the desync directly
    assert any(it.target_cache_after != it.seq_len_after - 1 for it in res.iterations)
