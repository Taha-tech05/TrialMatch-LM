"""Greedy speculative decoding, written out explicitly. Not model.generate, not assistant_model.

LIMITATION: batch size 1 only (one sequence, no padding). Batched speculation needs per-row accepted lengths and
ragged caches; out of scope here.

Setting: a small DRAFT model and the big TARGET model share one tokenizer. The output is exactly what plain greedy
decoding of the target would produce (token-identical up to floating-point ties); the draft only decides how many
target forwards are needed.

State invariant at the start of every iteration (seq = prompt + tokens emitted so far, last = seq[-1]):
    * TARGET cache holds seq[:-1]            -> length len(seq) - 1   (the last token is "pending", not yet cached)
    * DRAFT  cache holds seq[:draft_len]     -> draft_len <= len(seq) - 1 ("the draft may lag")
After an iteration with m accepted candidates the draft cache is len(seq)-1 long if m < k, and EXACTLY ONE shorter
(len(seq) - 2) if all k were accepted, because the last candidate d_k was proposed but never fed back to the draft.
At the next iteration the draft first feeds every token of seq[draft_len:] in one forward (1 or 2 tokens) -
this is the "catch-up" - and only then proposes.

One iteration (n = len(seq) - 1 = target cache length, k candidates):
  1. DRAFT : feed seq[draft_len:], argmax -> d_1; then feed d_1 -> d_2, ... -> d_k   (k-1 further 1-token forwards;
             d_k is never fed). Draft cache is now n + 1 + (k-1) = n + k long.
  2. VERIFY: ONE target forward on [last, d_1..d_k] at positions n..n+k. Logits row i is the target's prediction
             for the token AFTER input i, so g_i = argmax(row i) is "what greedy target would emit at that point":
             g_0 should equal d_1, g_1 should equal d_2, ..., g_k is the bonus.
  3. ACCEPT: m = longest prefix with d_{j+1} == g_j (j < m). Emit d_1..d_m, then g_m:
             - m < k: g_m is the CORRECTION (the target disagreed with d_{m+1}; d_{m+1} and later are thrown away),
             - m == k: g_m = g_k is the BONUS token (free: the same forward already scored the position after d_k).
             1..k+1 tokens per iteration, always at least one, so progress is guaranteed.
  4. CROP  : the target cache holds [.., last, d_1..d_k] (n + 1 + k); only last, d_1..d_m are real -> cut to n+1+m.
             The draft cache (n + k) is cut to min(n + k, n + 1 + m). Both then satisfy the invariant above for the
             new seq (length n + m + 2). Without this crop the next forward attends to rejected tokens' keys and the
             cache/mask/position lengths disagree (see tests: the "no crop" variant breaks).
When fewer than 2 tokens of budget remain, k is reduced to the budget (k_eff = 0 on the very last token: a plain
target step, draft skipped, so its cache may lag further - generation ends right after).
EOS inside the emitted run truncates the run after the EOS; the budget max_new_tokens truncates likewise. After
truncation the caches are cropped to len(seq) - 1 / min(draft_len, len(seq) - 1) so the invariants hold even then.
"""
import torch

from trialmatch.inference.kv_cache.cache_utils import cache_seq_length
from trialmatch.inference.kv_cache.decode_common import normalize_eos, now
from trialmatch.inference.speculative.cache_sync import crop_cache
from trialmatch.inference.speculative.stats import IterStats, SpecResult


def _forward(model, ids, start: int, past, total_len: int):
    """Forward `ids` [1, T] whose first token sits at absolute position `start`; the cache must hold `start`
    positions. position_ids / cache_position are explicit; the 2-D mask is all ones (no padding) of length
    start + T = cache length after the forward, so every new token sees the whole cache and (via the causal mask the
    model builds from cache_position) only earlier new tokens."""
    dev = ids.device
    T = ids.shape[1]
    assert total_len == start + T
    pos = torch.arange(start, start + T, device=dev)
    return model(input_ids=ids, attention_mask=torch.ones(1, total_len, dtype=torch.long, device=dev),
                 position_ids=pos[None], cache_position=pos, past_key_values=past, use_cache=True)


@torch.no_grad()
def speculative_greedy_decode(target, draft, input_ids, max_new_tokens: int = 64, k: int = 4, eos_token_id=None,
                              return_logits: bool = False, _skip_crop: bool = False) -> SpecResult:
    """Greedy speculative decoding of `target` with `draft` proposing k tokens per iteration. input_ids [1, T].

    `_skip_crop=True` is a TEST-ONLY sabotage switch: caches are not cropped after verification (the
    cache-synchronisation bug this module exists to avoid)."""
    assert input_ids.shape[0] == 1, "speculative decoding here supports batch size 1 only"
    assert k >= 1 and max_new_tokens >= 1
    dev = input_ids.device
    T = input_ids.shape[1]
    eos = normalize_eos(eos_token_id)
    res = SpecResult(tokens=[], prompt_len=T, k=k)

    # ---- prefill both models on the prompt; keep both caches ourselves
    t0 = now(dev)
    out_t = _forward(target, input_ids, 0, None, T)
    t_past = out_t.past_key_values
    out_d = _forward(draft, input_ids, 0, None, T)
    d_past = out_d.past_key_values
    row = out_t.logits[0, -1]
    first = int(row.argmax())
    res.tokens.append(first)
    if return_logits:
        res.step_logits.append(row.float().cpu())
    res.prefill_s = now(dev) - t0
    assert cache_seq_length(t_past) == T and cache_seq_length(d_past) == T
    if first in eos:
        res.finish_reason = "eos"
        return res
    seq = input_ids[0].tolist() + [first]             # prompt + emitted tokens (python ints)

    while len(res.tokens) < max_new_tokens:
        n = len(seq) - 1                                 # target cache length (last token pending)
        draft_len = cache_seq_length(d_past)
        if not _skip_crop:
            assert cache_seq_length(t_past) == n and draft_len <= n, (cache_seq_length(t_past), draft_len, n)
        remaining = max_new_tokens - len(res.tokens)
        k_eff = max(0, min(k, remaining - 1))            # an iteration emits up to k_eff + 1 tokens: never plan past the budget

        # ---- 1. DRAFT: catch up on seq[draft_len:], then propose k_eff tokens greedily with the draft's own cache
        t0 = now(dev)
        cand = []
        if k_eff:
            pending = torch.tensor([seq[draft_len:]], device=dev)          # 1 or 2 tokens (>= 1 since draft_len <= n)
            o = _forward(draft, pending, draft_len, d_past, len(seq))
            d_past = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(-1)                            # [1], stays on the device
            cand.append(nxt)
            for j in range(1, k_eff):                                      # d_{j+1} from feeding d_j
                o = _forward(draft, nxt[:, None], len(seq) + j - 1, d_past, len(seq) + j)
                d_past = o.past_key_values
                nxt = o.logits[:, -1, :].argmax(-1)
                cand.append(nxt)
        draft_s = now(dev) - t0

        # ---- 2. VERIFY: one target forward over [last, d_1..d_k]
        t0 = now(dev)
        ver_in = torch.cat([torch.tensor([[seq[-1]]], device=dev)] + [c[:, None] for c in cand], dim=1)  # [1, k_eff+1]
        o = _forward(target, ver_in, n, t_past, n + 1 + k_eff)
        t_past = o.past_key_values
        g = o.logits[0].argmax(-1)                                         # [k_eff+1] target greedy at each position
        # ---- 3. ACCEPT longest matching prefix
        if k_eff:
            d = torch.cat(cand)                                            # [k_eff]
            m = int((d == g[:k_eff]).long().cumprod(0).sum())
        else:
            d, m = None, 0
        g_list = g.tolist()
        run = (d[:m].tolist() if m else []) + [g_list[m]]                  # accepted + correction/bonus
        # truncate at EOS (inclusive) and at the token budget
        eos_cut = next((i + 1 for i, tok in enumerate(run) if tok in eos), None)   # keep the EOS itself
        cut = min(len(run), eos_cut or len(run), remaining)
        truncated = cut < len(run)
        if eos_cut is not None and eos_cut <= remaining:
            res.finish_reason = "eos"
        run = run[:cut]
        if return_logits:
            res.step_logits.extend(o.logits[0, i].float().cpu() for i in range(len(run)))   # row i produced run[i]
        seq.extend(run)
        res.tokens.extend(run)

        # ---- 4. CROP both caches back to the accepted length
        before_t, before_d = cache_seq_length(t_past), cache_seq_length(d_past)
        if not _skip_crop:
            t_past = crop_cache(t_past, len(seq) - 1)                      # real entries: last, d_1..d_m  (fewer if truncated)
            d_past = crop_cache(d_past, min(before_d, len(seq) - 1))
        verify_s = now(dev) - t0
        res.iterations.append(IterStats(
            k_proposed=k_eff, n_accepted=min(m, len(run)), n_emitted=len(run),
            draft_s=draft_s, verify_s=verify_s, target_cache_after=cache_seq_length(t_past),
            draft_cache_after=cache_seq_length(d_past), seq_len_after=len(seq),
            draft_cache_before_crop=before_d, target_cache_before_crop=before_t, truncated=truncated))
        if res.finish_reason == "eos":
            break
    return res
