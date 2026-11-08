"""Per-iteration statistics of speculative decoding and their aggregation."""
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class IterStats:
    k_proposed: int            # candidates the draft proposed (k, or fewer near the end of the token budget)
    n_accepted: int            # longest prefix with draft token == target argmax
    n_emitted: int             # tokens appended to the output this iteration (accepted + correction/bonus, after EOS/budget truncation)
    draft_s: float             # draft work: catching its cache up + proposing k tokens
    verify_s: float            # ONE target forward over [last token + k candidates] + accept/reject + both crops
    target_cache_after: int    # target cache length after the crop
    draft_cache_after: int     # draft cache length after the crop
    seq_len_after: int         # prompt + all emitted tokens
    draft_cache_before_crop: int = 0    # (diagnostic) lengths just before the crop, show what the crop removed
    target_cache_before_crop: int = 0
    truncated: bool = False    # EOS or max_new_tokens cut this iteration short


@dataclass
class SpecResult:
    tokens: list[int]
    prefill_s: float = 0.0     # both prompt prefills + first target argmax
    iterations: list[IterStats] = field(default_factory=list)
    step_logits: list = field(default_factory=list)   # target logits [V] (fp32 CPU) behind every emitted token, if requested
    prompt_len: int = 0
    k: int = 0
    finish_reason: str = "length"      # "eos" | "length"

    @property
    def n_new_tokens(self) -> int:
        return len(self.tokens)

    @property
    def decode_s(self) -> float:
        return sum(i.draft_s + i.verify_s for i in self.iterations)

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s


def aggregate(res: SpecResult) -> dict:
    """Summary numbers for one generation.

    acceptance_rate               accepted / proposed candidates (0 if nothing was proposed)
    mean_accepted_per_iter        average accepted prefix length
    mean_tokens_per_verify        average tokens emitted per verify forward (between 1 and k+1)
    mean_tokens_per_target_forward  all generated tokens / (verify forwards + the one prefill forward)
    accepted_run_hist             {accepted run length: number of iterations}
    draft_s / verify_s / prefill_s / draft_share   wall time split (draft_share = draft / (draft + verify))
    """
    its = res.iterations
    proposed = sum(i.k_proposed for i in its)
    accepted = sum(i.n_accepted for i in its)
    emitted_in_iters = sum(i.n_emitted for i in its)
    draft_s, verify_s = sum(i.draft_s for i in its), sum(i.verify_s for i in its)
    return {
        "n_iterations": len(its), "n_new_tokens": len(res.tokens), "k": res.k,
        "tokens_proposed": proposed, "tokens_accepted": accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "mean_accepted_per_iter": accepted / len(its) if its else 0.0,
        "mean_tokens_per_verify": emitted_in_iters / len(its) if its else 0.0,
        "mean_tokens_per_target_forward": len(res.tokens) / (len(its) + 1),
        "accepted_run_hist": dict(sorted(Counter(i.n_accepted for i in its).items())),
        "prefill_s": res.prefill_s, "draft_s": draft_s, "verify_s": verify_s,
        "draft_share": draft_s / (draft_s + verify_s) if draft_s + verify_s else 0.0,
        "finish_reason": res.finish_reason,
    }
