"""Mismatch investigator: run both decoders with logits recorded and report where they first disagree."""
from dataclasses import dataclass

from trialmatch.inference.kv_cache.cached_decode import cached_greedy_decode
from trialmatch.inference.kv_cache.reference_decode import reference_greedy_decode


@dataclass
class Comparison:
    equal: bool
    first_diff_step: int | None           # generated-token index of the first difference (row 0 for batches: first row that differs)
    first_diff_row: int | None
    max_logit_diff_at_first_diff: float | None
    max_logit_diff_per_step: list[float]  # over the steps both runs share
    cached_tokens: list[list[int]]
    reference_tokens: list[list[int]]

    def summary(self) -> str:
        if self.equal:
            return f"tokens identical; max logit diff over all steps = {max(self.max_logit_diff_per_step, default=0):.3e}"
        return (f"FIRST MISMATCH at step {self.first_diff_step} (row {self.first_diff_row}); max |logit diff| at that "
                f"step = {self.max_logit_diff_at_first_diff:.3e}")


def compare_decoders(model, input_ids, attention_mask=None, max_new_tokens=32, eos_token_id=None,
                     pad_token_id=None, **cached_kw) -> Comparison:
    ref = reference_greedy_decode(model, input_ids, attention_mask, max_new_tokens, eos_token_id, pad_token_id, True)
    cac = cached_greedy_decode(model, input_ids, attention_mask, max_new_tokens, eos_token_id, pad_token_id, True,
                               **cached_kw)
    n = min(len(ref.step_logits), len(cac.step_logits))
    per_step = [float((ref.step_logits[i] - cac.step_logits[i]).abs().max()) for i in range(n)]
    first = row = None
    for b, (r, c) in enumerate(zip(ref.sequences, cac.sequences)):
        for i in range(max(len(r), len(c))):
            if i >= len(r) or i >= len(c) or r[i] != c[i]:
                if first is None or i < first:
                    first, row = i, b
                break
    return Comparison(equal=ref.sequences == cac.sequences, first_diff_step=first, first_diff_row=row,
                      max_logit_diff_at_first_diff=per_step[first] if first is not None and first < n else None,
                      max_logit_diff_per_step=per_step, cached_tokens=cac.sequences, reference_tokens=ref.sequences)
