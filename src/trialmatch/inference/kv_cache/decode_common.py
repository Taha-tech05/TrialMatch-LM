"""Shared pieces of the cached and uncached greedy decoders."""
import time
from dataclasses import dataclass, field


def sync(device) -> None:
    """torch.cuda.synchronize() when the tensors live on a CUDA device; no-op on CPU."""
    import torch
    if torch.cuda.is_available() and str(getattr(device, "type", device)).startswith("cuda"):
        torch.cuda.synchronize()


def now(device) -> float:
    sync(device)
    return time.perf_counter()


def normalize_eos(eos) -> set[int]:
    if eos is None:
        return set()
    return {int(eos)} if isinstance(eos, int) else {int(e) for e in eos}


def positions_from_mask(mask):
    """position_ids [B, T] for a (possibly left-padded) attention mask: 0,1,2,... over real tokens only
    (padding gets 0). With no padding this is simply arange(T)."""
    return (mask.long().cumsum(-1) - 1).clamp(min=0)


@dataclass
class DecodeResult:
    sequences: list[list[int]]            # generated ids per row; a finished row ends with its EOS token
    prefill_s: float = 0.0                # prompt forward + first argmax
    decode_s: float = 0.0                 # all remaining steps
    step_logits: list = field(default_factory=list)   # per generated token: [B, V] fp32 CPU (if requested)
    cache_lengths: list[int] = field(default_factory=list)  # cache length after prefill and after each decode step
    final_cache: object = None            # the explicit past_key_values after the last forward (cached only)

    @property
    def n_new_tokens(self) -> int:
        return sum(len(s) for s in self.sequences)

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s
