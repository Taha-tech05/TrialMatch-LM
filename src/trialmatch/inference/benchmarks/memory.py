"""Memory benchmarking utilities: peak GPU memory per run (all no-ops on CPU)."""
import gc


def begin_run(device) -> int:
    """Free cached blocks, reset the peak counter, return the allocated bytes at start (weights etc.)."""
    import torch
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()
    return 0


def end_run(device) -> int | None:
    """Peak allocated bytes since begin_run (None on CPU)."""
    import torch
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()
    return None
