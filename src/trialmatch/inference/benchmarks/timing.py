"""Timing benchmarking utilities: summary statistics (torch-free)."""
import statistics


def summarize(values: list[float]) -> dict:
    """mean, std (sample, 0 for n=1), min, max, median."""
    v = [float(x) for x in values]
    return {"n": len(v), "mean": statistics.fmean(v), "std": statistics.stdev(v) if len(v) > 1 else 0.0,
            "min": min(v), "max": max(v), "median": statistics.median(v)}
