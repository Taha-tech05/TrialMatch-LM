"""Attention micro-benchmark: reference (materialised) vs tiled online-softmax vs torch SDPA backends.

Inputs are random q [1, 32, S, 128], k/v [1, 8, S, 128] (GQA 32/8), causal. Each configuration is run with warm-up,
`repeats` timed runs (synchronize around each), and peak CUDA memory is reset per run. Memory is reported as the peak
minus the bytes already allocated before the run (i.e. the working memory of the call, excluding q/k/v themselves).
OOM is caught per configuration and recorded. Error is measured against an fp32 reference_attention baseline
computed from the same (possibly fp16-valued) inputs, in head chunks so the baseline itself does not OOM.
"""
import csv
import json
import time
from pathlib import Path

from trialmatch.inference.attention.reference import reference_attention
from trialmatch.inference.attention.sdpa_backends import BACKENDS, try_sdpa
from trialmatch.inference.attention.tiled import tiled_attention
from trialmatch.inference.benchmarks import memory
from trialmatch.inference.benchmarks.timing import summarize

TOL = {"float32": 1e-4, "float16": 5e-3}   # tiled-vs-fp32-baseline pass criterion (max abs error)


def environment(device, dtype) -> dict:
    import platform

    import torch

    from trialmatch.training.logging_utils import git_commit
    env = {"torch": torch.__version__, "dtype": str(dtype), "device": str(device), "python": platform.python_version(),
           "platform": platform.platform(), "git_commit": git_commit(), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
        env["cuda"] = torch.version.cuda
    return env


def make_inputs(seq, heads, kv_heads, head_dim, device, dtype, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    mk = lambda h: torch.randn(1, h, seq, head_dim, generator=g).to(device=device, dtype=dtype)  # noqa: E731
    return mk(heads), mk(kv_heads), mk(kv_heads)


def fp32_baseline(q, k, v, causal=True):
    """reference_attention in fp32, one KV-head group at a time (same result as all heads at once)."""
    import torch
    n_rep = q.shape[1] // k.shape[1]
    outs = []
    for g in range(k.shape[1]):
        outs.append(reference_attention(q[:, g * n_rep:(g + 1) * n_rep].float(), k[:, g:g + 1].float(),
                                        v[:, g:g + 1].float(), causal=causal))
    return torch.cat(outs, dim=1)


class Unavailable(RuntimeError):
    """SDPA backend not usable on this machine (recorded, not fatal)."""


def is_oom(e: BaseException) -> bool:
    return isinstance(e, MemoryError) or "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def implementations(block_sizes):
    """name -> fn(q, k, v, causal). SDPA entries raise on failure so it is recorded as an error."""
    impls = {"reference": lambda q, k, v, c: reference_attention(q, k, v, causal=c)}
    for b in block_sizes:
        impls[f"tiled_{b}"] = (lambda b: lambda q, k, v, c: tiled_attention(q, k, v, causal=c, block_q=b, block_k=b))(b)
    for name in BACKENDS:
        def fn(q, k, v, c, name=name):
            out, err = try_sdpa(q, k, v, causal=c, backend=name)
            if err:
                raise Unavailable(err)
            return out
        impls[f"sdpa_{name}"] = fn
    return impls


def bench_one(fn, q, k, v, causal, device, warmup, repeats, baseline):
    import torch
    cuda = str(device).startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    rec = {"status": "ok"}
    try:
        out = None
        for _ in range(warmup):
            out = fn(q, k, v, causal)
        sync()
        times, peaks = [], []
        for _ in range(repeats):
            del out
            out = None
            start_alloc = memory.begin_run(device)
            sync()
            t0 = time.perf_counter()
            out = fn(q, k, v, causal)
            sync()
            times.append(time.perf_counter() - t0)
            peak = memory.end_run(device)
            if peak is not None:
                peaks.append(peak - start_alloc)
        rec["time_s"] = summarize(times)
        if peaks:
            rec["peak_mem_bytes"] = max(peaks)
        d = (out.float() - baseline.to(out.device)).abs()
        rec["max_abs_err"], rec["mean_abs_err"] = d.max().item(), d.mean().item()
        del out, d
    except Exception as e:  # noqa: BLE001
        msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
        rec = {"status": "oom" if is_oom(e) else ("unavailable" if isinstance(e, Unavailable) else "error"), "error": msg}
    finally:
        if cuda:
            torch.cuda.empty_cache()
    return rec


def run(seq_lens, block_sizes, heads, kv_heads, head_dim, device, dtype, warmup, repeats, causal=True, log=print):
    import torch
    impls = implementations(block_sizes)
    tol = TOL[str(dtype).replace("torch.", "")]
    results = []
    for s in seq_lens:
        log(f"--- seq_len={s}")
        try:
            q, k, v = make_inputs(s, heads, kv_heads, head_dim, device, dtype)
            baseline = fp32_baseline(q, k, v, causal)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            log(f"    inputs/baseline OOM at seq_len={s}; skipped")
            results.append({"seq_len": s, "impl": "(all)", "status": "oom", "error": "inputs or baseline"})
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            continue
        for name, fn in impls.items():
            r = bench_one(fn, q, k, v, causal, device, warmup, repeats, baseline)
            r.update(seq_len=s, impl=name, tolerance=tol)
            r["within_tolerance"] = (r["max_abs_err"] <= tol) if r["status"] == "ok" else None
            results.append(r)
            if r["status"] == "ok":
                mem = r.get("peak_mem_bytes")
                log(f"    {name:<14} {r['time_s']['mean'] * 1e3:9.2f} ms  "
                    f"{'' if mem is None else f'{mem / 2 ** 20:9.1f} MiB  '}max_err={r['max_abs_err']:.2e}")
            else:
                log(f"    {name:<14} {r['status']}: {r['error']}")
        del q, k, v, baseline
    return results


def mismatches(results):
    """tiled/reference runs whose error vs the fp32 baseline exceeds tolerance."""
    return [r for r in results if r["status"] == "ok" and r["impl"].startswith(("tiled", "reference"))
            and r["within_tolerance"] is False]


def write_results(out_dir: Path, env: dict, args: dict, results: list[dict], stamp: str | None = None):
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, cp, pp = (out_dir / f"attention_{stamp}{e}" for e in (".json", ".csv", ".png"))
    jp.write_text(json.dumps({"environment": env, "args": args, "mismatches": len(mismatches(results)),
                              "results": results}, indent=2), encoding="utf-8")
    cols = ["seq_len", "impl", "status", "time_mean_s", "time_std_s", "time_min_s", "peak_mem_mib", "max_abs_err",
            "mean_abs_err", "within_tolerance", "error"]
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        for r in results:
            t = r.get("time_s", {})
            w.writerow({"seq_len": r["seq_len"], "impl": r["impl"], "status": r["status"],
                        "time_mean_s": t.get("mean", ""), "time_std_s": t.get("std", ""), "time_min_s": t.get("min", ""),
                        "peak_mem_mib": r["peak_mem_bytes"] / 2 ** 20 if "peak_mem_bytes" in r else "",
                        "max_abs_err": r.get("max_abs_err", ""), "mean_abs_err": r.get("mean_abs_err", ""),
                        "within_tolerance": r.get("within_tolerance", ""), "error": r.get("error", "")})
    plot(results, pp, env)
    return jp, cp, pp


def plot(results, path: Path, env: dict) -> None:
    """time and peak memory vs sequence length (log-log); OOM / unavailable configurations are simply absent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in results if r["status"] == "ok"]
    if not ok:
        return
    impls = list(dict.fromkeys(r["impl"] for r in ok))
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, name in enumerate(impls):
        rows = sorted((r for r in ok if r["impl"] == name), key=lambda r: r["seq_len"])
        x = [r["seq_len"] for r in rows]
        style = dict(color=cmap(i % 10), marker="o", ms=4, lw=1.8, label=name,
                     ls="--" if name.startswith("sdpa") else "-")
        axes[0].plot(x, [r["time_s"]["mean"] * 1e3 for r in rows], **style)
        if all("peak_mem_bytes" in r for r in rows):
            axes[1].plot(x, [r["peak_mem_bytes"] / 2 ** 20 for r in rows], **style)
    for ax, yl in zip(axes, ("time per call (ms)", "peak extra memory (MiB)")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    if not axes[1].lines:
        axes[1].text(0.5, 0.5, "no CUDA: peak memory not measured", ha="center", transform=axes[1].transAxes)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Causal attention, GQA  ({env.get('gpu', env['device'])}, {env['dtype']})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
