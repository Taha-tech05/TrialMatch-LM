"""KV-cache benchmark: cached vs uncached greedy decoding on the SAME model and inputs.

Each (input_len, output_len) cell: warm-up runs, then N timed repeats per decoder. Prefill = prompt forward
(+ first argmax); decode = all later steps (for the uncached decoder: later full-sequence forwards).
Tokens/s = generated tokens / total time. Token equality cached vs uncached is checked for every repeat.
Synthetic inputs run a fixed number of new tokens (EOS disabled); val prompts stop on <|im_end|>.
"""
import csv
import json
import time
from pathlib import Path

from trialmatch.inference.benchmarks import memory
from trialmatch.inference.benchmarks.timing import summarize
from trialmatch.inference.kv_cache.cached_decode import cached_greedy_decode
from trialmatch.inference.kv_cache.compare import compare_decoders
from trialmatch.inference.kv_cache.reference_decode import reference_greedy_decode

DECODERS = {"cached": cached_greedy_decode, "uncached": reference_greedy_decode}
METRICS = ("prefill_s", "decode_s", "total_s", "tokens_per_s", "decode_ms_per_token", "peak_mem_bytes",
           "peak_mem_delta_bytes")


def environment(model, device, dtype) -> dict:
    import platform

    import torch
    import transformers

    from trialmatch.training.logging_utils import git_commit
    env = {"torch": torch.__version__, "transformers": transformers.__version__, "dtype": str(dtype),
           "device": str(device), "python": platform.python_version(), "platform": platform.platform(),
           "git_commit": git_commit(), "cuda_available": torch.cuda.is_available(),
           "model_class": type(model).__name__, "n_params": sum(p.numel() for p in model.parameters())}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_total_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    return env


def synthetic_input(length: int, vocab_size: int, seed: int, base_ids: list[int] | None = None):
    """[1, length] ids: a real val prompt's token ids tiled/cut to length when given, else seeded random ids
    (never ids 0-2, which tests treat as special). Timing depends on shape only."""
    import torch
    if base_ids:
        reps = -(-length // len(base_ids))
        return torch.tensor([(base_ids * reps)[:length]])
    g = torch.Generator().manual_seed(seed + length)
    return torch.randint(3, vocab_size, (1, length), generator=g)


def val_inputs(tok, n: int):
    """First n VAL prompts (chat template, thinking off) as token-id tensors. The test split is never read."""
    from trialmatch.evaluation.generate_eval import build_items
    from trialmatch.model.chat import chat_prompt_text
    items = build_items("val", allow_test=False, limit=n)
    out = []
    for it in items:
        enc = tok(chat_prompt_text(tok, it["prompt"]), return_tensors="pt", add_special_tokens=False)
        out.append((it["annotation_id"], enc["input_ids"]))
    return out


def run_once(fn, model, ids, n_new, eos, device):
    base = memory.begin_run(device)
    res = fn(model, ids, max_new_tokens=n_new, eos_token_id=eos)
    peak = memory.end_run(device)
    n = res.n_new_tokens
    total = res.total_s
    rec = {"prefill_s": res.prefill_s, "decode_s": res.decode_s, "total_s": total, "n_new_tokens": n,
           "tokens_per_s": n / total if total else 0.0,
           "decode_ms_per_token": 1000 * res.decode_s / (n - 1) if n > 1 else 0.0,
           "peak_mem_bytes": peak, "peak_mem_delta_bytes": None if peak is None else peak - base}
    return rec, res.sequences


def bench_case(model, ids, n_new, eos, device, warmup, repeats, compare_on_mismatch=True) -> dict:
    """One input, both decoders. Returns per-decoder stats plus the equality verdict."""
    import torch
    inp = ids.to(device)
    for fn in DECODERS.values():
        for _ in range(warmup):
            fn(model, inp, max_new_tokens=min(n_new, 4), eos_token_id=eos)
    runs = {k: [] for k in DECODERS}
    toks = {k: [] for k in DECODERS}
    for _ in range(repeats):
        for k, fn in DECODERS.items():             # interleaved so drift/thermal effects hit both equally
            rec, seqs = run_once(fn, model, inp, n_new, eos, device)
            runs[k].append(rec)
            toks[k].append(seqs)
    equal = all(c == u for c, u in zip(toks["cached"], toks["uncached"]))
    case = {"input_len": int(ids.shape[1]), "target_new_tokens": n_new, "tokens_equal": equal,
            "n_new_tokens": runs["cached"][0]["n_new_tokens"], "repeats": repeats, "decoders": {}}
    if not equal and compare_on_mismatch:
        case["mismatch_report"] = compare_decoders(model, inp, max_new_tokens=n_new, eos_token_id=eos).summary()
    for k in DECODERS:
        case["decoders"][k] = {"raw": runs[k], **{m: summarize([r[m] for r in runs[k]]) for m in METRICS
                                                  if runs[k][0][m] is not None}}
    return case


def run_grid(model, device, vocab_size, input_lengths, output_lengths, warmup, repeats, seed, base_ids=None):
    cases = []
    for L in input_lengths:
        ids = synthetic_input(L, vocab_size, seed, base_ids)
        for n in output_lengths:
            c = bench_case(model, ids, n, None, device, warmup, repeats)
            c["source"] = "synthetic"
            cases.append(c)
            d = c["decoders"]
            print(f"[synthetic in={L:5d} out={n:4d}] cached {d['cached']['total_s']['median']:.3f}s "
                  f"({d['cached']['tokens_per_s']['median']:.1f} tok/s) | uncached {d['uncached']['total_s']['median']:.3f}s "
                  f"({d['uncached']['tokens_per_s']['median']:.1f} tok/s) | equal={c['tokens_equal']}", flush=True)
    return cases


def run_val(model, tok, device, n_prompts, max_new, warmup, repeats):
    from trialmatch.model.chat import end_token_id
    eos = end_token_id(tok)
    cases = []
    for aid, ids in val_inputs(tok, n_prompts):
        c = bench_case(model, ids, max_new, eos, device, warmup, repeats)
        c.update(source="val", annotation_id=aid)
        cases.append(c)
        print(f"[val id={aid} in={c['input_len']} new={c['n_new_tokens']}] equal={c['tokens_equal']} "
              f"cached {c['decoders']['cached']['total_s']['median']:.3f}s uncached "
              f"{c['decoders']['uncached']['total_s']['median']:.3f}s", flush=True)
    return cases


def csv_rows(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        for k, d in c["decoders"].items():
            row = {"source": c["source"], "annotation_id": c.get("annotation_id", ""), "input_len": c["input_len"],
                   "target_new_tokens": c["target_new_tokens"], "n_new_tokens": c["n_new_tokens"], "decoder": k,
                   "repeats": c["repeats"], "tokens_equal": c["tokens_equal"]}
            for m in METRICS:
                for stat in ("mean", "std", "min", "max", "median"):
                    row[f"{m}_{stat}"] = d[m][stat] if m in d else ""
            rows.append(row)
    return rows


def write_results(out_dir: Path, env: dict, args: dict, cases: list[dict], stamp: str | None = None):
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, cp, pp = (out_dir / f"kv_cache_{stamp}{e}" for e in (".json", ".csv", ".png"))
    jp.write_text(json.dumps({"environment": env, "args": args, "all_tokens_equal": all(c["tokens_equal"] for c in cases),
                              "cases": cases}, indent=2), encoding="utf-8")
    rows = csv_rows(cases)
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(cases, pp, env)
    return jp, cp, pp


def plot(cases: list[dict], path: Path, env: dict) -> None:
    """tokens/s and decode time vs output length, cached vs uncached, one column per input length (synthetic)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    syn = [c for c in cases if c["source"] == "synthetic"]
    if not syn:
        return
    lens = sorted({c["input_len"] for c in syn})
    colors = {"cached": "#1f6fb2", "uncached": "#d95f1f"}
    fig, axes = plt.subplots(2, len(lens), figsize=(3.6 * len(lens), 6.2), squeeze=False, sharex=True)
    for j, L in enumerate(lens):
        sub = sorted((c for c in syn if c["input_len"] == L), key=lambda c: c["target_new_tokens"])
        for i, (metric, label) in enumerate((("tokens_per_s", "tokens / s"), ("decode_s", "decode time (s)"))):
            ax = axes[i][j]
            for k in ("cached", "uncached"):
                x = [c["target_new_tokens"] for c in sub]
                m = [c["decoders"][k][metric]["mean"] for c in sub]
                s = [c["decoders"][k][metric]["std"] for c in sub]
                ax.errorbar(x, m, yerr=s, color=colors[k], marker="o", lw=2, ms=5, capsize=3, label=k)
            ax.set_xticks([c["target_new_tokens"] for c in sub])
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(f"input {L} tokens", fontsize=10)
            if i == 1:
                ax.set_xlabel("output length (tokens)")
            if j == 0:
                ax.set_ylabel(label)
    axes[0][0].legend(frameon=False)
    fig.suptitle(f"KV-cache vs uncached greedy decoding  ({env.get('gpu', env['device'])}, {env['dtype']}, mean +/- std)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
