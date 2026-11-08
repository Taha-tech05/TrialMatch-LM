"""Speculative-decoding benchmark: plain cached greedy (Task 3 decoder) vs our speculative decoder, same prompts.

One CONFIGURATION = (source, prompt, output length, k) where k = 0 means the plain cached decoder (the baseline of
that prompt). After every configuration the whole result file is rewritten (atomically) as
speculative_<stamp>.json/.csv, so a Colab disconnect loses at most one configuration; `--resume` continues the
latest file and skips configurations already in it (a row is identified by `key`).

Per row: warm-up runs, `repeats` timed runs (cuda.synchronize inside the decoders' timers, peak memory reset per
run - both models are loaded, so the peak covers target + draft + both caches). Token equality vs the plain
decoder's tokens is recorded per row; a mismatch records the first differing step and the target-logit gap there.
"""
import csv
import json
import os
import time
from pathlib import Path

from trialmatch.inference.benchmarks import kv_cache_bench as kb
from trialmatch.inference.benchmarks import memory
from trialmatch.inference.benchmarks.timing import summarize
from trialmatch.inference.kv_cache.cached_decode import cached_greedy_decode
from trialmatch.inference.speculative.speculative_decode import speculative_greedy_decode
from trialmatch.inference.speculative.stats import aggregate

CSV_FIELDS = ["key", "source", "annotation_id", "input_len", "target_new_tokens", "n_new_tokens", "decoder", "k",
              "draft", "repeats", "tokens_equal", "first_diff_step", "max_logit_gap_at_diff", "total_s_median",
              "total_s_std", "prefill_s_median", "decode_s_median", "tokens_per_s_median", "speedup_vs_plain",
              "acceptance_rate", "mean_accepted_per_iter", "mean_tokens_per_verify", "mean_tokens_per_target_forward",
              "n_iterations", "draft_s_median", "verify_s_median", "draft_share", "peak_mem_bytes_median",
              "accepted_run_hist"]


def environment(target, draft, device, dtype, draft_label: str) -> dict:
    env = kb.environment(target, device, dtype)
    env.update(draft=draft_label, draft_class=type(draft).__name__, n_params_draft=sum(p.numel() for p in draft.parameters()),
               target_layers=target.config.num_hidden_layers, draft_layers=draft.config.num_hidden_layers)
    return env


def make_key(source, aid, input_len, n_new, k, draft) -> str:
    return f"{source}|{aid or '-'}|in{input_len}|out{n_new}|{'plain' if k == 0 else f'k{k}'}|{draft if k else '-'}"


def plan(inputs, outputs, ks, draft_label, val_cases=(), val_max_new=64) -> list[dict]:
    """Ordered configurations: for every prompt the plain baseline first, then each k (speedup needs the baseline)."""
    cfgs = []
    for L in inputs:
        for n in outputs:
            for k in (0, *ks):
                cfgs.append({"source": "synthetic", "aid": None, "input_len": L, "n_new": n, "k": k})
    for aid, ids in val_cases:
        for k in (0, *ks):
            cfgs.append({"source": "val", "aid": aid, "input_len": int(ids.shape[1]), "n_new": val_max_new, "k": k})
    for c in cfgs:
        c["draft"] = draft_label
    return cfgs


def _timed(fn, device):
    base = memory.begin_run(device)
    res = fn()
    peak = memory.end_run(device)
    return res, peak, (None if peak is None else peak - base)


def first_diff(a: list[int], b: list[int]):
    for i in range(max(len(a), len(b))):
        if i >= len(a) or i >= len(b) or a[i] != b[i]:
            return i
    return None


def run_config(cfg, target, draft, inp, eos, warmup, repeats, device, plain_tokens=None, plain_row=None) -> dict:
    """Measure one configuration. inp: [1, T] ids on `device`. For k>0, plain_tokens/plain_row come from the baseline."""
    k, n_new = cfg["k"], cfg["n_new"]
    if k == 0:
        fn = lambda: cached_greedy_decode(target, inp, max_new_tokens=n_new, eos_token_id=eos)   # noqa: E731
    else:
        fn = lambda: speculative_greedy_decode(target, draft, inp, max_new_tokens=n_new, k=k, eos_token_id=eos)  # noqa: E731
    for _ in range(warmup):
        (cached_greedy_decode(target, inp, max_new_tokens=min(n_new, 6), eos_token_id=eos) if k == 0 else
         speculative_greedy_decode(target, draft, inp, max_new_tokens=min(n_new, 6), k=k, eos_token_id=eos))
    runs, toks, aggs = [], [], []
    for _ in range(repeats):
        res, peak, delta = _timed(fn, device)
        toks.append(res.sequences[0] if k == 0 else res.tokens)
        n = len(toks[-1])
        rec = {"total_s": res.total_s, "prefill_s": res.prefill_s, "decode_s": res.decode_s,
               "tokens_per_s": n / res.total_s if res.total_s else 0.0, "peak": peak}
        if k:
            a = aggregate(res)
            rec.update(draft_s=a["draft_s"], verify_s=a["verify_s"])
            aggs.append(a)
        runs.append(rec)
    tokens = toks[0]
    row = {"key": make_key(cfg["source"], cfg["aid"], cfg["input_len"], cfg["n_new"], k, cfg["draft"]),
           "source": cfg["source"], "annotation_id": cfg["aid"] or "", "input_len": cfg["input_len"],
           "target_new_tokens": n_new, "n_new_tokens": len(tokens), "decoder": "plain" if k == 0 else "speculative",
           "k": k, "draft": cfg["draft"] if k else "", "repeats": repeats,
           "repeat_tokens_identical": all(t == tokens for t in toks)}
    for m in ("total_s", "prefill_s", "decode_s", "tokens_per_s"):
        s = summarize([r[m] for r in runs])
        row[f"{m}_median"], row[f"{m}_std"], row[f"{m}_mean"] = s["median"], s["std"], s["mean"]
    if runs[0]["peak"] is not None:
        row["peak_mem_bytes_median"] = summarize([r["peak"] for r in runs])["median"]
    if k == 0:
        row.update(tokens=tokens, tokens_equal=True, speedup_vs_plain=1.0)
    else:
        for m in ("draft_s", "verify_s"):
            row[f"{m}_median"] = summarize([r[m] for r in runs])["median"]
        a = aggs[0]
        row.update({x: a[x] for x in ("acceptance_rate", "mean_accepted_per_iter", "mean_tokens_per_verify",
                                      "mean_tokens_per_target_forward", "n_iterations", "accepted_run_hist",
                                      "draft_share", "finish_reason")})
        equal = tokens == plain_tokens
        row["tokens_equal"] = equal
        row["speedup_vs_plain"] = plain_row["total_s_median"] / row["total_s_median"] if plain_row else None
        if not equal:
            row.update(mismatch_report=mismatch_report(target, draft, inp, plain_tokens, n_new, k, eos))
            row["first_diff_step"] = row["mismatch_report"]["first_diff_step"]
            row["max_logit_gap_at_diff"] = row["mismatch_report"]["max_logit_gap_at_diff"]
    return row


def mismatch_report(target, draft, inp, plain_tokens, n_new, k, eos) -> dict:
    """Re-run both decoders with logits recorded; report the first differing step and the target-logit gap there
    (a gap near fp16 rounding means a near-tie flipped by different matmul shapes, not a logic bug)."""
    ref = cached_greedy_decode(target, inp, max_new_tokens=n_new, eos_token_id=eos, return_logits=True)
    sp = speculative_greedy_decode(target, draft, inp, max_new_tokens=n_new, k=k, eos_token_id=eos, return_logits=True)
    step = first_diff(ref.sequences[0], sp.tokens)
    gap = None
    if step is not None and step < min(len(ref.step_logits), len(sp.step_logits)):
        gap = float((ref.step_logits[step][0] - sp.step_logits[step]).abs().max())
        top2 = ref.step_logits[step][0].topk(2).values
        margin = float(top2[0] - top2[1])
    else:
        margin = None
    return {"first_diff_step": step, "max_logit_gap_at_diff": gap, "plain_top1_top2_margin_at_diff": margin,
            "plain_token": None if step is None or step >= len(ref.sequences[0]) else ref.sequences[0][step],
            "spec_token": None if step is None or step >= len(sp.tokens) else sp.tokens[step]}


# ---------------------------------------------------------------- incremental saving
def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_results(out_dir: Path, stamp: str, env: dict, args: dict, rows: list[dict]) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, cp = out_dir / f"speculative_{stamp}.json", out_dir / f"speculative_{stamp}.csv"
    write_atomic(jp, json.dumps({"environment": env, "args": args,
                                 "all_tokens_equal": all(r["tokens_equal"] for r in rows), "rows": rows}, indent=2))
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({**r, "accepted_run_hist": json.dumps(r.get("accepted_run_hist", ""))})
    write_atomic(cp, buf.getvalue())
    return jp, cp


def latest_results(out_dir: Path):
    """(stamp, loaded json) of the newest speculative_*.json, or (None, None)."""
    files = sorted(out_dir.glob("speculative_*.json"))
    if not files:
        return None, None
    f = files[-1]
    return f.stem.removeprefix("speculative_"), json.loads(f.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- driver
def run_benchmark(target, draft, cfgs, inputs_for, eos_for, out_dir, env, args, device, warmup, repeats,
                  resume=False, max_configs=None, log=print) -> list[dict]:
    """Run configurations in order, saving after each. inputs_for(cfg) -> [1, T] cpu ids; eos_for(cfg) -> eos or None.
    Returns all rows (resumed + new)."""
    stamp, prev = (latest_results(out_dir) if resume else (None, None))
    rows = list(prev["rows"]) if prev else []
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    done = {r["key"]: r for r in rows}
    if prev:
        log(f"resuming {stamp}: {len(done)} configurations already done")
    ran = 0
    for cfg in cfgs:
        key = make_key(cfg["source"], cfg["aid"], cfg["input_len"], cfg["n_new"], cfg["k"], cfg["draft"])
        if key in done:
            continue
        if max_configs is not None and ran >= max_configs:
            log(f"--max-configs {max_configs} reached; stopping")
            break
        inp = inputs_for(cfg).to(device)
        base_key = make_key(cfg["source"], cfg["aid"], cfg["input_len"], cfg["n_new"], 0, cfg["draft"])
        base = done.get(base_key)
        if cfg["k"] and base is None:
            raise RuntimeError(f"baseline row missing for {key}")
        row = run_config(cfg, target, draft, inp, eos_for(cfg), warmup, repeats, device,
                         base["tokens"] if base else None, base)
        rows.append(row)
        done[key] = row
        ran += 1
        save_results(out_dir, stamp, env, args, rows)                    # after EVERY configuration
        log(describe(row))
    args["stamp"] = stamp
    return rows


def describe(r) -> str:
    head = f"[{r['source']} {r['annotation_id'] or ''} in={r['input_len']} out={r['n_new_tokens']}]".replace("  ", " ")
    if r["decoder"] == "plain":
        return f"{head} plain  {r['total_s_median']:.3f}s {r['tokens_per_s_median']:.1f} tok/s"
    return (f"{head} k={r['k']} {r['total_s_median']:.3f}s {r['tokens_per_s_median']:.1f} tok/s "
            f"speedup {r['speedup_vs_plain'] or 0:.2f}x accept {r['acceptance_rate']:.2f} "
            f"tok/verify {r['mean_tokens_per_verify']:.2f} draft {100 * r['draft_share']:.0f}% equal={r['tokens_equal']}")


# ---------------------------------------------------------------- plot
def plot(rows: list[dict], path: Path, env: dict) -> None:
    """Speedup vs k and acceptance vs k; one line per (source, output length) / val aggregate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = [r for r in rows if r["decoder"] == "speculative"]
    if not spec:
        return
    groups: dict[str, dict[int, list]] = {}
    for r in spec:
        name = f"synthetic out={r['target_new_tokens']}" if r["source"] == "synthetic" else "val prompts"
        groups.setdefault(name, {}).setdefault(r["k"], []).append(r)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, byk in sorted(groups.items()):
        ks = sorted(byk)
        for ax, field in zip(axes, ("speedup_vs_plain", "acceptance_rate")):
            ys = [sum(x[field] or 0 for x in byk[k]) / len(byk[k]) for k in ks]
            ax.plot(ks, ys, marker="o", lw=2, label=name)
    axes[0].axhline(1.0, color="gray", ls="--", lw=1)
    for ax, t in zip(axes, ("speedup vs plain cached greedy (>1 = faster)", "acceptance rate (accepted / proposed)")):
        ax.set_xlabel("k (draft tokens per iteration)")
        ax.set_title(t, fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({r["k"] for r in spec}))
        ax.set_xticklabels(sorted({r["k"] for r in spec}))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0, 1.05)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Speculative decoding ({env.get('gpu', env['device'])}, {env['dtype']}, draft {env.get('draft')})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
