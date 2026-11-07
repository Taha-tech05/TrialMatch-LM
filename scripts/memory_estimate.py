"""ESTIMATE weight + KV-cache memory from config values. Nothing is measured; no weights are loaded.

Reads results/metrics/model_facts.json (run scripts/model_facts.py first).
KV cache = 2 x layers x kv_heads x head_dim x seq_len x bytes (fp16 = 2 bytes), times batch.
Excluded from every figure: activations, CUDA context, allocator fragmentation, quantisation scales.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trialmatch.model.measure import BYTES_PER_PARAM, GIB, kv_cache_bytes, weights_bytes
from trialmatch.utils.common import resolve

SEQ_LENS, BATCHES = (512, 1024, 2048, 4096), (1, 4)
T4_GIB = 16.0
PAIRS = [("Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B"), ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"),
         ("meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct")]


def fit(gib: float) -> str:
    return "fits" if gib <= 0.8 * T4_GIB else ("TIGHT" if gib <= T4_GIB else "NO")


def main():
    facts = {r["repo"]: r for r in json.loads(resolve("results/metrics/model_facts.json").read_text(encoding="utf-8"))
             if r["status"] == "ok"}
    out = {"note": "ESTIMATES only", "t4_gib": T4_GIB, "weights_gib": {}, "kv_gib": {}, "combined": []}
    md = ["# Memory estimates (ESTIMATES, not measurements)", "",
          "Weights = params x bytes/param (fp16 2, int8 1, 4-bit 0.5; quantisation scales ignored). "
          "KV cache = 2 x layers x kv_heads x head_dim x seq_len x bytes (fp16 KV, 2 bytes) x batch. "
          "Activations, CUDA context and fragmentation are NOT included (a T4 has 16 GB nominal, ~15 GB usable). "
          "Fit label: `fits` <= 80% of 16 GiB, `TIGHT` <= 16 GiB, `NO` above. GiB = 2^30 bytes.", "",
          "## Weights", "", "| model | params (B) | fp16 GiB | int8 GiB | 4-bit GiB |", "|---|---|---|---|---|"]
    for repo, f in facts.items():
        w = {p: weights_bytes(f["params"], p) / GIB for p in BYTES_PER_PARAM}
        out["weights_gib"][repo] = w
        md.append(f"| {repo} | {f['params'] / 1e9:.2f} | {w['fp16']:.2f} | {w['int8']:.2f} | {w['4bit']:.2f} |")
    md += ["", "## KV cache (fp16)", "", "| model | batch | " + " | ".join(f"seq {s}" for s in SEQ_LENS) + " |",
           "|---|---|" + "---|" * len(SEQ_LENS)]
    for repo, f in facts.items():
        for b in BATCHES:
            v = [kv_cache_bytes(f["layers"], f["kv_heads"], f["head_dim"], s, b) / GIB for s in SEQ_LENS]
            out["kv_gib"][f"{repo}|b{b}"] = dict(zip(map(str, SEQ_LENS), v))
            md.append(f"| {repo} | {b} | " + " | ".join(f"{x:.3f} GiB" for x in v) + " |")
    for t, d in PAIRS:
        if t not in facts or d not in facts:
            md += ["", f"## Target + draft: {t} + {d}", "", "Skipped (model config inaccessible)."]
            continue
        md += ["", f"## Target + draft together: {t} + {d}", "",
               "Total = weights(target) + weights(draft) + KV(target) + KV(draft), both KV caches fp16, same seq_len.", "",
               "| weights precision | batch | seq | weights GiB | KV GiB | total GiB | % of 16 GiB | fit |", "|---|---|---|---|---|---|---|---|"]
        for prec in BYTES_PER_PARAM:
            w = (weights_bytes(facts[t]["params"], prec) + weights_bytes(facts[d]["params"], prec)) / GIB
            for b in BATCHES:
                for s in SEQ_LENS:
                    kv = sum(kv_cache_bytes(facts[m]["layers"], facts[m]["kv_heads"], facts[m]["head_dim"], s, b)
                             for m in (t, d)) / GIB
                    tot = w + kv
                    out["combined"].append({"target": t, "draft": d, "precision": prec, "batch": b, "seq": s,
                                            "weights_gib": w, "kv_gib": kv, "total_gib": tot, "fit": fit(tot)})
                    md.append(f"| {prec} | {b} | {s} | {w:.2f} | {kv:.3f} | {tot:.2f} | {100 * tot / T4_GIB:.0f}% | {fit(tot)} |")
    resolve("results/metrics/memory_estimate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    text = "\n".join(md) + "\n"
    resolve("docs/memory_estimate.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
