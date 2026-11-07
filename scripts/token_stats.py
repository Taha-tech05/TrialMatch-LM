"""Token-length statistics per candidate tokenizer on TRAIN and VAL rows only (never test).

Prompt = model-agnostic prompt from trialmatch.model.prompt wrapped in the model's chat template
(Qwen3: enable_thinking=False, generation prompt appended). Target = JSON target tokens + 1 end-of-turn token.
Rows listed under known_skipped_rows (annotation 883) are excluded.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from trialmatch.model.measure import CANDIDATES, describe
from trialmatch.model.prompt import example_from_row, skip_reason
from trialmatch.utils.common import load_config, resolve

THRESHOLDS = (1024, 2048, 4096)
SPLITS = ("train", "val")  # test is deliberately never opened


def load_examples() -> list[dict]:
    cfg = load_config()
    c = cfg["columns"]
    skipped = {s["annotation_id"] for s in cfg["known_skipped_rows"]}
    out = []
    for split in SPLITS:
        df = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / f"{split}.parquet")
        for _, r in df.iterrows():
            if int(r[c["id"]]) in skipped or skip_reason(r[c["criterion_text"]]):
                continue
            ex = example_from_row(r.to_dict(), cfg)
            out.append({"id": int(r[c["id"]]), "split": split, "prompt": ex.prompt, "target": ex.target})
    return out


def count_tokens(tok, prompt: str, target: str, is_qwen: bool) -> tuple[int, int]:
    """Return (templated prompt tokens, target tokens incl. one end-of-turn token)."""
    kw = {"enable_thinking": False} if is_qwen else {}
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, **kw) if tok.chat_template else prompt
    return len(tok(text, add_special_tokens=False)["input_ids"]), len(tok(target, add_special_tokens=False)["input_ids"]) + 1


def stats_for_tokenizer(tok, examples: list[dict], is_qwen: bool) -> dict:
    rows = []
    for ex in examples:
        p, t = count_tokens(tok, ex["prompt"], ex["target"], is_qwen)
        rows.append((ex["id"], ex["split"], p, t, p + t))
    tot = [r[4] for r in rows]
    longest = max(rows, key=lambda r: r[4])
    return {
        "n_rows": len(rows),
        "prompt": describe([r[2] for r in rows]), "target": describe([r[3] for r in rows]),
        "prompt_plus_target": describe(tot),
        "rows_exceeding": {str(th): sum(x > th for x in tot) for th in THRESHOLDS},
        "longest_example": {"annotation_id": longest[0], "split": longest[1], "prompt": longest[2],
                            "target": longest[3], "total": longest[4]},
    }


def main():
    from transformers import AutoTokenizer
    examples = load_examples()
    res = {"splits": list(SPLITS), "n_rows": len(examples), "excluded_annotation_ids":
           [s["annotation_id"] for s in load_config()["known_skipped_rows"]], "tokenizers": {}}
    for role, repo in CANDIDATES.items():
        if role == "reference_alt_draft":
            continue  # same tokenizer family as reference_alt; avoid a second gated lookup
        try:
            tok = AutoTokenizer.from_pretrained(repo)
        except Exception as e:
            res["tokenizers"][repo] = {"status": "SKIPPED", "reason": f"{type(e).__name__}: {str(e).splitlines()[0][:150]}"}
            print(f"SKIPPED {repo}: {type(e).__name__}")
            continue
        res["tokenizers"][repo] = {"status": "ok", **stats_for_tokenizer(tok, examples, "Qwen" in repo)}
    resolve("results/metrics/token_stats.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"rows counted: {res['n_rows']} (splits {SPLITS})")
    for repo, s in res["tokenizers"].items():
        if s["status"] != "ok":
            continue
        print(f"\n{repo}")
        for k in ("prompt", "target", "prompt_plus_target"):
            d = s[k]
            print(f"  {k:19s} min {d['min']}  mean {d['mean']:.1f}  median {d['median']}  p95 {d['p95']:.1f}  max {d['max']}")
        print(f"  rows exceeding: {s['rows_exceeding']}  longest: {s['longest_example']}")


if __name__ == "__main__":
    main()
