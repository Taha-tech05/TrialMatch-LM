"""Check that draft and target tokenizers are interchangeable (needed for speculative decoding)."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from trialmatch.model.measure import COMPAT_PAIRS
from trialmatch.model.prompt import example_from_row, skip_reason
from trialmatch.utils.common import load_config, resolve

EDGE = ["", " ", "\n\n", "Café naïve – 5 mg/dL ≥ 3.5 × 10^9/L", "患者 45 岁",
        '{"label": "included", "evidence_ids": [0, 2]}', "😀 emoji", "tab\tsep", "ECOG PS 0-1; HbA1c < 7.5%"]


def sample_strings(n: int = 200, seed: int = 0) -> list[str]:
    """Deterministic samples from the TRAIN split only: prompts, targets, criteria, note sentences."""
    cfg = load_config()
    c = cfg["columns"]
    df = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / "train.parquet")
    df = df[[skip_reason(t) is None for t in df[c["criterion_text"]]]].sample(n=60, random_state=seed)
    out = list(EDGE)
    for _, r in df.iterrows():
        ex = example_from_row(r.to_dict())
        out += [ex.prompt, ex.target, str(r[c["criterion_text"]]),
                random.Random(int(r[c["id"]])).choice(str(r[c["note"]]).split("\n"))]
    return out[:n]


def _specials(t) -> dict:
    return {"special_tokens_map": {k: str(v) for k, v in t.special_tokens_map.items()},
            "added_tokens": {i: (x.content, x.special) for i, x in sorted(t.added_tokens_decoder.items())}}


def compare(repo_a: str, repo_b: str, strings: list[str]) -> dict:
    res = {"pair": [repo_a, repo_b]}
    try:
        from transformers import AutoTokenizer
        a, b = AutoTokenizer.from_pretrained(repo_a), AutoTokenizer.from_pretrained(repo_b)
    except Exception as e:
        return {**res, "status": "SKIPPED", "reason": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    mism = [i for i, s in enumerate(strings) if a.encode(s) != b.encode(s)]
    res.update(status="ok", len_a=len(a), len_b=len(b),
               vocab_identical=a.get_vocab() == b.get_vocab(),
               special_tokens_identical=_specials(a) == _specials(b),
               n_strings=len(strings), n_id_mismatches=len(mism), mismatch_indices=mism[:10])
    res["passed"] = res["vocab_identical"] and res["special_tokens_identical"] and not mism
    return res


def main():
    strings = sample_strings()
    results = [compare(a, b, strings) for a, b in COMPAT_PAIRS]
    # Model-side vocab (embedding/logit width) also matters for speculative decoding.
    facts_path = resolve("results/metrics/model_facts.json")
    if facts_path.exists():
        facts = {r["repo"]: r for r in json.loads(facts_path.read_text(encoding="utf-8"))}
        for r in results:
            r["config_vocab_sizes"] = [facts.get(p, {}).get("vocab_size_config") for p in r["pair"]]
    resolve("results/metrics/tokenizer_compat.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results:
        verdict = "SKIPPED" if r["status"] == "SKIPPED" else ("PASS" if r["passed"] else "FAIL")
        print(f"{verdict}: {r['pair'][0]} vs {r['pair'][1]}")
        for k, v in r.items():
            if k != "pair":
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
