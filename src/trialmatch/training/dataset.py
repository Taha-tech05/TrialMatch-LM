"""Tokenised train/val examples with prompt-masked labels (torch-free: plain lists of ints).

Row source: data/processed/{train,val}.parquet through the existing prompt formatter.
Annotation ids in configs/data.yaml known_skipped_rows (883) and null/empty criteria are skipped.
The test split can only be read by passing allow_test=True (used solely by the guarded evaluator).
"""
from typing import Iterator

import pandas as pd

from trialmatch.model.chat import chat_prompt_text, end_token_id
from trialmatch.model.prompt import example_from_row, skip_reason
from trialmatch.utils.common import load_config, resolve

IGNORE_INDEX = -100
TRAINING_SPLITS = ("train", "val")


class TruncationError(ValueError):
    """An example does not fit in max_seq_len. Examples are never truncated."""


def iter_rows(split: str, allow_test: bool = False) -> Iterator[dict]:
    """Yield processed rows of a split (as dicts), skipping known-skipped/empty-criterion rows."""
    if split == "test" and not allow_test:
        raise PermissionError("the test split is reserved for the final guarded evaluation")
    if split not in (*TRAINING_SPLITS, "test"):
        raise ValueError(f"unknown split {split!r}")
    cfg = load_config()
    c = cfg["columns"]
    skipped = {s["annotation_id"] for s in cfg["known_skipped_rows"]}
    df = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / f"{split}.parquet")
    for _, r in df.iterrows():
        if int(r[c["id"]]) in skipped or skip_reason(r[c["criterion_text"]]):
            continue
        yield r.to_dict()


def encode_example(tok, prompt: str, target: str, max_seq_len: int) -> dict:
    """input_ids = chat-templated prompt + target JSON + <|im_end|>; labels = -100 on the prompt.

    Prompt and target are tokenised separately, which is exactly what generation sees (the model
    continues after the templated prompt). Raises TruncationError instead of truncating.
    """
    p_ids = tok(chat_prompt_text(tok, prompt), add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"] + [end_token_id(tok)]
    ids = p_ids + t_ids
    if len(ids) > max_seq_len:
        raise TruncationError(f"example has {len(ids)} tokens > max_seq_len {max_seq_len}")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [IGNORE_INDEX] * len(p_ids) + t_ids, "n_prompt": len(p_ids)}


def build_dataset(tok, split: str, max_seq_len: int) -> list[dict]:
    """Tokenise every usable row of train/val; asserts nothing is truncated."""
    if split not in TRAINING_SPLITS:
        raise PermissionError(f"build_dataset only builds {TRAINING_SPLITS}, not {split!r}")
    out = []
    for row in iter_rows(split):
        ex = example_from_row(row)
        enc = encode_example(tok, ex.prompt, ex.target, max_seq_len)
        enc["annotation_id"] = int(row[load_config()["columns"]["id"]])
        out.append(enc)
    assert out, f"no examples built for {split}"
    assert all(len(e["input_ids"]) <= max_seq_len for e in out), "an example exceeds max_seq_len"
    return out
