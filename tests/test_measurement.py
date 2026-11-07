"""Tests for the measurement helpers (no network, no model weights)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from trialmatch.model.measure import GIB, describe, kv_cache_bytes, weights_bytes


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kv_cache_hand_computed():
    # Qwen3-4B: 2 x 36 layers x 8 kv heads x 128 head_dim x 1024 tokens x 2 bytes
    # = 72 * 8 * 128 * 1024 * 2 = 150,994,944 bytes = 144 MiB = 0.140625 GiB
    assert kv_cache_bytes(36, 8, 128, 1024) == 150_994_944
    assert kv_cache_bytes(36, 8, 128, 1024) / GIB == 0.140625
    assert kv_cache_bytes(36, 8, 128, 1024, batch=4) == 4 * 150_994_944
    assert kv_cache_bytes(36, 8, 128, 4096) == 4 * 150_994_944  # linear in seq_len


def test_weights_bytes():
    assert weights_bytes(1_000_000_000, "fp16") == 2e9
    assert weights_bytes(1_000_000_000, "int8") == 1e9
    assert weights_bytes(1_000_000_000, "4bit") == 0.5e9


def test_describe():
    d = describe([1, 2, 3, 4, 5])
    assert (d["min"], d["max"], d["mean"], d["median"], d["n"]) == (1, 5, 3, 3, 5)
    assert d["p95"] == pytest.approx(4.8)  # 0.95 * 4 = 3.8 -> 4 + 0.8 * (5 - 4)
    assert describe([1, 2, 3, 4])["median"] == 2.5


class FakeTok:
    chat_template = None  # no template: prompt used as-is; one token per whitespace word

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_token_stats_on_few_rows():
    ts = _load_script("token_stats")
    exs = [{"id": 1, "split": "train", "prompt": "a b c", "target": "x y"},
           {"id": 2, "split": "val", "prompt": "a b c d e f", "target": "x"}]
    s = ts.stats_for_tokenizer(FakeTok(), exs, is_qwen=False)
    assert s["n_rows"] == 2
    assert s["prompt"]["max"] == 6 and s["prompt"]["min"] == 3
    assert s["target"]["min"] == 2 and s["target"]["max"] == 3  # +1 end-of-turn token
    assert s["prompt_plus_target"]["max"] == 8  # 6 + (1 + 1)
    assert s["longest_example"]["annotation_id"] == 2
    assert s["rows_exceeding"] == {"1024": 0, "2048": 0, "4096": 0}


def test_load_examples_train_val_only_and_skips_883():
    ts = _load_script("token_stats")
    assert ts.SPLITS == ("train", "val")
    exs = ts.load_examples()
    assert 883 not in {e["id"] for e in exs}
    assert {e["split"] for e in exs} == {"train", "val"}
