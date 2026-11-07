"""Torch-free tests: label masking, truncation guard, config, logging guards, test-split guard, projection."""
import importlib.util
import json
from pathlib import Path

import pytest

from trialmatch.evaluation.generate_eval import (TEST_MARKER, TestGuardError, check_test_guard, summarize)
from trialmatch.model.chat import END_TOKEN, chat_prompt_text
from trialmatch.model.prompt import example_from_row
from trialmatch.training import config as tcfg
from trialmatch.training.dataset import (IGNORE_INDEX, TruncationError, build_dataset, encode_example, iter_rows)
from trialmatch.training.logging_utils import NonFiniteLossError, StepLogger, check_finite
from trialmatch.utils.common import load_config, resolve

CFG = load_config()
needs_data = pytest.mark.skipif(not (resolve(CFG["paths"]["processed_dir"]) / "train.parquet").exists(),
                                reason="processed data missing; run scripts/prepare_data.py")


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    except Exception as e:  # offline and not cached
        pytest.skip(f"Qwen tokenizer unavailable: {e}")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_data
def test_label_mask_on_few_rows(tok):
    end_id = tok.convert_tokens_to_ids(END_TOKEN)
    for row in list(iter_rows("train"))[:5]:
        ex = example_from_row(row)
        enc = encode_example(tok, ex.prompt, ex.target, 768)
        ids, labels, n_p = enc["input_ids"], enc["labels"], enc["n_prompt"]
        assert len(ids) == len(labels) == len(enc["attention_mask"])
        assert all(l == IGNORE_INDEX for l in labels[:n_p])             # prompt fully masked
        assert labels[n_p:] == ids[n_p:]                                 # answer trained as-is
        assert labels[-1] == end_id and ids[-1] == end_id                # end token is trained
        assert tok.decode(labels[n_p:]) == ex.target + END_TOKEN         # exactly answer JSON + end token
        # prompt part is the chat-templated prompt (thinking disabled), nothing else
        assert tok.decode(ids[:n_p]) == chat_prompt_text(tok, ex.prompt)
        assert "<think>\n\n</think>" in tok.decode(ids[:n_p])


@needs_data
def test_truncation_is_an_error_not_silent(tok):
    ex = example_from_row(next(iter(iter_rows("train"))))
    with pytest.raises(TruncationError):
        encode_example(tok, ex.prompt, ex.target, 100)


@needs_data
def test_build_dataset_skips_883_and_fits(tok):
    ds = build_dataset(tok, "val", 768)
    assert len(ds) == len(list(iter_rows("val")))
    tr = build_dataset(tok, "train", 768)
    assert 883 not in {e["annotation_id"] for e in tr}
    assert len(tr) == 602  # 603 train rows minus annotation 883
    assert max(len(e["input_ids"]) for e in tr + ds) <= 768
    with pytest.raises(PermissionError):
        build_dataset(tok, "test", 768)


def test_iter_rows_refuses_test_by_default():
    with pytest.raises(PermissionError):
        next(iter_rows("test"))


def test_config_values_and_validation(monkeypatch):
    cfg = tcfg.load_train_config()
    assert cfg["model"]["id"] == "Qwen/Qwen3-4B" and cfg["model"]["max_seq_len"] == 768
    assert cfg["train"]["seed"] == 42 and cfg["train"]["fp16"] is True
    assert set(cfg["lora"]["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert cfg["quantization"] in tcfg.QUANT_MODES
    with pytest.raises(ValueError):
        tcfg.load_train_config(quantization="int3")
    bad = json.loads(json.dumps(cfg))
    bad["lora"]["target_modules"].append("lm_head")
    with pytest.raises(ValueError):
        tcfg.validate(bad)


def test_output_root_env_override(monkeypatch, tmp_path):
    cfg = tcfg.load_train_config()
    monkeypatch.delenv("TRIALMATCH_OUT", raising=False)
    assert tcfg.output_root(cfg) == resolve(".").resolve()
    monkeypatch.setenv("TRIALMATCH_OUT", str(tmp_path))
    p = tcfg.run_paths(cfg)
    assert p["root"] == tmp_path and p["run_dir"] == tmp_path / "checkpoints" / "qwen3-4b_qlora_4bit"
    assert p["logs"] == tmp_path / "results" / "logs"
    assert tcfg.run_name(tcfg.load_train_config("Qwen/Qwen3-1.7B", "lora_fp16")) == "qwen3-1.7b_lora_fp16"


def test_nonfinite_guard():
    check_finite("train", 1.5, 3)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(NonFiniteLossError, match="STOPPING"):
            check_finite("train", bad, 7)


def test_step_logger_jsonl_csv_and_append(tmp_path):
    lg = StepLogger(tmp_path, "run")
    lg.write({"type": "env", "x": 1})
    lg.write({"step": 1, "epoch": 0.1, "split": "train", "loss": 2.0, "learning_rate": 1e-4, "grad_norm": 0.5})
    StepLogger(tmp_path, "run").write({"step": 5, "epoch": 1.0, "split": "val", "loss": 1.2})  # resume appends
    lines = (tmp_path / "run.jsonl").read_text().splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["type"] == "env"
    csv_lines = (tmp_path / "run.csv").read_text().splitlines()
    assert csv_lines[0].startswith("time,step,epoch,split,loss") and len(csv_lines) == 3  # header + 2 loss rows


def test_test_split_guard(tmp_path):
    assert check_test_guard("val", False, False, tmp_path) is None
    with pytest.raises(TestGuardError, match="confirm-final"):
        check_test_guard("test", False, False, tmp_path)
    marker = check_test_guard("test", True, False, tmp_path)
    assert marker == tmp_path / TEST_MARKER
    marker.write_text("{}")
    with pytest.raises(TestGuardError, match="already evaluated"):
        check_test_guard("test", True, False, tmp_path)
    assert check_test_guard("test", True, True, tmp_path) == marker  # --force


def test_summarize_counts_invalid_as_wrong():
    recs = [{"status": "ok", "pred_label": "a", "gold_label": "a"},
            {"status": "ok", "pred_label": "b", "gold_label": "a"},
            {"status": "invalid", "pred_label": None, "gold_label": "a"},
            {"status": "ok", "pred_label": "c", "gold_label": "c"}]
    s = summarize(recs)
    assert s["n"] == 4 and s["n_invalid"] == 1 and s["invalid_rate"] == 0.25
    assert s["accuracy"] == 0.5 and s["accuracy_among_valid"] == pytest.approx(2 / 3)


def test_smoke_projection_math():
    smoke = _script("smoke_test_train")
    cfg = tcfg.load_train_config()
    # per step 10 s after 2 ignored warmup steps; 602 train rows / 16 per step = 38 steps/epoch -> 114 steps
    p = smoke.project([30, 20, 10, 10, 10], n_train=602, n_val=204, cfg=cfg)
    assert p["sec_per_step"] == 10 and p["total_steps"] == 114
    assert p["train_min"] == pytest.approx(19.0)
    assert p["eval_min"] == pytest.approx(3 * 204 * (10 / 16) / 3 / 60)
    assert p["within_rule"] is True
    slow = smoke.project([100] * 5, 602, 204, cfg)
    assert slow["total_min"] > 90 and slow["within_rule"] is False
