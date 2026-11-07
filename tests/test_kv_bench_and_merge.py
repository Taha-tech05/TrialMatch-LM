"""Benchmark plumbing and adapter-merge tests (tiny random Qwen3, CPU, no downloads)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from trialmatch.inference.benchmarks import kv_cache_bench as kb
from trialmatch.inference.benchmarks.tiny_model import build_tiny_qwen3
from trialmatch.inference.benchmarks.timing import summarize
from trialmatch.utils.common import load_config, resolve


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 10.0])
    assert s["mean"] == 4.0 and s["median"] == 2.5 and s["min"] == 1.0 and s["max"] == 10.0 and s["n"] == 4
    assert summarize([5.0])["std"] == 0.0


def test_bench_case_and_outputs(tmp_path):
    model = build_tiny_qwen3()
    case = kb.bench_case(model, kb.synthetic_input(32, 256, 0), 6, None, "cpu", warmup=1, repeats=2)
    assert case["tokens_equal"] and case["n_new_tokens"] == 6 and case["repeats"] == 2
    for k in ("cached", "uncached"):
        d = case["decoders"][k]
        assert d["total_s"]["n"] == 2 and d["tokens_per_s"]["median"] > 0 and "peak_mem_bytes" not in d  # CPU
    case["source"] = "synthetic"
    env = kb.environment(model, "cpu", torch.float32)
    jp, cp, pp = kb.write_results(tmp_path, env, {}, [case], stamp="t")
    assert jp.exists() and cp.exists() and pp.exists()
    assert len(cp.read_text().splitlines()) == 3  # header + cached + uncached


def test_val_inputs_use_val_split_only():
    if not (resolve(load_config()["paths"]["processed_dir"]) / "val.parquet").exists():
        pytest.skip("processed data missing")

    class Tok:  # stub: no download needed
        def apply_chat_template(self, msgs, **kw):
            return msgs[0]["content"]

        def __call__(self, text, **kw):
            return {"input_ids": torch.tensor([[ord(c) % 200 + 3 for c in text[:50]]])}

    from trialmatch.evaluation.generate_eval import build_items
    got = kb.val_inputs(Tok(), 3)
    val_ids = {it["annotation_id"] for it in build_items("val", False)}
    assert len(got) == 3 and all(aid in val_ids for aid, _ in got)


def test_merge_adapter_matches_unmerged_model(tmp_path):
    peft = pytest.importorskip("peft")
    from trialmatch.model.load import merge_adapter_into_base
    base = build_tiny_qwen3(seed=3)
    pm = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"]))
    with torch.no_grad():  # make the adapter non-trivial (lora_B starts at zero)
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.3)
    ids = torch.randint(3, 256, (1, 20))
    before = pm.eval()(ids).logits.detach()
    pm.save_pretrained(tmp_path / "ad")
    fresh = build_tiny_qwen3(seed=3)
    merged = merge_adapter_into_base(fresh, tmp_path / "ad")
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    after = merged(ids).logits.detach()
    assert (before - after).abs().max() < 1e-4
    assert (build_tiny_qwen3(seed=3)(ids).logits - after).abs().max() > 1e-3   # the merge really changed the model
