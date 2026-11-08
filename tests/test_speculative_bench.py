"""Speculative benchmark plumbing: planning, incremental saving, resume, mismatch report, plot (tiny CPU models)."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from trialmatch.inference.benchmarks import kv_cache_bench as kb
from trialmatch.inference.benchmarks import speculative_bench as sb
from trialmatch.inference.benchmarks.tiny_model import build_tiny_draft, build_tiny_qwen3

TARGET, DRAFT = build_tiny_qwen3(0), build_tiny_draft()
ENV = sb.environment(TARGET, DRAFT, "cpu", torch.float32, "tiny")


def run(out_dir, cfgs, **kw):
    return sb.run_benchmark(TARGET, DRAFT, cfgs, lambda c: kb.synthetic_input(c["input_len"], 256, 0),
                            lambda c: None, out_dir, ENV, {}, "cpu", 0, 1, log=lambda *_: None, **kw)


def test_plan_puts_plain_baseline_before_each_k():
    cfgs = sb.plan([16], [8, 12], [1, 4], "d")
    assert [c["k"] for c in cfgs] == [0, 1, 4, 0, 1, 4]
    assert len({sb.make_key(c["source"], c["aid"], c["input_len"], c["n_new"], c["k"], c["draft"]) for c in cfgs}) == 6


def test_incremental_save_max_configs_and_resume(tmp_path):
    cfgs = sb.plan([16, 24], [6], [2, 4], "tiny")                  # 6 configurations
    run(tmp_path, cfgs, max_configs=2)
    jp = next(tmp_path.glob("speculative_*.json"))
    assert len(json.loads(jp.read_text())["rows"]) == 2              # saved after each configuration, stopped at 2
    run(tmp_path, cfgs, max_configs=2, resume=True)
    assert len(json.loads(jp.read_text())["rows"]) == 4
    rows = run(tmp_path, cfgs, resume=True)
    assert len(rows) == 6 and len(list(tmp_path.glob("speculative_*.json"))) == 1   # same file, no duplicates
    assert len({r["key"] for r in rows}) == 6
    spec = [r for r in rows if r["decoder"] == "speculative"]
    assert all(r["tokens_equal"] and r["speedup_vs_plain"] > 0 and 0 <= r["acceptance_rate"] <= 1 for r in spec)
    assert (tmp_path / jp.name.replace(".json", ".csv")).read_text().splitlines()[0].startswith("key,source")
    sb.plot(rows, tmp_path / "p.png", ENV)
    assert (tmp_path / "p.png").exists()


def test_mismatch_report_pinpoints_first_difference():
    ids = kb.synthetic_input(16, 256, 0)
    from trialmatch.inference.kv_cache.cached_decode import cached_greedy_decode
    plain = cached_greedy_decode(TARGET, ids, max_new_tokens=10).sequences[0]
    fake = plain[:4] + [(plain[4] + 1) % 256] + plain[5:]            # pretend the plain decoder differed at step 4
    rep = sb.mismatch_report(TARGET, DRAFT, ids, fake, 10, 2, None)
    assert rep["first_diff_step"] is None                            # real decoders still agree with each other
    assert sb.first_diff(plain, fake) == 4 and sb.first_diff(plain, plain) is None
