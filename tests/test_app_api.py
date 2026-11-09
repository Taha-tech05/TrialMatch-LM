"""API tests with FastAPI TestClient and a fake backend (no model, no GPU)."""
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from trialmatch.app.config import load_app_config
from trialmatch.app.examples import ExampleStore
from trialmatch.app.runtime import Runtime
from trialmatch.app.server import STATIC, create_app

NOTE = "0. Pt has diabetes.\n1. Pt is 40.\n2. No smoking."


class FakeBackend:
    mode = "real"

    def __init__(self):
        self.calls = []

    def ready(self):
        return True

    def info(self):
        return {"mode": "real", "phase": "ready", "model_name": "Fake", "device": "cpu", "dtype": "float32",
                "adapter": "fake/adapter", "models": [], "banner": None}

    def engines(self):
        return [{"id": e, "label": e, "available": e != "speculative", "reason": "no draft", "warning": None}
                for e in ("plain", "kv", "speculative")]

    def predict(self, note, ctype, text, engine="kv"):
        self.calls.append(text)
        if "boom" in text:
            raise RuntimeError("generation exploded")
        if "slow" in text:
            time.sleep(0.3)
        if "garbage" in text:
            return {"status": "invalid", "label": None, "evidence_ids": None, "raw_output": "not json",
                    "parse_error": "no valid JSON object found", "engine": engine, "latency_s": 0.01, "n_tokens": 3}
        label = "included" if ctype == "inclusion" else "not excluded"
        return {"status": "ok", "label": label, "evidence_ids": [0], "raw_output": '{"label": "%s"}' % label,
                "parse_error": None, "engine": engine, "latency_s": 0.01, "n_tokens": 9}


@pytest.fixture()
def client(tmp_path):
    cfg = load_app_config()
    cfg["reports_dir"] = str(tmp_path / "reports")
    rt = Runtime.with_backend(cfg, FakeBackend(), ExampleStore([]))
    return TestClient(create_app(rt, ))


def wait(c, job_id, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def crit(t="inclusion", x="c"):
    return {"type": t, "text": x}


def test_static_page_has_disclaimer_and_assets():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "Educational Prototype Notice: this is a research screening prototype, not a clinical enrollment decision or diagnostic tool." in html
    assert "#FEF3C7".lower() in (STATIC / "style.css").read_text(encoding="utf-8").lower()
    for f in ("app.js", "style.css"):
        assert (STATIC / f).exists()
    assert "http://" not in html.replace("http://www.w3.org", "") and "https://" not in html   # no CDN / external fonts


def test_index_and_status(client):
    assert client.get("/").status_code == 200
    s = client.get("/api/status").json()
    assert s["phase"] == "ready" and s["model_name"] == "Fake" and s["generation"]["max_new_tokens"] == 64
    assert "Educational Prototype" in s["disclaimer"]


def test_screening_flow_and_export(client):
    r = client.post("/api/screen", json={"note": NOTE, "criteria": [crit(), crit("exclusion"), crit(x="   ")], "engine": "kv"})
    assert r.status_code == 200 and len(r.json()["warnings"]) == 1
    j = wait(client, r.json()["job_id"])
    assert j["status"] == "done" and j["done"] == j["total"] == 3
    assert [x["status"] for x in j["results"]] == ["ok", "ok", "skipped"]
    assert j["results"][0]["cited"] == [{"id": 0, "text": "Pt has diabetes."}]
    assert j["aggregate"]["code"] == "eligible" and j["report_paths"]
    assert Path(j["report_paths"]["html"]).exists()
    for fmt, needle in (("json", "provenance"), ("html", "Educational Prototype Notice"), ("md", "Provenance")):
        e = client.get(f"/api/jobs/{j['id']}/export", params={"format": fmt})
        assert e.status_code == 200 and needle in e.text
    assert client.get(f"/api/jobs/{j['id']}/export", params={"format": "pdf"}).status_code == 400


def test_validation_errors_are_structured_json(client):
    for payload, code in [({"note": "", "criteria": [crit()]}, "invalid_input"),
                          ({"note": NOTE, "criteria": []}, "invalid_input"),
                          ({"note": NOTE, "criteria": [crit(x="y" * 1501)]}, "invalid_input"),
                          ({"note": NOTE, "criteria": [crit(t=None)]}, "invalid_input")]:
        r = client.post("/api/screen", json=payload)
        assert r.status_code == 422
        e = r.json()["error"]
        assert e["code"] == code and e["message"] and "Traceback" not in e["message"] and e["details"]
    r = client.post("/api/screen", content=b"\xff\xfe not json", headers={"Content-Type": "application/json"})
    assert r.status_code in (400, 422) and "error" in r.json()
    assert client.post("/api/screen", json={"note": NOTE, "criteria": [crit()], "engine": "speculative"}).json()["error"]["code"] == "engine_unavailable"
    assert client.get("/api/jobs/nope").json()["error"]["code"] == "not_found"


def test_per_criterion_error_and_invalid_continue(client):
    r = client.post("/api/screen", json={"note": NOTE, "criteria": [crit(x="boom"), crit(x="garbage"), crit(x="fine")]})
    j = wait(client, r.json()["job_id"])
    st = [x["status"] for x in j["results"]]
    assert st == ["error", "invalid", "ok"] and j["status"] == "done"
    assert j["results"][0]["flags"]["error"] and j["results"][1]["flags"]["invalid"]
    assert "exploded" in j["results"][0]["message"]
    assert j["aggregate"]["code"] == "undetermined"
    assert client.get("/api/status").status_code == 200                 # the server survived


def test_cancel_and_reset(client):
    r = client.post("/api/screen", json={"note": NOTE, "criteria": [crit(x="slow 1"), crit(x="slow 2"), crit(x="slow 3")]})
    jid = r.json()["job_id"]
    assert client.post("/api/screen", json={"note": NOTE, "criteria": [crit()]}).json()["error"]["code"] == "busy"
    client.post(f"/api/jobs/{jid}/cancel")
    j = wait(client, jid)
    assert j["status"] == "cancelled" and any(x.get("cancelled") for x in j["results"])
    assert client.post("/api/reset").json()["ok"]
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_parse_and_prepare_endpoints(client):
    p = client.post("/api/criteria/parse", json={"text": "Inclusion Criteria:\n- A\nExclusion Criteria:\n- B"}).json()
    assert [c["type"] for c in p["criteria"]] == ["inclusion", "exclusion"]
    n = client.post("/api/note/prepare", json={"text": "Hello there. General Kenobi."}).json()
    assert n["note"] == "0. Hello there.\n1. General Kenobi." and n["numbered_by_app"]
    assert client.post("/api/note/prepare", json={"text": " "}).status_code == 400
    assert client.post("/api/upload", content=b"\xff\xfe\xfa").json()["error"]["code"] == "unreadable_text"
    assert client.post("/api/upload", content="ok ✓".encode()).json()["text"] == "ok ✓"


def test_lab_endpoints_degrade_cleanly(client):
    assert client.post("/api/lab/run", json={"engines": ["kv"]}).json()["error"]["code"] == "needs_real"
    s = client.get("/api/lab/saved").json()
    assert set(s) == {"kv_cache", "speculative", "attention"}
    if s["kv_cache"]["available"]:
        assert "label" in s["kv_cache"]["provenance"] and s["kv_cache"]["provenance"]["gpu"]
    assert client.get("/api/lab/attention", params={"seq": 2}).status_code == 400


def test_attention_demo_matches_reference():
    pytest.importorskip("torch")
    from trialmatch.app.lab import attention_demo
    d = attention_demo(seq=32, block=8, row=20)
    assert d["max_abs_err"] < 1e-5 and d["trace_vs_tiled_err"] < 1e-5
    assert d["grid"][0][3] == "skipped" and d["grid"][3][0] == "full" and d["grid"][2][2] == "partial"
    assert attention_demo(32, 8, 5, causal=False)["n_skipped"] == 0


def test_saved_loader_missing_and_synthetic(tmp_path):
    from trialmatch.app import lab
    cfg = {"benchmarks": {"kv_cache": None, "speculative": "does/not/exist.json", "attention": None}}
    s = lab.load_saved(cfg)
    assert all(not v["available"] and v["reason"] for v in s.values())
    import json
    f = tmp_path / "attention_20260101_000000.json"
    f.write_text(json.dumps({"environment": {"gpu": "T4", "dtype": "torch.float16", "n_params": 4_000_000_000},
                             "args": {}, "mismatches": 0,
                             "results": [{"seq_len": 64, "impl": "tiled", "status": "ok", "time_s": {"mean": 0.001},
                                          "peak_mem_bytes": 2 ** 21, "max_abs_err": 1e-4, "within_tolerance": True}]}))
    out = lab._attn(json.loads(f.read_text()), f.name)
    assert out["results"][0]["time_ms"] == 1.0 and "T4" in out["provenance"]["label"] and "2026-01-01" in out["provenance"]["label"]


def test_real_backend_with_tiny_model_all_engines():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from trialmatch.app.backends import LoadedModel, RealBackend, run_engine
    from trialmatch.inference.benchmarks.tiny_model import build_tiny_draft, build_tiny_qwen3

    class Tok:                                          # byte-level stand-in; no downloads
        pad_token_id, unk_token_id = 0, -1
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return msgs[0]["content"][:200]
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            return {"input_ids": torch.tensor([[3 + (ord(c) % 200) for c in text]])}
        def convert_tokens_to_ids(self, t):
            return 2
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(32 + (i % 90)) for i in ids if not (skip_special_tokens and i == 2))

    tgt = LoadedModel("merged", build_tiny_qwen3(0), Tok(), "tiny", "cpu", "float32", 0.0)
    drf = LoadedModel("small", build_tiny_draft(), Tok(), "tinydraft", "cpu", "float32", 0.0)
    cfg = load_app_config()
    be = RealBackend(cfg, source={"kind": "merged", "path": "x"}, loader=lambda k: tgt if k == "merged" else drf)
    be.start(); be.load_model("small")
    assert be.wait(10)
    time.sleep(0.3)
    note = "0. Pt has diabetes.\n1. Pt is 40."
    outs = {e: be.predict(note, "inclusion", "Diabetic patients", e) for e in ("plain", "kv", "speculative")}
    assert all(o["status"] in ("ok", "invalid") and o["n_tokens"] > 0 for o in outs.values())
    assert outs["plain"]["raw_output"] == outs["kv"]["raw_output"] == outs["speculative"]["raw_output"]
    ids = be.encode(tgt, "hello world")
    toks = {e: run_engine(e, tgt, drf, ids, max_new=10, k=3)["tokens"] for e in ("plain", "kv", "speculative")}
    assert toks["plain"] == toks["kv"] == toks["speculative"]


# ---------------------------------------------------------------- draft loading (fake loader, no weights)
def _fake_loaders():
    from trialmatch.app.backends import LoadedModel
    made = []

    def loader(key):
        made.append(key)
        if key == "small" and loader.fail_draft:
            raise RuntimeError("draft boom")
        return LoadedModel(key, object(), object(), "Target-4B" if key == "merged" else "Draft-0.6B", "cpu", "float32", 0.0)
    loader.fail_draft = False
    return loader, made


def _wait_for(pred, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _spec(be):
    return next(e for e in be.engines() if e["id"] == "speculative")


def test_load_draft_default_off_then_on():
    from trialmatch.app.backends import RealBackend
    cfg = load_app_config()
    assert cfg["real"]["load_draft"] is False
    loader, made = _fake_loaders()
    be = RealBackend(cfg, source={"kind": "merged", "path": "x"}, loader=loader)
    be.start()
    assert _wait_for(be.ready) and made == ["merged"]
    s = _spec(be)
    assert not s["available"] and "--load-draft" in s["reason"]
    assert be.info()["draft"]["loaded"] is False and be.info()["draft"]["requested"] is False

    cfg2 = load_app_config()
    cfg2["real"]["load_draft"] = True
    loader2, made2 = _fake_loaders()
    be2 = RealBackend(cfg2, source={"kind": "merged", "path": "x"}, loader=loader2)
    be2.start()
    assert _wait_for(lambda: be2.ready() and "small" in be2.models)
    assert made2 == ["merged", "small"]                       # one thread, sequential
    assert _spec(be2)["available"]
    info = be2.info()
    assert info["target"]["name"] == "Target-4B" and info["draft"]["name"] == "Draft-0.6B"
    be2.select("small")                                        # draft as the active model: speculative off, with a reason
    s2 = _spec(be2)
    assert not s2["available"] and "merged 4B as the active model" in s2["reason"]


def test_draft_failure_does_not_block_target():
    from trialmatch.app.backends import RealBackend
    cfg = load_app_config()
    cfg["real"]["load_draft"] = True
    loader, made = _fake_loaders()
    loader.fail_draft = True
    be = RealBackend(cfg, source={"kind": "merged", "path": "x"}, loader=loader)
    be.start()
    assert _wait_for(lambda: "small" in be.errors)
    assert be.ready() and be.phase() == "ready"
    s = _spec(be)
    assert not s["available"] and "failed to load" in s["reason"]
    assert be.info()["draft"]["error"]


def test_status_exposes_both_models_via_api():
    from trialmatch.app.backends import RealBackend
    cfg = load_app_config()
    cfg["real"]["load_draft"] = True
    loader, _ = _fake_loaders()
    be = RealBackend(cfg, source={"kind": "merged", "path": "x"}, loader=loader)
    be.start()
    assert _wait_for(lambda: len(be.models) == 2)
    c = TestClient(create_app(Runtime.with_backend(cfg, be, ExampleStore([]))))
    s = c.get("/api/status").json()
    assert s["target"]["loaded"] and s["draft"]["loaded"]
    assert next(e for e in s["engines"] if e["id"] == "speculative")["available"]
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "Draft:" in js and "Target:" in js                    # header renders both models


def test_launch_cli_has_load_draft_flag():
    src = (Path(__file__).resolve().parents[1] / "scripts" / "launch_app.py").read_text(encoding="utf-8")
    assert '"--load-draft"' in src and 'cfg["real"]["load_draft"] = True' in src


# ---------------------------------------------------------------- examples / benchmark empty states
def test_missing_val_parquet_gives_clear_message(monkeypatch):
    def boom(cls):
        raise FileNotFoundError("data/processed/val.parquet")
    monkeypatch.setattr(ExampleStore, "from_val", classmethod(boom))
    st = ExampleStore.safe_from_val()
    assert st.rows == [] and "python scripts/prepare_data.py" in st.error
    rt = Runtime(load_app_config(), "demo")
    rt._examples = None
    assert "prepare_data.py" in rt.examples.public()["error"]
    assert "prepare_data.py" in ExampleStore([], ExampleStore.MISSING_MSG).public()["error"]
    assert ExampleStore([]).public()["error"] is None


def test_official_benchmark_files_configured_and_missing_is_empty_state(tmp_path):
    from trialmatch.app import lab
    cfg = load_app_config()
    assert cfg["benchmarks"]["kv_cache"].endswith("kv_cache_20261107_192136.json")
    assert cfg["benchmarks"]["speculative"].endswith("speculative_20261108_113154.json")
    s = lab.load_saved(cfg)
    assert s["kv_cache"]["available"] and s["speculative"]["available"]
    s = lab.load_saved({"benchmarks": {"kv_cache": "no/such.json", "speculative": None, "attention": None}})
    assert all(not v["available"] for v in s.values())
    assert "not found" in s["kv_cache"]["reason"] and "benchmarks.kv_cache" in s["kv_cache"]["reason"]
    assert "scripts/bench_attention.py" in s["attention"]["reason"]
