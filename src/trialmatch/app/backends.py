"""Prediction backends. Common interface:

    predict(note, criterion_type, criterion_text, engine) -> record
    info() -> dict (mode, phase, model name, device, dtype, ...)
    engines() -> [{'id','label','available','reason','warning'}]

record keys: status ('ok' | 'invalid' | 'unavailable'), label, evidence_ids, raw_output, parse_error, engine,
latency_s, n_tokens, message, banner. Errors raised by predict() are turned into 'error' records by the caller.

RealBackend loads the model ONCE in a background thread (torch/transformers are imported lazily, only then).
DemoBackend serves cached REAL validation predictions and never invents an output.
"""
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from trialmatch.app.config import DEMO_BANNER, ENGINE_LABELS, ENGINES, generation_settings
from trialmatch.utils.common import PROJECT_ROOT, resolve

log = logging.getLogger("trialmatch.app")

SMALL_NOTE = "untuned, low quality, fast demo only"


def norm(s) -> str:
    return " ".join(str(s or "").split())


class BackendBusy(RuntimeError):
    """The model is in use by another request."""


# ----------------------------------------------------------------------------------------------- demo
class DemoBackend:
    mode = "demo"

    def __init__(self, records: list[dict], source: str = "cached validation predictions"):
        """records: joined val predictions (note, criterion_type, criterion_text, status, raw_output, parse_error,
        pred_label, pred_evidence). Keyed by whitespace-normalised (note, type, criterion text)."""
        self.source, self._index, self._ambiguous = source, {}, set()
        for r in records:
            key = (norm(r["note"]), r["criterion_type"], norm(r["criterion_text"]))
            if key in self._index and self._index[key] != r:
                if (self._index[key]["pred_label"], self._index[key]["pred_evidence"]) != (r["pred_label"], r["pred_evidence"]):
                    self._ambiguous.add(key)
                continue
            self._index.setdefault(key, r)
        self.n_records = len(records)
        self.lock = threading.Lock()

    @classmethod
    def from_results(cls, cfg: dict) -> "DemoBackend":
        from trialmatch.evaluation.val_data import load_joined  # val only; the test split is refused there
        last = None
        for rel in cfg["demo"]["predictions"]:
            p = resolve(rel)
            if p.exists():
                tag = p.name.replace("_val_predictions.jsonl", "")
                try:
                    return cls(load_joined(tag, "val"), source=rel)
                except Exception as e:  # missing val data etc.
                    last = e
        raise FileNotFoundError(f"no usable cached validation predictions ({last or 'files not found'})")

    def info(self) -> dict:
        return {"mode": "demo", "phase": "ready", "model_key": "demo", "model_name": "cached validation predictions",
                "device": "n/a (no inference)", "dtype": "n/a", "message": f"{self.n_records} cached VAL predictions "
                f"from {self.source}", "banner": DEMO_BANNER, "adapter": None, "models": [], "source": self.source}

    def engines(self) -> list[dict]:
        return [{"id": e, "label": ENGINE_LABELS[e], "available": False,
                 "reason": "Demo mode serves cached predictions; no engine runs.", "warning": None} for e in ENGINES]

    def ready(self) -> bool:
        return True

    def predict(self, note, criterion_type, criterion_text, engine="kv") -> dict:
        base = {"engine": "cached (demo)", "latency_s": None, "n_tokens": None, "banner": DEMO_BANNER}
        key = (norm(note), criterion_type, norm(criterion_text))
        if key in self._ambiguous:
            return {**base, "status": "unavailable", "label": None, "evidence_ids": None, "raw_output": None,
                    "parse_error": None, "message": "Ambiguous cached entry: cannot predict in demo mode, "
                                                    "switch to real mode."}
        r = self._index.get(key)
        if r is None:
            return {**base, "status": "unavailable", "label": None, "evidence_ids": None, "raw_output": None,
                    "parse_error": None, "message": "Cannot predict in demo mode: this note + criterion is not in the "
                                                    "cached validation predictions. Switch to real mode."}
        ok = r["status"] == "ok"
        return {**base, "status": "ok" if ok else "invalid", "label": r["pred_label"] if ok else None,
                "evidence_ids": list(r["pred_evidence"]) if ok else None, "raw_output": r["raw_output"],
                "parse_error": r.get("parse_error"), "message": None}


# ----------------------------------------------------------------------------------------------- real
def cpu_supports_bf16() -> bool:
    try:
        import torch
        for fn in ("_is_avx512_bf16_supported", "_is_amx_tile_supported"):
            f = getattr(torch.cpu, fn, None)
            if f is not None and f():
                return True
    except Exception:
        pass
    return False


def pick_runtime(cfg: dict):
    """-> (device str, torch dtype, human reason). CUDA fp16; CPU bf16 only with hardware support, else fp32."""
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.float16, "CUDA available: fp16"
    if cfg["real"].get("cpu_allow_bf16", True) and cpu_supports_bf16():
        return "cpu", torch.bfloat16, "CPU with bf16 hardware support: bf16"
    return "cpu", torch.float32, "CPU without bf16 support: fp32 (~16 GB for the 4B model)"


@dataclass
class LoadedModel:
    key: str
    model: object
    tok: object
    name: str
    device: str
    dtype: str
    load_s: float
    note: str = ""


def resolve_source(cfg: dict, merged_dir: str | None = None) -> dict | None:
    """Where the 4B model comes from; None if nothing local/configured exists (nothing is downloaded silently)."""
    real = cfg["real"]
    cand = merged_dir or os.environ.get("TRIALMATCH_MERGED_DIR") or real.get("merged_dir")
    if cand:
        return {"kind": "merged", "path": str(Path(cand).expanduser())}
    try:
        from trialmatch.model.load import default_merged_dir
        d = default_merged_dir()
        if (d / "config.json").exists():
            return {"kind": "merged", "path": str(d)}
    except Exception:
        pass
    adapter = os.environ.get("TRIALMATCH_ADAPTER") or real.get("adapter") or real.get("adapter_hub_id")
    if adapter:
        return {"kind": "adapter", "adapter": str(adapter)}
    return None


class RealBackend:
    mode = "real"

    def __init__(self, cfg: dict, merged_dir: str | None = None, loader=None, source: dict | None = None):
        self.cfg = cfg
        self.source = source if source is not None else resolve_source(cfg, merged_dir)
        self._loader = loader or self._default_loader
        self.gen = generation_settings()
        self.models: dict[str, LoadedModel] = {}
        self.loading: set[str] = set()
        self.errors: dict[str, str] = {}
        self.active = "merged"
        self.lock = threading.Lock()          # one inference at a time (screening, live benchmark, streaming)
        self._state = threading.Lock()
        self.runtime_note = ""
        self.merge_info = self._read_merge_info()

    # ---- loading
    def _read_merge_info(self) -> dict:
        if self.source and self.source["kind"] == "merged":
            p = Path(self.source["path"]) / "merge_info.json"
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    return {}
        return {}

    def _default_loader(self, key: str) -> LoadedModel:
        t0 = time.perf_counter()
        dev, dt, why = pick_runtime(self.cfg)
        self.runtime_note = why
        log.info("device/dtype choice: %s", why)
        if key == "merged":
            from trialmatch.model import load as L
            if self.source is None:
                raise FileNotFoundError("no model source: pass --merged-dir, set TRIALMATCH_MERGED_DIR, or configure "
                                        "real.adapter / real.adapter_hub_id in configs/app.yaml")
            if self.source["kind"] == "merged":
                model, tok = L.load_saved_merged(self.source["path"], device=dev, dtype=dt)
            else:
                model, tok, _ = L.load_merged(adapter=self.source["adapter"], device=dev, dtype=dt)
            name = "Qwen3-4B + LoRA (merged)"
            note = "fine-tuned"
        else:
            from trialmatch.inference.speculative.loading import load_draft
            from trialmatch.model.loading import load_tokenizer
            sm = self.cfg["real"]["small_model"]
            model = load_draft(sm, device=dev, dtype=dt)
            tok = load_tokenizer(sm, padding_side="left")
            name, note = sm.split("/")[-1] + " base", SMALL_NOTE
        return LoadedModel(key, model, tok, name, dev, str(dt).replace("torch.", ""), time.perf_counter() - t0, note)

    def start(self) -> None:
        """Load the target (merged 4B) in a background thread; with real.load_draft also load the draft (0.6B)
        in the SAME thread afterwards, so peak RAM is one load at a time and Speculative becomes available
        once both are ready. A draft failure is recorded but never blocks the target."""
        if not self.cfg["real"].get("load_draft"):
            self.load_model("merged")
            return
        with self._state:
            todo = [k for k in ("merged", "small") if k not in self.models and k not in self.loading]
            self.loading.update(todo)
            for k in todo:
                self.errors.pop(k, None)
        if todo:
            threading.Thread(target=lambda: [self._load_thread(k) for k in todo], daemon=True, name="load-all").start()

    def load_model(self, key: str) -> bool:
        """Start loading `key` in a background thread (idempotent). Returns False if already loaded/loading."""
        if key not in ("merged", "small"):
            raise ValueError(f"unknown model key {key!r}")
        with self._state:
            if key in self.models or key in self.loading:
                return False
            self.loading.add(key)
            self.errors.pop(key, None)
        threading.Thread(target=self._load_thread, args=(key,), daemon=True, name=f"load-{key}").start()
        return True

    def _load_thread(self, key: str) -> None:
        try:
            lm = self._loader(key)
            with self._state:
                self.models[key] = lm
            log.info("loaded %s in %.1fs on %s/%s", lm.name, lm.load_s, lm.device, lm.dtype)
        except BaseException as e:  # never kill the server because the model failed to load
            log.exception("model load failed: %s", key)
            with self._state:
                self.errors[key] = f"{type(e).__name__}: {e}"
        finally:
            with self._state:
                self.loading.discard(key)

    def select(self, key: str) -> None:
        """Make `key` the model used for screening (loads it first if needed)."""
        if key not in ("merged", "small"):
            raise ValueError(f"unknown model key {key!r}")
        self.active = key
        self.load_model(key)

    def wait(self, timeout: float = 60.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.active in self.models or self.active in self.errors:
                return self.active in self.models
            time.sleep(0.05)
        return False

    # ---- state
    def ready(self) -> bool:
        return self.active in self.models

    def phase(self) -> str:
        if self.active in self.models:
            return "ready"
        if self.active in self.loading:
            return "loading"
        return "error" if self.active in self.errors else "idle"

    def info(self) -> dict:
        lm = self.models.get(self.active)
        models = []
        for key, label in (("merged", "Merged Qwen3-4B (fine-tuned, default)"),
                           ("small", f"{self.cfg['real']['small_model'].split('/')[-1]} base ({SMALL_NOTE})")):
            m = self.models.get(key)
            models.append({"key": key, "label": label, "loaded": m is not None, "loading": key in self.loading,
                           "error": self.errors.get(key), "active": key == self.active,
                           "device": m.device if m else None, "dtype": m.dtype if m else None,
                           "load_s": round(m.load_s, 1) if m else None})
        adapter = self.merge_info.get("adapter") or (self.source or {}).get("adapter")
        tgt, drf = self.models.get("merged"), self.models.get("small")
        draft = {"loaded": drf is not None, "loading": "small" in self.loading, "error": self.errors.get("small"),
                 "name": drf.name if drf else None, "device": drf.device if drf else None,
                 "dtype": drf.dtype if drf else None, "requested": bool(self.cfg["real"].get("load_draft"))}
        target = {"loaded": tgt is not None, "loading": "merged" in self.loading, "error": self.errors.get("merged"),
                  "name": tgt.name if tgt else None, "device": tgt.device if tgt else None,
                  "dtype": tgt.dtype if tgt else None}
        return {"mode": "real", "target": target, "draft": draft, "phase": self.phase(), "model_key": self.active,
                "model_name": lm.name if lm else None, "model_note": lm.note if lm else "",
                "device": lm.device if lm else None, "dtype": lm.dtype if lm else None,
                "message": self.errors.get(self.active) or self.runtime_note, "banner": None,
                "adapter": adapter, "source": self.source, "models": models, "runtime_note": self.runtime_note}

    def engines(self) -> list[dict]:
        ready = self.ready()
        spec_ok = ready and self.active == "merged" and "small" in self.models
        out = []
        for e in ENGINES:
            avail, reason, warn = ready, None if ready else "Model not loaded yet.", None
            if e == "speculative":
                avail = spec_ok
                if not spec_ok:
                    if not ready:
                        reason = "Model not loaded yet."
                    elif self.active != "merged":
                        reason = "Speculative decoding needs the merged 4B as the active model (the 0.6B is only the draft)."
                    elif "small" in self.loading:
                        reason = "The 0.6B draft model is still loading."
                    elif self.errors.get("small"):
                        reason = "The 0.6B draft model failed to load: " + self.errors["small"][:160]
                    else:
                        reason = ("The 0.6B draft model is not loaded. Start with --load-draft (or set real.load_draft: true), "
                                  "or select the 0.6B once in the model selector, then switch back to the merged 4B.")
                warn = "On CPU speculative decoding may be slower than plain KV-cache decoding."
            out.append({"id": e, "label": ENGINE_LABELS[e], "available": avail, "reason": reason, "warning": warn})
        return out

    # ---- inference
    def encode(self, lm: LoadedModel, prompt: str):
        from trialmatch.model.chat import chat_prompt_text
        text = chat_prompt_text(lm.tok, prompt)
        return lm.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(lm.model.device)

    def predict(self, note, criterion_type, criterion_text, engine="kv") -> dict:
        from trialmatch.model import schema
        from trialmatch.model.parse import parse_output
        from trialmatch.model.prompt import build_prompt
        if not self.ready():
            raise RuntimeError("The model is not loaded yet.")
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r}")
        lm = self.models[self.active]
        if not self.lock.acquire(blocking=False):
            raise BackendBusy("The model is busy with another request.")
        try:
            prompt = build_prompt(note=note, criterion_type=criterion_type, criterion_text=criterion_text)
            ids = self.encode(lm, prompt)
            t0 = time.perf_counter()
            out = run_engine(engine, lm, self.models.get("small") if self.active == "merged" else None, ids,
                             max_new=self.gen["max_new_tokens"], k=self.cfg["real"]["speculative_k"])
            latency = time.perf_counter() - t0
        finally:
            self.lock.release()
        text = lm.tok.decode(out["tokens"], skip_special_tokens=True)
        pred = parse_output(text, criterion_type, schema.count_note_sentences(note))
        return {"status": "ok" if pred.parse_ok else "invalid", "label": pred.label, "evidence_ids": pred.evidence_ids,
                "raw_output": text, "parse_error": pred.error, "engine": engine, "latency_s": round(latency, 3),
                "n_tokens": len(out["tokens"]), "message": None, "banner": None,
                "prefill_s": out["prefill_s"], "decode_s": out["decode_s"]}


class TapModel:
    """Transparent wrapper: calls the model unchanged and reports every forward pass to `hook(kwargs, output)`.
    Lets the Inference Lab stream tokens without touching the decoders."""

    def __init__(self, model, hook):
        self._m, self._hook = model, hook

    def __call__(self, *a, **kw):
        out = self._m(*a, **kw)
        self._hook(kw, out)
        return out

    def __getattr__(self, name):
        return getattr(self._m, name)


def run_engine(engine: str, target: LoadedModel, draft: LoadedModel | None, input_ids, *, max_new: int, k: int = 4,
               target_hook=None, draft_hook=None) -> dict:
    """Run one of the project's greedy decoders. Returns {'tokens','prefill_s','decode_s','spec'(stats or None)}."""
    from trialmatch.model.chat import end_token_id
    eos = end_token_id(target.tok)
    pad = target.tok.pad_token_id
    tm = TapModel(target.model, target_hook) if target_hook else target.model
    if engine == "plain":
        from trialmatch.inference.kv_cache.reference_decode import reference_greedy_decode
        r = reference_greedy_decode(tm, input_ids, None, max_new_tokens=max_new, eos_token_id=eos, pad_token_id=pad)
        return {"tokens": r.sequences[0], "prefill_s": r.prefill_s, "decode_s": r.decode_s, "spec": None}
    if engine == "kv":
        from trialmatch.inference.kv_cache.cached_decode import cached_greedy_decode
        r = cached_greedy_decode(tm, input_ids, None, max_new_tokens=max_new, eos_token_id=eos, pad_token_id=pad)
        return {"tokens": r.sequences[0], "prefill_s": r.prefill_s, "decode_s": r.decode_s, "spec": None}
    if engine == "speculative":
        if draft is None:
            raise RuntimeError("speculative decoding needs the draft model to be loaded")
        from trialmatch.inference.speculative.speculative_decode import speculative_greedy_decode
        from trialmatch.inference.speculative.stats import aggregate
        dm = TapModel(draft.model, draft_hook) if draft_hook else draft.model
        r = speculative_greedy_decode(tm, dm, input_ids, max_new_tokens=max_new, k=k, eos_token_id=eos)
        return {"tokens": list(r.tokens), "prefill_s": r.prefill_s, "decode_s": r.decode_s, "spec": aggregate(r)}
    raise ValueError(f"unknown engine {engine!r}")
