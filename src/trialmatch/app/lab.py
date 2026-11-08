"""Inference Lab: saved benchmark loader, live mini-benchmark / token stream, tiled-attention demo.

Saved results are read from the files named in configs/app.yaml (benchmarks.*): nothing is hard-coded here.
Live runs use the project's own decoders (via backends.run_engine) on whatever device is present.
"""
import json
import math
import re
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from trialmatch.app.config import ENGINES
from trialmatch.utils.common import resolve

MIB = 2 ** 20


# ------------------------------------------------------------------------------------------ saved results
def _read(rel):
    if not rel:
        return None, "no file configured (set it under benchmarks: in configs/app.yaml)"
    p = resolve(rel)
    if not p.exists():
        return None, f"file not found: {rel}"
    try:
        return json.loads(p.read_text(encoding="utf-8")), None
    except Exception as e:
        return None, f"could not read {rel}: {type(e).__name__}"


def _provenance(data: dict, rel: str) -> dict:
    env, args = data.get("environment", {}), data.get("args", {})
    m = re.search(r"_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", rel or "")
    date = f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}" if m else None
    n = env.get("n_params") or 0
    model = "merged Qwen3-4B" if 3.5e9 < n < 4.5e9 else (f"merged {n / 1e9:.1f}B-param model" if n else "model unknown")
    colab = str(args.get("merged_dir", "")).startswith("/content")
    gpu, dtype = env.get("gpu") or env.get("device"), str(env.get("dtype", "")).replace("torch.", "")
    commit = env.get("git_commit")
    label = f"measured on {'Colab ' if colab else ''}{gpu}, {dtype}, {model}, {date or 'date unknown'}, git {(commit or 'unknown')[:7]}"
    return {"label": label, "gpu": gpu, "dtype": dtype, "model": model, "date": date, "git_commit": commit,
            "file": rel, "draft": env.get("draft"), "repeats": args.get("repeats")}


def _med(d, key="median"):
    return d.get(key) if isinstance(d, dict) else None


def _kv(data: dict, rel: str) -> dict:
    cases, mism = [], []
    for c in data.get("cases", []):
        row = {"source": c.get("source"), "input_len": c["input_len"], "out_len": c["target_new_tokens"],
               "tokens_equal": c.get("tokens_equal"), "mismatch_report": c.get("mismatch_report")}
        for name, dec in c.get("decoders", {}).items():
            ms = _med(dec.get("decode_ms_per_token"))
            row[name] = {"prefill_s": _med(dec.get("prefill_s")), "tok_s": _med(dec.get("tokens_per_s")),
                         "decode_tok_s": 1000.0 / ms if ms else None,
                         "mem_delta_mib": (_med(dec.get("peak_mem_delta_bytes")) or 0) / MIB if dec.get("peak_mem_delta_bytes") else None}
        cases.append(row)
        if c.get("tokens_equal") is False:
            mism.append({"input_len": row["input_len"], "out_len": row["out_len"], "source": row["source"],
                         "detail": c.get("mismatch_report")})
    return {"available": True, "provenance": _provenance(data, rel), "cases": cases,
            "equality": {"all_equal": data.get("all_tokens_equal"), "n_cases": len(cases),
                         "n_equal": sum(1 for c in cases if c["tokens_equal"]), "mismatches": mism}}


def _spec(data: dict, rel: str) -> dict:
    rows, mism = [], []
    for r in data.get("rows", []):
        n, dec_s = r.get("n_new_tokens"), r.get("decode_s_median")
        row = {"source": r.get("source"), "input_len": r["input_len"], "out_len": r["target_new_tokens"],
               "decoder": r["decoder"], "k": r.get("k"), "prefill_s": r.get("prefill_s_median"),
               "total_s": r.get("total_s_median"), "tok_s": r.get("tokens_per_s_median"),
               "decode_tok_s": (n - 1) / dec_s if n and dec_s else None,
               "peak_mem_mib": (r.get("peak_mem_bytes_median") or 0) / MIB or None,
               "acceptance_rate": r.get("acceptance_rate"), "mean_tokens_per_verify": r.get("mean_tokens_per_verify"),
               "speedup": r.get("speedup_vs_plain"), "tokens_equal": r.get("tokens_equal")}
        rows.append(row)
        if r["decoder"] == "speculative" and r.get("tokens_equal") is False:
            mism.append({"input_len": row["input_len"], "out_len": row["out_len"], "k": row["k"],
                         "source": row["source"], "detail": r.get("mismatch_report")})
    spec_rows = [r for r in rows if r["decoder"] == "speculative"]
    return {"available": True, "provenance": _provenance(data, rel), "rows": rows,
            "equality": {"all_equal": data.get("all_tokens_equal"), "n_rows": len(spec_rows),
                         "n_equal": sum(1 for r in spec_rows if r["tokens_equal"]), "mismatches": mism}}


def _attn(data: dict, rel: str) -> dict:
    res = [{"seq_len": r["seq_len"], "impl": r["impl"], "status": r.get("status"),
            "time_ms": (r.get("time_s", {}).get("mean") or 0) * 1e3 or None,
            "peak_mem_mib": r["peak_mem_bytes"] / MIB if "peak_mem_bytes" in r else None,
            "max_abs_err": r.get("max_abs_err"), "within_tolerance": r.get("within_tolerance")}
           for r in data.get("results", [])]
    return {"available": True, "provenance": _provenance(data, rel), "results": res,
            "mismatches": data.get("mismatches")}


def load_saved(cfg: dict) -> dict:
    out = {}
    for key, fn in (("kv_cache", _kv), ("speculative", _spec), ("attention", _attn)):
        rel = (cfg.get("benchmarks") or {}).get(key)
        data, why = _read(rel)
        try:
            out[key] = fn(data, rel) if data is not None else {"available": False, "reason": why}
        except Exception as e:
            out[key] = {"available": False, "reason": f"unexpected file layout in {rel} ({type(e).__name__})"}
    return out


# ------------------------------------------------------------------------------------------ tiled attention
def attention_demo(seq: int = 32, block: int = 8, row: int = 20, causal: bool = True, head_dim: int = 16,
                   seed: int = 0) -> dict:
    """One head: reference vs the project's tiled attention, the block grid, and the online-softmax trace of one row."""
    try:
        import torch

        from trialmatch.inference.attention import reference_attention, tiled_attention
    except ImportError as e:
        raise RuntimeError("PyTorch is not installed; the tiled attention demo needs it.") from e
    if not 4 <= seq <= 128:
        raise ValueError("sequence length must be between 4 and 128")
    if not 1 <= block <= seq:
        raise ValueError("block size must be between 1 and the sequence length")
    if not 0 <= row < seq:
        raise ValueError("query row must be inside the sequence")
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(1, 1, seq, head_dim, generator=g) for _ in range(3))
    t0 = time.perf_counter()
    ref = reference_attention(q, k, v, causal=causal)
    t1 = time.perf_counter()
    til = tiled_attention(q, k, v, causal=causal, block_q=block, block_k=block)
    t2 = time.perf_counter()
    nb = math.ceil(seq / block)
    # Mirrors tiled.py: a key block is skipped for a query block when it starts after the last visible key of any
    # query in the block (ks > qe - 1); it is 'partial' when it straddles the diagonal (ke - 1 > qs).
    grid = []
    for qb in range(nb):
        qs, qe = qb * block, min(qb * block + block, seq)
        line = []
        for kb in range(nb):
            ks, ke = kb * block, min(kb * block + block, seq)
            line.append("skipped" if causal and ks > qe - 1 else ("partial" if causal and ke - 1 > qs else "full"))
        grid.append(line)
    qb_row, scale = row // block, 1.0 / math.sqrt(head_dim)
    m, l = torch.tensor(float("-inf")), torch.tensor(0.0)
    o = torch.zeros(head_dim)
    steps = []
    for kb in range(nb):
        ks, ke = kb * block, min(kb * block + block, seq)
        if grid[qb_row][kb] == "skipped":
            steps.append({"key_block": kb, "status": "skipped"})
            continue
        s = (k[0, 0, ks:ke] @ q[0, 0, row]) * scale
        if causal:
            s = s.masked_fill(torch.arange(ks, ke) > row, float("-inf"))
        m_new = torch.maximum(m, s.max())
        m_ref = torch.zeros(()) if torch.isinf(m_new) else m_new
        alpha = torch.exp(m - m_ref)
        p = torch.exp(s - m_ref)
        l = alpha * l + p.sum()
        o = alpha * o + p @ v[0, 0, ks:ke]
        m = m_new
        fin = lambda x: None if not math.isfinite(float(x)) else float(x)
        steps.append({"key_block": kb, "status": grid[qb_row][kb], "visible_keys": int((~torch.isinf(s)).sum()),
                      "running_max": fin(m), "alpha": float(alpha), "normalizer": float(l)})
    trace_out = o / l.clamp_min(1e-30)
    return {"seq": seq, "block": block, "row": row, "causal": causal, "n_blocks": nb, "grid": grid,
            "query_block": qb_row, "steps": steps,
            "max_abs_err": float((ref - til).abs().max()), "trace_vs_tiled_err": float((trace_out - til[0, 0, row]).abs().max()),
            "n_skipped": sum(c == "skipped" for r in grid for c in r), "n_blocks_total": nb * nb,
            "ms_reference": (t1 - t0) * 1e3, "ms_tiled": (t2 - t1) * 1e3, "head_dim": head_dim}


# ------------------------------------------------------------------------------------------ live runs
class RssSampler:
    """Peak process RSS while a block runs (psutil optional)."""

    def __init__(self):
        try:
            import psutil
            self._p = psutil.Process()
        except Exception:
            self._p = None
        self.peak = self.start = 0
        self._stop = threading.Event()

    def __enter__(self):
        if self._p:
            self.start = self.peak = self._p.memory_info().rss
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def _loop(self):
        while not self._stop.wait(0.02):
            try:
                self.peak = max(self.peak, self._p.memory_info().rss)
            except Exception:
                return

    def __exit__(self, *a):
        self._stop.set()
        if self._p:
            self._t.join(1)
            self.peak = max(self.peak, self._p.memory_info().rss)

    def result(self):
        return None if not self._p else {"kind": "rss", "peak_mib": self.peak / MIB, "delta_mib": (self.peak - self.start) / MIB}


class LabJob:
    def __init__(self, kind: str, params: dict):
        self.id, self.kind, self.params = uuid.uuid4().hex[:8], kind, params
        self.status, self.events, self.results = "running", [], {}
        self.label = ""
        self.cancel_flag = threading.Event()
        self._lock = threading.Lock()

    def emit(self, type_: str, **data):
        with self._lock:
            self.events.append({"seq": len(self.events), "type": type_, **data})

    def since(self, n: int) -> list[dict]:
        with self._lock:
            return self.events[n:]

    def public(self) -> dict:
        return {"id": self.id, "kind": self.kind, "status": self.status, "label": self.label,
                "results": self.results, "n_events": len(self.events)}


class LabManager:
    def __init__(self, keep: int = 10):
        self.jobs: "OrderedDict[str, LabJob]" = OrderedDict()
        self.keep = keep
        self._lock = threading.Lock()

    def active(self):
        return next((j for j in self.jobs.values() if j.status == "running"), None)

    def add(self, job: LabJob):
        with self._lock:
            self.jobs[job.id] = job
            while len(self.jobs) > self.keep:
                self.jobs.popitem(last=False)

    def get(self, job_id):
        return self.jobs.get(job_id)


def _tap_factory(job, name, lm, t0, max_new, eos, k_state, record):
    """Hook for the TARGET model: turns each forward into streamed tokens (accepted / corrected / bonus)."""
    st = {"n": 0, "done": False}

    def push(tok_id, kind):
        if st["done"] or st["n"] >= max_new:
            return
        st["n"] += 1
        record.append(int(tok_id))
        if int(tok_id) == eos:
            st["done"] = True
        job.emit("token", engine=name, id=int(tok_id), text=lm.tok.decode([int(tok_id)], skip_special_tokens=False),
                 kind=kind, t=round(time.perf_counter() - t0, 4), n=st["n"])

    def hook(kw, out):
        logits, ids = out.logits, kw.get("input_ids")
        if name != "speculative":
            push(logits[0, -1].argmax(), "target")
        elif kw.get("past_key_values") is None:
            push(logits[0, -1].argmax(), "target")                      # prefill: first token
        else:
            g = logits[0].argmax(-1).tolist()
            cand = ids[0, 1:].tolist()
            m = 0
            while m < len(cand) and cand[m] == g[m]:
                m += 1
            for c in cand[:m]:
                push(c, "accepted")
            push(g[m], "corrected" if m < len(cand) else "bonus")
    return hook


def run_lab_job(job: LabJob, backend, prompt: dict, engines: list[str], model_key: str, max_new: int, k: int,
                stream: bool) -> None:
    """Worker: run the selected engines one after another (so timings are not distorted by contention)."""
    from trialmatch.app.backends import run_engine
    from trialmatch.model.prompt import build_prompt
    try:
        lm = backend.models[model_key]
        draft = backend.models.get("small") if model_key == "merged" else None
        ids = backend.encode(lm, build_prompt(note=prompt["note"], criterion_type=prompt["criterion_type"],
                                              criterion_text=prompt["criterion_text"]))
        job.label = f"live, {lm.device}, {lm.dtype}"
        job.results["meta"] = {"label": job.label, "model": lm.name, "prompt_tokens": int(ids.shape[1]),
                               "max_new_tokens": max_new, "k": k, "stream": stream}
        if not backend.lock.acquire(blocking=False):
            raise RuntimeError("The model is busy with a screening run. Try again when it finishes.")
        try:
            order = [e for e in ("plain", "kv", "speculative") if e in engines]
            import torch
            from trialmatch.model.chat import end_token_id
            eos = end_token_id(lm.tok)
            for e in order:
                if job.cancel_flag.is_set():
                    break
                job.emit("engine_start", engine=e)
                streamed: list[int] = []
                t0 = time.perf_counter()
                hook = _tap_factory(job, e, lm, t0, max_new, eos, None, streamed) if stream else None
                cuda = lm.device == "cuda" and torch.cuda.is_available()
                if cuda:
                    from trialmatch.inference.benchmarks.memory import begin_run, end_run
                    base = begin_run("cuda")
                with RssSampler() as rss:
                    out = run_engine(e, lm, draft, ids, max_new=max_new, k=k, target_hook=hook)
                mem = ({"kind": "cuda", "peak_mib": end_run("cuda") / MIB, "delta_mib": (end_run("cuda") - base) / MIB}
                       if cuda else rss.result())
                n, dec = len(out["tokens"]), out["decode_s"]
                total = out["prefill_s"] + dec
                res = {"engine": e, "n_tokens": n, "prefill_s": out["prefill_s"], "decode_s": dec, "total_s": total,
                       "decode_tok_s": (n - 1) / dec if dec > 0 and n > 1 else None, "tok_s": n / total if total else None,
                       "memory": mem, "tokens": out["tokens"],
                       "text": lm.tok.decode(out["tokens"], skip_special_tokens=True),
                       "acceptance_rate": out["spec"]["acceptance_rate"] if out["spec"] else None,
                       "mean_tokens_per_verify": out["spec"]["mean_tokens_per_verify"] if out["spec"] else None,
                       "stream_consistent": (streamed == out["tokens"]) if stream else None}
                job.results[e] = res
                job.emit("engine_end", engine=e, **{x: y for x, y in res.items() if x != "tokens"})
            plain = job.results.get("plain")
            for e in order:
                r = job.results.get(e)
                if r:
                    r["tokens_equal_to_plain"] = None if plain is None else r["tokens"] == plain["tokens"]
                    r["speedup_vs_plain"] = (plain["total_s"] / r["total_s"]) if plain and r["total_s"] else None
            job.emit("summary", results={e: {x: y for x, y in job.results[e].items() if x != "tokens"}
                                         for e in order if e in job.results})
        finally:
            backend.lock.release()
        job.status = "cancelled" if job.cancel_flag.is_set() else "done"
        job.emit("done", status=job.status)
    except BaseException as e:
        job.status = "failed"
        job.emit("error", message=f"The run failed ({type(e).__name__}: {str(e)[:200]}).")
        job.emit("done", status="failed")
