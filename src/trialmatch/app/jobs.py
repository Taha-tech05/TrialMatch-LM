"""Background screening jobs: one criterion at a time, progress polled by the UI, per-criterion error isolation."""
import functools
import threading
import time
import uuid
from collections import OrderedDict

from trialmatch.app.aggregate import aggregate
from trialmatch.app.backends import BackendBusy
from trialmatch.app.config import generation_settings
from trialmatch.app.explain import explain
from trialmatch.app.label_badges import badge_for
from trialmatch.app.report import build_report, save_report


@functools.lru_cache(maxsize=1)
def _git_commit():
    from trialmatch.training.logging_utils import git_commit
    return git_commit()


def call_with_timeout(fn, timeout_s: float):
    """Run fn in a daemon thread; TimeoutError if it does not finish. (The worker cannot be killed; the model lock
    keeps later requests from overlapping with it, and decoding is bounded by max_new_tokens anyway.)"""
    box = {}

    def target():
        try:
            box["v"] = fn()
        except BaseException as e:
            box["e"] = e
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"no result after {int(timeout_s)} s")
    if "e" in box:
        raise box["e"]
    return box["v"]


def build_result(index: int, crit: dict, rec: dict, sentences: list[dict]) -> dict:
    by_id = {s["id"]: s["text"] for s in sentences}
    status = rec.get("status", "error")
    ids = rec.get("evidence_ids") if status == "ok" else None
    cited = [{"id": i, "text": by_id[i]} for i in (ids or []) if i in by_id]
    ctype, label = crit["type"], rec.get("label") if status == "ok" else None
    err = rec.get("parse_error") or rec.get("message")
    return {
        "index": index, "criterion_type": ctype, "criterion_text": crit["text"], "status": status, "label": label,
        "evidence_ids": ids, "cited": cited, "badge": badge_for(ctype, label, status),
        "explanation": explain(ctype, label, status, cited, err),
        "missing_info": status == "ok" and label == "not enough information",
        "flags": {"invalid": status == "invalid", "error": status in ("error", "unavailable")},
        "raw_output": rec.get("raw_output"), "parse_error": rec.get("parse_error"), "message": rec.get("message"),
        "engine": rec.get("engine"), "latency_s": rec.get("latency_s"), "n_tokens": rec.get("n_tokens"),
        "banner": rec.get("banner"), "elapsed_s": rec.get("elapsed_s"),
    }


class ScreeningJob:
    def __init__(self, validated: dict, engine: str, backend, cfg: dict):
        self.id = uuid.uuid4().hex[:8]
        self.v, self.engine, self.backend, self.cfg = validated, engine, backend, cfg
        self.status = "queued"
        self.results: list[dict | None] = [None] * len(validated["criteria"])
        self.current = None
        self.t_start = self.t_end = None
        self.cancel_flag = threading.Event()
        self.run = None
        self.report = None
        self.report_paths = None
        self.save_error = None
        self.error = None
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._lock = threading.Lock()

    # ---- worker
    def start(self):
        threading.Thread(target=self._work, daemon=True, name=f"screen-{self.id}").start()

    def _work(self):
        self.status, self.t_start = "running", time.time()
        timeout = float(self.cfg["real"].get("criterion_timeout_s", 900))
        try:
            for i, c in enumerate(self.v["criteria"]):
                if self.cancel_flag.is_set():
                    break
                self.current = i
                t0 = time.time()
                if c["skip"]:
                    rec = {"status": "skipped", "message": "empty criterion text"}
                else:
                    rec = self._predict(c, timeout)
                rec["elapsed_s"] = round(time.time() - t0, 2)
                res = build_result(i, c, rec, self.v["sentences"])
                with self._lock:
                    self.results[i] = res
            self.current = None
            cancelled = self.cancel_flag.is_set()
            for i, c in enumerate(self.v["criteria"]):
                if self.results[i] is None:
                    self.results[i] = build_result(i, c, {"status": "skipped", "message": "run cancelled"},
                                                   self.v["sentences"])
                    self.results[i]["cancelled"] = True
            self._finish(cancelled)               # aggregate + report first, so "done" always has them
            self.status = "cancelled" if cancelled else "done"
        except BaseException as e:  # a bug here must not kill the server; the UI shows a plain message
            self.status, self.error = "failed", f"The run stopped unexpectedly ({type(e).__name__}). Press Reset and try again."
        finally:
            self.t_end = time.time()

    def _predict(self, c: dict, timeout: float) -> dict:
        deadline = time.time() + timeout
        while True:
            try:
                return call_with_timeout(
                    lambda: self.backend.predict(self.v["note"], c["type"], c["text"], self.engine),
                    max(1.0, deadline - time.time()))
            except BackendBusy:
                if time.time() > deadline:
                    return {"status": "error", "message": "the model stayed busy with another request"}
                time.sleep(1.0)
            except TimeoutError as e:
                return {"status": "error", "message": f"timed out: {e}", "engine": self.engine}
            except Exception as e:
                return {"status": "error", "message": f"generation failed ({type(e).__name__}: {str(e)[:200]})",
                        "engine": self.engine}

    def _finish(self, cancelled: bool):
        done = [r for r in self.results if not r.get("cancelled")]
        agg = aggregate(done)
        if cancelled:
            agg["notes"].append("The run was cancelled; this outcome covers only the criteria completed so far.")
        info = self.backend.info()
        self.run = {
            "run_id": self.id, "created_at": self.created_at, "mode": self.backend.mode,
            "backend": {"model_name": info.get("model_name"), "device": info.get("device"),
                        "dtype": info.get("dtype"), "adapter": info.get("adapter")},
            "engine": self.engine, "note": self.v["note"], "sentences": self.v["sentences"],
            "results": self.results, "aggregate": agg, "warnings": self.v["warnings"],
            "generation": generation_settings(), "git_commit": _git_commit(),
        }
        self.report = build_report(self.run)
        try:
            from trialmatch.utils.common import PROJECT_ROOT
            rd = self.cfg.get("reports_dir", "outputs/reports")
            from pathlib import Path
            rd = Path(rd) if Path(rd).is_absolute() else PROJECT_ROOT / rd
            self.report_paths = save_report(self.report, rd)
        except Exception as e:
            self.save_error = f"Could not save the report to disk ({type(e).__name__}); export from the browser instead."

    # ---- view
    def public(self) -> dict:
        with self._lock:
            results = [r if r is not None else {"index": i, "status": "pending",
                                                 "criterion_type": c["type"], "criterion_text": c["text"]}
                       for i, (r, c) in enumerate(zip(self.results, self.v["criteria"]))]
        n_done = sum(1 for r in results if r["status"] != "pending")
        elapsed = (self.t_end or time.time()) - self.t_start if self.t_start else 0.0
        times = [r["elapsed_s"] for r in results if r.get("elapsed_s") and r["status"] not in ("pending", "skipped")]
        remaining = sum(1 for c, r in zip(self.v["criteria"], results) if r["status"] == "pending" and not c["skip"])
        eta = (sum(times) / len(times)) * remaining if times and self.status == "running" else None
        return {"id": self.id, "status": self.status, "total": len(results), "done": n_done, "current": self.current,
                "elapsed_s": round(elapsed, 1), "eta_s": round(eta, 1) if eta is not None else None,
                "engine": self.engine, "mode": self.backend.mode, "results": results,
                "aggregate": self.run["aggregate"] if self.run else None, "warnings": self.v["warnings"],
                "note": self.v["note"], "sentences": self.v["sentences"], "error": self.error,
                "report_paths": self.report_paths, "save_error": self.save_error}


class JobManager:
    def __init__(self, keep: int = 20):
        self.jobs: "OrderedDict[str, ScreeningJob]" = OrderedDict()
        self.keep, self._lock = keep, threading.Lock()

    def active(self):
        with self._lock:
            return next((j for j in self.jobs.values() if j.status in ("queued", "running")), None)

    def submit(self, job: ScreeningJob) -> ScreeningJob:
        with self._lock:
            self.jobs[job.id] = job
            while len(self.jobs) > self.keep:
                self.jobs.popitem(last=False)
        job.start()
        return job

    def get(self, job_id: str):
        return self.jobs.get(job_id)

    def reset(self):
        for j in list(self.jobs.values()):
            j.cancel_flag.set()
        with self._lock:
            self.jobs.clear()
