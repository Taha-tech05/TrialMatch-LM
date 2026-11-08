"""FastAPI application. All errors are structured JSON {"error": {"code", "message", "details"}} in plain language."""
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from trialmatch.app import lab as labmod
from trialmatch.app.config import ENGINES
from trialmatch.app.criteria_parser import parse_criteria
from trialmatch.app.jobs import JobManager, ScreeningJob
from trialmatch.app.report import render_html, render_markdown
from trialmatch.app.runtime import Runtime
from trialmatch.app.sentence_splitter import prepare_note
from trialmatch.app.validation import InputError, check_text, decode_bytes, validate_screening

log = logging.getLogger("trialmatch.app")
STATIC = Path(__file__).parent / "static"


def err(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "details": details or []}}, status_code=status)


async def body(request: Request) -> dict:
    try:
        data = await request.json()
    except Exception:
        raise ApiError(400, "bad_request", "The request was not valid UTF-8 JSON.")
    if not isinstance(data, dict):
        raise ApiError(400, "bad_request", "The request body must be a JSON object.")
    return data


class ApiError(Exception):
    def __init__(self, status, code, message, details=None):
        self.status, self.code, self.message, self.details = status, code, message, details


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="TrialMatch screening prototype", docs_url=None, redoc_url=None)
    jobs, labs = JobManager(), labmod.LabManager()
    cfg = runtime.cfg

    # ------------------------------------------------------------------ errors
    @app.exception_handler(ApiError)
    async def _api(_, e: ApiError):
        return err(e.status, e.code, e.message, e.details)

    @app.exception_handler(RequestValidationError)
    async def _val(_, e):
        return err(422, "bad_request", "The request was malformed. Check the inputs and try again.")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_, e):
        return err(e.status_code, "not_found" if e.status_code == 404 else "http_error",
                   "Not found." if e.status_code == 404 else "The request could not be handled.")

    @app.exception_handler(Exception)
    async def _any(_, e):
        log.exception("unhandled error")
        return err(500, "internal_error", "Something went wrong on the server. The app is still running; try again.")

    # ------------------------------------------------------------------ pages / status
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/status")
    def status():
        return runtime.status()

    @app.get("/api/examples")
    def examples():
        return runtime.examples.public()

    @app.post("/api/note/prepare")
    async def note_prepare(request: Request):
        d = await body(request)
        try:
            text = check_text(d.get("text"))
        except InputError as e:
            raise ApiError(400, "unreadable_text", str(e))
        prep = prepare_note(text)
        if not prep["sentences"]:
            raise ApiError(400, "empty_note", "The patient note is empty. Load an example or paste a note.")
        return prep

    @app.post("/api/criteria/parse")
    async def criteria_parse(request: Request):
        d = await body(request)
        try:
            text = check_text(d.get("text"))
        except InputError as e:
            raise ApiError(400, "unreadable_text", str(e))
        return parse_criteria(text)

    @app.post("/api/upload")
    async def upload(request: Request):
        raw = await request.body()
        if len(raw) > 2_000_000:
            raise ApiError(413, "too_large", "The file is too large (limit 2 MB).")
        try:
            return {"text": decode_bytes(raw)}
        except InputError as e:
            raise ApiError(400, "unreadable_text", str(e))

    # ------------------------------------------------------------------ screening
    @app.post("/api/screen")
    async def screen(request: Request):
        d = await body(request)
        be = runtime.backend
        if be is None or not be.ready():
            raise ApiError(409, "model_loading", "The model is still loading. Wait for the header to show READY.")
        v = validate_screening(d.get("note"), d.get("criteria"), cfg["limits"])
        if not v["ok"]:
            raise ApiError(422, "invalid_input", v["errors"][0]["message"], v["errors"])
        engine = d.get("engine") or "kv"
        if be.mode == "real":
            ok = {e["id"]: e for e in be.engines()}
            if engine not in ok:
                raise ApiError(400, "bad_engine", f"Unknown engine '{engine}'.")
            if not ok[engine]["available"]:
                raise ApiError(400, "engine_unavailable", ok[engine]["reason"] or "That engine is not available.")
        if jobs.active():
            raise ApiError(409, "busy", "A screening run is already in progress. Cancel it or wait for it to finish.")
        job = jobs.submit(ScreeningJob(v, engine, be, cfg))
        return {"job_id": job.id, "warnings": v["warnings"]}

    def _job(job_id):
        j = jobs.get(job_id)
        if j is None:
            raise ApiError(404, "not_found", "That run no longer exists (it may have been reset).")
        return j

    @app.get("/api/jobs/{job_id}")
    def job_state(job_id: str):
        return _job(job_id).public()

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        _job(job_id).cancel_flag.set()
        return {"ok": True, "note": "Cancel takes effect after the criterion currently running."}

    @app.get("/api/jobs/{job_id}/export")
    def job_export(job_id: str, format: str = "json"):
        j = _job(job_id)
        if j.report is None:
            raise ApiError(409, "not_finished", "The run has not finished yet; export is available when it completes.")
        name = f"trial_screening_{j.id}"
        if format == "json":
            return Response(json.dumps(j.report, indent=2, ensure_ascii=False), media_type="application/json",
                            headers={"Content-Disposition": f'attachment; filename="{name}.json"'})
        if format == "md":
            return Response(render_markdown(j.report), media_type="text/markdown; charset=utf-8",
                            headers={"Content-Disposition": f'attachment; filename="{name}.md"'})
        if format in ("html", "print"):
            return Response(render_html(j.report), media_type="text/html; charset=utf-8")
        raise ApiError(400, "bad_format", "Export format must be json, html or md.")

    @app.post("/api/reset")
    def reset():
        jobs.reset()
        for j in list(labs.jobs.values()):
            j.cancel_flag.set()
        return {"ok": True}

    @app.post("/api/model")
    async def model_select(request: Request):
        d = await body(request)
        be = runtime.backend
        if be is None or be.mode != "real" or not hasattr(be, "select"):
            raise ApiError(400, "no_real_backend", "Model selection needs real mode.")
        if jobs.active():
            raise ApiError(409, "busy", "Finish or cancel the running screening first.")
        try:
            be.select(d.get("key"))
        except ValueError:
            raise ApiError(400, "bad_model", "Unknown model.")
        return runtime.status()

    @app.post("/api/model/load")
    async def model_load(request: Request):
        """Load a model (e.g. the 0.6B draft) in the background without making it the active one."""
        d = await body(request)
        be = runtime.backend
        if be is None or not hasattr(be, "load_model"):
            raise ApiError(400, "no_real_backend", "Loading models needs real mode.")
        try:
            be.load_model(d.get("key"))
        except ValueError:
            raise ApiError(400, "bad_model", "Unknown model.")
        return runtime.status()

    # ------------------------------------------------------------------ inference lab
    @app.get("/api/lab/saved")
    def lab_saved():
        return labmod.load_saved(cfg)

    @app.get("/api/lab/prompts")
    def lab_prompts():
        return {"prompts": runtime.examples.prompts()}

    @app.get("/api/lab/attention")
    def lab_attention(seq: int = 32, block: int = 8, row: int = 20, causal: bool = True, seed: int = 0):
        try:
            return labmod.attention_demo(seq, block, row, causal, cfg["lab"]["attention_head_dim"], seed)
        except ValueError as e:
            raise ApiError(400, "bad_params", str(e))
        except RuntimeError as e:
            raise ApiError(503, "no_torch", str(e))

    @app.post("/api/lab/run")
    async def lab_run(request: Request):
        d = await body(request)
        be = runtime.backend
        if be is None or not hasattr(be, "models"):
            raise ApiError(400, "needs_real", "Live runs need real mode with a loaded model. Demo mode serves cached predictions only.")
        key = d.get("model", "merged")
        if key not in be.models:
            raise ApiError(409, "model_not_loaded", "That model is not loaded yet. Load it from the model selector first.")
        engines = [e for e in (d.get("engines") or []) if e in ENGINES]
        if not engines:
            raise ApiError(400, "no_engines", "Select at least one engine.")
        if "speculative" in engines and (key != "merged" or "small" not in be.models):
            raise ApiError(400, "engine_unavailable", "Speculative needs the merged 4B target AND the 0.6B draft loaded.")
        try:
            max_new = max(1, min(int(d.get("max_new_tokens", cfg["lab"]["default_new_tokens"])), cfg["lab"]["max_new_tokens"]))
            k = max(1, min(int(d.get("k", cfg["real"]["speculative_k"])), 8))
        except (TypeError, ValueError):
            raise ApiError(400, "bad_params", "Output length and k must be whole numbers.")
        p = d.get("prompt") or {}
        if p.get("source") == "val":
            r = runtime.examples.by_id.get(p.get("annotation_id"))
            if r is None:
                raise ApiError(400, "bad_prompt", "Unknown validation prompt.")
            prompt = {"note": r["note"], "criterion_type": r["criterion_type"], "criterion_text": r["criterion_text"]}
        else:
            v = validate_screening(p.get("note"), [{"type": p.get("criterion_type"), "text": p.get("criterion_text")}],
                                   cfg["limits"])
            if not v["ok"]:
                raise ApiError(422, "invalid_input", v["errors"][0]["message"], v["errors"])
            prompt = {"note": v["note"], "criterion_type": v["criteria"][0]["type"], "criterion_text": v["criteria"][0]["text"]}
        if labs.active() or jobs.active():
            raise ApiError(409, "busy", "The model is busy with another run.")
        job = labmod.LabJob("stream" if d.get("kind") == "stream" else "bench", d)
        labs.add(job)
        threading.Thread(target=labmod.run_lab_job, daemon=True, name=f"lab-{job.id}",
                         args=(job, be, prompt, engines, key, max_new, k, job.kind == "stream")).start()
        return {"job_id": job.id}

    def _lab(job_id):
        j = labs.get(job_id)
        if j is None:
            raise ApiError(404, "not_found", "That run no longer exists.")
        return j

    @app.get("/api/lab/jobs/{job_id}")
    def lab_state(job_id: str):
        return _lab(job_id).public()

    @app.post("/api/lab/jobs/{job_id}/cancel")
    def lab_cancel(job_id: str):
        _lab(job_id).cancel_flag.set()
        return {"ok": True, "note": "Cancel takes effect between engines."}

    @app.get("/api/lab/jobs/{job_id}/events")
    def lab_events(job_id: str, request: Request):
        job = _lab(job_id)
        try:
            n = int(request.headers.get("last-event-id", "-1")) + 1
        except ValueError:
            n = 0

        def gen():
            nonlocal n
            last = time.time()
            while True:
                evs = job.since(n)
                for e in evs:
                    n += 1
                    yield f"id: {e['seq']}\ndata: {json.dumps(e)}\n\n"
                    if e["type"] == "done":
                        return
                if evs:
                    last = time.time()
                elif job.status != "running" and not job.since(n):
                    return
                elif time.time() - last > 15:
                    yield ": keep-alive\n\n"
                    last = time.time()
                time.sleep(0.05)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app
