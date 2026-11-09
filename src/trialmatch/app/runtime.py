"""Runtime: chooses the active backend (real / demo / auto) and exposes a status dict. No web imports."""
import logging
import threading

from trialmatch.app.backends import DemoBackend, RealBackend
from trialmatch.app.config import DEMO_BANNER, DISCLAIMER, generation_settings
from trialmatch.app.examples import ExampleStore
from trialmatch.app.label_badges import all_badges

log = logging.getLogger("trialmatch.app")


class Runtime:
    def __init__(self, cfg: dict, requested: str = "auto", merged_dir: str | None = None, real=None, demo=None,
                 examples: ExampleStore | None = None):
        if requested not in ("real", "demo", "auto"):
            raise ValueError("mode must be real, demo or auto")
        self.cfg, self.requested, self.merged_dir = cfg, requested, merged_dir
        self.real, self.demo, self._examples = real, demo, examples
        self.backend = None
        self.fallback_reason = None
        self.startup_error = None

    @classmethod
    def with_backend(cls, cfg: dict, backend, examples: ExampleStore | None = None) -> "Runtime":
        """Pre-built backend (tests, embedding). No model loading is started."""
        rt = cls(cfg, "real" if backend.mode == "real" else "demo", examples=examples)
        rt.backend = backend
        return rt

    # ---- startup
    def start(self) -> None:
        if self.requested == "demo":
            return self._use_demo("demo mode requested")
        self.real = self.real or RealBackend(self.cfg, self.merged_dir)
        if self.requested == "auto" and self.real.source is None:
            return self._use_demo("auto mode: no model source configured (no merged dir / adapter), using demo")
        self.backend = self.real
        self.real.start()
        if self.requested == "auto":
            threading.Thread(target=self._watch, daemon=True, name="auto-fallback").start()

    def _watch(self):
        self.real.wait(timeout=24 * 3600)
        if not self.real.ready():
            self._use_demo(f"auto mode: the model did not load ({self.real.errors.get('merged', 'unknown error')}), using demo")

    def _use_demo(self, why: str) -> None:
        try:
            self.demo = self.demo or DemoBackend.from_results(self.cfg)
            self.backend, self.fallback_reason = self.demo, why
            log.info(why)
        except Exception as e:
            self.startup_error = f"Demo predictions could not be loaded: {e}"
            self.fallback_reason = why

    # ---- views
    @property
    def examples(self) -> ExampleStore:
        if self._examples is None:
            self._examples = ExampleStore.safe_from_val()
        return self._examples

    def status(self) -> dict:
        if self.backend is None:
            info = {"mode": self.requested, "phase": "error", "model_name": None, "device": None, "dtype": None,
                    "message": self.startup_error or "Starting...", "banner": None, "models": [], "adapter": None}
            engines = []
        else:
            info, engines = self.backend.info(), self.backend.engines()
        return {**info, "requested_mode": self.requested, "fallback_reason": self.fallback_reason, "engines": engines,
                "limits": self.cfg["limits"], "generation": generation_settings(), "disclaimer": DISCLAIMER,
                "demo_banner": DEMO_BANNER, "badges": all_badges(),
                "lab": {"default_new_tokens": self.cfg["lab"]["default_new_tokens"],
                        "max_new_tokens": self.cfg["lab"]["max_new_tokens"],
                        "default_k": self.cfg["real"]["speculative_k"]},
                "live_available": bool(self.backend is not None and hasattr(self.backend, "models"))}
