"""App configuration (configs/app.yaml) and a few shared constants. No web imports."""
import copy
from functools import lru_cache
from pathlib import Path

import yaml

from trialmatch.utils.common import PROJECT_ROOT

APP_CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"

DISCLAIMER = ("Educational Prototype Notice: this is a research screening prototype, "
              "not a clinical enrollment decision or diagnostic tool.")
DEMO_BANNER = "DEMO MODE: cached validation prediction, not live inference"
ENGINES = ("plain", "kv", "speculative")
ENGINE_LABELS = {"plain": "Plain (no cache)", "kv": "KV-cache", "speculative": "Speculative (draft + target)"}


@lru_cache(maxsize=None)
def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_app_config(path: Path | None = None) -> dict:
    """Fresh copy, so callers (and tests) may modify it."""
    return copy.deepcopy(_load(Path(path) if path else APP_CONFIG_PATH))


def generation_settings() -> dict:
    """Decoding settings come from configs/train.yaml so the app can never drift from training/eval."""
    train = _load(PROJECT_ROOT / "configs" / "train.yaml")
    g = train["generation"]
    return {"greedy": not g.get("do_sample", False), "max_new_tokens": int(g["max_new_tokens"])}
