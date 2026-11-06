"""Shared utilities."""
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


@lru_cache(maxsize=None)
def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Cached config load; callers must not mutate the result."""
    return _load(path)


def resolve(rel: str) -> Path:
    """Resolve a config path relative to the project root."""
    return PROJECT_ROOT / rel
