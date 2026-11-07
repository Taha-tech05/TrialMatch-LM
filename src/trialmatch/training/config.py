"""Training/eval config loading (torch-free)."""
import copy
import os
import re
from pathlib import Path

from trialmatch.utils.common import PROJECT_ROOT, load_config

TRAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"
QUANT_MODES = ("qlora_4bit", "lora_fp16")
FORBIDDEN_LORA_TARGETS = ("lm_head", "embed_tokens")


def load_train_config(model_id: str | None = None, quantization: str | None = None,
                      num_epochs: int | None = None) -> dict:
    cfg = copy.deepcopy(load_config(TRAIN_CONFIG_PATH))
    if model_id:
        cfg["model"]["id"] = model_id
    if quantization:
        cfg["quantization"] = quantization
    if num_epochs:
        cfg["train"]["num_epochs"] = num_epochs
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    if cfg["quantization"] not in QUANT_MODES:
        raise ValueError(f"quantization must be one of {QUANT_MODES}, got {cfg['quantization']!r}")
    bad = [m for m in cfg["lora"]["target_modules"] if any(f in m for f in FORBIDDEN_LORA_TARGETS)]
    if bad:
        raise ValueError(f"LoRA must not target tied embeddings/lm_head: {bad}")


def output_root(cfg: dict) -> Path:
    """TRIALMATCH_OUT (if set) else <project>/<default_root>."""
    env = os.environ.get(cfg["output"]["env_var"])
    return Path(env).expanduser() if env else (PROJECT_ROOT / cfg["output"]["default_root"]).resolve()


def run_name(cfg: dict) -> str:
    slug = re.sub(r"[^a-z0-9.]+", "-", cfg["model"]["id"].split("/")[-1].lower())
    return f"{slug}_{cfg['quantization']}"


def run_paths(cfg: dict) -> dict[str, Path]:
    root = output_root(cfg)
    return {"root": root, "run_dir": root / "checkpoints" / run_name(cfg),
            "logs": root / "results" / "logs", "eval": root / "results" / "eval"}
