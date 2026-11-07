"""Torch-free logging helpers: JSONL + CSV step logs, non-finite loss guard, environment info."""
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

from trialmatch.utils.common import PROJECT_ROOT

CSV_FIELDS = ["time", "step", "epoch", "split", "loss", "learning_rate", "grad_norm"]


class NonFiniteLossError(RuntimeError):
    pass


def check_finite(kind: str, value, step) -> None:
    if value is None:
        return
    if not math.isfinite(float(value)):
        raise NonFiniteLossError(
            f"STOPPING: {kind} loss is {value} at step {step}. Training diverged (NaN/inf). "
            "Typical fixes: lower the learning rate, check fp16 overflow (try lora_fp16 vs qlora_4bit), "
            "or resume from the last good checkpoint. The log up to this step was kept.")


class StepLogger:
    """Appends records to <stem>.jsonl and (loss rows only) <stem>.csv. Appending makes resume safe."""

    def __init__(self, logs_dir: Path, stem: str):
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl, self.csv = logs_dir / f"{stem}.jsonl", logs_dir / f"{stem}.csv"
        if not self.csv.exists():
            with open(self.csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, CSV_FIELDS).writeheader()

    def write(self, record: dict) -> None:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if record.get("split") in ("train", "val"):
            with open(self.csv, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, CSV_FIELDS, extrasaction="ignore").writerow(record)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_info(cfg: dict) -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "git_commit": git_commit(),
            "config": cfg}
    for mod in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "total_memory_gib": round(p.total_memory / 1024 ** 3, 2),
                           "capability": list(torch.cuda.get_device_capability(0)),
                           "bf16_supported": bool(torch.cuda.is_bf16_supported())}
    except Exception:
        info["cuda_available"] = None
    return info
