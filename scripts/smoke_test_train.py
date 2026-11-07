"""Smoke test: a few optimiser steps, then seconds/step, peak GPU memory and a 3-epoch projection.

  python scripts/smoke_test_train.py [--steps 15] [--model ...] [--quantization ...]
Writes no checkpoints. Projection is a ROUGH estimate (model load time excluded; validation passes
estimated at one third of the per-example training cost).
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WARMUP_STEPS_IGNORED = 2


def project(step_seconds: list[float], n_train: int, n_val: int, cfg: dict) -> dict:
    """Pure function (tested): projected minutes for the full run from measured step times."""
    t = cfg["train"]
    steady = step_seconds[WARMUP_STEPS_IGNORED:] or step_seconds
    sec_step = sum(steady) / len(steady)
    eff_batch = t["per_device_batch_size"] * t["grad_accum_steps"]
    total_steps = math.ceil(n_train / eff_batch) * t["num_epochs"]
    train_min = sec_step * total_steps / 60
    eval_min = t["num_epochs"] * n_val * (sec_step / eff_batch) / 3 / 60
    total = train_min + eval_min
    return {"sec_per_step": sec_step, "total_steps": total_steps, "train_min": train_min,
            "eval_min": eval_min, "total_min": total, "within_rule": total <= cfg["rule"]["max_projected_minutes"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--model")
    ap.add_argument("--quantization", choices=["qlora_4bit", "lora_fp16"])
    a = ap.parse_args()

    import torch
    from transformers import TrainerCallback

    from trialmatch.training.config import load_train_config
    from trialmatch.training.train import run_training

    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found: this smoke test must run on the Colab T4.")
    cfg = load_train_config(a.model, a.quantization)

    class StepTimer(TrainerCallback):
        def __init__(self):
            self.times, self.t0 = [], None

        def on_step_begin(self, args, state, control, **kw):
            torch.cuda.synchronize()
            self.t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kw):
            torch.cuda.synchronize()
            self.times.append(time.perf_counter() - self.t0)

    timer = StepTimer()
    torch.cuda.reset_peak_memory_stats()
    info = run_training(cfg, smoke_steps=a.steps, extra_callbacks=[timer])
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    p = project(timer.times, info["n_train"], info["n_val"], cfg)

    mid, quant = cfg["model"]["id"], cfg["quantization"]
    print("\n===== SMOKE TEST RESULT =====")
    print(f"model {mid} | {quant} | fp16={cfg['train']['fp16']} | grad ckpt={cfg['train']['gradient_checkpointing']}")
    print(f"steps timed: {len(timer.times)} (first {WARMUP_STEPS_IGNORED} ignored); "
          f"one step = {cfg['train']['per_device_batch_size']}x{cfg['train']['grad_accum_steps']} examples")
    print(f"seconds per step: {p['sec_per_step']:.2f}")
    print(f"peak GPU memory (allocated): {peak:.2f} GiB of {total_gib:.1f} GiB")
    print(f"projected {cfg['train']['num_epochs']}-epoch run ({p['total_steps']} steps): "
          f"train {p['train_min']:.0f} min + val ~{p['eval_min']:.1f} min = ~{p['total_min']:.0f} min (ESTIMATE)")
    limit = cfg["rule"]["max_projected_minutes"]
    if p["within_rule"]:
        print(f"RECOMMENDATION: KEEP {mid} (projected <= {limit} min).")
    elif mid != cfg["model"]["fallback_id"]:
        print(f"RECOMMENDATION: SWITCH to {cfg['model']['fallback_id']} (projected > {limit} min). "
              f"Re-run: python scripts/smoke_test_train.py --model {cfg['model']['fallback_id']}, "
              f"then python scripts/train.py --model {cfg['model']['fallback_id']}")
    else:
        print(f"WARNING: even the fallback {mid} is projected > {limit} min; reduce epochs or batch settings.")
    if peak > 0.92 * total_gib:
        print("WARNING: peak memory is close to the GPU limit; lower per_device_batch_size (raise grad_accum_steps).")


if __name__ == "__main__":
    main()
