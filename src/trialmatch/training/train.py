"""LoRA / QLoRA training with the plain transformers Trainer (needs torch, peft, transformers).

Features: fp16 on T4, optional gradient checkpointing, val loss every epoch, adapter checkpoint every
epoch, resume from the latest checkpoint, per-step JSONL+CSV logs, NaN/inf guard, environment log.
Never touches the test split (build_dataset only accepts train/val).
"""
import json
import math
from pathlib import Path

from trialmatch.model.loading import attach_lora, load_base_model, load_tokenizer
from trialmatch.training.config import run_name, run_paths
from trialmatch.training.dataset import build_dataset
from trialmatch.training.logging_utils import (NonFiniteLossError, StepLogger, check_finite,
                                               environment_info)


def make_collator(pad_id: int):
    import torch

    def collate(batch):
        n, m = len(batch), max(len(b["input_ids"]) for b in batch)
        ids = torch.full((n, m), pad_id, dtype=torch.long)
        att = torch.zeros((n, m), dtype=torch.long)
        lab = torch.full((n, m), -100, dtype=torch.long)
        for i, b in enumerate(batch):  # right padding
            k = len(b["input_ids"])
            ids[i, :k], att[i, :k], lab[i, :k] = (torch.tensor(b["input_ids"]), torch.tensor(b["attention_mask"]),
                                                  torch.tensor(b["labels"]))
        return {"input_ids": ids, "attention_mask": att, "labels": lab}
    return collate


def steps_per_epoch(n_train: int, cfg: dict) -> int:
    t = cfg["train"]
    return math.ceil(n_train / (t["per_device_batch_size"] * t["grad_accum_steps"]))


def make_training_args(cfg: dict, run_dir: Path, n_train: int, smoke_steps: int | None = None):
    from transformers import TrainingArguments
    t = cfg["train"]
    spe = steps_per_epoch(n_train, cfg)
    warmup = max(1, round(t["warmup_ratio"] * spe * t["num_epochs"]))
    kw = dict(
        output_dir=str(run_dir), per_device_train_batch_size=t["per_device_batch_size"],
        per_device_eval_batch_size=t["per_device_batch_size"], gradient_accumulation_steps=t["grad_accum_steps"],
        learning_rate=t["learning_rate"], num_train_epochs=t["num_epochs"], warmup_steps=warmup,
        weight_decay=t["weight_decay"], max_grad_norm=t["max_grad_norm"], lr_scheduler_type=t["lr_scheduler"],
        seed=t["seed"], data_seed=t["seed"], fp16=t["fp16"], optim="adamw_torch",
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps", logging_steps=t["logging_steps"],
        logging_nan_inf_filter=False,  # the default would silently replace NaN losses; the guard needs them
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=t["save_total_limit"],
        report_to="none", remove_unused_columns=False, label_names=["labels"], dataloader_num_workers=0,
    )
    if smoke_steps:  # short timing run: no eval, no checkpoints
        kw.update(max_steps=smoke_steps, eval_strategy="no", save_strategy="no", warmup_steps=1)
    return TrainingArguments(**kw)


def make_logging_callback(logger: StepLogger):
    from transformers import TrainerCallback

    class LossLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            base = {"step": state.global_step, "epoch": logs.get("epoch", state.epoch)}
            if "loss" in logs:
                logger.write({**base, "split": "train", "loss": logs["loss"],
                              "learning_rate": logs.get("learning_rate"), "grad_norm": logs.get("grad_norm")})
                check_finite("train", logs["loss"], state.global_step)
            if "eval_loss" in logs:
                logger.write({**base, "split": "val", "loss": logs["eval_loss"]})
                check_finite("validation", logs["eval_loss"], state.global_step)

    return LossLogger()


def latest_checkpoint(run_dir: Path):
    from transformers.trainer_utils import get_last_checkpoint
    return get_last_checkpoint(str(run_dir)) if run_dir.is_dir() else None


def run_training(cfg: dict, smoke_steps: int | None = None, extra_callbacks=(), model=None, tok=None,
                 resume: bool = True) -> dict:
    """Train (or smoke-test when smoke_steps is set). model/tok may be injected (used by offline tests)."""
    from transformers import Trainer

    paths = run_paths(cfg)
    name = run_name(cfg) + ("_smoke" if smoke_steps else "")
    max_len = cfg["model"]["max_seq_len"]

    tok = tok or load_tokenizer(cfg["model"]["id"])
    train_ds, val_ds = build_dataset(tok, "train", max_len), build_dataset(tok, "val", max_len)
    longest = max(len(e["input_ids"]) for e in train_ds + val_ds)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} longest={longest} tokens (max_seq_len {max_len}, no truncation)")

    logger = StepLogger(paths["logs"], f"train_{name}")
    env = environment_info(cfg)
    logger.write({"type": "env", **env})
    (paths["logs"] / f"env_{name}.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    print(f"[env] gpu={env.get('gpu')} torch={env['torch']} transformers={env['transformers']} "
          f"peft={env['peft']} commit={env['git_commit']}")

    if model is None:
        model = attach_lora(load_base_model(cfg["model"]["id"], cfg["quantization"]), cfg)
    model.print_trainable_parameters()

    args = make_training_args(cfg, paths["run_dir"], len(train_ds), smoke_steps)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=make_collator(tok.pad_token_id),
                      callbacks=[make_logging_callback(logger), *extra_callbacks])
    ckpt = latest_checkpoint(paths["run_dir"]) if (resume and not smoke_steps) else None
    print(f"[train] {'resuming from ' + ckpt if ckpt else 'starting fresh'}; logs: {logger.jsonl}")
    trainer.train(resume_from_checkpoint=ckpt)

    if not smoke_steps:
        final = paths["run_dir"] / "adapter_final"
        trainer.save_model(str(final))
        tok.save_pretrained(str(final))
        print(f"[done] final adapter: {final}")
    return {"n_train": len(train_ds), "n_val": len(val_ds), "steps_per_epoch": steps_per_epoch(len(train_ds), cfg),
            "run_name": name, "run_dir": str(paths["run_dir"])}


def main(model_id=None, quantization=None, num_epochs=None, resume: bool = True) -> int:
    from trialmatch.training.config import load_train_config
    cfg = load_train_config(model_id, quantization, num_epochs)
    try:
        run_training(cfg, resume=resume)
    except NonFiniteLossError as e:
        print(f"\n{e}")
        return 2
    return 0
