"""Merged fp16 model: Qwen3-4B base + LoRA adapter folded in with merge_and_unload (imports are lazy).

All inference benchmarks (Tasks 3-5) use this ONE model so cached/uncached/tiled/speculative numbers are
comparable. The saved merged model is ~8 GB: it goes to $TRIALMATCH_OUT/merged/... and is NOT committed.
Note: the adapter was trained on a 4-bit (NF4) base; merging into fp16 base weights changes the model very
slightly, so a few predictions can differ from the 4-bit-base evaluation (see docs/decisions.md).
"""
import json
import time
from pathlib import Path

from trialmatch.training.config import output_root
from trialmatch.utils.common import PROJECT_ROOT


def inference_config() -> dict:
    import yaml
    with open(PROJECT_ROOT / "configs" / "inference.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_device_dtype():
    """fp16 on GPU (T4 has no bf16), fp32 on CPU."""
    import torch
    return ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)


def default_adapter(icfg: dict | None = None) -> str:
    icfg = icfg or inference_config()
    m = icfg["merge"]
    if m.get("adapter"):
        return str(m["adapter"])
    from trialmatch.training.config import load_train_config
    return str(output_root(load_train_config()) / m["adapter_default_relative"])


def default_merged_dir(icfg: dict | None = None) -> Path:
    from trialmatch.training.config import load_train_config
    icfg = icfg or inference_config()
    return output_root(load_train_config()) / icfg["merge"]["merged_default_relative"]


def _check_adapter(adapter: str) -> None:
    """Fail early on a local-looking adapter path that does not exist (a Hub id looks like 'user/name')."""
    if adapter.startswith(("/", ".", "~")) or ":" in adapter[:3]:
        p = Path(adapter).expanduser()
        if not (p / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"adapter not found at {p} (no adapter_config.json). Is TRIALMATCH_OUT set in this session "
                "(run the 'OUT = ...' Colab cell) and did training finish? Pass --adapter <dir or Hub id>.")


def _check_val_data() -> None:
    from trialmatch.utils.common import load_config, resolve
    path = resolve(load_config()["paths"]["processed_dir"]) / "val.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `python scripts/prepare_data.py` first "
                                "(processed data is not in git), or use --verify-prompts 0.")


def _dtype_kw(dtype) -> dict:
    import transformers
    from packaging.version import Version
    return {("dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"): dtype}


def _device_map(device: str):
    return {"": 0} if device == "cuda" else {"": "cpu"}


def _peft_attach(base_model, adapter):
    """PeftModel.from_pretrained with a readable hint for the Colab torchao/peft version conflict."""
    from peft import PeftModel
    try:
        return PeftModel.from_pretrained(base_model, str(adapter))
    except ImportError as e:
        if "torchao" in str(e):
            raise ImportError(f"{e}\nFix: run `pip uninstall -y torchao` (unused here) or `pip install -U torchao`, "
                              "then re-run (restart the runtime if peft was already imported).") from e
        raise


def merge_adapter_into_base(base_model, adapter: str | Path):
    """Attach the LoRA adapter to an already loaded base model, merge it, return the plain model.
    Raises if any LoRA parameter is left."""
    merged = _peft_attach(base_model, adapter).merge_and_unload()
    left = [n for n, _ in merged.named_parameters() if "lora_" in n]
    if left:
        raise RuntimeError(f"LoRA parameters remain after merge: {left[:3]}")
    return merged.eval()


def _last_logits(model, tok, texts, device):
    import torch
    rows = []
    for t in texts:
        enc = tok(t, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            rows.append(model(**enc).logits[0, -1].float().cpu())
    return torch.stack(rows)


def load_merged(base_id: str | None = None, adapter: str | None = None, save_dir: str | Path | None = None,
                verify_prompts: int = 0, device: str | None = None, dtype=None):
    """Load base in fp16, attach adapter, merge_and_unload. Returns (model, tokenizer, report).

    verify_prompts > 0 compares last-position logits of the UNMERGED (base+adapter) model with the merged
    one on that many VAL prompts (same weights, so differences should only be fp16 rounding)."""
    from transformers import AutoModelForCausalLM

    from trialmatch.model.loading import load_tokenizer
    icfg = inference_config()
    base_id = base_id or icfg["merge"]["base_model"]
    adapter = adapter or default_adapter(icfg)
    _check_adapter(str(adapter))          # fail before downloading/loading 8 GB of weights
    if verify_prompts:
        _check_val_data()
    dev, dt = pick_device_dtype()
    device, dtype = device or dev, dtype or dt
    tok = load_tokenizer(base_id, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(base_id, device_map=_device_map(device), **_dtype_kw(dtype))
    report = {"base_model": base_id, "adapter": str(adapter), "dtype": str(dtype).replace("torch.", ""),
              "device": device}
    before, texts = None, []
    if verify_prompts:
        from trialmatch.evaluation.generate_eval import build_items
        from trialmatch.model.chat import chat_prompt_text
        texts = [chat_prompt_text(tok, it["prompt"]) for it in build_items("val", False, verify_prompts)]
        peft_model = _peft_attach(base, adapter).eval()
        before = _last_logits(peft_model, tok, texts, device)
        merged = peft_model.merge_and_unload().eval()
    else:
        merged = merge_adapter_into_base(base, adapter)
    left = [n for n, _ in merged.named_parameters() if "lora_" in n]
    report["lora_params_left"] = len(left)
    if left:
        raise RuntimeError(f"LoRA parameters remain after merge: {left[:3]}")
    report["n_params"] = sum(p.numel() for p in merged.parameters())
    if before is not None:
        after = _last_logits(merged, tok, texts, device)
        report["verify"] = {"n_prompts": len(texts),
                            "max_abs_logit_diff": float((before - after).abs().max()),
                            "top1_agreement": float((before.argmax(-1) == after.argmax(-1)).float().mean())}
    merged.config.use_cache = True
    if save_dir:
        save_merged(merged, tok, save_dir, report)
    return merged, tok, report


def save_merged(model, tok, save_dir: str | Path, report: dict | None = None) -> Path:
    """Save merged model (~8 GB) + tokenizer + merge_info.json. Point this OUTSIDE the git tree ($TRIALMATCH_OUT)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tok.save_pretrained(save_dir)
    from trialmatch.training.logging_utils import git_commit
    info = {**(report or {}), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_commit()}
    (save_dir / "merge_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return save_dir


def load_saved_merged(merged_dir: str | Path, device: str | None = None, dtype=None):
    """Load a previously saved merged model (fp16 on GPU, fp32 on CPU). Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM

    from trialmatch.model.loading import load_tokenizer
    if not (Path(merged_dir) / "config.json").exists():
        raise FileNotFoundError(
            f"no saved merged model at '{merged_dir}' (no config.json). Run scripts/merge_adapter.py --save first, "
            "and make sure TRIALMATCH_OUT is set in this session (an empty value gives paths like '/merged/...').")
    dev, dt = pick_device_dtype()
    device, dtype = device or dev, dtype or dt
    model = AutoModelForCausalLM.from_pretrained(str(merged_dir), device_map=_device_map(device), **_dtype_kw(dtype))
    model.config.use_cache = True
    return model.eval(), load_tokenizer(str(merged_dir), padding_side="left")


def load_merged_for_inference(adapter: str | None = None, merged_dir: str | None = None):
    """Saved merged dir if given (or present at the default location and no adapter given), else merge now."""
    if merged_dir:
        return load_saved_merged(merged_dir)
    default = default_merged_dir()
    if adapter is None and (default / "config.json").exists():
        return load_saved_merged(default)
    model, tok, _ = load_merged(adapter=adapter)
    return model, tok
