"""Base model, LoRA and adapter loading (needs torch/transformers/peft; imports are lazy)."""
from pathlib import Path

from trialmatch.training.config import FORBIDDEN_LORA_TARGETS


def load_tokenizer(model_id: str, padding_side: str = "right"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, padding_side=padding_side)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_id: str, quantization: str):
    """fp16 compute everywhere (T4 has no bf16). qlora_4bit = NF4 + double quant; lora_fp16 = fp16 weights."""
    import torch
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kw = {"device_map": {"": 0 if torch.cuda.is_available() else "cpu"}}
    kw["dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"] = torch.float16
    if quantization == "qlora_4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    elif quantization != "lora_fp16":
        raise ValueError(f"unknown quantization {quantization!r}")
    return AutoModelForCausalLM.from_pretrained(model_id, **kw)


def attach_lora(model, cfg: dict):
    """Wrap the base model with trainable LoRA adapters (attention + MLP projections only)."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lc, tc = cfg["lora"], cfg["train"]
    assert not any(f in m for m in lc["target_modules"] for f in FORBIDDEN_LORA_TARGETS)
    gc_kw = {"use_reentrant": False}
    if cfg["quantization"] == "qlora_4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=tc["gradient_checkpointing"], gradient_checkpointing_kwargs=gc_kw)
    elif tc["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kw)
        model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules"],
        bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():  # fp16 AMP needs fp32 trainable weights (GradScaler rejects fp16 grads)
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


def load_with_adapter(model_id: str, quantization: str, adapter_dir: str | Path | None):
    """Base model, plus a trained adapter if adapter_dir is given. Returned in eval mode."""
    model = load_base_model(model_id, quantization)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.config.use_cache = True
    return model.eval()
