"""Draft model loading for speculative decoding (imports are lazy).

Default draft = the untuned Qwen/Qwen3-0.6B (same tokenizer as the Qwen3-4B target, verified by
scripts/tokenizer_compat.py). Config (configs/inference.yaml -> speculative):
  draft_model    Hub id of the draft base                      (default Qwen/Qwen3-0.6B)
  draft_dir      optional local directory of a FULL fine-tuned draft (merged weights); overrides draft_model
  draft_adapter  optional LoRA adapter dir/Hub id merged into draft_model (a draft fine-tuned later; not trained here)
"""
from pathlib import Path


def load_draft(draft_model: str = "Qwen/Qwen3-0.6B", draft_dir: str | Path | None = None,
               draft_adapter: str | Path | None = None, device: str | None = None, dtype=None):
    """Return the draft model in eval mode on `device` (fp16 on GPU, fp32 on CPU by default)."""
    from transformers import AutoModelForCausalLM

    from trialmatch.model.load import _dtype_kw, pick_device_dtype
    dev0, dt0 = pick_device_dtype()
    device, dtype = device or dev0, dtype or dt0
    src = str(draft_dir) if draft_dir else draft_model
    model = AutoModelForCausalLM.from_pretrained(src, **_dtype_kw(dtype))
    if draft_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(draft_adapter)).merge_and_unload()
    return model.to(device).eval()


def check_compatible(target, draft) -> None:
    """Speculation compares token ids, so both models must share the vocabulary (embedding size and special ids)."""
    tc, dc = target.config, draft.config
    if tc.vocab_size != dc.vocab_size:
        raise ValueError(f"vocab size mismatch: target {tc.vocab_size} vs draft {dc.vocab_size}")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(tc, name, None) != getattr(dc, name, None):
            # Qwen3-0.6B and 4B differ in config-level bos/eos listing in some releases; the tokenizer is what matters
            import logging
            logging.getLogger(__name__).warning("config %s differs: target %s vs draft %s", name,
                                                getattr(tc, name, None), getattr(dc, name, None))
