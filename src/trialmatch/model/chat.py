"""Qwen3 chat formatting shared by training and generation (torch-free).

Training and evaluation MUST build the prompt through chat_prompt_text so the model sees the
same tokens at train and test time. Thinking is disabled (enable_thinking=False), which makes the
template end the prompt with an empty think block; the answer JSON follows directly.
"""
END_TOKEN = "<|im_end|>"


def chat_prompt_text(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def end_token_id(tok) -> int:
    tid = tok.convert_tokens_to_ids(END_TOKEN)
    if tid is None or tid == tok.unk_token_id:
        raise ValueError(f"tokenizer has no {END_TOKEN} token")
    return tid
