"""Greedy generation over a split, parsing with the existing parser, accuracy + invalid rate only.

Same prompt (chat_prompt_text) and the same decoding settings (configs/train.yaml `generation`) are used
for the base model and for base+adapter, so before/after numbers are comparable. The full metrics module
comes later. The test split needs --confirm-final and refuses a second run without --force.
"""
import json
import time
from pathlib import Path

from trialmatch.model import schema
from trialmatch.model.chat import chat_prompt_text, end_token_id
from trialmatch.model.parse import parse_output
from trialmatch.model.prompt import example_from_row
from trialmatch.training.dataset import iter_rows
from trialmatch.training.logging_utils import git_commit
from trialmatch.utils.common import load_config

TEST_MARKER = "test_split_evaluated.json"


class TestGuardError(RuntimeError):
    __test__ = False  # not a pytest class


def check_test_guard(split: str, confirm_final: bool, force: bool, eval_dir: Path) -> Path | None:
    """Raise unless a test-split run is explicitly confirmed and has not happened before.
    Returns the marker path to write for test runs, else None."""
    if split != "test":
        return None
    marker = eval_dir / TEST_MARKER
    if not confirm_final:
        raise TestGuardError("The test split is reserved for the single final evaluation: pass --confirm-final.")
    if marker.exists() and not force:
        raise TestGuardError(f"The test split was already evaluated ({marker}). Refusing to run again; "
                             "use --force only if you really mean to.")
    return marker


def summarize(records: list[dict]) -> dict:
    n = len(records)
    valid = [r for r in records if r["status"] == "ok"]
    correct = sum(r["pred_label"] == r["gold_label"] for r in valid)
    return {"n": n, "n_valid": len(valid), "n_invalid": n - len(valid),
            "invalid_rate": (n - len(valid)) / n if n else None,
            "accuracy": correct / n if n else None,  # invalid outputs count as wrong
            "accuracy_among_valid": correct / len(valid) if valid else None}


def build_items(split: str, allow_test: bool, limit: int | None = None) -> list[dict]:
    cfg = load_config()
    c = cfg["columns"]
    items = []
    for row in iter_rows(split, allow_test=allow_test):
        ex = example_from_row(row)
        items.append({"annotation_id": int(row[c["id"]]), "criterion_type": row[c["criterion_type"]],
                      "n_sentences": schema.count_note_sentences(row[c["note"]]), "prompt": ex.prompt,
                      "gold_label": row[c["label"]], "gold_target": ex.target})
        if limit and len(items) >= limit:
            break
    return items


def generate(model, tok, items: list[dict], gen_cfg: dict) -> list[str]:
    """Greedy decoding, left-padded batches (sorted by length to cut padding; order restored)."""
    import torch
    end_id = end_token_id(tok)
    texts = [chat_prompt_text(tok, it["prompt"]) for it in items]
    order = sorted(range(len(items)), key=lambda i: -len(texts[i]))
    outs = [None] * len(items)
    bs = gen_cfg["batch_size"]
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=gen_cfg["max_new_tokens"],
                                 eos_token_id=end_id, pad_token_id=tok.pad_token_id,
                                 temperature=None, top_p=None, top_k=None)
        for i, text in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            outs[i] = text
        print(f"  generated {min(s + bs, len(order))}/{len(order)}", flush=True)
    return outs


def run_eval(model, tok, cfg: dict, split: str, tag: str, eval_dir: Path, adapter: str | None,
             confirm_final: bool = False, force: bool = False, limit: int | None = None) -> dict:
    marker = check_test_guard(split, confirm_final, force, eval_dir)
    if marker is not None and limit:
        raise TestGuardError("--limit is not allowed on the test split.")
    eval_dir.mkdir(parents=True, exist_ok=True)
    if marker is not None:  # written BEFORE reading test data so a crash still counts as a run
        marker.write_text(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag,
                                      "adapter": adapter}, indent=2), encoding="utf-8")
    items = build_items(split, allow_test=(split == "test"), limit=limit)
    gen_cfg = cfg["generation"]
    t0 = time.time()
    raws = generate(model, tok, items, gen_cfg)
    records = []
    for it, raw in zip(items, raws):
        pred = parse_output(raw, it["criterion_type"], it["n_sentences"])
        records.append({"annotation_id": it["annotation_id"], "criterion_type": it["criterion_type"],
                        "gold_label": it["gold_label"], "gold_target": it["gold_target"], "raw_output": raw,
                        "status": pred.status, "pred_label": pred.label, "pred_evidence_ids": pred.evidence_ids,
                        "parse_error": pred.error})
    pred_path, sum_path = eval_dir / f"{tag}_{split}_predictions.jsonl", eval_dir / f"{tag}_{split}_summary.json"
    pred_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    summary = {"tag": tag, "split": split, "model": cfg["model"]["id"], "quantization": cfg["quantization"],
               "adapter": adapter, "generation": {**gen_cfg, "greedy": True}, "limit": limit,
               "git_commit": git_commit(), "seconds": round(time.time() - t0, 1), **summarize(records)}
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved: {pred_path}\n       {sum_path}")
    return summary
