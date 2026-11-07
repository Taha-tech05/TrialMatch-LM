"""CLI: greedy-generation evaluation of the base model or base+adapter (accuracy + invalid rate).

  python scripts/evaluate.py --tag base                                   # base model, val split
  python scripts/evaluate.py --adapter <run_dir>/adapter_final --tag ft   # fine-tuned, val split
  python scripts/evaluate.py --adapter ... --tag ft --split test --confirm-final   # ONCE, at the very end
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trialmatch.evaluation.generate_eval import TestGuardError, check_test_guard, run_eval
from trialmatch.training.config import load_train_config, run_paths

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--confirm-final", action="store_true", help="required for --split test")
    ap.add_argument("--force", action="store_true", help="allow a repeated test-split run")
    ap.add_argument("--adapter", help="adapter directory; omit to evaluate the base model")
    ap.add_argument("--tag", help="output name prefix (default: 'base' or the adapter folder name)")
    ap.add_argument("--model")
    ap.add_argument("--quantization", choices=["qlora_4bit", "lora_fp16"])
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--limit", type=int, help="first N rows only (val only; for quick checks)")
    a = ap.parse_args()

    cfg = load_train_config(a.model, a.quantization)
    if a.batch_size:
        cfg["generation"]["batch_size"] = a.batch_size
    eval_dir = run_paths(cfg)["eval"]
    try:  # fail fast, before loading any weights
        check_test_guard(a.split, a.confirm_final, a.force, eval_dir)
    except TestGuardError as e:
        sys.exit(f"REFUSED: {e}")

    from trialmatch.model.loading import load_tokenizer, load_with_adapter
    tok = load_tokenizer(cfg["model"]["id"], padding_side="left")
    model = load_with_adapter(cfg["model"]["id"], cfg["quantization"], a.adapter)
    tag = a.tag or (Path(a.adapter).name if a.adapter else "base")
    try:
        run_eval(model, tok, cfg, a.split, tag, eval_dir, a.adapter, a.confirm_final, a.force, a.limit)
    except TestGuardError as e:
        sys.exit(f"REFUSED: {e}")
