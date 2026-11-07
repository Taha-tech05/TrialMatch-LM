"""CLI entry point: LoRA/QLoRA fine-tuning. Resumes automatically from the latest checkpoint.

  python scripts/train.py                              # settings from configs/train.yaml
  python scripts/train.py --model Qwen/Qwen3-1.7B      # fallback model
  python scripts/train.py --quantization lora_fp16
Set TRIALMATCH_OUT to redirect checkpoints/logs (e.g. to Google Drive).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trialmatch.training.train import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="override model.id")
    ap.add_argument("--quantization", choices=["qlora_4bit", "lora_fp16"])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints")
    a = ap.parse_args()
    sys.exit(main(a.model, a.quantization, a.epochs, resume=not a.no_resume))
