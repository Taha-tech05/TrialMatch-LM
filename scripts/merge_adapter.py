"""CLI: merge the LoRA adapter into the fp16 base (merge_and_unload), verify, optionally save (~8 GB, NOT committed).

  python scripts/merge_adapter.py --adapter $TRIALMATCH_OUT/checkpoints/qwen3-4b_qlora_4bit/adapter_final --save
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", help="adapter dir or HF Hub id (default: configs/inference.yaml)")
    ap.add_argument("--base", help="base model id (default: configs/inference.yaml)")
    ap.add_argument("--save", action="store_true", help="save the merged model (~8 GB) to --out-dir")
    ap.add_argument("--out-dir", help="default: $TRIALMATCH_OUT/merged/qwen3-4b_merged_fp16")
    ap.add_argument("--verify-prompts", type=int, help="val prompts for the unmerged-vs-merged logit check (0 = skip)")
    a = ap.parse_args()

    from trialmatch.model.load import default_merged_dir, inference_config, load_merged
    from trialmatch.training.config import load_train_config, output_root
    icfg = inference_config()
    n = icfg["merge"]["verify_prompts"] if a.verify_prompts is None else a.verify_prompts
    out = Path(a.out_dir) if a.out_dir else default_merged_dir(icfg)
    model, tok, report = load_merged(a.base, a.adapter, save_dir=out if a.save else None, verify_prompts=n)
    log = output_root(load_train_config()) / "results" / "logs" / "merge_verify.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved report: {log}" + (f"\nsaved merged model: {out}" if a.save else "\n(merged model not saved; add --save)"))
