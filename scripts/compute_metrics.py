"""Validation metrics for saved predictions (base, finetuned) + baselines. VALIDATION split only.

  python scripts/compute_metrics.py                 # tags base finetuned
  python scripts/compute_metrics.py --tags base finetuned --train-log results/logs/train_qwen3-4b_qlora_4bit.csv
Reads results/eval/<tag>_val_predictions.jsonl; needs no torch/GPU and never reads the test split.
Writes results/metrics/val_metrics_<tag>.json, val_comparison.md and results/figures/*.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trialmatch.evaluation.baselines import make_baseline
from trialmatch.evaluation.reporting import (build_metrics, comparison_markdown, plot_confusion,
                                             plot_training_curves, write_confusion_csv)
from trialmatch.evaluation.val_data import load_joined
from trialmatch.utils.common import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["base", "finetuned"])
    ap.add_argument("--train-log", default="results/logs/train_qwen3-4b_qlora_4bit.csv")
    a = ap.parse_args()
    out_m, out_f = resolve("results/metrics"), resolve("results/figures")
    out_m.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    records = {t: load_joined(t, "val") for t in a.tags}
    ids = [sorted(r["annotation_id"] for r in rs) for rs in records.values()]
    assert all(i == ids[0] for i in ids), "tags cover different annotation ids"
    base_recs, maj = make_baseline(next(iter(records.values())))
    metrics = {t: build_metrics(t, rs) for t, rs in records.items()}
    metrics["baseline"] = build_metrics("baseline", base_recs, {"majority_labels_from_train": maj})

    for t, m in metrics.items():
        (out_m / f"val_metrics_{t}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        for scope, rep in m["classification"].items():
            write_confusion_csv(out_f / f"confusion_{t}_{scope}_val.csv", rep["confusion"])
            plot_confusion(out_f / f"confusion_{t}_{scope}_val.png", rep["confusion"], f"{t} - {scope} (val, n={rep['n']})")
    md = comparison_markdown(metrics)
    (out_m / "val_comparison.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"majority labels (train): {maj}")
    log = resolve(a.train_log)
    if log.exists():
        info = plot_training_curves(log, out_f / "training_curves.png")
        print("training curves:", json.dumps(info))
    else:
        print(f"[skip] training log not found: {log}")


if __name__ == "__main__":
    main()
