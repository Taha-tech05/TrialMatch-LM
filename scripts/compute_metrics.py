"""Metrics for saved predictions (base, finetuned) + baselines. Validation by default.

  python scripts/compute_metrics.py                 # val, tags base finetuned
  python scripts/compute_metrics.py --tags base finetuned --train-log results/logs/train_qwen3-4b_qlora_4bit.csv
  python scripts/compute_metrics.py --split test --confirm-final    # FINAL evaluation only
Reads results/eval/<tag>_<split>_predictions.jsonl; needs no torch/GPU. The test split is read ONLY with
--split test --confirm-final (the flag alone, or --split test alone, is refused).
Writes results/metrics/<split>_metrics_<tag>.json, <split>_comparison.md and results/figures/confusion_*.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trialmatch.evaluation.baselines import make_baseline
from trialmatch.evaluation.reporting import (build_metrics, comparison_markdown, plot_confusion,
                                             plot_training_curves, write_confusion_csv)
from trialmatch.evaluation.val_data import check_metrics_split, load_joined
from trialmatch.utils.common import load_config, resolve


TEST_DEFAULT_TAGS = ["base", "finetuned_merged"]
OPTIONAL_TEST_TAGS = ["finetuned_4bit"]          # included only if its test predictions file exists


def gold_support_note(records: list[dict], cfg: dict) -> list[str]:
    """Data-driven note listing label classes with zero gold rows (per criterion type)."""
    lines = []
    for ctype in ("inclusion", "exclusion"):
        rows = [r for r in records if r["criterion_type"] == ctype]
        for lab in cfg["labels"][ctype]:
            n = sum(r["gold_label"] == lab for r in rows)
            if n == 0:
                lines.append(f"**Note:** the `{lab}` class has **zero gold rows** among the {len(rows)} {ctype} rows of this "
                             f"split, so macro-F1 (present classes) and macro-F1 (full label set) are both reported and differ.")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--confirm-final", action="store_true", help="required to read the test split")
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--train-log", default="results/logs/train_qwen3-4b_qlora_4bit.csv")
    a = ap.parse_args()
    allow_test = check_metrics_split(a.split, a.confirm_final)       # raises before any test data is touched
    sp = a.split
    tags = a.tags
    if tags is None:
        tags = ["base", "finetuned"] if sp == "val" else TEST_DEFAULT_TAGS + [
            t for t in OPTIONAL_TEST_TAGS if resolve(f"results/eval/{t}_test_predictions.jsonl").exists()]
    out_m, out_f = resolve("results/metrics"), resolve("results/figures")
    out_m.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    records = {t: load_joined(t, sp, allow_test=allow_test) for t in tags}
    ids = [sorted(r["annotation_id"] for r in rs) for rs in records.values()]
    assert all(i == ids[0] for i in ids), "tags cover different annotation ids"
    base_recs, maj = make_baseline(next(iter(records.values())))
    metrics = {t: build_metrics(t, rs, split=sp) for t, rs in records.items()}
    metrics["baseline"] = build_metrics("baseline", base_recs, {"majority_labels_from_train": maj}, split=sp)
    notes = gold_support_note(next(iter(records.values())), load_config()) if sp == "test" else []
    if notes:
        for m in metrics.values():
            m["notes"] = notes

    for t, m in metrics.items():
        (out_m / f"{sp}_metrics_{t}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        for scope, rep in m["classification"].items():
            write_confusion_csv(out_f / f"confusion_{t}_{scope}_{sp}.csv", rep["confusion"])
            plot_confusion(out_f / f"confusion_{t}_{scope}_{sp}.png", rep["confusion"], f"{t} - {scope} ({sp}, n={rep['n']})")
    md = comparison_markdown(metrics, split=sp, preamble=notes + ([""] if notes else []))
    (out_m / f"{sp}_comparison.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"majority labels (train): {maj}")
    if sp == "val":
        log = resolve(a.train_log)
        if log.exists():
            info = plot_training_curves(log, out_f / "training_curves.png")
            print("training curves:", json.dumps(info))
        else:
            print(f"[skip] training log not found: {log}")


if __name__ == "__main__":
    main()
