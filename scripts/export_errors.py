"""Export errors of the fine-tuned model for manual analysis. Validation by default.

  python scripts/export_errors.py
  python scripts/export_errors.py --split test --confirm-final   # FINAL: writes only test_finetuned_merged_errors.csv
Writes results/error_analysis/: val_finetuned_errors.csv, val_finetuned_error_summary.csv,
val_finetuned_candidate_errors.csv (12 rows covering distinct categories), and the two base-vs-finetuned
disagreement files. No explanations are generated: the my_notes column is left blank.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trialmatch.evaluation.error_analysis import (CSV_COLUMNS, category_counts, error_rows, export_row,
                                                  pick_candidates)
from trialmatch.evaluation.evidence import row_prf
from trialmatch.evaluation.val_data import check_metrics_split, load_joined
from trialmatch.utils.common import resolve


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel opens UTF-8 correctly
        w = csv.DictWriter(f, columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def label_ok(r: dict) -> bool:
    return r["status"] == "ok" and r["pred_label"] == r["gold_label"]


def evid_f1(r: dict) -> float:
    return row_prf(set(r["gold_evidence"]), None if r["pred_evidence"] is None else set(r["pred_evidence"]))[2]


def main_test(tag: str = "finetuned_merged"):
    out = resolve("results/error_analysis")
    out.mkdir(parents=True, exist_ok=True)
    recs = load_joined(tag, "test", allow_test=True)
    rows = error_rows(recs)
    write_csv(out / f"test_{tag}_errors.csv", rows, CSV_COLUMNS)
    counts = category_counts(rows, len(recs))
    print(f"test rows: {len(recs)}; {tag} rows with any error (label or evidence): {len(rows)}")
    for c in counts:
        print(f"{c['category']:58s} {c['n_rows']:5d} {c['n_rows_as_primary']:8d} {c['pct_of_all_rows']:9.1f}")
    print(f"wrote {out / f'test_{tag}_errors.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--confirm-final", action="store_true")
    a = ap.parse_args()
    if check_metrics_split(a.split, a.confirm_final):
        return main_test()
    out = resolve("results/error_analysis")
    out.mkdir(parents=True, exist_ok=True)
    ft, base = load_joined("finetuned", "val"), load_joined("base", "val")
    n = len(ft)

    rows = error_rows(ft)
    write_csv(out / "val_finetuned_errors.csv", rows, CSV_COLUMNS)
    counts = category_counts(rows, n)
    write_csv(out / "val_finetuned_error_summary.csv", counts, ["category", "n_rows", "n_rows_as_primary", "pct_of_all_rows"])
    cand = pick_candidates(rows, 12)
    write_csv(out / "val_finetuned_candidate_errors.csv", cand, ["selected_for_category", *CSV_COLUMNS])

    # Base vs finetuned disagreements on the LABEL (the primary task); evidence F1 shown for context.
    b_by_id = {r["annotation_id"]: r for r in base}
    extra = ["base_pred_label", "base_pred_evidence_ids", "base_evidence_f1", "finetuned_evidence_f1"]
    def disagree(pred):
        res = []
        for r in ft:
            b = b_by_id[r["annotation_id"]]
            if pred(label_ok(b), label_ok(r)):
                row = export_row(r)
                row.update(base_pred_label=b["pred_label"] or "invalid", base_pred_evidence_ids=str(b["pred_evidence"]),
                           base_evidence_f1=round(evid_f1(b), 3), finetuned_evidence_f1=round(evid_f1(r), 3))
                res.append(row)
        return res
    b_right = disagree(lambda b, f: b and not f)
    f_right = disagree(lambda b, f: f and not b)
    write_csv(out / "val_base_right_finetuned_wrong.csv", b_right, CSV_COLUMNS[:-1] + extra + ["my_notes"])
    write_csv(out / "val_finetuned_right_base_wrong.csv", f_right, CSV_COLUMNS[:-1] + extra + ["my_notes"])

    print(f"val rows: {n}; fine-tuned rows with any error (label or evidence): {len(rows)}")
    print(f"{'category':58s} {'rows':>5s} {'primary':>8s} {'% of val':>9s}")
    for c in counts:
        print(f"{c['category']:58s} {c['n_rows']:5d} {c['n_rows_as_primary']:8d} {c['pct_of_all_rows']:9.1f}")
    print("\ncandidate rows (annotation_id | selected for | primary):")
    for r in cand:
        print(f"  {r['annotation_id']:4d} | {r['selected_for_category']} | {r['primary_category']}")
    print(f"\nlabel right in base but wrong in finetuned: {len(b_right)}; right in finetuned but wrong in base: {len(f_right)}")
    print(f"files in {out}")


if __name__ == "__main__":
    main()
