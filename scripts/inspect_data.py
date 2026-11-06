"""Inspect the raw dataset; write results/metrics/data_inspection.json and docs/data_inspection.md.

Nothing is assumed: label/evidence columns and leakage columns are detected from
column names and values, and unclear columns are flagged for human review.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from trialmatch.data.loading import load_raw
from trialmatch.utils.common import CONFIG_PATH, load_config, resolve

LEAK_PAT = re.compile(r"gpt|llm|model|predict|generated|explanation|correctness", re.I)


def is_json_int_list(s) -> bool:
    try:
        v = json.loads(s)
        return isinstance(v, list) and all(isinstance(x, int) for x in v)
    except Exception:
        return False


def stats(s: pd.Series) -> dict:
    return {k: float(v) for k, v in s.describe(percentiles=[0.5, 0.9, 0.99]).items()}


def main():
    cfg = load_config()
    c = cfg["columns"]
    df = load_raw(cfg)
    out: dict = {}
    out["rows"] = len(df)
    out["columns"] = list(df.columns)
    out["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}
    out["unique_patients"] = int(df[c["patient"]].nunique())
    out["unique_trials"] = int(df[c["trial"]].nunique())
    rpp = df.groupby(c["patient"]).size()
    out["rows_per_patient"] = {"min": int(rpp.min()), "max": int(rpp.max()), "mean": float(rpp.mean())}
    out["criterion_type_counts"] = df[c["criterion_type"]].value_counts().to_dict()
    out["annotation_id_unique"] = bool(df[c["id"]].is_unique)

    # --- detect leakage columns / label column / evidence columns
    leak = [col for col in df.columns if LEAK_PAT.search(col)]
    out["leakage_columns"] = leak
    low_card = [col for col in df.columns if df[col].dtype == object and df[col].nunique() <= 10
                and col != c["criterion_type"]]
    out["low_cardinality_string_columns"] = {col: df[col].value_counts().to_dict() for col in low_card}
    gold_label_cols = [col for col in low_card if col not in leak and re.search(r"eligib|label", col, re.I)]
    evid_cols = [col for col in df.columns if df[col].dtype == object and is_json_int_list(df[col].iloc[0])]
    gold_evid_cols = [col for col in evid_cols if col not in leak]
    out["detected"] = {"label_column": gold_label_cols, "evidence_columns_all": evid_cols,
                       "gold_evidence_column": gold_evid_cols}
    if gold_label_cols != [c["label"]] or gold_evid_cols != [c["evidence"]]:
        print(f"WARNING: detected label/evidence {gold_label_cols}/{gold_evid_cols} differs from config "
              f"{c['label']}/{c['evidence']} - review!")
    lab, ev = gold_label_cols[0], gold_evid_cols[0]

    out["label_column"] = lab
    out["label_distribution_overall"] = df[lab].value_counts().to_dict()
    out["label_distribution_by_criterion_type"] = {
        t: g[lab].value_counts().to_dict() for t, g in df.groupby(c["criterion_type"])}
    for lc in [x for x in leak if x in low_card]:
        out.setdefault("leakage_label_distributions", {})[lc] = df[lc].value_counts().to_dict()

    # --- evidence
    parsed = df[ev].map(json.loads)
    out["evidence"] = {
        "column": ev, "stored_python_type": type(df[ev].iloc[0]).__name__,
        "storage": "JSON-encoded list of ints inside a string", "example": df[ev].iloc[0],
        "empty_fraction": float((parsed.map(len) == 0).mean()),
        "empty_fraction_by_label": {k: float((parsed[g.index].map(len) == 0).mean()) for k, g in df.groupby(lab)},
        "n_sentences_cited_stats": stats(parsed.map(len)),
    }
    # --- notes
    note = df[c["note"]]
    first = df.drop_duplicates(c["patient"])
    uniq_notes = first[c["note"]]
    out["note"] = {
        "chars_per_row": stats(note.str.len()),
        "chars_per_unique_patient": stats(uniq_notes.str.len()),
        "whitespace_tokens_per_unique_patient": stats(uniq_notes.str.split().map(len)),
        "sentence_numbering": "lines prefixed 'N. ' (N starting at 0) separated by newlines",
        "one_note_per_patient": bool(df.groupby(c["patient"])[c["note"]].nunique().max() == 1),
    }
    numbered = uniq_notes.map(lambda t: [int(m) for m in re.findall(r"(?m)^(\d+)\. ", t)])
    out["note"]["numbering_starts_at_zero_and_contiguous_fraction"] = float(
        numbered.map(lambda n: len(n) > 0 and n == list(range(len(n)))).mean())
    n_sent = dict(zip(first[c["patient"]], numbered.map(len)))
    out["note"]["n_sentences_per_note"] = stats(pd.Series(list(n_sent.values())))
    out["evidence"]["rows_with_out_of_range_sentence_ids"] = int(sum(
        any(i >= n_sent[p] or i < 0 for i in ids) for ids, p in zip(parsed, df[c["patient"]])))
    # --- criterion
    out["null_counts"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    ct = df[c["criterion_text"]].dropna()
    out["criterion_text"] = {"chars": stats(ct.str.len()), "whitespace_tokens": stats(ct.str.split().map(len))}
    # --- columns of unclear purpose
    known = {c["id"], c["patient"], c["trial"], c["note"], c["criterion_text"], c["criterion_type"],
             lab, ev, "trial_title"}
    review = {}
    for col in df.columns:
        if col in leak or col in known:
            continue
        review[col] = {"dtype": str(df[col].dtype), "example_values": df[col].value_counts().head(5).to_dict()}
    if "training" in review:
        pt = df.groupby(c["patient"])["training"]
        review["training"]["patients_with_mixed_values"] = int((pt.nunique() > 1).sum())
        review["training"]["patients_true"] = int(pt.first().sum())
    out["needs_human_review"] = review
    out["duplicate_patient_trial_criterion_rows"] = int(
        df.duplicated([c["patient"], c["trial"], c["criterion_text"]]).sum())

    # --- print
    print("=" * 60)
    for k in ("rows", "columns", "dtypes", "unique_patients", "unique_trials", "rows_per_patient",
              "criterion_type_counts", "null_counts", "label_column", "label_distribution_overall",
              "label_distribution_by_criterion_type", "evidence", "note", "criterion_text"):
        print(f"\n## {k}\n{json.dumps(out[k], indent=2, default=str)}")
    print("\n" + "!" * 60 + "\nLEAKAGE COLUMNS (previous model predictions/explanations):\n  " + "\n  ".join(leak))
    print("!" * 60)
    print("\nFLAGGED FOR YOUR REVIEW (purpose unclear):")
    print(json.dumps(review, indent=2, default=str))

    # --- save json + markdown
    jp = resolve(cfg["paths"]["inspection_json"])
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    md = ["# Data inspection", "", "Auto-generated by `scripts/inspect_data.py`. Do not edit by hand.", "",
          "## LEAKAGE COLUMNS", "", *[f"- `{x}`" for x in leak], ""]
    for k, v in out.items():
        md += [f"## {k}", "```json", json.dumps(v, indent=2, default=str), "```", ""]
    mp = resolve(cfg["paths"]["inspection_md"])
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text("\n".join(md), encoding="utf-8")

    # --- persist forbidden_columns into the config (replace only that block)
    txt = CONFIG_PATH.read_text(encoding="utf-8")
    block = "forbidden_columns:\n" + "".join(f"  - {x}\n" for x in leak)
    new = re.sub(r"forbidden_columns:\n(?:  - .*\n)+", lambda m: block, txt)
    if new != txt:
        CONFIG_PATH.write_text(new, encoding="utf-8")
        print("Updated forbidden_columns in configs/data.yaml")
    else:
        print("forbidden_columns in configs/data.yaml already match detection")
    print(f"\nSaved {jp} and {mp}")


if __name__ == "__main__":
    main()
