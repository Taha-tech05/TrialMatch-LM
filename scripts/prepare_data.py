"""CLI entry point: download raw data, make the fixed patient-grouped split, write manifest."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trialmatch.data import leakage as lk
from trialmatch.data.loading import download_raw, load_raw
from trialmatch.data.split import assign_rows, split_patients
from trialmatch.utils.common import load_config, resolve

METHOD = ("Unique patient IDs sorted; shuffled with np.random.default_rng(seed); first 37 -> train, "
          "next 8 -> val, last 8 -> test; every annotation row follows its patient.")
SEPARATE_NAME = "forbidden_previous_predictions_NOT_FOR_MODEL_USE.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download the raw dataset")
    args = ap.parse_args()
    cfg = load_config()
    c, sp = cfg["columns"], cfg["split"]
    download_raw(cfg, force=args.force)
    df = load_raw(cfg)

    patients = split_patients(df[c["patient"]], sp["seed"], sp["sizes"])
    rows = assign_rows(df, patients, c["patient"])

    lk.check_split_sizes(patients, sp["sizes"])
    lk.check_no_patient_overlap(rows, c["patient"])
    lk.check_full_coverage(df, rows, c["id"])
    lk.warn_duplicate_notes_across_splits(rows, c["patient"], c["note"])

    forb = [f for f in cfg["forbidden_columns"] if f in df.columns]
    proc = resolve(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    clean = {}
    for name, d in rows.items():
        d = d.sort_values(c["id"]).reset_index(drop=True)
        clean[name] = d.drop(columns=forb)
        lk.check_forbidden_columns_absent(clean[name], cfg["forbidden_columns"], f"{name} processed")
        clean[name].to_parquet(proc / f"{name}.parquet", index=False)
    # previous-model outputs kept apart, clearly not for model use
    df[[c["id"], c["patient"], *forb]].sort_values(c["id"]).to_parquet(proc / SEPARATE_NAME, index=False)

    manifest = {
        "seed": sp["seed"], "method": METHOD, "dataset": cfg["dataset"]["hf_id"],
        "patients": patients,
        "patient_counts": {k: len(v) for k, v in patients.items()},
        "row_counts": {k: len(d) for k, d in clean.items()},
        "label_column": c["label"],
        "label_distribution": {k: {t: g[c["label"]].value_counts().to_dict()
                                   for t, g in d.groupby(c["criterion_type"])} for k, d in clean.items()},
        "forbidden_columns_removed": forb,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    mp = resolve(cfg["paths"]["manifest"])
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("patient_counts", "row_counts", "label_distribution")}, indent=2))
    print(f"[ok] wrote {mp}")


if __name__ == "__main__":
    main()
