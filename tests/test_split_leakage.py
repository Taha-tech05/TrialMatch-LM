"""Tests for split integrity and leakage. Skipped if the raw data has not been downloaded."""
import json

import pandas as pd
import pytest

from trialmatch.data import leakage as lk
from trialmatch.data.split import assign_rows, split_patients
from trialmatch.utils.common import load_config, resolve

CFG = load_config()
C = CFG["columns"]
RAW = resolve(CFG["paths"]["raw"])
PROC = resolve(CFG["paths"]["processed_dir"])
MANIFEST = resolve(CFG["paths"]["manifest"])

pytestmark = pytest.mark.skipif(
    not RAW.exists(), reason=f"raw data missing ({RAW}); run `python scripts/prepare_data.py` first")


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(RAW)


@pytest.fixture(scope="module")
def splits(df):
    return split_patients(df[C["patient"]], CFG["split"]["seed"], CFG["split"]["sizes"])


def test_split_deterministic(df):
    a = split_patients(df[C["patient"]], 42, CFG["split"]["sizes"])
    b = split_patients(df[C["patient"]].sample(frac=1, random_state=1), 42, CFG["split"]["sizes"])
    assert a == b  # same seed -> same result, regardless of row order


def test_split_counts(splits):
    assert {k: len(v) for k, v in splits.items()} == {"train": 37, "val": 8, "test": 8}


def test_wrong_patient_count_fails_loudly():
    with pytest.raises(ValueError, match="53"):
        split_patients([f"p{i}" for i in range(10)], 42, {"train": 37, "val": 8, "test": 8})


def test_no_patient_overlap_and_full_coverage(df, splits):
    rows = assign_rows(df, splits, C["patient"])
    lk.check_no_patient_overlap(rows, C["patient"])
    lk.check_full_coverage(df, rows, C["id"])


def test_overlap_check_detects_leak(df, splits):
    rows = assign_rows(df, splits, C["patient"])
    rows["test"] = pd.concat([rows["test"], rows["train"].head(1)])
    with pytest.raises(lk.LeakageError):
        lk.check_no_patient_overlap(rows, C["patient"])


@pytest.mark.skipif(not (PROC / "train.parquet").exists(), reason="processed files missing; run prepare_data.py")
def test_forbidden_columns_absent_from_processed():
    for name in ("train", "val", "test"):
        lk.check_forbidden_columns_absent(pd.read_parquet(PROC / f"{name}.parquet"),
                                          CFG["forbidden_columns"], name)


@pytest.mark.skipif(not MANIFEST.exists(), reason="manifest missing; run prepare_data.py")
def test_manifest_matches_files(df):
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["seed"] == CFG["split"]["seed"]
    files = {n: pd.read_parquet(PROC / f"{n}.parquet") for n in ("train", "val", "test")}
    for n, d in files.items():
        assert sorted(d[C["patient"]].unique()) == sorted(m["patients"][n])
        assert len(d) == m["row_counts"][n]
        assert m["patient_counts"][n] == len(m["patients"][n])
        got = {t: g[C["label"]].value_counts().to_dict() for t, g in d.groupby(C["criterion_type"])}
        assert got == m["label_distribution"][n]
    lk.check_no_patient_overlap(files, C["patient"])
    lk.check_full_coverage(df, files, C["id"])
    assert m["patients"] == split_patients(df[C["patient"]], m["seed"], CFG["split"]["sizes"])
