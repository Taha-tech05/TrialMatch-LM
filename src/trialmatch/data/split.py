"""Fixed patient-grouped train/val/test split.

Rule (exact):
 1. Take unique patient IDs and sort them.
 2. Shuffle with np.random.default_rng(seed).
 3. First 37 -> train, next 8 -> val, remaining 8 -> test.
 4. Every annotation row of a patient goes to that patient's split.
"""
import numpy as np
import pandas as pd

EXPECTED_PATIENTS = 53


def split_patients(patient_ids, seed: int, sizes: dict) -> dict[str, list[str]]:
    unique = sorted(set(patient_ids))
    total = sum(sizes.values())
    if len(unique) != EXPECTED_PATIENTS or total != EXPECTED_PATIENTS:
        raise ValueError(
            f"Expected {EXPECTED_PATIENTS} unique patients and split sizes summing to "
            f"{EXPECTED_PATIENTS}; got {len(unique)} patients and sizes {sizes} (sum {total}). "
            "The dataset differs from what this project assumes - stop and review."
        )
    arr = np.array(unique, dtype=object)
    np.random.default_rng(seed).shuffle(arr)  # in-place shuffle of the sorted ID array
    shuffled = [str(x) for x in arr]
    n_tr, n_va, n_te = sizes["train"], sizes["val"], sizes["test"]
    splits = {
        "train": shuffled[:n_tr],
        "val": shuffled[n_tr:n_tr + n_va],
        "test": shuffled[n_tr + n_va:],
    }
    assert [len(splits[k]) for k in ("train", "val", "test")] == [n_tr, n_va, n_te], "bad split sizes"
    return splits


def assign_rows(df: pd.DataFrame, splits: dict[str, list[str]], patient_col: str) -> dict[str, pd.DataFrame]:
    return {name: df[df[patient_col].isin(set(ids))].copy() for name, ids in splits.items()}
