"""Leakage checks. Every function raises LeakageError on failure."""
import warnings

import pandas as pd


class LeakageError(AssertionError):
    pass


def check_no_patient_overlap(splits: dict[str, pd.DataFrame], patient_col: str = "patient_id") -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = set(splits[a][patient_col]) & set(splits[b][patient_col])
            if both:
                raise LeakageError(f"Patients in both '{a}' and '{b}': {sorted(both)}")


def check_full_coverage(full: pd.DataFrame, splits: dict[str, pd.DataFrame], id_col: str = "annotation_id") -> None:
    if full[id_col].duplicated().any():
        raise LeakageError(f"Full dataset has duplicated {id_col}")
    all_ids = pd.concat([d[id_col] for d in splits.values()])
    if all_ids.duplicated().any():
        raise LeakageError(f"Duplicated {id_col} across/within splits: {all_ids[all_ids.duplicated()].tolist()[:10]}")
    missing = set(full[id_col]) - set(all_ids)
    extra = set(all_ids) - set(full[id_col])
    if missing or extra:
        raise LeakageError(f"Coverage mismatch: dropped={sorted(missing)[:10]} extra={sorted(extra)[:10]}")


def check_split_sizes(patient_lists: dict[str, list], expected: dict[str, int]) -> None:
    got = {k: len(v) for k, v in patient_lists.items()}
    if got != dict(expected):
        raise LeakageError(f"Split patient counts {got} != expected {dict(expected)}")


def check_forbidden_columns_absent(df: pd.DataFrame, forbidden: list[str], name: str = "dataframe") -> None:
    present = [c for c in forbidden if c in df.columns]
    if present:
        raise LeakageError(f"Forbidden columns present in {name}: {present}")


def warn_duplicate_notes_across_splits(splits: dict[str, pd.DataFrame], patient_col="patient_id", note_col="note") -> list:
    """Warn (not raise) if identical note text is used under different patient IDs in different splits."""
    owner: dict[str, set] = {}
    for sname, d in splits.items():
        for note, pid in zip(d[note_col], d[patient_col]):
            owner.setdefault(note, set()).add((pid, sname))
    dups = [(n, v) for n, v in owner.items() if len({p for p, _ in v}) > 1 and len({s for _, s in v}) > 1]
    if dups:
        warnings.warn(f"{len(dups)} identical note text(s) appear under different patient IDs across splits")
    return dups
