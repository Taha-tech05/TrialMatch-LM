"""Trivial baselines evaluated on val.

  * Majority-class: for each criterion type, the most frequent gold label in TRAIN (ties broken by label-set
    order) is predicted for every val row of that type.
  * Always-empty evidence: predicts evidence_ids = [] for every row.
The two are combined into one prediction set ("baseline") so the same metric code scores it; its
classification numbers are the majority baseline and its evidence numbers are the always-empty baseline.
"""
from collections import Counter

from trialmatch.evaluation.classification import label_order
from trialmatch.evaluation.val_data import load_gold
from trialmatch.utils.common import load_config


def majority_labels(train_gold, cfg: dict | None = None) -> dict[str, str]:
    """Most frequent train label per criterion type (train rows only)."""
    cfg = cfg or load_config()
    c = cfg["columns"]
    out = {}
    for ctype in ("inclusion", "exclusion"):
        counts = Counter(train_gold.loc[train_gold[c["criterion_type"]] == ctype, c["label"]])
        order = label_order(ctype, cfg)
        out[ctype] = max(order, key=lambda l: (counts.get(l, 0), -order.index(l)))
    return out


def baseline_records(val_records: list[dict], majority: dict[str, str]) -> list[dict]:
    """Same record shape as joined model predictions."""
    return [{**{k: v for k, v in r.items() if k not in ("raw_output", "parse_error")},
             "status": "ok", "pred_label": majority[r["criterion_type"]], "pred_evidence": []}
            for r in val_records]


def make_baseline(val_records: list[dict], cfg: dict | None = None) -> tuple[list[dict], dict[str, str]]:
    maj = majority_labels(load_gold("train", cfg), cfg)
    return baseline_records(val_records, maj), maj
