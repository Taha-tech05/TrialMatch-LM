"""Structured output schema: {"label": str, "evidence_ids": list[int]}.

Explanations are deliberately NOT part of the model output: no gold explanations exist
and the dataset's GPT-4 explanations are forbidden. The app builds explanations later
from the label plus the cited sentences only.
"""
import re
from dataclasses import dataclass
from typing import Optional

from trialmatch.utils.common import load_config

OUTPUT_KEYS = ("label", "evidence_ids")
CRITERION_TYPES = ("inclusion", "exclusion")
INVALID = "invalid"
OK = "ok"


def label_sets(cfg: Optional[dict] = None) -> dict[str, tuple[str, ...]]:
    cfg = cfg or load_config()
    return {t: tuple(cfg["labels"][t]) for t in CRITERION_TYPES}


def verify_label_sets(df, cfg: Optional[dict] = None) -> None:
    """Raise if the labels seen in the data differ from the configured sets."""
    cfg = cfg or load_config()
    c, sets = cfg["columns"], label_sets(cfg)
    for t, g in df.groupby(c["criterion_type"]):
        seen = set(g[c["label"]].unique())
        if seen != set(sets[t]):
            raise ValueError(f"Label set mismatch for '{t}': data={sorted(seen)} config={sorted(sets[t])}")


def count_note_sentences(note: str) -> int:
    """Notes number sentences as lines starting 'N. ' with N from 0."""
    ids = [int(m) for m in re.findall(r"(?m)^(\d+)\. ", note)]
    if ids != list(range(len(ids))):
        raise ValueError("note sentence numbering is not contiguous from 0")
    return len(ids)


@dataclass
class Prediction:
    """A parsed model output. status == 'invalid' must be counted separately by metrics."""
    parse_ok: bool
    status: str = OK                          # 'ok' | 'invalid'
    label: Optional[str] = None               # only set when valid for the criterion type
    evidence_ids: Optional[list[int]] = None  # only set when valid
    error: Optional[str] = None
    raw: str = ""


def validation_errors(label, evidence_ids, criterion_type: str, n_sentences: int,
                      sets: Optional[dict] = None) -> list[str]:
    sets = sets or label_sets()
    if criterion_type not in sets:
        return [f"unknown criterion_type {criterion_type!r}"]
    errs = []
    if not isinstance(label, str) or label not in sets[criterion_type]:
        errs.append(f"label {label!r} not allowed for {criterion_type} (allowed: {list(sets[criterion_type])})")
    if not isinstance(evidence_ids, list):
        errs.append("evidence_ids is not a list")
        return errs
    if any(isinstance(i, bool) or not isinstance(i, int) for i in evidence_ids):
        errs.append("evidence_ids contains non-integers")
        return errs
    if len(set(evidence_ids)) != len(evidence_ids):
        errs.append("evidence_ids has duplicates")
    if evidence_ids != sorted(evidence_ids):
        errs.append("evidence_ids not sorted ascending")
    bad = [i for i in evidence_ids if i < 0 or i >= n_sentences]
    if bad:
        errs.append(f"evidence_ids out of range [0, {n_sentences - 1}]: {bad}")
    return errs
