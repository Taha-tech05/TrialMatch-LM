"""Join saved predictions to gold labels/notes from the processed train/val files (never test).

Only an allow-list of columns is ever read from parquet, and the file schema is asserted free of the
forbidden previous-prediction columns before reading.
"""
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from trialmatch.model.prompt import parse_gold_evidence
from trialmatch.utils.common import load_config, resolve

READ_SPLITS = ("train", "val")
SENTENCE_RE_PREFIX = "The patient will provide informed consent"


def load_gold(split: str = "val", cfg: dict | None = None, allow_test: bool = False) -> pd.DataFrame:
    """Gold rows for train/val with the columns needed for metrics and error analysis, skipping
    known-skipped annotations. The test split is refused unless allow_test=True (final evaluation only)."""
    if split == "test" and allow_test:
        pass
    elif split not in READ_SPLITS:
        raise PermissionError(f"load_gold reads only {READ_SPLITS}, not {split!r} (test needs the final-evaluation flag)")
    cfg = cfg or load_config()
    c = cfg["columns"]
    path = resolve(cfg["paths"]["processed_dir"]) / f"{split}.parquet"
    leaked = set(cfg["forbidden_columns"]) & set(pq.read_schema(path).names)
    if leaked:
        raise AssertionError(f"forbidden columns present in {path.name}: {sorted(leaked)}")
    cols = [c["id"], c["patient"], c["trial"], c["criterion_type"], c["criterion_text"], c["note"],
            c["label"], c["evidence"]]
    df = pd.read_parquet(path, columns=cols)
    skipped = {s["annotation_id"] for s in cfg["known_skipped_rows"]}
    df = df[~df[c["id"]].isin(skipped)].reset_index(drop=True)
    df["gold_evidence"] = df[c["evidence"]].map(parse_gold_evidence)
    return df


def load_predictions(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def join_predictions(preds: list[dict], gold: pd.DataFrame, cfg: dict | None = None) -> list[dict]:
    """One merged record per gold row. Asserts every gold annotation appears exactly once in the
    predictions (and nothing else), and that gold labels/evidence in the file match the processed data."""
    cfg = cfg or load_config()
    c = cfg["columns"]
    ids = [p["annotation_id"] for p in preds]
    assert len(ids) == len(set(ids)), "duplicate annotation_id in predictions"
    expected = set(gold[c["id"]])
    missing, extra = expected - set(ids), set(ids) - expected
    assert not missing and not extra, f"prediction ids mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
    by_id = {p["annotation_id"]: p for p in preds}
    out = []
    for _, g in gold.iterrows():
        p = by_id[int(g[c["id"]])]
        assert p["gold_label"] == g[c["label"]], f"gold label mismatch for {g[c['id']]}"
        assert json.loads(p["gold_target"])["evidence_ids"] == g["gold_evidence"], f"gold evidence mismatch {g[c['id']]}"
        ok = p["status"] == "ok"
        out.append({
            "annotation_id": int(g[c["id"]]), "patient_id": g[c["patient"]], "trial_id": g[c["trial"]],
            "criterion_type": g[c["criterion_type"]], "criterion_text": g[c["criterion_text"]], "note": g[c["note"]],
            "gold_label": g[c["label"]], "gold_evidence": list(g["gold_evidence"]),
            "status": p["status"], "raw_output": p["raw_output"], "parse_error": p.get("parse_error"),
            "pred_label": p["pred_label"] if ok else None,
            "pred_evidence": p["pred_evidence_ids"] if ok else None})
    return out


def load_joined(tag: str, split: str = "val", eval_dir: Path | None = None, allow_test: bool = False) -> list[dict]:
    cfg = load_config()
    eval_dir = eval_dir or resolve("results/eval")
    gold = load_gold(split, cfg, allow_test=allow_test)
    return join_predictions(load_predictions(eval_dir / f"{tag}_{split}_predictions.jsonl"), gold, cfg)


def check_metrics_split(split: str, confirm_final: bool) -> bool:
    """Guard for scripts/compute_metrics.py and scripts/export_errors.py: returns allow_test.
    Test data is read only with split == 'test' AND confirm_final; the flag alone never reads it."""
    if split not in ("val", "test"):
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and not confirm_final:
        raise PermissionError("Refusing to read the test split: pass --split test --confirm-final "
                              "(final evaluation only).")
    if split == "val" and confirm_final:
        raise ValueError("--confirm-final only applies to --split test")
    return split == "test"
