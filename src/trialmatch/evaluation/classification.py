"""Classification metrics for criterion-level labels (pure Python; no torch, no sklearn).

Definitions (also written into every report):
  * An unparseable/invalid model output is the separate prediction "invalid". It is never mapped to a
    valid label, is always wrong, and counts as a false negative of its gold class (it is a false
    positive for no class). The confusion matrix gets an extra "invalid" column.
  * accuracy = correct / n, with invalid outputs in the denominator.
  * precision = tp/(tp+fp), recall = tp/(tp+fn), F1 = 2PR/(P+R). zero_division=0: a ratio with a zero
    denominator is reported as 0.0 (precision when the class is never predicted, recall when it has no
    gold support, F1 when P+R == 0), and the entry is flagged "defined": false.
  * macro-F1 (present): mean F1 over classes with gold support > 0.
    macro-F1 (full): mean F1 over the FULL label set of the scope, so a class absent from gold counts as
    F1 = 0 (zero_division=0) and pulls the average down. Both are always reported.
  * Label sets: inclusion and exclusion have their own sets. The "overall" scope pools both criterion
    types and uses the UNION of the two label sets, ordered: inclusion labels, then exclusion-only labels.
    "not enough information" and "not applicable" are therefore pooled across types in the overall scope.
"""
from trialmatch.utils.common import load_config

INVALID = "invalid"
SCOPES = ("overall", "inclusion", "exclusion")


def label_order(scope: str, cfg: dict | None = None) -> list[str]:
    """Fixed label order for a scope."""
    cfg = cfg or load_config()
    inc, exc = list(cfg["labels"]["inclusion"]), list(cfg["labels"]["exclusion"])
    if scope == "inclusion":
        return inc
    if scope == "exclusion":
        return exc
    if scope == "overall":
        return inc + [l for l in exc if l not in inc]
    raise ValueError(f"unknown scope {scope!r}")


def accuracy(gold: list[str], pred: list[str]) -> float:
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else float("nan")


def confusion_matrix(gold: list[str], pred: list[str], labels: list[str]) -> list[list[int]]:
    """Rows = gold labels (in `labels` order); columns = predicted labels in `labels` order plus a final
    "invalid" column. A predicted label outside `labels` (and not "invalid") is an error."""
    cols = list(labels) + [INVALID]
    idx_c = {l: i for i, l in enumerate(cols)}
    idx_r = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(cols) for _ in labels]
    for g, p in zip(gold, pred):
        if g not in idx_r:
            raise ValueError(f"gold label {g!r} not in label set {labels}")
        if p not in idx_c:
            raise ValueError(f"predicted label {p!r} not in label set {labels}")
        m[idx_r[g]][idx_c[p]] += 1
    return m


def _div(num: float, den: float):
    """(value, defined) with zero_division = 0."""
    return (num / den, True) if den else (0.0, False)


def per_class(gold: list[str], pred: list[str], labels: list[str]) -> dict[str, dict]:
    out = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))  # includes predicted "invalid"
        prec, pd_ = _div(tp, tp + fp)
        rec, rd = _div(tp, tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn, "tp": tp, "fp": fp,
                    "fn": fn, "precision_defined": pd_, "recall_defined": rd, "f1_defined": pd_ and rd}
    return out


def macro_f1(pc: dict[str, dict], present_only: bool) -> float:
    vals = [v["f1"] for v in pc.values() if (v["support"] > 0 or not present_only)]
    return sum(vals) / len(vals) if vals else float("nan")


def classification_report(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    n = len(gold)
    pc = per_class(gold, pred, labels)
    n_inv = sum(p == INVALID for p in pred)
    return {"n": n, "accuracy": accuracy(gold, pred), "n_invalid": n_inv,
            "invalid_rate": n_inv / n if n else float("nan"),
            "macro_f1_present_classes": macro_f1(pc, True), "macro_f1_full_label_set": macro_f1(pc, False),
            "n_classes_present_in_gold": sum(v["support"] > 0 for v in pc.values()), "n_classes_full": len(labels),
            "per_class": pc,
            "confusion": {"gold_labels_rows": list(labels), "pred_labels_cols": list(labels) + [INVALID],
                          "matrix": confusion_matrix(gold, pred, labels)}}


def report_by_scope(records: list[dict], cfg: dict | None = None) -> dict[str, dict]:
    """records: dicts with criterion_type, gold_label, pred_label (None/"invalid" -> invalid), status."""
    out = {}
    for scope in SCOPES:
        rs = records if scope == "overall" else [r for r in records if r["criterion_type"] == scope]
        gold = [r["gold_label"] for r in rs]
        pred = [r["pred_label"] if r.get("status", "ok") == "ok" and r["pred_label"] else INVALID for r in rs]
        out[scope] = classification_report(gold, pred, label_order(scope, cfg))
    return out
