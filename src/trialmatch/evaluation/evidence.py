"""Evidence-sentence metrics (pure Python).

Rules (decided earlier, see docs/decisions.md):
  * Per row, on sets of sentence IDs: gold and prediction both empty -> P = R = F1 = 1;
    exactly one empty -> 0; otherwise the usual set precision/recall/F1.
  * An INVALID output has no usable evidence: it scores 0 on every row (even when gold is empty, so a
    broken output never earns the both-empty credit) and contributes its gold IDs as false negatives.
  * micro = pooled sentence counts (tp, fp, fn) over rows; macro = mean of per-row scores. Pooled counts
    use the same rule lifted to totals: no gold and no predicted sentences in the pool -> 1.0; a ratio whose
    denominator is zero otherwise -> 0.0. Both-empty rows add nothing to pooled counts, so micro and macro
    differ most when many rows have empty gold.
  * Every figure is reported INCLUDING and EXCLUDING rows with empty gold. Excluding drops those rows
    entirely (so over-citing on empty-gold rows is not penalised in that view).
  * Absent evidence is never treated as proof of eligibility or ineligibility: empty evidence is only a
    statement that no note sentence was cited, and is scored purely as above.
"""
NOTE = ("Absent evidence is never treated as proof of eligibility or ineligibility. Empty gold evidence is "
        "a valid answer. Both empty = P=R=F1=1; exactly one empty = 0; invalid output = 0.")
SCOPES = ("overall", "inclusion", "exclusion")


def row_prf(gold: set[int], pred: set[int] | None) -> tuple[float, float, float]:
    """Per-row precision/recall/F1. pred=None means an invalid output (always 0)."""
    if pred is None:
        return 0.0, 0.0, 0.0
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold or not pred:
        return 0.0, 0.0, 0.0
    tp = len(gold & pred)
    p, r = tp / len(pred), tp / len(gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _ratio(num: int, den: int, empty_pool: bool) -> float:
    if den:
        return num / den
    return 1.0 if empty_pool else 0.0


def pooled_prf(rows: list[tuple[set[int], set[int] | None]]) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for gold, pred in rows:
        pred = pred or set()  # invalid (None) has no predicted sentences
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    # an invalid output never earns the "nothing to cite, nothing cited" credit
    empty_pool = (tp + fp + fn) == 0 and all(pred is not None for _, pred in rows)
    p, r = _ratio(tp, tp + fp, empty_pool), _ratio(tp, tp + fn, empty_pool)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _summary(rows: list[tuple[set[int], set[int] | None]]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    scores = [row_prf(g, p) for g, p in rows]
    mp, mr, mf = pooled_prf(rows)
    return {"n": n,
            "micro": {"precision": mp, "recall": mr, "f1": mf},
            "macro": {"precision": sum(s[0] for s in scores) / n, "recall": sum(s[1] for s in scores) / n,
                      "f1": sum(s[2] for s in scores) / n}}


def _breakdown(rows: list[tuple[set[int], set[int] | None]]) -> dict:
    empty = [(g, p) for g, p in rows if not g]
    nonempty = [(g, p) for g, p in rows if g]
    valid = lambda rs: [(g, p) for g, p in rs if p is not None]
    out = {"empty_gold": {"n": len(empty)}, "nonempty_gold": {"n": len(nonempty)}}
    if empty:
        ok = sum(1 for g, p in empty if p is not None and not p)
        out["empty_gold"].update(n_pred_empty_correct=ok, rate_correctly_empty=ok / len(empty),
                                 n_pred_nonempty=sum(1 for g, p in valid(empty) if p),
                                 n_invalid=sum(1 for g, p in empty if p is None))
    if nonempty:
        exact = sum(1 for g, p in nonempty if p is not None and p == g)
        out["nonempty_gold"].update(
            n_exact_match=exact, rate_exact_match=exact / len(nonempty),
            n_pred_empty_missed=sum(1 for g, p in valid(nonempty) if not p),
            n_invalid=sum(1 for g, p in nonempty if p is None),
            **{k: v for k, v in _summary(nonempty).items() if k != "n"})
    return out


def evidence_report(records: list[dict]) -> dict:
    """records: dicts with criterion_type, gold_evidence (iterable of int), pred_evidence (iterable or None
    for invalid output)."""
    out = {"note": NOTE}
    for scope in SCOPES:
        rs = records if scope == "overall" else [r for r in records if r["criterion_type"] == scope]
        rows = [(set(r["gold_evidence"]), None if r["pred_evidence"] is None else set(r["pred_evidence"]))
                for r in rs]
        out[scope] = {"including_empty_gold": _summary(rows),
                      "excluding_empty_gold": _summary([x for x in rows if x[0]]),
                      "breakdown": _breakdown(rows)}
    return out
