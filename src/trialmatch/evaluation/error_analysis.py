"""Automatic error categorisation and export tables (no explanations or conclusions are generated).

A row is an "error" if its output is invalid, its label differs from gold, or its evidence set differs from
gold. Categories (a row may have several; `primary_category` is the first by the priority below):
  invalid_output
  label_confusion:nei_gold_decided_pred   gold "not enough information", predicted a decided label
  label_confusion:decided_gold_nei_pred   gold decided label, predicted "not enough information"
  label_confusion:polarity                the two decided labels swapped (included<->not included,
                                          excluded<->not excluded)
  label_confusion:applicable_vs_not_applicable   "not applicable" involved on exactly one side
  evidence_wrong    prediction has both missing and extra sentences vs gold (includes disjoint sets)
  evidence_missing  prediction is a strict subset of gold (includes predicting [] for non-empty gold)
  evidence_extra    prediction is a strict superset of gold (includes citing anything for empty gold)
  cites_boilerplate_consent_sentence   prediction cites the boilerplate consent sentence and gold does not
"""
import re

NEI, NA = "not enough information", "not applicable"
DECIDED = {"inclusion": ("included", "not included"), "exclusion": ("excluded", "not excluded")}
CONSENT_PREFIX = "The patient will provide informed consent"
PRIORITY = ["invalid_output", "label_confusion:applicable_vs_not_applicable", "label_confusion:nei_gold_decided_pred",
            "label_confusion:decided_gold_nei_pred", "label_confusion:polarity", "evidence_wrong",
            "evidence_missing", "evidence_extra", "cites_boilerplate_consent_sentence"]
CSV_COLUMNS = ["annotation_id", "patient_id", "trial_id", "criterion_type", "criterion_text", "gold_label",
               "pred_label", "gold_evidence_ids", "pred_evidence_ids", "gold_evidence_text", "pred_evidence_text",
               "error_categories", "primary_category", "gold_cites_consent", "pred_cites_consent", "my_notes"]


def note_sentences(note: str) -> dict[int, str]:
    """Numbered note lines 'N. text' -> {N: text}."""
    return {int(m.group(1)): m.group(2) for m in re.finditer(r"(?m)^(\d+)\. (.*)$", note)}


def consent_index(sents: dict[int, str]) -> int | None:
    hits = [i for i, t in sents.items() if t.startswith(CONSENT_PREFIX)]
    return hits[-1] if hits else None


def evidence_text(ids, sents: dict[int, str]) -> str:
    return " || ".join(f"[{i}] {sents.get(i, '<out of range>')}" for i in (ids or []))


def label_category(ctype: str, gold: str, pred: str) -> str | None:
    if gold == pred:
        return None
    if NA in (gold, pred):
        return "label_confusion:applicable_vs_not_applicable"
    if gold == NEI:
        return "label_confusion:nei_gold_decided_pred"
    if pred == NEI:
        return "label_confusion:decided_gold_nei_pred"
    return "label_confusion:polarity"


def evidence_category(gold: set[int], pred: set[int]) -> str | None:
    if gold == pred:
        return None
    missing, extra = gold - pred, pred - gold
    if missing and extra:
        return "evidence_wrong"
    return "evidence_missing" if missing else "evidence_extra"


def categorize(rec: dict) -> list[str]:
    """Categories for one joined record; [] means exactly right (label and evidence)."""
    if rec["status"] != "ok" or rec["pred_label"] is None:
        return ["invalid_output"]
    cats = []
    lc = label_category(rec["criterion_type"], rec["gold_label"], rec["pred_label"])
    if lc:
        cats.append(lc)
    gold, pred = set(rec["gold_evidence"]), set(rec["pred_evidence"])
    ec = evidence_category(gold, pred)
    if ec:
        cats.append(ec)
    ci = consent_index(note_sentences(rec["note"]))
    if ci is not None and ci in pred and ci not in gold:
        cats.append("cites_boilerplate_consent_sentence")
    return sorted(cats, key=PRIORITY.index)


def export_row(rec: dict) -> dict:
    sents = note_sentences(rec["note"])
    ci = consent_index(sents)
    cats = categorize(rec)
    pred_ev = rec["pred_evidence"]
    return {
        "annotation_id": rec["annotation_id"], "patient_id": rec["patient_id"], "trial_id": rec["trial_id"],
        "criterion_type": rec["criterion_type"], "criterion_text": rec["criterion_text"],
        "gold_label": rec["gold_label"], "pred_label": rec["pred_label"] if rec["pred_label"] else "invalid",
        "gold_evidence_ids": str(rec["gold_evidence"]), "pred_evidence_ids": "" if pred_ev is None else str(pred_ev),
        "gold_evidence_text": evidence_text(rec["gold_evidence"], sents),
        "pred_evidence_text": evidence_text(pred_ev, sents),
        "error_categories": ";".join(cats), "primary_category": cats[0] if cats else "",
        "gold_cites_consent": ci is not None and ci in rec["gold_evidence"],
        "pred_cites_consent": ci is not None and ci in (pred_ev or []), "my_notes": ""}


def error_rows(records: list[dict]) -> list[dict]:
    return [r for r in (export_row(x) for x in records) if r["error_categories"]]


def category_counts(rows: list[dict], n_total: int) -> list[dict]:
    """Rows per category (multi-label: a row counts once in each of its categories) plus primary counts."""
    out = []
    for cat in PRIORITY:
        k = sum(cat in r["error_categories"].split(";") for r in rows)
        p = sum(r["primary_category"] == cat for r in rows)
        out.append({"category": cat, "n_rows": k, "n_rows_as_primary": p, "pct_of_all_rows": round(100 * k / n_total, 1)})
    out.append({"category": "ANY_ERROR (rows with >=1 category)", "n_rows": len(rows), "n_rows_as_primary": len(rows),
                "pct_of_all_rows": round(100 * len(rows) / n_total, 1)})
    return out


def pick_candidates(rows: list[dict], k: int = 12) -> list[dict]:
    """Deterministic pick covering distinct categories: rarest categories first, one row each (lowest
    annotation_id not yet chosen, preferring a patient not yet represented); remaining slots are filled
    round-robin over categories, rarest first. Each picked row records the category it was picked for."""
    cats = [c for c in PRIORITY if any(c in r["error_categories"].split(";") for r in rows)]
    cats.sort(key=lambda c: sum(c in r["error_categories"].split(";") for r in rows))
    chosen, used_ids, used_patients = [], set(), set()

    def take(cat):
        pool = sorted((r for r in rows if cat in r["error_categories"].split(";")
                       and r["annotation_id"] not in used_ids), key=lambda r: r["annotation_id"])
        pool.sort(key=lambda r: r["patient_id"] in used_patients)  # stable: unseen patients first
        if not pool:
            return False
        r = pool[0]
        chosen.append({**r, "selected_for_category": cat})
        used_ids.add(r["annotation_id"])
        used_patients.add(r["patient_id"])
        return True

    for c in cats:
        if len(chosen) < k:
            take(c)
    progress = True
    while len(chosen) < k and progress:
        progress = False
        for c in cats:
            if len(chosen) < k and take(c):
                progress = True
    return chosen
