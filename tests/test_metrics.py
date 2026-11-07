"""Hand-computed tests for classification/evidence metrics, baselines, error categories. No torch."""
import json

import pandas as pd
import pytest

from trialmatch.evaluation import error_analysis as ea
from trialmatch.evaluation.baselines import baseline_records, majority_labels
from trialmatch.evaluation.classification import (INVALID, accuracy, classification_report, confusion_matrix,
                                                  label_order, per_class, report_by_scope)
from trialmatch.evaluation.evidence import evidence_report, pooled_prf, row_prf
from trialmatch.evaluation.val_data import join_predictions, load_gold, load_joined, load_predictions
from trialmatch.utils.common import load_config, resolve

INC = ["included", "not included", "not enough information", "not applicable"]
# gold:  inc  inc  inc  notinc  NEI  NEI
# pred:  inc  inc  NEI  inc     NEI  invalid
GOLD = ["included", "included", "included", "not included", "not enough information", "not enough information"]
PRED = ["included", "included", "not enough information", "included", "not enough information", INVALID]


def test_accuracy_hand_computed():
    assert accuracy(GOLD, PRED) == 0.5  # rows 0, 1, 4 correct; the invalid row is wrong


def test_per_class_hand_computed():
    pc = per_class(GOLD, PRED, INC)
    inc = pc["included"]  # tp 2, fp 1 (row 3), fn 1 (row 2)
    assert (inc["tp"], inc["fp"], inc["fn"], inc["support"]) == (2, 1, 1, 3)
    assert inc["precision"] == pytest.approx(2 / 3) and inc["recall"] == pytest.approx(2 / 3) and inc["f1"] == pytest.approx(2 / 3)
    ni = pc["not included"]  # never predicted, one gold -> P undefined (0), R = 0
    assert (ni["precision"], ni["recall"], ni["f1"], ni["support"]) == (0.0, 0.0, 0.0, 1)
    assert ni["precision_defined"] is False and ni["recall_defined"] is True
    nei = pc["not enough information"]  # tp 1, fp 1 (row 2), fn 1 (row 5: invalid counts as a miss)
    assert (nei["tp"], nei["fp"], nei["fn"]) == (1, 1, 1)
    assert nei["precision"] == 0.5 and nei["recall"] == 0.5 and nei["f1"] == 0.5
    na = pc["not applicable"]  # absent from gold and never predicted
    assert na["support"] == 0 and na["f1"] == 0.0 and na["recall_defined"] is False


def test_both_macro_f1_variants_with_class_absent_from_gold():
    rep = classification_report(GOLD, PRED, INC)
    # F1s: 2/3, 0, 1/2, (absent) 0 -> present classes = 3, full label set = 4
    assert rep["macro_f1_present_classes"] == pytest.approx((2 / 3 + 0 + 1 / 2) / 3)   # 7/18
    assert rep["macro_f1_full_label_set"] == pytest.approx((2 / 3 + 0 + 1 / 2) / 4)    # 7/24
    assert rep["n_classes_present_in_gold"] == 3 and rep["n_classes_full"] == 4
    assert rep["invalid_rate"] == pytest.approx(1 / 6) and rep["n_invalid"] == 1


def test_confusion_matrix_order_and_invalid_column():
    m = confusion_matrix(GOLD, PRED, INC)
    # rows = gold in INC order, cols = INC order + invalid
    assert m == [[2, 0, 1, 0, 0],
                 [1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 1],
                 [0, 0, 0, 0, 0]]
    rep = classification_report(GOLD, PRED, INC)
    assert rep["confusion"]["pred_labels_cols"] == INC + [INVALID]
    with pytest.raises(ValueError):
        confusion_matrix(["included"], ["excluded"], INC)  # a label from the other type is not silently accepted


def test_invalid_never_mapped_to_a_label():
    recs = [{"criterion_type": "inclusion", "gold_label": "included", "pred_label": None, "status": "invalid"},
            {"criterion_type": "inclusion", "gold_label": "included", "pred_label": "included", "status": "ok"}]
    rep = report_by_scope(recs)["inclusion"]
    assert rep["accuracy"] == 0.5 and rep["invalid_rate"] == 0.5
    assert rep["per_class"]["included"]["recall"] == 0.5 and rep["per_class"]["included"]["precision"] == 1.0
    assert sum(map(sum, rep["confusion"]["matrix"])) == 2 and rep["confusion"]["matrix"][0][-1] == 1


def test_label_order_overall_is_union():
    assert label_order("overall") == ["included", "not included", "not enough information", "not applicable",
                                      "excluded", "not excluded"]
    assert label_order("exclusion") == ["excluded", "not excluded", "not enough information", "not applicable"]


def test_scopes_split_and_overall_union():
    recs = [{"criterion_type": "inclusion", "gold_label": "included", "pred_label": "included", "status": "ok"},
            {"criterion_type": "exclusion", "gold_label": "not excluded", "pred_label": "excluded", "status": "ok"},
            {"criterion_type": "exclusion", "gold_label": "not excluded", "pred_label": "not excluded", "status": "ok"}]
    r = report_by_scope(recs)
    assert (r["overall"]["n"], r["inclusion"]["n"], r["exclusion"]["n"]) == (3, 1, 2)
    assert r["overall"]["accuracy"] == pytest.approx(2 / 3) and r["exclusion"]["accuracy"] == 0.5
    assert len(r["overall"]["confusion"]["matrix"]) == 6  # union of labels


def test_matches_sklearn_cross_check():
    sk = pytest.importorskip("sklearn.metrics")
    p, r, f, s = sk.precision_recall_fscore_support(GOLD, PRED, labels=INC, zero_division=0)
    pc = per_class(GOLD, PRED, INC)
    for i, lab in enumerate(INC):
        assert pc[lab]["precision"] == pytest.approx(p[i]) and pc[lab]["recall"] == pytest.approx(r[i])
        assert pc[lab]["f1"] == pytest.approx(f[i]) and pc[lab]["support"] == s[i]
    rep = classification_report(GOLD, PRED, INC)
    assert rep["macro_f1_full_label_set"] == pytest.approx(sk.f1_score(GOLD, PRED, labels=INC, average="macro", zero_division=0))
    present = [l for l in INC if l in GOLD]
    assert rep["macro_f1_present_classes"] == pytest.approx(sk.f1_score(GOLD, PRED, labels=present, average="macro", zero_division=0))
    assert accuracy(GOLD, PRED) == pytest.approx(sk.accuracy_score(GOLD, PRED))


# ---------------------------------------------------------------- evidence
def test_row_prf_cases():
    assert row_prf(set(), set()) == (1.0, 1.0, 1.0)                    # both empty
    assert row_prf({1}, set()) == (0.0, 0.0, 0.0)                      # exactly one empty
    assert row_prf(set(), {2}) == (0.0, 0.0, 0.0)
    assert row_prf({4}, {4}) == (1.0, 1.0, 1.0)                        # exact match
    p, r, f = row_prf({0, 1, 2}, {1, 2, 3})                            # partial overlap: tp = 2
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3) and f == pytest.approx(2 / 3)
    p, r, f = row_prf({0, 1, 2, 3}, {0})                               # subset: P = 1, R = 1/4
    assert (p, r) == (1.0, 0.25) and f == pytest.approx(2 * 0.25 / 1.25)
    assert row_prf(set(), None) == (0.0, 0.0, 0.0)                     # invalid output never earns the both-empty credit


def _ev(rows, ctype="inclusion"):
    return [{"criterion_type": ctype, "gold_evidence": g, "pred_evidence": p} for g, p in rows]


def test_micro_vs_macro_evidence_hand_computed():
    # A exact {0,1,2,3}; B gold {5} predicted []; C both empty; D gold [] predicted {7}
    rows = [([0, 1, 2, 3], [0, 1, 2, 3]), ([5], []), ([], []), ([], [7])]
    ev = evidence_report(_ev(rows))["overall"]
    inc = ev["including_empty_gold"]
    assert inc["n"] == 4
    assert inc["macro"]["f1"] == pytest.approx(0.5)            # per-row F1: 1, 0, 1, 0
    assert inc["macro"]["precision"] == pytest.approx(0.5) and inc["macro"]["recall"] == pytest.approx(0.5)
    # pooled: tp 4, fp 1 (row D), fn 1 (row B) -> P = R = F1 = 0.8
    assert inc["micro"]["precision"] == pytest.approx(0.8) and inc["micro"]["f1"] == pytest.approx(0.8)
    exc = ev["excluding_empty_gold"]                           # rows C, D dropped
    assert exc["n"] == 2 and exc["macro"]["f1"] == pytest.approx(0.5)
    assert exc["micro"]["precision"] == pytest.approx(1.0)     # tp 4, fp 0
    assert exc["micro"]["recall"] == pytest.approx(0.8)        # fn 1
    assert exc["micro"]["f1"] == pytest.approx(2 * 1.0 * 0.8 / 1.8)
    b = ev["breakdown"]
    assert b["empty_gold"]["n"] == 2 and b["empty_gold"]["n_pred_empty_correct"] == 1
    assert b["empty_gold"]["rate_correctly_empty"] == 0.5
    assert b["nonempty_gold"]["n"] == 2 and b["nonempty_gold"]["n_exact_match"] == 1
    assert b["nonempty_gold"]["n_pred_empty_missed"] == 1
    assert "never treated as proof" in evidence_report(_ev(rows))["note"]


def test_pooled_edge_cases_and_invalid():
    assert pooled_prf([(set(), set()), (set(), set())]) == (1.0, 1.0, 1.0)   # nothing to cite, nothing cited
    assert pooled_prf([({1, 2}, None)]) == (0.0, 0.0, 0.0)                    # invalid: misses gold, no credit
    rep = evidence_report(_ev([([], None)]))["overall"]["including_empty_gold"]
    assert rep["macro"]["f1"] == 0.0 and rep["micro"]["f1"] == 0.0           # invalid on empty gold is not "correct"


def test_evidence_scopes_separate():
    recs = _ev([([1], [1])], "inclusion") + _ev([([2], [3])], "exclusion")
    ev = evidence_report(recs)
    assert ev["inclusion"]["including_empty_gold"]["macro"]["f1"] == 1.0
    assert ev["exclusion"]["including_empty_gold"]["macro"]["f1"] == 0.0
    assert ev["overall"]["including_empty_gold"]["macro"]["f1"] == 0.5


# ---------------------------------------------------------------- baselines
def test_majority_labels_use_given_train_rows_and_tie_break():
    cfg = load_config()
    c = cfg["columns"]
    df = pd.DataFrame({c["criterion_type"]: ["inclusion"] * 4 + ["exclusion"] * 3,
                       c["label"]: ["included", "included", "not enough information", "not enough information",
                                    "not excluded", "excluded", "not excluded"]})
    maj = majority_labels(df, cfg)
    assert maj["inclusion"] == "included"      # tie 2-2 -> first in label-set order
    assert maj["exclusion"] == "not excluded"
    val = [{"criterion_type": "inclusion", "gold_label": "included", "gold_evidence": [1], "status": "ok",
            "pred_label": "x", "pred_evidence": [9], "raw_output": "r", "parse_error": None}]
    b = baseline_records(val, maj)[0]
    assert b["pred_label"] == "included" and b["pred_evidence"] == [] and "raw_output" not in b


# ---------------------------------------------------------------- error analysis
def test_label_and_evidence_categories():
    NEI, NA = "not enough information", "not applicable"
    assert ea.label_category("inclusion", "included", "included") is None
    assert ea.label_category("inclusion", NEI, "included") == "label_confusion:nei_gold_decided_pred"
    assert ea.label_category("exclusion", "excluded", NEI) == "label_confusion:decided_gold_nei_pred"
    assert ea.label_category("inclusion", "included", "not included") == "label_confusion:polarity"
    assert ea.label_category("exclusion", "not excluded", NA) == "label_confusion:applicable_vs_not_applicable"
    assert ea.label_category("exclusion", NA, NEI) == "label_confusion:applicable_vs_not_applicable"
    assert ea.evidence_category({1, 2}, {1, 2}) is None
    assert ea.evidence_category({1, 2}, {1}) == "evidence_missing"
    assert ea.evidence_category({1, 2}, set()) == "evidence_missing"
    assert ea.evidence_category({1}, {1, 2}) == "evidence_extra"
    assert ea.evidence_category(set(), {3}) == "evidence_extra"
    assert ea.evidence_category({1, 2}, {2, 3}) == "evidence_wrong"
    assert ea.evidence_category({1}, {4}) == "evidence_wrong"


NOTE = ("0. First sentence.\n1. Second sentence.\n"
        "2. The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")


def _rec(**kw):
    base = {"annotation_id": 1, "patient_id": "p", "trial_id": "t", "criterion_type": "inclusion", "criterion_text": "c",
            "note": NOTE, "gold_label": "included", "gold_evidence": [0], "status": "ok", "pred_label": "included",
            "pred_evidence": [0], "raw_output": "", "parse_error": None}
    return {**base, **kw}


def test_categorize_and_consent_and_text():
    assert ea.note_sentences(NOTE)[1] == "Second sentence." and ea.consent_index(ea.note_sentences(NOTE)) == 2
    assert ea.categorize(_rec()) == []
    assert ea.categorize(_rec(status="invalid", pred_label=None, pred_evidence=None)) == ["invalid_output"]
    assert ea.categorize(_rec(pred_evidence=[0, 2])) == ["evidence_extra", "cites_boilerplate_consent_sentence"]
    assert ea.categorize(_rec(gold_evidence=[0, 2], pred_evidence=[0, 2])) == []   # gold cites it too: not flagged
    cats = ea.categorize(_rec(pred_label="not included", pred_evidence=[1]))
    assert cats == ["label_confusion:polarity", "evidence_wrong"]
    row = ea.export_row(_rec(pred_evidence=[0, 2]))
    assert row["pred_cites_consent"] is True and row["gold_cites_consent"] is False and row["my_notes"] == ""
    assert row["gold_evidence_text"] == "[0] First sentence." and row["pred_evidence_text"].startswith("[0] First sentence. || [2] The patient")
    assert set(ea.CSV_COLUMNS) <= set(row)


def test_pick_candidates_covers_distinct_categories_deterministically():
    rows = []
    for i, cats in enumerate([["label_confusion:polarity"], ["evidence_missing"], ["evidence_missing"], ["evidence_extra"],
                              ["evidence_extra", "cites_boilerplate_consent_sentence"], ["invalid_output"]]):
        rows.append({"annotation_id": i, "patient_id": f"p{i % 2}", "error_categories": ";".join(cats), "primary_category": cats[0]})
    cand = ea.pick_candidates(rows, 4)
    assert len(cand) == 4 and len({r["annotation_id"] for r in cand}) == 4
    assert {r["selected_for_category"] for r in cand} == {"invalid_output", "label_confusion:polarity",
                                                          "cites_boilerplate_consent_sentence", "evidence_extra"} or \
        len({r["selected_for_category"] for r in cand}) == 4
    assert cand == ea.pick_candidates(rows, 4)       # deterministic
    assert len(ea.pick_candidates(rows, 12)) == 6    # never more than available


# ---------------------------------------------------------------- data join (real files, val only)
needs_files = pytest.mark.skipif(
    not ((resolve("data/processed/val.parquet")).exists() and resolve("results/eval/finetuned_val_predictions.jsonl").exists()),
    reason="val data or prediction files missing")


def test_test_split_is_refused():
    with pytest.raises(PermissionError):
        load_gold("test")


@needs_files
def test_join_covers_every_val_row_once_and_matches_summary():
    gold = load_gold("val")
    recs = load_joined("finetuned", "val")
    assert sorted(r["annotation_id"] for r in recs) == sorted(gold["annotation_id"]) and len(recs) == len(gold)
    assert not set(load_config()["forbidden_columns"]) & (set(recs[0]) | set(gold.columns))
    summary = json.loads(resolve("results/eval/finetuned_val_summary.json").read_text())
    rep = report_by_scope(recs)["overall"]
    assert rep["accuracy"] == pytest.approx(summary["accuracy"]) and rep["invalid_rate"] == pytest.approx(summary["invalid_rate"])


@needs_files
def test_join_rejects_missing_and_duplicate_ids():
    gold = load_gold("val")
    preds = load_predictions(resolve("results/eval/finetuned_val_predictions.jsonl"))
    with pytest.raises(AssertionError, match="mismatch"):
        join_predictions(preds[1:], gold)
    with pytest.raises(AssertionError, match="duplicate"):
        join_predictions(preds + preds[:1], gold)
