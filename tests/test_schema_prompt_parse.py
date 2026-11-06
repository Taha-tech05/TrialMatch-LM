"""Tests for label sets, schema validation, prompt formatter and output parser."""
import json

import pandas as pd
import pytest

from trialmatch.model import schema
from trialmatch.model.parse import parse_output
from trialmatch.model.prompt import (SkipRow, build_prompt, build_target, example_from_row,
                                     parse_gold_evidence)
from trialmatch.utils.common import load_config, resolve

CFG = load_config()
C = CFG["columns"]
PROC = resolve(CFG["paths"]["processed_dir"])
SKIP_IDS = {k["annotation_id"] for k in CFG["known_skipped_rows"]}

needs_data = pytest.mark.skipif(not (PROC / "train.parquet").exists(),
                                reason="processed files missing; run scripts/prepare_data.py")


@pytest.fixture(scope="module")
def splits():
    return {n: pd.read_parquet(PROC / f"{n}.parquet") for n in ("train", "val", "test")}


@pytest.fixture(scope="module")
def all_rows(splits):
    return pd.concat(splits.values(), ignore_index=True)


def usable(df):
    return df[~df[C["id"]].isin(SKIP_IDS)]


# ---------- gold data checks
@needs_data
def test_label_sets_match_data(all_rows):
    schema.verify_label_sets(all_rows)


@needs_data
def test_gold_roundtrip_all_splits(all_rows):
    n = 0
    for _, row in usable(all_rows).iterrows():
        ex = example_from_row(row.to_dict())
        pred = parse_output(ex.target, row[C["criterion_type"]], schema.count_note_sentences(row[C["note"]]))
        assert pred.parse_ok, pred.error
        assert pred.label == row[C["label"]]
        assert pred.evidence_ids == parse_gold_evidence(row[C["evidence"]])
        n += 1
    assert n == len(all_rows) - len(SKIP_IDS)


@needs_data
def test_gold_labels_valid_and_evidence_in_range(all_rows):
    for _, r in all_rows.iterrows():
        assert r[C["label"]] in schema.label_sets()[r[C["criterion_type"]]]
        ids = parse_gold_evidence(r[C["evidence"]])
        assert all(0 <= i < schema.count_note_sentences(r[C["note"]]) for i in ids)


@needs_data
def test_prompt_contents_and_no_forbidden_text(all_rows):
    raw = pd.read_parquet(resolve(CFG["paths"]["processed_dir"]) /
                          "forbidden_previous_predictions_NOT_FOR_MODEL_USE.parquet")
    expl = raw["gpt4_explanation"].dropna().unique()
    for _, r in usable(all_rows).head(60).iterrows():
        p = example_from_row(r.to_dict()).prompt
        assert r[C["note"]] in p and r[C["criterion_text"]].strip() in p
        assert f"Criterion type: {r[C['criterion_type']]}" in p
        assert all(l in p for l in schema.label_sets()[r[C["criterion_type"]]])
        own_inputs = r[C["note"]] + "\n" + r[C["criterion_text"]]
        # an explanation may legitimately equal a generic criterion sentence; it must not come from elsewhere
        assert not any(e in p and e not in own_inputs for e in expl)


@needs_data
def test_forbidden_columns_rejected_by_formatter(all_rows):
    row = all_rows.iloc[0].to_dict()
    row["gpt4_explanation"] = "leak"
    with pytest.raises(AssertionError, match="forbidden"):
        example_from_row(row)


@needs_data
def test_null_criterion_row_handled(all_rows):
    r = all_rows[all_rows[C["id"]].isin(SKIP_IDS)]
    assert len(r) == len(SKIP_IDS) == 1 and r[C["criterion_text"]].isna().all()
    with pytest.raises(SkipRow):
        example_from_row(r.iloc[0].to_dict())


def test_empty_criterion_text_skipped():
    for bad in (None, "", "   ", float("nan")):
        with pytest.raises(SkipRow):
            build_prompt(note="0. a.", criterion_type="inclusion", criterion_text=bad)


# ---------- parser
NOTE_N = 5


def P(text, ctype="inclusion"):
    return parse_output(text, ctype, NOTE_N)


def test_parse_wellformed():
    p = P('{"label": "included", "evidence_ids": [0, 2]}')
    assert p.parse_ok and p.status == "ok" and p.label == "included" and p.evidence_ids == [0, 2]


def test_parse_surrounding_text():
    p = P('Sure! Here you go: {"label": "not applicable", "evidence_ids": []} hope it helps {x}')
    assert p.parse_ok and p.label == "not applicable" and p.evidence_ids == []


def test_parse_target_builder_output_roundtrip():
    assert P(build_target("included", [1, 3])).evidence_ids == [1, 3]


@pytest.mark.parametrize("text", ["", "   ", None, "no json here", '{"label": "included", "evidence_ids": [0,',
                                  '{"label": "included"}', '[1,2]', '{"label": "included", "evidence_ids": [], "x": 1}'])
def test_parse_malformed_never_crashes(text):
    p = P(text)
    assert not p.parse_ok and p.status == "invalid" and p.label is None and p.error


def test_parse_wrong_label_for_type():
    p = P('{"label": "not excluded", "evidence_ids": []}', "inclusion")
    assert p.status == "invalid" and p.label is None and "not allowed" in p.error
    assert P('{"label": "not excluded", "evidence_ids": []}', "exclusion").parse_ok


def test_parse_unknown_label_not_remapped():
    p = P('{"label": "Included!", "evidence_ids": []}')
    assert p.status == "invalid" and p.label is None


@pytest.mark.parametrize("ids", ["[5]", "[-1]", "[0, 0]", "[2, 1]", '["1"]', "[1.5]", "[true]", "3"])
def test_parse_bad_evidence(ids):
    p = P('{"label": "included", "evidence_ids": %s}' % ids)
    assert p.status == "invalid" and p.evidence_ids is None


def test_parse_empty_evidence_valid():
    assert P('{"label": "not enough information", "evidence_ids": []}').parse_ok
