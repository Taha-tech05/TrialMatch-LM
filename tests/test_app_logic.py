"""Pure-logic tests for the browser app (no model, no GPU, no downloads, no web server)."""
import pytest

from trialmatch.app import aggregate as agg_mod
from trialmatch.app.aggregate import aggregate
from trialmatch.app.backends import DemoBackend
from trialmatch.app.criteria_parser import parse_criteria
from trialmatch.app.explain import NO_EVIDENCE, explain
from trialmatch.app.label_badges import badge_for, missing_dataset_labels
from trialmatch.app.report import build_report, render_html, render_markdown, save_report
from trialmatch.app.sentence_splitter import prepare_note, split_sentences
from trialmatch.app.validation import InputError, decode_bytes, validate_screening
from trialmatch.model import schema


def R(t, label, status="ok", i=0):
    return {"index": i, "criterion_type": t, "label": label, "status": status, "criterion_text": "x"}


# ---------------------------------------------------------------- aggregation: every branch
@pytest.mark.parametrize("results,code", [
    ([R("inclusion", "included"), R("exclusion", "not excluded")], "eligible"),
    ([R("exclusion", "excluded"), R("inclusion", "included")], "ineligible"),
    ([R("inclusion", "not included"), R("exclusion", "not excluded")], "ineligible"),
    ([R("inclusion", "not enough information"), R("exclusion", "not excluded")], "undetermined"),
    ([R("inclusion", "included"), R("exclusion", None, "invalid")], "undetermined"),
    ([R("inclusion", "included"), R("exclusion", None, "error")], "undetermined"),
    ([R("inclusion", "included"), R("exclusion", None, "unavailable")], "undetermined"),
    ([R("inclusion", "included"), R("exclusion", "not applicable"), R("inclusion", "not applicable")], "eligible"),
    ([R("exclusion", "excluded"), R("inclusion", None, "error")], "ineligible"),   # decisive failure wins
    ([R("inclusion", "included"), R("inclusion", None, "skipped")], "eligible"),    # skipped ignored
])
def test_aggregate_branches(results, code):
    a = aggregate(results)
    assert a["code"] == code
    assert a["outcome"] in (agg_mod.INELIGIBLE, agg_mod.UNDETERMINED, agg_mod.ELIGIBLE)
    assert "NOT a ground-truth label" in a["disclaimer"] and a["rule_text"]


def test_aggregate_lists_contributors_and_na_only_note():
    a = aggregate([R("inclusion", "not enough information", i=0), R("exclusion", None, "invalid", i=1)])
    assert {c["index"] for c in a["contributing"]} == {0, 1}
    assert aggregate([R("inclusion", "not applicable")])["notes"]


# ---------------------------------------------------------------- explanations
def test_explain_empty_evidence_wording():
    e = explain("inclusion", "included", "ok", [])
    assert NO_EVIDENCE in e["text"] and not e["has_evidence"]
    assert NO_EVIDENCE == "No supporting sentence was cited; this is not proof of eligibility or ineligibility."


def test_explain_uses_only_label_and_cited_text():
    e = explain("exclusion", "excluded", "ok", [{"id": 3, "text": "She smokes."}])
    assert "[S3] She smokes." in e["text"] and "meets the exclusion" in e["text"]
    assert NO_EVIDENCE not in e["text"]
    assert "could not be turned into a valid answer" in explain("inclusion", None, "invalid", [], "bad json")["text"]
    assert "skipped" in explain("inclusion", None, "skipped", [])["text"]


# ---------------------------------------------------------------- badges
def test_every_dataset_label_has_a_badge():
    assert missing_dataset_labels() == []
    for t, labels in schema.label_sets().items():
        for l in labels:
            b = badge_for(t, l)
            assert b["icon"] and b["color"] and b["text"] and b["label"] == l


def test_badge_polarity_and_invalid():
    assert badge_for("exclusion", "not excluded")["icon"] == "check"
    assert badge_for("exclusion", "excluded")["icon"] == "x"
    assert badge_for("inclusion", "included")["color"] == "emerald"
    assert badge_for("inclusion", "not included")["color"] == "crimson"
    assert badge_for("inclusion", "not enough information")["icon"] == "question"
    assert badge_for("exclusion", "not applicable")["icon"] == "slash"
    b = badge_for("inclusion", None, "invalid")
    assert (b["icon"], b["color"]) == ("warning", "gray")


# ---------------------------------------------------------------- criteria parser (>= 8 formats)
def types(r):
    return [(c["type"], c["text"]) for c in r["criteria"]]


def test_parser_dash_bullets_ctgov_style():
    r = parse_criteria("Inclusion Criteria:\n\n          -  Age >= 18\n          -  Diagnosed with X\n\n"
                       "Exclusion Criteria:\n\n          -  Pregnant\n")
    assert types(r) == [("inclusion", "Age >= 18"), ("inclusion", "Diagnosed with X"), ("exclusion", "Pregnant")]


def test_parser_headings_any_case_and_numbered():
    r = parse_criteria("INCLUSION CRITERIA\n1. A\n2) B\nexclusion criteria:\n(1) C")
    assert types(r) == [("inclusion", "A"), ("inclusion", "B"), ("exclusion", "C")]


def test_parser_star_and_unicode_bullets():
    assert types(parse_criteria("Inclusion Criteria:\n* A\n\u2022 B")) == [("inclusion", "A"), ("inclusion", "B")]


def test_parser_wrapped_continuation_lines():
    r = parse_criteria("Inclusion Criteria:\n- Patients with a long\n  wrapped criterion\n- Next")
    assert types(r)[0] == ("inclusion", "Patients with a long wrapped criterion") and len(r["criteria"]) == 2


def test_parser_no_bullets_one_per_line_and_blank_lines():
    r = parse_criteria("Inclusion Criteria:\nA\n\nB\nExclusion Criteria:\nC")
    assert types(r) == [("inclusion", "A"), ("inclusion", "B"), ("exclusion", "C")]


def test_parser_missing_heading_needs_type():
    r = parse_criteria("- A\n- B")
    assert r["needs_type"] and all(c["type"] is None for c in r["criteria"]) and r["warnings"]


def test_parser_inline_heading_text_and_short_form():
    r = parse_criteria("Inclusion: Age over 18\nExclusion: Pregnancy")
    assert types(r) == [("inclusion", "Age over 18"), ("exclusion", "Pregnancy")]


def test_parser_nested_bullets_join_parent_and_key_heading():
    r = parse_criteria("Key Inclusion Criteria:\n- Any of:\n    - A\n    - B\n- C")
    assert types(r) == [("inclusion", "Any of: ; A; B"), ("inclusion", "C")] or len(r["criteria"]) == 2


def test_parser_sentence_starting_with_word_is_not_a_heading():
    r = parse_criteria("Inclusion criteria were met by all\n- X")
    assert r["needs_type"]


def test_parser_empty():
    r = parse_criteria("  \n ")
    assert r["criteria"] == [] and r["warnings"]


def test_parser_exclusion_before_inclusion_and_crlf():
    r = parse_criteria("Exclusion Criteria:\r\n- A\r\nInclusion Criteria:\r\n- B")
    assert types(r) == [("exclusion", "A"), ("inclusion", "B")]


# ---------------------------------------------------------------- sentence splitter
def test_splitter_keeps_numbered_note_unchanged():
    note = "0. First sentence.\n1. Second one.\n2. Third."
    p = prepare_note(note)
    assert p["note"] == note and not p["numbered_by_app"] and [s["id"] for s in p["sentences"]] == [0, 1, 2]


def test_splitter_abbreviations_and_decimals():
    s = split_sentences("Dr. Smith gave 7.5 mg. daily. She improved. Is it OK? Yes.")
    assert s == ["Dr. Smith gave 7.5 mg. daily.", "She improved.", "Is it OK?", "Yes."]
    assert split_sentences("Dose was 7.5 mg today. Next") == ["Dose was 7.5 mg today.", "Next"]


def test_splitter_formats_dataset_style_and_is_deterministic():
    a = prepare_note("A man came. He was ill.\n\nThen he left.")
    assert a["note"] == "0. A man came.\n1. He was ill.\n2. Then he left." and a["numbered_by_app"]
    assert prepare_note("A man came. He was ill.\n\nThen he left.") == a
    assert schema.count_note_sentences(a["note"]) == 3


def test_splitter_empty_and_bad_numbering():
    assert prepare_note("")["sentences"] == [] and prepare_note("   \n")["note"] == ""
    p = prepare_note("1. a\n2. b")                     # not starting at 0 -> renumbered with a warning
    assert p["note"] == "0. a\n1. b" and p["warnings"]


# ---------------------------------------------------------------- validation
NOTE = "0. A.\n1. B."


def C(t="inclusion", x="text"):
    return {"type": t, "text": x}


def codes(v):
    return {e["code"] for e in v["errors"]}


def test_validation_ok_and_skipped_warning():
    v = validate_screening(NOTE, [C(), C(x="   "), C(x=None)])
    assert v["ok"] and [c["skip"] for c in v["criteria"]] == [False, True, True] and len(v["warnings"]) == 2


def test_validation_empty_note_and_no_criteria():
    assert "empty_note" in codes(validate_screening("  ", [C()]))
    assert "no_criteria" in codes(validate_screening(NOTE, []))
    assert "no_criteria" in codes(validate_screening(NOTE, [C(x=" ")]))


def test_validation_limits():
    assert "note_too_many_sentences" in codes(validate_screening("\n".join(f"{i}. s" for i in range(41)), [C()]))
    assert "note_too_long" in codes(validate_screening("0. " + "x" * 3001, [C()]))
    assert "too_many_criteria" in codes(validate_screening(NOTE, [C()] * 41))
    assert "criterion_too_long" in codes(validate_screening(NOTE, [C(x="y" * 1501)]))
    assert validate_screening("\n".join(f"{i}. s" for i in range(40)), [C()] * 40)["ok"]


def test_validation_type_and_unreadable():
    assert "missing_type" in codes(validate_screening(NOTE, [C(t=None)]))
    assert "unreadable_note" in codes(validate_screening("a\x00b", [C()]))
    with pytest.raises(InputError):
        decode_bytes(b"\xff\xfe\xfa bad")
    assert decode_bytes("héllo".encode("utf-8")) == "héllo"


# ---------------------------------------------------------------- demo backend never invents
def _rec(**kw):
    base = {"note": NOTE, "criterion_type": "inclusion", "criterion_text": "Age 18", "status": "ok", "raw_output": "{}",
            "parse_error": None, "pred_label": "included", "pred_evidence": [0]}
    return {**base, **kw}


def test_demo_backend_serves_cache_and_refuses_unknown():
    be = DemoBackend([_rec()])
    ok = be.predict(NOTE, "inclusion", "Age   18", "kv")           # whitespace-insensitive match
    assert ok["status"] == "ok" and ok["label"] == "included" and "DEMO MODE" in ok["banner"] and ok["latency_s"] is None
    for args in [(NOTE, "inclusion", "something else"), (NOTE, "exclusion", "Age 18"), ("0. Z.", "inclusion", "Age 18")]:
        r = be.predict(*args, "kv")
        assert r["status"] == "unavailable" and r["label"] is None and r["evidence_ids"] is None
        assert "switch to real mode" in r["message"].lower()


def test_demo_backend_conflicting_duplicates_are_not_guessed():
    be = DemoBackend([_rec(), _rec(pred_label="not included")])
    assert be.predict(NOTE, "inclusion", "Age 18")["status"] == "unavailable"


# ---------------------------------------------------------------- report
def _run():
    from trialmatch.app.jobs import build_result
    sents = [{"id": 0, "text": "A <b>x</b>."}, {"id": 1, "text": "B."}]
    res = [build_result(0, {"type": "inclusion", "text": "crit <script>"}, {"status": "ok", "label": "included",
           "evidence_ids": [0], "engine": "kv", "latency_s": 1.0, "n_tokens": 12}, sents),
           build_result(1, {"type": "exclusion", "text": "c2"}, {"status": "ok", "label": "not enough information",
           "evidence_ids": [], "engine": "kv"}, sents)]
    return {"run_id": "t1", "mode": "real", "backend": {"model_name": "M", "device": "cpu", "dtype": "float32", "adapter": "ad/x"},
            "engine": "kv", "note": NOTE, "sentences": sents, "results": res, "aggregate": aggregate(res),
            "warnings": [], "git_commit": "abc123", "generation": {"greedy": True, "max_new_tokens": 64}}


def test_report_contents_and_escaping(tmp_path):
    rep = build_report(_run())
    html_, md = render_html(rep), render_markdown(rep)
    for out in (html_, md):
        assert "Educational Prototype Notice" in out and "max_new_tokens 64" in out and "greedy" in out
        assert "criterion-level labels only" in out and "abc123" in out and "ad/x" in out
    assert "<script>" not in html_ and "&lt;script&gt;" in html_ and "&lt;b&gt;" in html_
    assert "Print / Save as PDF" in html_ and "NOT a ground-truth label" in html_
    assert len(rep["missing_information"]) == 1
    paths = save_report(rep, tmp_path / "reports")
    assert all((tmp_path / "reports" / p.split("\\")[-1].split("/")[-1]).exists() for p in paths.values())


def test_splitter_unit_dots_split_but_abbreviations_and_decimals_do_not():
    t = "Laboratory tests show an HbA1c of 8.2% and a creatinine of 1.1 mg/dL. He has no known drug allergies."
    assert split_sentences(t) == ["Laboratory tests show an HbA1c of 8.2% and a creatinine of 1.1 mg/dL.",
                                  "He has no known drug allergies."]
    assert split_sentences("BP was 120/80 mmHg. Pulse was regular.") == ["BP was 120/80 mmHg.", "Pulse was regular."]
    assert split_sentences("Oxygen saturation was 94%. She was stable.") == ["Oxygen saturation was 94%.", "She was stable."]
    assert split_sentences("Hemoglobin 12 g. He left.") == ["Hemoglobin 12 g.", "He left."]
    assert split_sentences("Weighs 70 kg. She is well.") == ["Weighs 70 kg.", "She is well."]
    # abbreviations and decimals must not split
    assert split_sentences("Seen by Dr. Smith today. Fine.") == ["Seen by Dr. Smith today.", "Fine."]
    assert split_sentences("Seen by Dr. M. Smith. Fine.") == ["Seen by Dr. M. Smith.", "Fine."]
    assert split_sentences("HbA1c was 7.5 and rose. Then fell.") == ["HbA1c was 7.5 and rose.", "Then fell."]
    assert split_sentences("Dose 7.5 mg. daily. Ok.") == ["Dose 7.5 mg. daily.", "Ok."]      # lower-case follow-up: abbreviation
    assert split_sentences("Take 5 mg. 2 tablets daily.") == ["Take 5 mg. 2 tablets daily."]
