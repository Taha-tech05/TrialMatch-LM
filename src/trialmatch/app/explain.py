"""Template explanations built ONLY from the label and the cited sentences. Never calls a model, never adds facts."""

NO_EVIDENCE = "No supporting sentence was cited; this is not proof of eligibility or ineligibility."

_TEMPLATES = {
    ("inclusion", "included"): "The model labelled this inclusion criterion 'included': the cited sentences were "
                               "judged to show that the patient meets it.",
    ("inclusion", "not included"): "The model labelled this inclusion criterion 'not included': the cited sentences "
                                   "were judged to show that the patient does not meet it.",
    ("exclusion", "excluded"): "The model labelled this exclusion criterion 'excluded': the cited sentences were "
                               "judged to show that the patient meets the exclusion (this counts against eligibility).",
    ("exclusion", "not excluded"): "The model labelled this exclusion criterion 'not excluded': the cited sentences "
                                   "were judged to show that the patient does not meet the exclusion (favourable).",
}
_NEI = ("The model labelled this criterion 'not enough information': the note was judged not to contain "
        "enough to decide it.")
_NA = "The model labelled this criterion 'not applicable' to this patient."
_FAIL = {"invalid": "The model output could not be turned into a valid answer",
         "error": "This criterion could not be evaluated",
         "unavailable": "No prediction is available for this criterion"}


def explain(criterion_type: str, label: str | None, status: str, cited: list[dict], error: str | None = None) -> dict:
    """cited: [{'id': int, 'text': str}]. Returns {'text', 'summary', 'evidence_lines', 'has_evidence'}."""
    if status == "skipped":
        s = "This criterion was empty and was skipped; no prediction was made."
        return {"text": s, "summary": s, "evidence_lines": [], "has_evidence": False}
    if status != "ok":
        what = _FAIL.get(status, "This criterion has no valid result")
        s = f"{what}: {error}." if error else f"{what}."
        if status != "unavailable":
            s += " No label is available."
        return {"text": s, "summary": s, "evidence_lines": [], "has_evidence": False}
    if label == "not enough information":
        summary = _NEI
    elif label == "not applicable":
        summary = _NA
    else:
        summary = _TEMPLATES.get((criterion_type, label), f"The model labelled this {criterion_type} criterion '{label}'.")
    lines = [f"[S{c['id']}] {c['text']}" for c in cited]
    text = summary + (" Cited evidence: " + " ".join(lines) if lines else " " + NO_EVIDENCE)
    return {"text": text, "summary": summary, "evidence_lines": lines, "has_evidence": bool(lines),
            "no_evidence_text": None if lines else NO_EVIDENCE}
