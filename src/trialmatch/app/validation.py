"""Input validation for the screening request (pure). Messages are plain language, never stack traces."""
from trialmatch.app.sentence_splitter import prepare_note

DEFAULT_LIMITS = {"max_sentences": 40, "max_note_chars": 3000, "max_criteria": 40, "max_criterion_chars": 1500}
CRITERION_TYPES = ("inclusion", "exclusion")


class InputError(ValueError):
    """User-facing input problem (bad encoding etc.)."""


def decode_bytes(data: bytes) -> str:
    """Strict UTF-8 (a BOM is tolerated). Raises InputError with a plain message otherwise."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputError("The file is not valid UTF-8 text. Save it as plain UTF-8 text and try again.") from None
    return check_text(text)


def check_text(text) -> str:
    """Reject non-strings, NUL/binary content and undecodable replacement characters."""
    if text is None:
        return ""
    if not isinstance(text, str):
        raise InputError("The text is not readable as plain text.")
    if "\x00" in text:
        raise InputError("The text contains binary (NUL) characters and looks unreadable. Paste plain text.")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise InputError("The text contains characters that cannot be read as UTF-8.") from None
    if text.count("�") >= 3:
        raise InputError("The text contains several unreadable characters (U+FFFD); it was probably not saved as UTF-8.")
    return text


def _err(code, field, message):
    return {"code": code, "field": field, "message": message}


def validate_screening(note_text, criteria, limits: dict | None = None) -> dict:
    """-> {'ok', 'errors', 'warnings', 'note', 'sentences', 'criteria': [{'index','type','text','skip'}]}.
    criteria: list of {'type','text'} (text may be None / whitespace: skipped with a warning)."""
    lim = {**DEFAULT_LIMITS, **(limits or {})}
    errors, warnings = [], []
    note, sentences = "", []
    try:
        raw = check_text(note_text)
        if not raw.strip():
            errors.append(_err("empty_note", "note", "The patient note is empty. Load an example or paste a note."))
        else:
            prep = prepare_note(raw)
            note, sentences = prep["note"], prep["sentences"]
            warnings += [_w("note", m) for m in prep["warnings"]]
            if not sentences:
                errors.append(_err("empty_note", "note", "No sentences could be found in the note."))
            if len(sentences) > lim["max_sentences"]:
                errors.append(_err("note_too_many_sentences", "note",
                                   f"The note has {len(sentences)} sentences; the limit is {lim['max_sentences']}."))
            if len(note) > lim["max_note_chars"]:
                errors.append(_err("note_too_long", "note",
                                   f"The note has {len(note)} characters; the limit is {lim['max_note_chars']}."))
    except InputError as e:
        errors.append(_err("unreadable_note", "note", str(e)))

    out, n_real = [], 0
    if not isinstance(criteria, list):
        criteria = []
    if len(criteria) > lim["max_criteria"]:
        errors.append(_err("too_many_criteria", "criteria",
                           f"There are {len(criteria)} criteria; the limit is {lim['max_criteria']}."))
    for i, c in enumerate(criteria):
        c = c if isinstance(c, dict) else {}
        ctype, text = c.get("type"), c.get("text")
        try:
            text = check_text(text)
        except InputError as e:
            errors.append(_err("unreadable_criterion", f"criteria[{i}]", f"Criterion {i + 1}: {e}"))
            text = ""
        text = text.strip()
        skip = not text
        if skip:
            warnings.append(_w(f"criteria[{i}]", f"Criterion {i + 1} is empty and was skipped (the dataset has the "
                                                  "same case, annotation 883)."))
        else:
            n_real += 1
            if ctype not in CRITERION_TYPES:
                errors.append(_err("missing_type", f"criteria[{i}]",
                                   f"Criterion {i + 1}: choose a type (inclusion or exclusion)."))
            if len(text) > lim["max_criterion_chars"]:
                errors.append(_err("criterion_too_long", f"criteria[{i}]",
                                   f"Criterion {i + 1} has {len(text)} characters; the limit is "
                                   f"{lim['max_criterion_chars']}."))
        out.append({"index": i, "type": ctype if ctype in CRITERION_TYPES else None, "text": text, "skip": skip})
    if n_real == 0 and not any(e["code"] == "too_many_criteria" for e in errors):
        errors.append(_err("no_criteria", "criteria", "There are no criteria to screen. Paste or add at least one."))
    return {"ok": not errors, "errors": errors, "warnings": warnings, "note": note, "sentences": sentences,
            "criteria": out}


def _w(field, message):
    return {"field": field, "message": message}
