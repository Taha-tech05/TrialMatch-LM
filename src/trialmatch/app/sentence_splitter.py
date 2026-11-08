"""Deterministic sentence splitter producing the dataset's numbered note format ("0. sentence\\n1. sentence").

Already-numbered notes (lines "0. ...", "1. ...", contiguous from 0) are kept as they are. No ML, no randomness.
"""
import re

_NUM_LINE = re.compile(r"^\s*(\d+)\s*\.\s+(.*\S)\s*$")
# Tokens (lower-case, without the final dot) after which a full stop does NOT end a sentence.
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "approx", "no", "fig", "inc", "ltd", "dept",
    "mg", "mcg", "ml", "kg", "g", "l", "dl", "mm", "cm", "hr", "hrs", "min", "sec", "mmhg", "yo", "y.o", "wk",
    "e.g", "i.e", "a.m", "p.m", "cf", "ca", "pt", "pts", "hx", "dx", "tx", "rx", "sx", "w/", "ft", "lb", "lbs", "oz",
}
_BOUNDARY = re.compile(r"([.!?]+[\"')\]]*)(\s+)(?=[\"'(\[]?[A-Z0-9])")


def _is_abbrev(before: str) -> bool:
    m = re.search(r"([A-Za-z][A-Za-z.]*)$", before)
    if not m:
        return False
    tok = m.group(1).lower().rstrip(".")
    if tok in ABBREVIATIONS:
        return True
    # single capital initial ("J. Smith") or dotted forms like "U.S"
    return bool(re.fullmatch(r"[a-z]", tok)) or bool(re.fullmatch(r"(?:[a-z]\.)+[a-z]", tok))


def split_sentences(text: str) -> list[str]:
    """Split free text into sentences. Blank lines are hard boundaries; single newlines are plain whitespace."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        para = " ".join(para.split())
        if not para:
            continue
        start = 0
        for m in _BOUNDARY.finditer(para):
            end_punct = m.group(1)
            before = para[start:m.start()]
            if end_punct.startswith(".") and _is_abbrev(before):
                continue
            out.append(para[start:m.start() + len(end_punct)].strip())
            start = m.end()
        tail = para[start:].strip()
        if tail:
            out.append(tail)
    return [s for s in out if s]


def parse_numbered(text: str) -> list[tuple[int, str]] | None:
    """If `text` is a dataset-style numbered note (ids contiguous from 0), return [(id, sentence)], else None.
    Unnumbered lines after a numbered one are treated as wrapped continuations of that sentence."""
    lines = [l for l in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if l.strip()]
    if not lines:
        return None
    items: list[list] = []
    for l in lines:
        m = _NUM_LINE.match(l)
        if m and int(m.group(1)) == len(items):
            items.append([len(items), m.group(2).strip()])
        elif items and not m:
            items[-1][1] += " " + " ".join(l.split())
        else:
            return None
    return [(i, s) for i, s in items] if items else None


def format_numbered(sentences: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))


def prepare_note(text: str) -> dict:
    """-> {'note': numbered text, 'sentences': [{'id','text'}], 'numbered_by_app': bool, 'warnings': [...]}.
    Never raises on odd input; an empty result has note == ''."""
    text = text or ""
    warnings: list[str] = []
    parsed = parse_numbered(text)
    if parsed is not None:
        sents = [s for _, s in parsed]
        return {"note": format_numbered(sents), "sentences": [{"id": i, "text": s} for i, s in parsed],
                "numbered_by_app": False, "warnings": warnings}
    # Numbered but not from 0 / not contiguous: strip the prefixes and renumber, with a warning.
    raw_lines = [l for l in text.replace("\r\n", "\n").split("\n") if l.strip()]
    if raw_lines and all(_NUM_LINE.match(l) for l in raw_lines):
        sents = [_NUM_LINE.match(l).group(2).strip() for l in raw_lines]
        warnings.append("The note's own numbering does not run 0, 1, 2, ... without gaps; it was renumbered from 0 "
                        "to match the dataset format the model expects.")
    else:
        sents = split_sentences(text)
        if sents:
            warnings.append(f"The note was not numbered like the dataset; it was split into {len(sents)} "
                            "sentence(s) and numbered from 0. Check the split below.")
    return {"note": format_numbered(sents), "sentences": [{"id": i, "text": s} for i, s in enumerate(sents)],
            "numbered_by_app": bool(sents), "warnings": warnings}
