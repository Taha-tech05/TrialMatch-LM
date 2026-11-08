"""Split a pasted eligibility-criteria block into inclusion / exclusion criteria (pure, deterministic).

Handles headings in any case ("Inclusion Criteria:", "EXCLUSION CRITERIA", "Key inclusion criteria", "Inclusion:"),
bullets (-, *, bullet dot, "1.", "1)", "(1)", "a)"), wrapped continuation lines, blank lines, ClinicalTrials.gov
indentation, and a missing heading (criteria then come back with type None: the user must choose).
"""
import re

_HEADING = re.compile(
    r"^\s*[#*_\s]*(?:(?:key|main|major|general|additional|other)\s+)?(inclusion|exclusion)"
    r"(\s+criteria)?\s*(?:\([^)]*\))?\s*(:?)\s*[*_]*\s*(.*)$", re.I)
_BULLET = re.compile(r"^(\s*)(?:[-*•–—·]|\d+[.)]|\(\d+\)|\(?[a-zA-Z]\))\s+(.*)$")


def _heading(line: str):
    """-> (type, inline_rest) if `line` is a section heading, else None."""
    m = _HEADING.match(line)
    if not m:
        return None
    kind, has_criteria, colon, rest = m.group(1).lower(), m.group(2), m.group(3), m.group(4).strip()
    if rest and not colon:
        return None                      # a sentence that merely starts with the word
    if not has_criteria and not colon:
        return None                      # bare word "Inclusion" needs a colon
    return kind, rest


def _clean(t: str) -> str:
    return " ".join(t.split()).strip(" \t;")


def parse_criteria(text: str) -> dict:
    """-> {'criteria': [{'type': 'inclusion'|'exclusion'|None, 'text': str}], 'warnings': [...], 'needs_type': bool}"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    sections: list[dict] = [{"type": None, "lines": [], "implicit": True}]
    for line in text.split("\n"):
        h = _heading(line) if line.strip() else None
        if h:
            sections.append({"type": h[0], "lines": [h[1]] if h[1] else [], "implicit": False})
        else:
            sections[-1]["lines"].append(line)

    criteria: list[dict] = []
    warnings: list[str] = []
    for sec in sections:
        items = _items(sec["lines"])
        if sec["implicit"] and items and len(sections) > 1:
            warnings.append(f"{len(items)} line(s) before the first heading have no type; choose one for each.")
        criteria += [{"type": sec["type"], "text": t} for t in items]
    if not criteria:
        warnings.append("No criteria were found. Paste a block with 'Inclusion Criteria:' / 'Exclusion Criteria:' "
                        "headings and one criterion per line or bullet.")
    elif any(c["type"] is None for c in criteria) and len(sections) == 1:
        warnings.append("No 'Inclusion Criteria:' / 'Exclusion Criteria:' heading was found; choose a type "
                        "(inclusion or exclusion) for each criterion.")
    return {"criteria": criteria, "warnings": warnings, "needs_type": any(c["type"] is None for c in criteria)}


def _items(lines: list[str]) -> list[str]:
    bulleted = any(_BULLET.match(l) for l in lines if l.strip())
    items: list[list] = []                  # [text, indent]
    blank_before = True
    base_indent = None
    for line in lines:
        if not line.strip():
            blank_before = True
            continue
        indent = len(line) - len(line.lstrip())
        m = _BULLET.match(line)
        if m:
            ind = len(m.group(1))
            if base_indent is None:
                base_indent = ind
            if items and ind >= base_indent + 2:          # nested bullet: part of its parent
                items[-1][0] += "; " + m.group(2).strip()
            else:
                items.append([m.group(2).strip(), ind])
        elif not items:
            items.append([line.strip(), indent])
        elif bulleted:
            if blank_before:
                items.append([line.strip(), indent])      # new paragraph-level criterion
            else:
                items[-1][0] += " " + line.strip()        # wrapped continuation
        else:                                             # no bullets: one criterion per line, unless indented deeper
            if indent > items[-1][1] + 1 and not blank_before:
                items[-1][0] += " " + line.strip()
            else:
                items.append([line.strip(), indent])
        blank_before = False
    return [c for c in (_clean(t) for t, _ in items) if c]
