"""Label -> badge mapping (pure). Badges are chosen by OUTCOME, not by the raw label string.

For an exclusion criterion 'not excluded' is the favourable outcome and 'excluded' the unfavourable one, so the
mapping is by (criterion_type, label). Every badge carries an icon name, a colour key AND text: never colour alone.
"""
from trialmatch.model import schema

INVALID_KEY = "invalid"

# outcome key -> (icon, colour class, human text)
_OUTCOMES = {
    "met": ("check", "emerald", "Favourable"),
    "unmet": ("x", "crimson", "Unfavourable"),
    "unknown": ("question", "amber", "Not enough information"),
    "na": ("slash", "gray", "Not applicable"),
    INVALID_KEY: ("warning", "gray", "Invalid / error"),
}

# (criterion_type, label) -> outcome key
_LABEL_OUTCOME = {
    ("inclusion", "included"): "met",
    ("inclusion", "not included"): "unmet",
    ("inclusion", "not enough information"): "unknown",
    ("inclusion", "not applicable"): "na",
    ("exclusion", "not excluded"): "met",
    ("exclusion", "excluded"): "unmet",
    ("exclusion", "not enough information"): "unknown",
    ("exclusion", "not applicable"): "na",
}


def badge_for(criterion_type: str, label: str | None, status: str = "ok") -> dict:
    """Badge dict {key, icon, color, text, label}. status != 'ok' (invalid / error / skipped / unavailable)
    or an unknown label gives the grey warning badge. `label` is the exact label string (or None)."""
    key = _LABEL_OUTCOME.get((criterion_type, label)) if status == "ok" else None
    key = key or INVALID_KEY
    icon, color, text = _OUTCOMES[key]
    if key == INVALID_KEY:
        text = {"error": "Error", "skipped": "Skipped", "unavailable": "Unavailable"}.get(status, "Invalid output")
    return {"key": key, "icon": icon, "color": color, "text": text, "label": label if status == "ok" else None}


def all_badges() -> dict:
    """The full mapping, so the UI (and tests) can show a legend without duplicating it."""
    return {f"{t}|{l}": badge_for(t, l, "ok") for (t, l) in _LABEL_OUTCOME}


def missing_dataset_labels() -> list[tuple[str, str]]:
    """(type, label) pairs of the dataset label sets with no badge (must be empty; unit-tested)."""
    return [(t, l) for t, labels in schema.label_sets().items() for l in labels if (t, l) not in _LABEL_OUTCOME]
