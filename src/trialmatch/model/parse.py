"""Robust parser for raw model text. Never raises; invalid output is flagged, never remapped."""
import json

from trialmatch.model import schema
from trialmatch.model.schema import INVALID, OK, Prediction


def _first_json_object(text: str):
    dec = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _ = dec.raw_decode(text, pos)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        pos = text.find("{", pos + 1)
    return None


def parse_output(text, criterion_type: str, n_sentences: int) -> Prediction:
    raw = text if isinstance(text, str) else repr(text)

    def bad(msg):
        return Prediction(parse_ok=False, status=INVALID, label=None, evidence_ids=None, error=msg, raw=raw)

    try:
        if not isinstance(text, str) or not text.strip():
            return bad("empty output")
        obj = _first_json_object(text)
        if obj is None:
            return bad("no valid JSON object found")
        if set(obj) != set(schema.OUTPUT_KEYS):
            return bad(f"keys {sorted(obj)} != required {sorted(schema.OUTPUT_KEYS)}")
        label, ids = obj["label"], obj["evidence_ids"]
        errs = schema.validation_errors(label, ids, criterion_type, n_sentences)
        if errs:
            return bad("; ".join(errs))
        return Prediction(parse_ok=True, status=OK, label=label, evidence_ids=ids, raw=raw)
    except Exception as e:  # the parser must never crash
        return bad(f"unexpected parser error: {type(e).__name__}: {e}")
