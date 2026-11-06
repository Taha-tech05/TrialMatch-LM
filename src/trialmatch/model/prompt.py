"""Model-agnostic prompt/target formatting. Plain strings only: no tokenizer, no chat template."""
import json
from dataclasses import dataclass
from typing import Mapping, Optional

from trialmatch.model import schema
from trialmatch.utils.common import load_config


class SkipRow(ValueError):
    """Row cannot be formatted (e.g. null criterion_text); the caller must skip it."""


@dataclass(frozen=True)
class TrainingExample:
    prompt: str
    target: str


def parse_gold_evidence(raw: str) -> list[int]:
    """The ONLY place that converts the stored gold string (e.g. '[0, 1, 2]') to list[int]."""
    v = json.loads(raw)
    if not isinstance(v, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in v):
        raise ValueError(f"gold evidence is not a list of ints: {raw!r}")
    return v


def skip_reason(criterion_text) -> Optional[str]:
    if criterion_text is None or (isinstance(criterion_text, float) and criterion_text != criterion_text) \
            or not str(criterion_text).strip():
        return "null/empty criterion_text"
    return None


def apply_template(prompt: str) -> str:
    """Single hook where a model-specific chat template may be applied later. Identity for now."""
    return prompt


def build_prompt(*, note: str, criterion_type: str, criterion_text: str) -> str:
    """Build model input from ONLY note, criterion_type, criterion_text."""
    reason = skip_reason(criterion_text)
    if reason:
        raise SkipRow(reason)
    sets = schema.label_sets()
    if criterion_type not in sets:
        raise ValueError(f"unknown criterion_type {criterion_type!r}")
    labels = "\n".join(f"- {l}" for l in sets[criterion_type])
    prompt = (
        "You are helping with a clinical trial eligibility research task (not medical advice).\n"
        "Read the patient note. Its sentences are numbered, starting from 0.\n\n"
        f"Patient note:\n{note}\n\n"
        f"Criterion type: {criterion_type}\n"
        f"Criterion: {str(criterion_text).strip()}\n\n"
        f"Decide the patient's status for this {criterion_type} criterion. Allowed labels:\n{labels}\n\n"
        "Cite the numbers of the note sentences that support your decision "
        "(an empty list if there is no supporting sentence).\n"
        "Respond with only a JSON object with exactly these keys:\n"
        '{"label": "<one allowed label>", "evidence_ids": [<integer sentence numbers, ascending, may be empty>]}'
    )
    return apply_template(prompt)


def build_target(label: str, evidence_ids: list[int]) -> str:
    return json.dumps({"label": label, "evidence_ids": list(evidence_ids)})


def build_training_example(*, note: str, criterion_type: str, criterion_text: str,
                           gold_label: str, gold_evidence_ids: list[int]) -> TrainingExample:
    prompt = build_prompt(note=note, criterion_type=criterion_type, criterion_text=criterion_text)
    errs = schema.validation_errors(gold_label, gold_evidence_ids, criterion_type,
                                    schema.count_note_sentences(note))
    if errs:
        raise ValueError(f"invalid gold target: {errs}")
    return TrainingExample(prompt, build_target(gold_label, gold_evidence_ids))


def example_from_row(row: Mapping, cfg: Optional[dict] = None) -> TrainingExample:
    """Build an example from a processed-file row. Refuses rows carrying forbidden columns and
    passes ONLY the three input fields to the prompt; gold targets are passed separately."""
    cfg = cfg or load_config()
    c = cfg["columns"]
    leaked = [k for k in cfg["forbidden_columns"] if k in row]
    if leaked:
        raise AssertionError(f"forbidden columns reached the formatter: {leaked}")
    return build_training_example(
        note=row[c["note"]], criterion_type=row[c["criterion_type"]], criterion_text=row[c["criterion_text"]],
        gold_label=row[c["label"]], gold_evidence_ids=parse_gold_evidence(row[c["evidence"]]))
