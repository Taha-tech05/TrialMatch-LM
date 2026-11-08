"""Rule-based preliminary outcome from per-criterion results (pure function).

Rules, in order of precedence:
  1. any exclusion 'excluded' or inclusion 'not included'           -> Likely ineligible (preliminary)
  2. else any 'not enough information' or an invalid/error result   -> Undetermined: more information needed
  3. else                                                           -> Likely eligible (preliminary)
'not applicable' is ignored; skipped criteria (empty text) are ignored. A result with status 'ok' and a label is
evaluated; any other status (invalid, error, unavailable) counts as unresolved.
"""

RULE_TEXT = (
    "Rule: if any exclusion criterion is 'excluded' or any inclusion criterion is 'not included', the preliminary "
    "outcome is 'Likely ineligible'. Otherwise, if any criterion is 'not enough information' or its output was "
    "invalid or errored, the outcome is 'Undetermined: more information needed'. Otherwise it is 'Likely eligible'. "
    "'not applicable' criteria are ignored."
)
NOT_GROUND_TRUTH = ("This is a rule-based aggregation of per-criterion model outputs. It is NOT a ground-truth label "
                    "and has not been validated against expert patient-trial eligibility verdicts.")

INELIGIBLE = "Likely ineligible (preliminary)"
UNDETERMINED = "Undetermined: more information needed"
ELIGIBLE = "Likely eligible (preliminary)"


def aggregate(results: list[dict]) -> dict:
    """results: dicts with criterion_type, status, label, criterion_text (optional), index (optional)."""
    blockers, unresolved, ignored, satisfied = [], [], [], []
    n_skipped = 0
    for i, r in enumerate(results):
        item = {"index": r.get("index", i), "criterion_type": r.get("criterion_type"),
                "text": r.get("criterion_text", r.get("text", "")), "label": r.get("label"),
                "status": r.get("status")}
        status, label, ctype = r.get("status"), r.get("label"), r.get("criterion_type")
        if status == "skipped":
            n_skipped += 1
        elif status != "ok" or label is None:
            unresolved.append({**item, "reason": f"output {status or 'missing'}: no usable label"})
        elif ctype == "exclusion" and label == "excluded":
            blockers.append({**item, "reason": "exclusion criterion is met ('excluded')"})
        elif ctype == "inclusion" and label == "not included":
            blockers.append({**item, "reason": "inclusion criterion is not met ('not included')"})
        elif label == "not enough information":
            unresolved.append({**item, "reason": "'not enough information'"})
        elif label == "not applicable":
            ignored.append(item)
        else:
            satisfied.append(item)
    if blockers:
        code, outcome, contributing = "ineligible", INELIGIBLE, blockers
    elif unresolved:
        code, outcome, contributing = "undetermined", UNDETERMINED, unresolved
    else:
        code, outcome, contributing = "eligible", ELIGIBLE, satisfied
    notes = []
    if code == "eligible" and not satisfied:
        notes.append("No criterion was positively assessed (all were 'not applicable' or skipped); "
                     "this outcome carries no information.")
    if code == "ineligible" and unresolved:
        notes.append(f"{len(unresolved)} other criteria are unresolved; the ineligible outcome stands because a "
                     "decisive criterion failed.")
    return {"code": code, "outcome": outcome, "contributing": contributing, "unresolved": unresolved,
            "counts": {"decisive_negative": len(blockers), "unresolved": len(unresolved),
                       "not_applicable": len(ignored), "favourable": len(satisfied), "skipped": n_skipped,
                       "total": len(results)},
            "rule_text": RULE_TEXT, "disclaimer": NOT_GROUND_TRUTH, "notes": notes}
