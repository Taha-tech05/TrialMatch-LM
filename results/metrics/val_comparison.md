# Validation comparison (val split only)

## Overall

| system | n | accuracy | macro-F1 (present) | macro-F1 (full) | invalid rate | evid. micro F1 (incl.) | evid. macro F1 (incl.) | evid. micro F1 (excl.) | evid. macro F1 (excl.) |
|---|---|---|---|---|---|---|---|---|---|
| base | 206 | 0.626 | 0.312 | 0.312 | 0.000 | 0.359 | 0.610 | 0.519 | 0.508 |
| finetuned | 206 | 0.748 | 0.549 | 0.549 | 0.000 | 0.529 | 0.737 | 0.637 | 0.599 |
| baseline | 206 | 0.621 | 0.215 | 0.215 | 0.000 | 0.000 | 0.563 | 0.000 | 0.000 |

## Inclusion only

| system | n | accuracy | macro-F1 (present) | macro-F1 (full) | invalid rate | evid. micro F1 (incl.) | evid. macro F1 (incl.) | evid. micro F1 (excl.) | evid. macro F1 (excl.) |
|---|---|---|---|---|---|---|---|---|---|
| base | 66 | 0.455 | 0.286 (4/4 classes) | 0.286 | 0.000 | 0.487 | 0.608 | 0.558 | 0.643 |
| finetuned | 66 | 0.697 | 0.365 (4/4 classes) | 0.365 | 0.000 | 0.676 | 0.728 | 0.700 | 0.741 |
| baseline | 66 | 0.409 | 0.145 (4/4 classes) | 0.145 | 0.000 | 0.000 | 0.242 | 0.000 | 0.000 |

## Exclusion only

| system | n | accuracy | macro-F1 (present) | macro-F1 (full) | invalid rate | evid. micro F1 (incl.) | evid. macro F1 (incl.) | evid. micro F1 (excl.) | evid. macro F1 (excl.) |
|---|---|---|---|---|---|---|---|---|---|
| base | 140 | 0.707 | 0.211 (4/4 classes) | 0.211 | 0.000 | 0.225 | 0.611 | 0.447 | 0.339 |
| finetuned | 140 | 0.771 | 0.606 (4/4 classes) | 0.606 | 0.000 | 0.362 | 0.742 | 0.535 | 0.421 |
| baseline | 140 | 0.721 | 0.210 (4/4 classes) | 0.210 | 0.000 | 0.000 | 0.714 | 0.000 | 0.000 |

## Per-class precision / recall / F1 (support)

### Inclusion

| class | support | base P / R / F1 | finetuned P / R / F1 | baseline P / R / F1 |
|---|---|---|---|---|
| included | 35 | 0.82 / 0.66 / 0.73 | 0.86 / 0.69 / 0.76 | 0.00 / 0.00 / 0.00 |
| not included | 3 | 0.08 / 0.67 / 0.14 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| not enough information | 27 | 0.56 / 0.19 / 0.28 | 0.61 / 0.81 / 0.70 | 0.41 / 1.00 / 0.58 |
| not applicable | 1 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |

### Exclusion

| class | support | base P / R / F1 | finetuned P / R / F1 | baseline P / R / F1 |
|---|---|---|---|---|
| excluded | 2 | 0.00 / 0.00 / 0.00 | 1.00 / 0.50 / 0.67 | 0.00 / 0.00 / 0.00 |
| not excluded | 101 | 0.74 / 0.98 / 0.84 | 0.82 / 0.92 / 0.87 | 0.72 / 1.00 / 0.84 |
| not enough information | 26 | 0.00 / 0.00 / 0.00 | 0.75 / 0.35 / 0.47 | 0.00 / 0.00 / 0.00 |
| not applicable | 11 | 0.00 / 0.00 / 0.00 | 0.38 / 0.45 / 0.42 | 0.00 / 0.00 / 0.00 |

## Evidence breakdown by gold emptiness

| system | scope | empty-gold rows | predicted empty (correct) | non-empty-gold rows | exact match | predicted empty (missed) | macro F1 on non-empty gold |
|---|---|---|---|---|---|---|---|
| base | overall | 116 | 80 (0.69) | 90 | 30 (0.33) | 25 | 0.508 |
| base | inclusion | 16 | 8 (0.50) | 50 | 23 (0.46) | 5 | 0.643 |
| base | exclusion | 100 | 72 (0.72) | 40 | 7 (0.17) | 20 | 0.339 |
| finetuned | overall | 116 | 98 (0.84) | 90 | 42 (0.47) | 23 | 0.599 |
| finetuned | inclusion | 16 | 11 (0.69) | 50 | 29 (0.58) | 5 | 0.741 |
| finetuned | exclusion | 100 | 87 (0.87) | 40 | 13 (0.33) | 18 | 0.421 |
| baseline | overall | 116 | 116 (1.00) | 90 | 0 (0.00) | 90 | 0.000 |
| baseline | inclusion | 16 | 16 (1.00) | 50 | 0 (0.00) | 50 | 0.000 |
| baseline | exclusion | 100 | 100 (1.00) | 40 | 0 (0.00) | 40 | 0.000 |

## Definitions
- **Invalid outputs** are a separate prediction ("invalid"): never mapped to a valid label, always wrong, counted in the accuracy denominator.
- **Macro-F1 (present)**: mean F1 over classes with gold support > 0. **Macro-F1 (full)**: mean over the full label set of the scope; classes absent from gold get F1 = 0 (zero_division = 0). Inclusion and exclusion use their own label sets; **overall** pools both types using the union of the two label sets (so "not enough information" and "not applicable" are pooled).
- **Evidence**: per-row set P/R/F1 on sentence IDs; both empty = 1, exactly one empty = 0, invalid output = 0. Micro = pooled sentence counts, macro = mean of per-row scores. "incl." / "excl." = including / excluding rows whose gold evidence is empty.
- Absent evidence is never treated as proof of eligibility or ineligibility. Empty gold evidence is a valid answer. Both empty = P=R=F1=1; exactly one empty = 0; invalid output = 0.
- Baseline = majority label per criterion type learned from TRAIN only, plus always-empty evidence.
- Validation split only (n small, 8 patients): differences of a few rows are within noise.

