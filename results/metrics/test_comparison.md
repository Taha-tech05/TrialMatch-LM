# Test comparison (test split)

**Note:** the `excluded` class has **zero gold rows** among the 131 exclusion rows of this split, so macro-F1 (present classes) and macro-F1 (full label set) are both reported and differ.

## Overall

| system | n | accuracy | macro-F1 (present) | macro-F1 (full) | invalid rate | evid. micro F1 (incl.) | evid. macro F1 (incl.) | evid. micro F1 (excl.) | evid. macro F1 (excl.) |
|---|---|---|---|---|---|---|---|---|---|
| base | 206 | 0.665 | 0.523 | 0.435 | 0.000 | 0.376 | 0.693 | 0.550 | 0.503 |
| finetuned_merged | 206 | 0.767 | 0.726 | 0.605 | 0.000 | 0.578 | 0.814 | 0.663 | 0.585 |
| baseline | 206 | 0.626 | 0.272 | 0.226 | 0.000 | 0.000 | 0.704 | 0.000 | 0.000 |

## Inclusion only

| system | n | accuracy | macro-F1 (present) | macro-F1 (full) | invalid rate | evid. micro F1 (incl.) | evid. macro F1 (incl.) | evid. micro F1 (excl.) | evid. macro F1 (excl.) |
|---|---|---|---|---|---|---|---|---|---|
| base | 75 | 0.640 | 0.565 (4/4 classes) | 0.565 | 0.000 | 0.441 | 0.554 | 0.583 | 0.590 |
| finetuned_merged | 75 | 0.813 | 0.705 (4/4 classes) | 0.705 | 0.000 | 0.676 | 0.720 | 0.754 | 0.689 |
| baseline | 75 | 0.507 | 0.168 (4/4 classes) | 0.168 | 0.000 | 0.000 | 0.400 | 0.000 | 0.000 |

## Exclusion only

| system | n | accuracy | macro-F1 (present) | macro-F1 (full) | invalid rate | evid. micro F1 (incl.) | evid. macro F1 (incl.) | evid. micro F1 (excl.) | evid. macro F1 (excl.) |
|---|---|---|---|---|---|---|---|---|---|
| base | 131 | 0.679 | 0.313 (3/4 classes) | 0.235 | 0.000 | 0.227 | 0.772 | 0.440 | 0.258 |
| finetuned_merged | 131 | 0.740 | 0.564 (3/4 classes) | 0.423 | 0.000 | 0.314 | 0.868 | 0.390 | 0.292 |
| baseline | 131 | 0.695 | 0.273 (3/4 classes) | 0.205 | 0.000 | 0.000 | 0.878 | 0.000 | 0.000 |

## Per-class precision / recall / F1 (support)

### Inclusion

| class | support | base P / R / F1 | finetuned_merged P / R / F1 | baseline P / R / F1 |
|---|---|---|---|---|
| included | 27 | 0.76 / 0.81 / 0.79 | 0.92 / 0.81 / 0.86 | 0.00 / 0.00 / 0.00 |
| not included | 7 | 0.21 / 0.57 / 0.31 | 0.80 / 0.57 / 0.67 | 0.00 / 0.00 / 0.00 |
| not enough information | 38 | 0.91 / 0.53 / 0.67 | 0.82 / 0.87 / 0.85 | 0.51 / 1.00 / 0.67 |
| not applicable | 3 | 0.40 / 0.67 / 0.50 | 0.33 / 0.67 / 0.44 | 0.00 / 0.00 / 0.00 |

### Exclusion

| class | support | base P / R / F1 | finetuned_merged P / R / F1 | baseline P / R / F1 |
|---|---|---|---|---|
| excluded | 0 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| not excluded | 91 | 0.70 / 0.97 / 0.81 | 0.75 / 0.95 / 0.84 | 0.69 / 1.00 / 0.82 |
| not enough information | 28 | 0.00 / 0.00 / 0.00 | 0.75 / 0.11 / 0.19 | 0.00 / 0.00 / 0.00 |
| not applicable | 12 | 0.25 / 0.08 / 0.12 | 0.67 / 0.67 / 0.67 | 0.00 / 0.00 / 0.00 |

## Evidence breakdown by gold emptiness

| system | scope | empty-gold rows | predicted empty (correct) | non-empty-gold rows | exact match | predicted empty (missed) | macro F1 on non-empty gold |
|---|---|---|---|---|---|---|---|
| base | overall | 145 | 112 (0.77) | 61 | 19 (0.31) | 16 | 0.503 |
| base | inclusion | 30 | 15 (0.50) | 45 | 17 (0.38) | 7 | 0.590 |
| base | exclusion | 115 | 97 (0.84) | 16 | 2 (0.12) | 9 | 0.258 |
| finetuned_merged | overall | 145 | 132 (0.91) | 61 | 29 (0.48) | 21 | 0.585 |
| finetuned_merged | inclusion | 30 | 23 (0.77) | 45 | 25 (0.56) | 10 | 0.689 |
| finetuned_merged | exclusion | 115 | 109 (0.95) | 16 | 4 (0.25) | 11 | 0.292 |
| baseline | overall | 145 | 145 (1.00) | 61 | 0 (0.00) | 61 | 0.000 |
| baseline | inclusion | 30 | 30 (1.00) | 45 | 0 (0.00) | 45 | 0.000 |
| baseline | exclusion | 115 | 115 (1.00) | 16 | 0 (0.00) | 16 | 0.000 |

## Definitions
- **Invalid outputs** are a separate prediction ("invalid"): never mapped to a valid label, always wrong, counted in the accuracy denominator.
- **Macro-F1 (present)**: mean F1 over classes with gold support > 0. **Macro-F1 (full)**: mean over the full label set of the scope; classes absent from gold get F1 = 0 (zero_division = 0). Inclusion and exclusion use their own label sets; **overall** pools both types using the union of the two label sets (so "not enough information" and "not applicable" are pooled).
- **Evidence**: per-row set P/R/F1 on sentence IDs; both empty = 1, exactly one empty = 0, invalid output = 0. Micro = pooled sentence counts, macro = mean of per-row scores. "incl." / "excl." = including / excluding rows whose gold evidence is empty.
- Absent evidence is never treated as proof of eligibility or ineligibility. Empty gold evidence is a valid answer. Both empty = P=R=F1=1; exactly one empty = 0; invalid output = 0.
- Baseline = majority label per criterion type learned from TRAIN only, plus always-empty evidence.
- Test split (8 patients, evaluated once): differences of a few rows are within noise.

