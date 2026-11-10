# Error analysis

> **DRAFT written by Claude Code for the author to review.** The error rows and counts come from `results/error_analysis/*.csv` (named in HTML comments). The causes are my hypotheses read from the note and criterion text, hedged with "likely" because I cannot inspect the model. The short per-error drafts are also written into the `my_notes` column (prefixed "DRAFT (unverified ...)") of `val_finetuned_errors.csv`, `val_finetuned_candidate_errors.csv` (4 rows) and `test_finetuned_merged_errors.csv`; no other column was changed.

Val errors are from the 4-bit-base fine-tuned run, test errors from the merged fp16 model (`test_finetuned_merged_errors.csv`), so the two splits are not the same model. <!-- results/eval/finetuned_val_summary.json, results/eval/finetuned_merged_test_summary.json -->

## How errors are counted

A row counts as an error if its label differs from gold or its evidence set differs from gold. A row can fall in several categories. <!-- src/trialmatch/evaluation/error_analysis.py docstring -->

| category | val rows (of 206) | test rows (of 206) |
|---|---|---|
| label: NEI gold, decided label predicted | 18 | 26 |
| label: decided gold, NEI predicted | 16 | 7 |
| label: not-applicable vs applicable | 16 | 13 |
| label: polarity (included/not included, excluded/not excluded swapped) | 2 | 2 |
| evidence_wrong (missing and extra sentences) | 7 | 3 |
| evidence_missing (strict subset of gold) | 27 | 22 |
| evidence_extra (strict superset of gold) | 32 | 20 |
| cites the boilerplate consent sentence (gold does not) | 4 | 1 |
| any error | 92 (44.7%) | 81 (39.3%) |

<!-- val: results/error_analysis/val_finetuned_error_summary.csv. test: counts of the error_categories column in results/error_analysis/test_finetuned_merged_errors.csv, computed by me; the test percentage is 81/206. The test file has no summary CSV. -->

Label-wrong rows add up to 52 on val (16 + 18 + 16 + 2) and 48 on test (13 + 26 + 7 + 2), which matches the accuracies 0.748 and 0.767 of 206 rows. The other errors are rows with the right label and the wrong evidence set. <!-- results/eval/finetuned_val_summary.json, results/eval/finetuned_merged_test_summary.json; the sums are my arithmetic -->

## What stands out

1. **Exclusion NEI answered as 'not excluded'.** On test, 26 rows have gold NEI and a decided prediction, and 18 of them (primary category) are in one patient, `sigir-201421`, a patient with 25 of the 81 test errors. <!-- test_finetuned_merged_errors.csv, primary_category by patient_id; computed by me --> Most of those rows have empty gold and empty predicted evidence. Likely the model defaults to 'not excluded' when a criterion is never mentioned in the note, in line with the training prior ('not excluded' is 75.26% of train exclusion rows). <!-- docs/label_distribution.md --> This matches the fine-tuned NEI recall for exclusion criteria on test of 0.11. <!-- results/metrics/test_comparison.md -->
2. **Evidence over- and under-citation** are the largest categories: `evidence_extra` and `evidence_missing` together touch 59 val rows and 42 test rows (with overlaps with label errors). <!-- tables above; the sums are my arithmetic, they may double count rows that have two categories -->
3. **Few polarity errors** (2 per split), but the 'excluded' class is rare: 10 rows overall, 8 in train, 2 in val and 0 in test. <!-- docs/label_distribution.md -->
4. **Consent sentence:** the model cites the boilerplate last sentence (consent) in 4 val rows but only 1 test row. <!-- val_finetuned_error_summary.csv; test file --> The decision log planned to check this. <!-- docs/decisions.md 2026-11-06 -->

## Drafted errors (14)

Annotation ids are those of the CSV files. "Gold" and "pred" are label and cited sentence numbers.

| # | split | id | type | gold -> pred | category | draft cause (hedged) |
|---|---|---|---|---|---|---|
| 1 | val | 8 | exclusion | NEI [0,1,2] -> not excluded [0-7] | NEI gold, decided pred; evidence_extra | "Unstable angina" is not stated; model likely treats no stated diagnosis as 'not excluded' and cites the whole note instead of picking sentences. |
| 2 | val | 19 | exclusion | NEI [] -> not excluded [] | NEI gold, decided pred | Beta-blocker contraindication is probably never mentioned; likely a default to the majority label. |
| 3 | val | 263 | exclusion | excluded [5] -> not excluded [1,4] | polarity; evidence_wrong | Gold relies on "vesicular mole vs fibroid degeneration" (sentence 5) as uterine pathology other than leiomyoma; the model likely did not make that inference and cited exam and ultrasound instead. |
| 4 | val | 935 | inclusion | not included [3] -> included [3] | polarity | Same sentence cited by gold and model, so the difference is in the reading of it: likely "culture of the sputum revealed an organism" was matched to "Mycobacterium culture positive" without checking the organism. This is my reading of the sentence, not a clinical judgement. |
| 5 | val | 375 | inclusion | NEI [0,1] -> included [0,1,6] | NEI gold, decided pred; evidence_extra; consent | Cites the boilerplate consent sentence [6] as support; the criterion's abbreviation is not explained in the cited sentences. |
| 6 | val | 33 | inclusion | not applicable [] -> NEI [6] | N/A vs applicable; evidence_extra; consent | The criterion text is garbled (contains a full-width comma); the model likely fell back to NEI and the final sentence. |
| 7 | val | 13 | exclusion | not excluded [4] -> not excluded [] | evidence_missing | Right label; the gold cites "She denies ... diabetes" and the model cites nothing, likely because most 'not excluded' rows have empty gold evidence (0.756 in `docs/data_inspection.md`). |
| 8 | val | 5 | exclusion | not excluded [] -> not excluded [3] | evidence_extra | The model cites "known to have hypertension" for a >170/100 threshold; likely over-citing a related sentence. |
| 9 | test | 392 | exclusion | NEI [] -> not excluded [] | NEI gold, decided pred | One of the 18 NEI rows in patient `sigir-201421` (drug-exposure windows such as "Intravenous cyclophosphamide within 6 months"); likely the same default. |
| 10 | test | 185 | inclusion | included [2,3] -> not included [] | polarity; evidence_missing | The criterion asks for shock or ARDS; the gold cites tachypnoea, tachycardia and BP 80/60; the model likely did not infer shock from vital signs. |
| 11 | test | 184 | exclusion | not excluded [] -> excluded [0,1,2] | polarity; evidence_extra | "Other inappropriate situation" is vague; the model likely mapped it to the acute complication in the note. |
| 12 | test | 381 | inclusion | included [0] -> included [4] | evidence_wrong; consent | The criterion mentions "signing the informed consent form", so the model likely matched on those words and cited the boilerplate consent sentence instead of the age sentence. |
| 13 | test | 193 | exclusion | NEI [0-4] -> not excluded [] | NEI gold, decided pred; evidence_missing | "Moribund" is not stated; the model likely judged nothing indicates it, while the annotators saw the note as insufficient. |
| 14 | test | 163 | inclusion | not included [] -> NEI [0,1,2] | decided gold, NEI pred; evidence_extra | The note concerns a femoral-artery complication; the criterion is about aneurysm repair; the model likely hedged to NEI because of related vascular terms. This is my reading of the note. |

<!-- all rows: results/error_analysis/val_finetuned_errors.csv (ids 8, 19, 263, 935, 375, 33, 13, 5) and results/error_analysis/test_finetuned_merged_errors.csv (ids 392, 185, 184, 381, 193, 163) -->

The set covers exclusion NEI misses (1, 2, 9, 13), polarity cases (3, 4, 10, 11), `evidence_extra` (1, 5, 8, 11, 14), `evidence_missing` (7, 10, 13), `evidence_wrong` (3, 12) and the consent sentence (5, 6, 12).

## Observations from the app (reported by the author, not recorded in the repo)

These two came from using the browser app in real mode on a Colab T4. The exact criteria and output are not saved in any file in the repo (`outputs/reports/` holds only one demo-mode report), so the wording below is from the author's description. Screenshot `demo/11_mi_3_months_not_excluded.png` shows the first case itself: the exclusion criterion "Myocardial infarction within the last 6 months" is labelled "Favourable · not excluded" with the cited sentence [S2] "He had a myocardial infarction 3 months ago and underwent stenting." (latency 1.0 s). `demo/05_summary.png` shows 0 decisive-negative, 1 unresolved, 1 not applicable and 8 favourable criteria for that note, so no exclusion criterion fired; with an "excluded" card the rule would have returned "likely ineligible" instead of "undetermined". The same screenshot shows the BMI inclusion criterion labelled NEI with the missing-information flag and "Pregnant or breastfeeding" labelled not applicable for a male patient. The second observation (a never-mentioned criterion labelled "not excluded") has no screenshot.

- **A myocardial infarction three months ago labelled 'not excluded' for a criterion that excludes MI within six months.** If this is accurate, it is a polarity-style error. A plausible cause is that the model does not do date arithmetic or compare a time window in the criterion with a time in the note, or that it ignored the note sentence. I did not run it.
- **A criterion never mentioned in the note labelled 'not excluded'** (instead of 'not enough information'). This fits the main pattern above (NEI answered as 'not excluded'), as seen in the test errors for patient `sigir-201421` and val annotation 19.

Both observations point at the same weakness: for exclusion criteria the model treats missing information as absence. The app's preliminary outcome rule ignores 'not excluded' rows when judging eligibility, so a wrongly confident 'not excluded' can push the result towards "Likely eligible (preliminary)". <!-- docs/decisions.md 2026-11-08, overall outcome rule -->

## Limits of this analysis

- The categories are assigned automatically; the causes are my reading of 14 rows out of 92 (val) and 81 (test).
- The val and test errors come from two slightly different model variants (4-bit base vs merged fp16; accuracy 0.748 vs 0.733 on val). <!-- results/eval/finetuned_val_summary.json, finetuned_merged_val_summary.json -->
- Test errors were looked at only after the single permitted test run; whether they influenced any later change is not recorded. <!-- docs/decisions.md 2026-11-07 and 2026-11-09, results/eval/test_split_evaluated.json -->
