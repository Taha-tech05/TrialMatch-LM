# TrialMatch Transformer

A Transformer-based system that matches patient descriptions to clinical trial inclusion/exclusion criteria, using the TrialGPT Criterion Annotations dataset. Includes fine-tuning, a manual KV-cache decoder, tiled online-softmax attention, speculative decoding, and a browser app.

> **Disclaimer:** This is an educational prototype. It is **not a clinical enrollment decision** and must not be used for real patient care or trial enrollment.

## Overview

## Setup

## Data Preparation

```bash
pip install -r requirements.txt
python scripts/prepare_data.py            # download (skipped if present; --force to redo), split, manifest
python scripts/inspect_data.py            # writes results/metrics/data_inspection.json, docs/data_inspection.md
python -m pytest tests/test_split_leakage.py
```

Paths, seed (42) and split sizes (37/8/8 patients) live in `configs/data.yaml`. Processed splits are in
`data/processed/{train,val,test}.parquet`; the manifest is `data/splits/split_manifest.json` (git-tracked).
Previous GPT-4 outputs are stored separately in `forbidden_previous_predictions_NOT_FOR_MODEL_USE.parquet`.

## Prompt and Output Schema

The model sees only the note (sentences numbered from 0), the criterion type and the criterion text.
It must answer with a strict JSON object with exactly two keys: `label` (one of the labels allowed for that
criterion type; no collapsing) and `evidence_ids` (ascending, unique, in-range integer sentence numbers; an empty
list is a valid answer). The model does **not** produce explanations: no gold explanations exist and the dataset's
GPT-4 explanations are forbidden, so the app will build explanations only from the label and the cited sentences.

- Inclusion labels: `included`, `not included`, `not enough information`, `not applicable`
- Exclusion labels: `excluded`, `not excluded`, `not enough information`, `not applicable`

Code: `src/trialmatch/model/{schema,prompt,parse}.py`. Invalid model output is never remapped to a valid label;
the parser returns `status="invalid"` with a reason so metrics can count it separately.

Example (train split, annotation_id 49):

```text
You are helping with a clinical trial eligibility research task (not medical advice).
Read the patient note. Its sentences are numbered, starting from 0.

Patient note:
0. A 2-year-old boy is brought to the emergency department by his parents for 5 days of high fever and irritability.
1. The physical exam reveals conjunctivitis, strawberry tongue, inflammation of the hands and feet, desquamation of the skin of the fingers and toes, and cervical lymphadenopathy with the smallest node at 1.5 cm.
2. The abdominal exam demonstrates tenderness and enlarged liver.
3. Laboratory tests report elevated alanine aminotransferase, white blood cell count of 17,580/mm, albumin 2.1 g/dL, C-reactive protein 4.5 mg, erythrocyte sedimentation rate 60 mm/h, mild normochromic, normocytic anemia, and leukocytes in urine of 20/mL with no bacteria identified.
4. The echocardiogram shows moderate dilation of the coronary arteries with possible coronary artery aneurysm.
5. The patient will provide informed consent, and will comply with the trial protocol without any practical issues.

Criterion type: inclusion
Criterion: Patients, male and female, at any age ≥ 3 months (5 kg) of life, with KD according to the American Heart Association definition for complete or incomplete KD. fever ≥ 5 days and ≥ 4 of 5 main clinical signs: modification of the extremities, polymorphic exanthema, bilateral bulbar not exudative conjunctivitis, erythema of the lips or oral cavity, and cervical lymph nodes usually unilateral > 1.5 cm in diameter. In the presence of less than 4 clinical criteria and 5 days of fever, the diagnosis of disease KD is proposed in case of coronary abnormalities (at least one dilated coronary artery with internal diameter ≥ 2,5 SD from the mean normalized for body surface area (Z score) as determined by echocardiography. For indicative purpose, in case of incomplete KD, other biological supportive criteria for incomplete KD can help to ensure the diagnosis: leucocytosis, elevated CRP, elevated ESR, anaemia, hyponatremia, elevated ASAT, ALAT and gGT, hyperlipidaemia.

Decide the patient's status for this inclusion criterion. Allowed labels:
- included
- not included
- not enough information
- not applicable

Cite the numbers of the note sentences that support your decision (an empty list if there is no supporting sentence).
Respond with only a JSON object with exactly these keys:
{"label": "<one allowed label>", "evidence_ids": [<integer sentence numbers, ascending, may be empty>]}
```

Target:

```json
{"label": "included", "evidence_ids": [0, 1, 2, 3, 4]}
```

Also in `docs/prompt_example.md`.

## Training

LoRA/QLoRA fine-tuning of `Qwen/Qwen3-4B` (fallback `Qwen/Qwen3-1.7B`). All settings are in `configs/train.yaml`:
`max_seq_len` 768 (longest measured prompt+target is 618 tokens; nothing is truncated and this is asserted),
fp16 compute (the Colab T4 has no bf16), LoRA r=16 / alpha=32 on attention and MLP projections only (not
`lm_head`/embeddings: they are tied), seed 42, `quantization: qlora_4bit | lora_fp16`. The loss is computed on the
answer JSON plus `<|im_end|>` only; the chat-templated prompt (`enable_thinking=False`) is masked. Annotation 883 is skipped.

Outputs go under `$TRIALMATCH_OUT` if set (e.g. Google Drive), else the project root:
`checkpoints/<run>/` (an adapter per epoch plus `adapter_final/`), `results/logs/train_<run>.{jsonl,csv}`
(train loss every optimiser step, val loss every epoch) and `results/eval/`.

Install GPU dependencies (torch is preinstalled on Colab; local installs stay light without this file):

```bash
pip install -r requirements.txt -r requirements-gpu.txt && pip install -e .
python scripts/prepare_data.py                  # processed splits are not in git; this rebuilds them
```

On a Colab T4 (`notebooks/colab_runner.ipynb` runs exactly these):

```bash
python scripts/smoke_test_train.py              # ~15 steps: sec/step, peak GPU memory, 3-epoch projection, 4B-vs-1.7B advice
python scripts/train.py                         # re-run the same command to resume after a disconnect
python scripts/train.py --model Qwen/Qwen3-1.7B # fallback if the projection exceeds 90 minutes
```

Training stops with a clear message if the loss becomes NaN/inf (exit code 2); the log is kept.

## Evaluation

Greedy decoding (`max_new_tokens` 64), the same prompt and settings for the base model and the adapter, parsed
with the project parser. This step reports only accuracy and the invalid-output rate; the full metrics come later.

```bash
python scripts/evaluate.py --tag base                                            # base model, val split
python scripts/evaluate.py --adapter $TRIALMATCH_OUT/checkpoints/qwen3-4b_qlora_4bit/adapter_final --tag finetuned
# final, once only (refuses to run a second time without --force):
python scripts/evaluate.py --adapter <adapter> --tag finetuned --split test --confirm-final
```

Raw outputs and parsed predictions: `results/eval/<tag>_<split>_predictions.jsonl`; summary: `..._summary.json`.

### Metrics, baselines and error analysis (validation split; no torch/GPU needed)

```bash
pip install -r requirements.txt
python scripts/compute_metrics.py    # results/metrics/val_metrics_<tag>.json, val_comparison.md, results/figures/*
python scripts/export_errors.py      # results/error_analysis/*.csv (fine-tuned errors, 12 candidates, base-vs-finetuned diffs)
python -m pytest tests/test_metrics.py
```

`compute_metrics.py` joins `results/eval/{base,finetuned}_val_predictions.jsonl` to the gold labels in
`data/processed/val.parquet` (asserting every val annotation appears exactly once), and adds a baseline built from
the majority label per criterion type in TRAIN plus always-empty evidence. It reports accuracy, per-class
precision/recall/F1/support, macro-F1 over classes present in gold and over the full label set, confusion
matrices (CSV and PNG in `results/figures/`, fixed label order, extra `invalid` column), invalid rate, and evidence
precision/recall/F1 (micro and macro, with and without empty-gold rows), each overall, inclusion and exclusion.
The overall scope pools both criterion types and uses the union of the two label sets. Training curves are written to
`results/figures/training_curves.png`. Definitions are at the bottom of `results/metrics/val_comparison.md`.
These scripts only ever read train/val files; the test split is not touched.

## KV-Cache/Attention/Speculative Benchmarks

## Running the App

## Model Weights Download

## Reproducibility

## Repository Structure

## AI Use Acknowledgment
