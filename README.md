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

## Training

## Evaluation

## KV-Cache/Attention/Speculative Benchmarks

## Running the App

## Model Weights Download

## Reproducibility

## Repository Structure

## AI Use Acknowledgment
