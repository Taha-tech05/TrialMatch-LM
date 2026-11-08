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

## KV-Cache Benchmarks

All inference benchmarks (KV cache, tiled attention, speculative decoding) use ONE model: Qwen3-4B + the LoRA
adapter **merged into fp16 weights** (`merge_and_unload`, no quantization). The merged model is ~8 GB: it is saved
under `$TRIALMATCH_OUT/merged/` (Google Drive on Colab) and **never committed**. The adapter was trained on a 4-bit base,
so merged-fp16 predictions can differ slightly from the earlier 4-bit-base val result (0.748 accuracy); re-run val to compare.

Code: `src/trialmatch/inference/kv_cache/` (`reference_decode.py` = uncached greedy, every step a full forward;
`cached_decode.py` = explicit prefill + one-token decode steps with explicit `position_ids`, `cache_position`, growing
mask and `past_key_values`; `cache_utils.py` = length/shape/bytes helpers; `compare.py` = mismatch investigator).
CPU tests (tiny random Qwen3, fp32, no downloads): `python -m pytest tests/test_cache_equivalence.py tests/test_kv_bench_and_merge.py`.
Batch size 1 is the benchmarked case; left-padded batches are also supported and tested.

Colab T4 commands (run after the setup cells of `notebooks/colab_runner.ipynb`; `OUT` is `$TRIALMATCH_OUT`):

```bash
# 1. merge + verify (unmerged-vs-merged logits on 4 val prompts) + save the merged fp16 model
python scripts/merge_adapter.py --adapter $TRIALMATCH_OUT/checkpoints/qwen3-4b_qlora_4bit/adapter_final --save
# 2. re-evaluate VAL on the merged fp16 model (same prompt + greedy settings) and compare with finetuned (4-bit base)
python scripts/evaluate.py --merged --merged-dir $TRIALMATCH_OUT/merged/qwen3-4b_merged_fp16 --tag finetuned_merged
python scripts/compute_metrics.py
# 3. benchmark cached vs uncached: synthetic lengths 128/256/512/1024 x 16/64/128 AND real val prompts
python scripts/bench_kv_cache.py --merged-dir $TRIALMATCH_OUT/merged/qwen3-4b_merged_fp16 --mode both
# quicker check first: python scripts/bench_kv_cache.py --merged-dir ... --quick
```

Results: `results/benchmarks/kv_cache_<timestamp>.{json,csv,png}` (under `$TRIALMATCH_OUT`): prefill / decode / total time,
tokens/s, peak GPU memory, mean/std/min/max/median over repeats (default 5, 1 warm-up), cached-vs-uncached token equality
per case, and the environment (GPU, torch/transformers versions, dtype, git commit). The uncached runs at 1024 input x 128 output are slow
(a full forward per token): use `--repeats 3` if the session is short. Test split is never read.

## Attention Benchmarks

Code: `src/trialmatch/inference/attention/` - `reference.py` (explicit scores -> additive mask -> max-subtracted softmax -> V,
plus `repeat_kv`), `tiled.py` (plain-PyTorch online-softmax attention over query/key blocks; update equations in its docstring),
`sdpa_backends.py` (torch SDPA with selectable backend, comparison only, not used by the other two or by the model).
Conventions: causal mask with offset (query i sees key j iff `j <= i + seq_k - seq_q`), fp32 accumulation with output in the
input dtype, all-zero output for a query that sees no key, GQA by repeating each KV head for its group (query head h uses KV head h // group).

```bash
python -m pytest tests/test_attention_correctness.py -q      # CPU; tiled vs reference, SDPA, HF eager, stress, hand example
python tests/test_attention_correctness.py                   # prints the max/mean error table (shapes x block sizes)
python scripts/bench_attention.py --quick                    # small sweep (CPU ok, fp32)
python scripts/bench_attention.py                            # GPU: fp16, seq 128..4096, heads 32 / kv heads 8 / head_dim 128, blocks 32/64/128/256, 10 repeats
```

Output: `results/benchmarks/attention_<timestamp>.{json,csv,png}` (under `$TRIALMATCH_OUT` if set): time and peak memory per
implementation and sequence length, max/mean absolute error against an fp32 reference, OOM / unavailable backends recorded
instead of crashing, and the environment (GPU, versions, dtype, git commit). Exit code 1 if tiled/reference differ from the fp32
baseline by more than the tolerance. Notes to fill in: `docs/attention_notes.md`.

## One-click Colab run

After the setup cells of `notebooks/colab_runner.ipynb` (clone, install, `TRIALMATCH_OUT` set), run the last cell, or:

```bash
python scripts/colab_all.py            # env log -> merge (skipped if merged dir exists) -> merged VAL eval -> compute_metrics -> KV-cache bench -> attention bench -> speculative bench
python scripts/colab_all.py --quick    # --quick / fewer repeats in the benchmarks and the merge check
python scripts/colab_all.py --stop-on-failure --skip bench_kv_cache
```

Logs go to `$TRIALMATCH_OUT/logs/<step>.log`, results to `$TRIALMATCH_OUT/results/`, and the per-step outcome to
`results/benchmarks/run_summary.json`. A benchmark exiting with code 1 (mismatch) is recorded as `mismatch` and the run continues;
a setup-step failure stops the run unless `--keep-going`. `--resume-bench` continues the latest KV-cache and speculative result files after a disconnect. The test
split is never evaluated (the script refuses any command containing `--confirm-final` or `--split test`).

## Speculative Decoding Benchmarks

Code: `src/trialmatch/inference/speculative/` - `speculative_decode.py` (explicit greedy speculative decoding: both models
prefilled, draft proposes k tokens, ONE target forward verifies them, longest matching prefix accepted, target's own token as
correction/bonus, both KV caches cropped back), `cache_sync.py` (crop with the transformers 5.x API plus a slicing fallback),
`stats.py` (acceptance rate, tokens per target forward, accepted-run histogram, draft/verify time), `loading.py` (draft loader),
`hf_assisted_reference.py` (HF `assistant_model`, **cross-check in tests only**). **Batch size 1 only.** Output is token-identical to
the plain cached greedy decoder (fp32 tests; fp16 near-ties are reported by the benchmark, not hidden).
Target = merged fp16 Qwen3-4B; draft = untuned `Qwen/Qwen3-0.6B` by default. Change it in `configs/inference.yaml`
(`speculative_benchmark.draft_model / draft_dir / draft_adapter`) or with `--draft-model / --draft-dir / --draft-adapter`
(a fine-tuned draft is not trained in this step).

CPU tests (tiny random Qwen3 target + smaller draft, fp32, no downloads):
`python -m pytest tests/test_speculative_equivalence.py tests/test_speculative_bench.py`.

Colab T4 commands (after the merged model exists, see KV-Cache Benchmarks; `$TRIALMATCH_OUT` set):

```bash
M=$TRIALMATCH_OUT/merged/qwen3-4b_merged_fp16
python scripts/bench_speculative.py --merged-dir $M --quick --max-configs 6    # short sanity run first
python scripts/bench_speculative.py --merged-dir $M                            # full: 128/256/512 x 16/64/128, k 1 2 4 8, 8 val prompts, 3 repeats
python scripts/bench_speculative.py --merged-dir $M --resume                   # after a disconnect: continue the latest file
python scripts/bench_speculative.py --merged-dir $M --draft-adapter <path>     # later: a fine-tuned draft adapter
python scripts/colab_all.py --only bench_speculative [--quick] [--resume-bench]
python scripts/bench_kv_cache.py --merged-dir $M --resume                      # the KV-cache benchmark resumes the same way
```

Results: `results/benchmarks/speculative_<timestamp>.{json,csv,png}` (under `$TRIALMATCH_OUT`), rewritten after EVERY
configuration. Per row: acceptance rate, mean accepted per iteration, wall latency, tokens/s, speedup vs plain cached decoding,
draft vs verify time, peak memory (both models loaded), token-equality flag (a mismatch records the first differing step and
the logit gap), plus the environment (GPU, versions, dtype, git commit). Exit code 1 if any output differs from plain greedy.
Test split is never read. Notes to fill in: `docs/speculative_notes.md`.

## KV-Cache/Attention/Speculative Benchmarks

## Running the App

## Model Weights Download

## Reproducibility

## Repository Structure

## AI Use Acknowledgment
