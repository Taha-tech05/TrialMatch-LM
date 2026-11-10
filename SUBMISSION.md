# Assignment 2 submission: TrialMatch Transformer

**Author:** Muhammad Taha Irfan, Department of Computer Science (BSCS), FAST NUCES, Islamabad, Pakistan, tahairfan549@gmail.com
**Disclaimer:** educational screening prototype, not a clinical enrollment decision.

## 1. Deliverables at a glance

| Item | Where |
|---|---|
| Report (IEEE, LaTeX) | PDF: `report/main.pdf` (source: `report/main.tex`, `report/references.bib`) |
| GitHub repository | https://github.com/Taha-tech05/TrialMatch-LM |
| Trained adapter (download link) | https://huggingface.co/Taha549/trialmatch-qwen3-4b-qlora (LoRA adapter for `Qwen/Qwen3-4B`) |
| Split manifest | `data/splits/split_manifest.json` (patient IDs per split, counts, seed 42, forbidden columns removed) |
| Benchmark logs | `results/benchmarks/` (KV cache, attention, speculative), `results/logs/` (training, merge, environment) |
| Evaluation outputs | `results/eval/` (predictions), `results/metrics/` (metrics and comparison tables), `results/error_analysis/` |
| Recorded demonstration (5 to 7 min) | **Video link:** https://youtu.be/e3O8ud70Kxo |
| This summary | `SUBMISSION.md` and `SUBMISSION.pdf` (source `SUBMISSION.tex`) |
| AI-use acknowledgment | Appendix A of the report; `docs/ai_use_log.md`; README section "AI Use Acknowledgment" |

The benchmark files used in the report are `results/benchmarks/kv_cache_20261107_192136.json`, `attention_20261107_193238.json` and `speculative_20261108_113154.json` (each also as `.csv` and `.png`).

## 2. Reproducible execution instructions

Full details are in `README.md` (sections "Setup", "Reproducibility", "Running the App"). Short version:

### 2.1 Environment
```bash
git clone https://github.com/Taha-tech05/TrialMatch-LM
cd TrialMatch-LM
pip install -r requirements.txt -r requirements-app.txt     # data, metrics, tests, demo-mode app (CPU, Python >= 3.10)
pip install -r requirements-gpu.txt && pip install -e .    # training, merging, benchmarks, real-mode app (GPU)
```
Training and all benchmarks were run on a Colab Tesla T4 (14.56 GiB), Python 3.13, torch 2.14.1+cu130, transformers 5.22.1, peft 0.21.2, bitsandbytes 0.50.2 (`results/logs/env_qwen3-4b_qlora_4bit.json`). `notebooks/colab_runner.ipynb` runs the Colab setup; set `TRIALMATCH_OUT` (for example a Google Drive folder) to keep checkpoints and results.

### 2.2 Data and split (CPU)
```bash
python scripts/prepare_data.py          # downloads the TrialGPT annotations, writes the patient-grouped split and the manifest
python scripts/inspect_data.py
python -m pytest tests/test_split_leakage.py
```
Split: sorted unique patient IDs, shuffled with `np.random.default_rng(42)`, first 37 train, next 8 validation, last 8 test (603 / 206 / 206 rows). The four GPT-4 columns are removed from the model inputs and stored separately.

### 2.3 Train, evaluate, benchmark (GPU)
```bash
python scripts/smoke_test_train.py
python scripts/train.py
python scripts/evaluate.py --tag base
python scripts/evaluate.py --adapter $TRIALMATCH_OUT/checkpoints/qwen3-4b_qlora_4bit/adapter_final --tag finetuned
python scripts/compute_metrics.py && python scripts/export_errors.py
python scripts/merge_adapter.py --adapter $TRIALMATCH_OUT/checkpoints/qwen3-4b_qlora_4bit/adapter_final --save
python scripts/bench_kv_cache.py --merged-dir $TRIALMATCH_OUT/merged/qwen3-4b_merged_fp16 --mode both
python scripts/bench_attention.py
python scripts/bench_speculative.py --merged-dir $TRIALMATCH_OUT/merged/qwen3-4b_merged_fp16
```
The test split is evaluated once, with `--split test --confirm-final` (the runner refuses a second run without `--force`). The full ordered command list is in `README.md`, "Reproducibility".

### 2.4 Download the weights
The adapter is on the Hugging Face Hub (link in section 1); the base model `Qwen/Qwen3-4B` and the draft `Qwen/Qwen3-0.6B` are public. A loading snippet using `PeftModel.from_pretrained` is in `README.md`, "Model Weights Download". The merged fp16 model (about 8 GB) is not hosted; `scripts/merge_adapter.py` rebuilds it from the adapter.

### 2.5 Start the browser application
```bash
python scripts/launch_app.py                                  # then open http://localhost:7860
python scripts/launch_app.py --mode demo                      # cached validation predictions only, no model needed
python scripts/launch_app.py --mode real --merged-dir PATH    # merged Qwen3-4B (GPU recommended)
python scripts/launch_app.py --mode real --merged-dir PATH --load-draft   # also loads Qwen3-0.6B for speculative decoding
python scripts/launch_app.py --mode real --merged-dir PATH --share        # public link via cloudflared or ngrok if installed
TRIALMATCH_ADAPTER=Taha549/trialmatch-qwen3-4b-qlora python scripts/launch_app.py --mode real   # base + Hub adapter, no local merged model
```
Demo mode answers only from cached real validation predictions and states "cannot predict in demo mode" for anything else. Real mode screens any pasted note and criteria. Workflow, screenshots and limits: `docs/app_notes.md`, `demo/README.md`.

## 3. Quick map from assignment tasks to files

| Task | Code | Results |
|---|---|---|
| 1 Research, model, fine-tuning | `src/trialmatch/model/`, `src/trialmatch/training/`, `scripts/train.py`, `configs/train.yaml` | `results/logs/train_*.jsonl`, `results/figures/training_curves.png`, `docs/research_notes.md`, `docs/model_comparison.md`, `docs/decisions.md` |
| 2 Eligibility and evidence | `src/trialmatch/evaluation/` | `results/metrics/`, `results/figures/confusion_*`, `results/error_analysis/`, `docs/error_analysis.md` |
| 3 KV cache | `src/trialmatch/inference/kv_cache/` | `results/benchmarks/kv_cache_20261107_192136.*`, `docs/kv_cache_notes.md` |
| 4 Tiled attention | `src/trialmatch/inference/attention/` | `results/benchmarks/attention_20261107_193238.*`, `docs/attention_notes.md` |
| 5 Speculative decoding | `src/trialmatch/inference/speculative/` | `results/benchmarks/speculative_20261108_113154.*`, `docs/speculative_notes.md` |
| 6 Application | `src/trialmatch/app/`, `scripts/launch_app.py` | `demo/`, `docs/app_notes.md` |
| Tests (CPU) | `tests/` | `python -m pytest tests` |
