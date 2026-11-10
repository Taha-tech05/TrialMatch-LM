# App notes

## Workflow
1. Start: `python scripts/launch_app.py` (see README "Running the App"). The header shows the mode (REAL/DEMO), model, device and dtype.
2. **Screening workspace.** Load a VAL note (or paste one: un-numbered text is split deterministically and shown numbered
   in the dataset format, IDs `[S0]`, `[S1]`, ... exactly as the model sees them). Load an example trial or paste criteria
   ("Inclusion Criteria:" / "Exclusion Criteria:" sections, bullets, one per line), fix types, choose an engine, Run.
3. Cards appear per criterion: type, label badge (icon + colour + text + exact label), cited `[S#]` pills, cited sentence
   text, template explanation, missing-information / invalid / error flags, and a Details drawer (raw output, parse status,
   engine, latency, tokens). Click a pill to jump to the sentence; click a sentence to highlight citing cards.
4. **Summary** tab: rule-based preliminary outcome with the rule text (not a ground-truth label).
5. Export (JSON / HTML / Markdown / Print-to-PDF). Each finished run is also saved to `outputs/reports/`.
6. **Inference Lab**: saved T4 benchmarks (files named in `configs/app.yaml` -> `benchmarks`), live mini-benchmark and
   token stream (real mode only), tiled-attention visualizer (CPU, milliseconds).

## Demo mode limits
Only validation notes with their own trial criteria are answerable (the "Example trial" dropdown loads the matching note).
Anything else returns "cannot predict in demo mode".

## Screenshot checklist (for the paper; to be taken by the author)
- [x] Header in REAL mode (model, device, dtype): `demo/01_header_real.png`
- [ ] Header in DEMO mode (banner): not taken yet
- [ ] Workspace after a run, light theme: not taken yet (dark-theme versions: `demo/03_evidence_sync.png`, `demo/09_speculative_run_16_criteria.png`)
- [x] Same in dark theme: `demo/03_evidence_sync.png`, `demo/09_speculative_run_16_criteria.png`
- [x] Evidence sync: a cited pill clicked, sentence glowing cyan: `demo/03_evidence_sync.png`
- [x] One card with the Details drawer open (raw output, latency): `demo/04_details_drawer.png`
- [ ] Cards showing each badge: only `included` (`demo/03_evidence_sync.png`) and not enough information with the missing flag (`demo/09_speculative_run_16_criteria.png`) are captured; the others are not
- [x] Summary tab with outcome, rule text and not-ground-truth statement: `demo/05_summary.png`
- [ ] Progress bar mid-run with "criterion k/N" and estimate: not taken (`demo/09_speculative_run_16_criteria.png` shows the finished bar, 16/16 in 22 s)
- [x] Exported HTML report, provenance table: `demo/06_report_export.png` (browser chrome cropped; printing to PDF not shown)
- [x] Inference Lab: saved T4 cards + charts with the "measured on ..." label; token equality (`tokens = plain`): `demo/07_inference_lab_saved.png`
- [x] Live benchmark table (labelled "live, cuda, float16"): `demo/10_live_kv_run.png` (KV-cached engine only, not next to the saved block)
- [ ] Token stream with speculative accepted (indigo) / corrected (amber) tokens: not captured (the speculative panel is empty in `demo/10_live_kv_run.png`)
- [x] Tiled attention grid with skipped blocks and the running max/normalizer table: `demo/08_live_bench_tiled.png`

## Speculative engine, draft model, empty states
- Speculative needs the merged 4B as the *active* model and the 0.6B draft loaded. Use `--load-draft` (or `real.load_draft: true`);
  the engine hint under the toolbar states exactly what is missing.
- Saved benchmarks come from `benchmarks.*` in `configs/app.yaml` (kv_cache_20261107_192136, speculative_20261108_113154, attention_20261107_193238).
  A missing or unset file yields an empty state with a message, never an error.
- If `data/processed/val.parquet` is missing the example dropdowns show "Run python scripts/prepare_data.py".
- Sentence splitter: a full stop after a unit ("mg/dL.", "mmHg.", "kg.") followed by a capitalised word starts a new sentence;
  titles ("Dr."), decimals ("7.5") and a unit dot followed by lower-case/number ("7.5 mg. daily") do not.
