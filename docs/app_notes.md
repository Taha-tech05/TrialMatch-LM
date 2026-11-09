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
- [ ] Header in REAL mode (model, device, dtype) and in DEMO mode (banner)
- [ ] Workspace after a run: note left, criteria cards right, light theme
- [ ] Same in dark theme
- [ ] Evidence sync: a cited pill clicked, sentence glowing cyan, citing card outlined
- [ ] One card with the Details drawer open (raw output, latency)
- [ ] Cards showing each badge: included/not excluded, not included/excluded, not enough information (+ missing flag), not applicable, invalid/error
- [ ] Summary tab with outcome, rule text and not-ground-truth statement
- [ ] Progress bar mid-run with "criterion k/N" and estimate
- [ ] Exported HTML report (printed to PDF), incl. provenance section
- [ ] Inference Lab: saved T4 cards + charts with the "measured on ..." label; token-equality panel
- [ ] Live benchmark table (labelled "live, <device>, <dtype>") next to the saved block
- [ ] Token stream with speculative accepted (indigo) / corrected (amber) tokens
- [ ] Tiled attention grid with skipped blocks and the running max/normalizer table

## Speculative engine, draft model, empty states
- Speculative needs the merged 4B as the *active* model and the 0.6B draft loaded. Use `--load-draft` (or `real.load_draft: true`);
  the engine hint under the toolbar states exactly what is missing.
- Saved benchmarks come from `benchmarks.*` in `configs/app.yaml` (kv_cache_20261107_192136, speculative_20261108_113154).
  `benchmarks.attention` is still unset because no attention benchmark file exists in `results/benchmarks/`; the Lab shows an empty
  state saying how to produce it. A missing file also yields an empty state with a message, never an error.
- If `data/processed/val.parquet` is missing the example dropdowns show "Run python scripts/prepare_data.py".
- Sentence splitter: a full stop after a unit ("mg/dL.", "mmHg.", "kg.") followed by a capitalised word starts a new sentence;
  titles ("Dr."), decimals ("7.5") and a unit dot followed by lower-case/number ("7.5 mg. daily") do not.
