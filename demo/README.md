# Demo notes: screenshots and demo video link

The application was run in **real mode** (Qwen3-4B + LoRA merged, cuda, float16, draft Qwen3-0.6B base) on a Colab Tesla T4 and reached through a public tunnel. All screenshots are dark theme and were taken on 2026-11-10. `06_report_export.png` is cropped to remove the browser toolbar. Numbers quoted in `report/main.tex` from these images are read off the screenshots.

**Demo video (5 to 7 minutes):** https://youtu.be/e3O8ud70Kxo

| File | What it shows |
|---|---|
| `01_header_real.png` | Empty workspace, REAL header (model, device, dtype, draft), paste areas for note and criteria |
| `02_note_numbered_criteria.png` | Unnumbered pasted note split into 8 sentences `[S0]..[S7]`; 10 parsed criteria |
| `03_evidence_sync.png` | Per-criterion cards with badge, cited sentence and latency; a clicked pill highlights sentence S2 in the note |
| `04_details_drawer.png` | Details drawer: status, parse, engine, latency, tokens and the raw JSON answer |
| `05_summary.png` | Summary tab: "Undetermined", the rule text, "not a ground-truth label", counts |
| `06_report_export.png` | Exported HTML report: run details (mode, model, device, engine, git commit, adapter, run id) |
| `07_inference_lab_saved.png` | Inference Lab: saved T4 benchmark cards (512 input, 64 output) with the "measured on ..." label |
| `08_live_bench_tiled.png` | Live mini-benchmark controls and the tiled-attention visualiser (6 of 16 blocks skipped) |
| `09_speculative_run_16_criteria.png` | Validation note screened with the speculative engine: 16/16 criteria in 22 s, saved report, NEI with missing-information flag |
| `10_live_kv_run.png` | Live KV-cached run: prompt 323 tokens, 17.13 decode tok/s, prefill 0.236 s |
| `11_mi_3_months_not_excluded.png` | Exclusion criterion "Myocardial infarction within the last 6 months" labelled "not excluded" while citing "3 months ago" (cropped); also BMI inclusion = NEI with the missing flag, "Pregnant or breastfeeding" = not applicable |

Not yet captured (see the checklist in `docs/app_notes.md`): DEMO-mode header, light theme, the other badge types, a mid-run progress bar, and a token stream with accepted/corrected speculative tokens.
