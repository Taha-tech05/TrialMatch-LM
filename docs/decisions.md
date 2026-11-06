# Technical Decisions Log

| Date | Decision | Reasoning | Alternatives considered |
|------|----------|-----------|-------------------------|
| 2026-11-06 | Patient-grouped split: sort unique patient IDs, shuffle with `np.random.default_rng(42)`, first 37 train / next 8 val / last 8 test; all rows follow their patient. Forbidden (previous-prediction) columns: `gpt4_explanation`, `explanation_correctness`, `gpt4_sentences`, `gpt4_eligibility` - removed from processed files, stored separately. `training` column flagged, not yet resolved. | Prevents patient/note leakage across splits; GPT-4 outputs would leak a prior model's answers. | Row-level random split (leaks notes); stratified split (breaks the exact rule requested). |
