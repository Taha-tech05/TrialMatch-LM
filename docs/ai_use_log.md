# AI Use Log

| Date | Tool | Purpose | What I verified/changed |
|------|------|---------|-------------------------|
| 2026-11-06 | Claude Code | Data download, inspection, patient-grouped split, manifest, leakage checks, tests | Reviewed detected label/evidence/leakage columns against the raw data; ran tests (7 passed); flagged `training` column for my decision |
| 2026-11-06 | Claude Code | Label table verification, output schema, prompt formatter, output parser, tests, decision records | Re-checked label sums against row counts (earlier summary omitted 'not enough information' rows from the table); gold roundtrip test over all rows; adjusted one test where a GPT-4 explanation coincidentally equalled a criterion sentence |
| 2026-11-07 | Claude Code | Model measurement scripts (facts, tokenizer compatibility, memory estimate, token stats), tests, model decision record | Checked KV-cache test value by hand (36x8x128x1024x2x2 = 150,994,944 bytes); confirmed no weights downloaded and test split not read; Llama repos were gated so their rows are reported as skipped, not filled in |
