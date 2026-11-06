# AI Use Log

| Date | Tool | Purpose | What I verified/changed |
|------|------|---------|-------------------------|
| 2026-11-06 | Claude Code | Data download, inspection, patient-grouped split, manifest, leakage checks, tests | Reviewed detected label/evidence/leakage columns against the raw data; ran tests (7 passed); flagged `training` column for my decision |
| 2026-11-06 | Claude Code | Label table verification, output schema, prompt formatter, output parser, tests, decision records | Re-checked label sums against row counts (earlier summary omitted 'not enough information' rows from the table); gold roundtrip test over all rows; adjusted one test where a GPT-4 explanation coincidentally equalled a criterion sentence |
