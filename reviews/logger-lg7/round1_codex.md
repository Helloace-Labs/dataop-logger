request changes

1. **High — Subtask label changes leave exports marked current.**  
   Files: `src/aftershock_logger/export.py`, prompt planning and `training_sets()`; `src/aftershock_logger/review.py`, `subtask_segments()`.  
   Export now depends on subtask IDs, but its review digest still hashes `effective()`, which omits recorded-step mappings. Changing a mapped ID from `unresolved` to a real subtask changes the required training data while the old export remains marked current. Add an opt-in digest covering effective subtask IDs and ranges. Compare it during stale checks. Test mapping-only changes while preserving default export records.

2. **High — The compatibility test cannot detect changed default bytes.**  
   File: `tests/test_export_prompts.py`, `test_the_default_export_is_unchanged()`.  
   Both calls use the new implementation. A regression affecting both calls equally still passes. The test also excludes the export record from its byte comparison and exercises only `teleop`. Add fixed public baseline outputs from `origin/main`, including LoRA and stable record content. Compare the default export against those baselines.

Tests and scripts were not run. `docs/WORKFLOW.md` and `docs/CODEX.md` are absent from this checkout.