request changes

1. Fix 1 is correct and complete by inspection. In `src/aftershock_logger/export.py`, `subtask_digest()` hashes the effective IDs and ranges used for prompt planning. `export()` records it only for `task+subtasks`. `training_sets()` detects mapping-only changes. Default records gain no key, and older records retain the previous review-digest check. No adjacent regression found.

2. **High — The baseline still misses export-record regressions.**  
   Files and sections: `tests/export_baseline.py`, `_runs()` and `summary()`; `tests/data/export_default_baseline.json`; `tests/test_export_prompts.py`, baseline comparison.

   Fix 2 is partly correct but incomplete. It adds a fixed baseline for all five recipes. However, `_runs()` removes digest fields without retaining their key names. Removing `review_digest` from a saved run record would therefore pass this baseline comparison, although `training_sets()` needs that field. `summary()` also compares source key names only, ignoring stable values such as `status` and `lists_sha256`.

   Preserve nested key sets while normalizing volatile values. Compare stable source values. Regenerate the baseline using the logger on main before this change. No runtime regression was found; the remaining problem is incomplete verification of the default record.

Tests and scripts were not run.