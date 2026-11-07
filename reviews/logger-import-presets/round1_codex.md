request changes

1. **High — Generic transition columns invent success labels.**  
   `src/aftershock_logger/importer.py`, `detect()` and `PRESETS["lerobot-hil-record"]`.

   Any dataset with `next.reward` and `next.done` selects the HIL preset. Those columns do not establish that positive reward means task success. A failed episode with positive shaping rewards becomes a success, affecting success rates and export selection.

   Require identifying evidence for this preset. Otherwise select `plain` and require an explicit preset or mapping. Add a synthetic dataset with dense rewards and a failed outcome to test this ambiguity.

2. **High — Invalid rewards produce confident outcomes.**  
   `src/aftershock_logger/importer.py`, `_build_run()`, outcome threshold handling at lines 428–430.

   An episode containing only NaN rewards becomes a failure. A positive infinity becomes a success. The reward handling then replaces these values with zero and marks them invalid, so validation cannot catch the unsupported outcome. This violates the unknown handling rule in `SPEC.md`.

   Check sample validity before deriving outcomes. Accept success only from a finite value above the threshold. Keep the outcome unknown when invalid samples prevent establishing failure. Report the invalid data and add tests for NaN and infinity.

3. **Low — The design table retains the superseded outcome rule.**  
   `docs/DESIGN_batches_import_export.md`, section 2, “Field mapping.”

   The table still describes `next.success` on the last frame as the standard outcome source. This conflicts with the updated README and importer.

   Document the reward threshold used by the HIL presets. Describe `next.success` as an optional custom mapping and align its any-frame behavior.

Tests were read, not run. No files were changed.