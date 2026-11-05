approve

2. **Correct and complete.** Files: `src/aftershock_logger/batches.py`, `tests/test_batches_files.py`. Logger recovery uses `_choose_for_commit` and falls back to the unnamed collecting choice. It cannot reach the legacy name lookup. The regression test covers closure during recording and recovery before join. No adjacent regression found.

4. **Correct and complete.** Files: `src/aftershock_logger/export.py`, `tests/test_batches_files.py`. `training_sets` combines matching export stale flags from every source dataset. Moving a run back cannot make the set current again. The regression test checks both datasets with zero moved runs. No adjacent regression found.

No High, Medium or Low findings. Static review of `d5f34d4` only. Tests were read, not run.