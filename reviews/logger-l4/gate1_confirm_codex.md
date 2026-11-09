Verdict: Both fixes are correct and complete for the stated issues. No adjacent regressions found by inspection.

- **High — resolved.** `src/aftershock_logger/metadata.py:659–664` rejects different content at the same revision and a higher revision with rewritten history. Both checks run under the lock, before saving. `tests/test_run_metadata.py:507` checks both conflicts, unchanged file bytes, an accepted extension, and an identical retry. The conflict assertions would fail without their respective guards. Lower revisions still raise `StaleRevError`; locked-batch behavior remains intact.

- **Medium — resolved.** `src/aftershock_logger/metadata.py:567–570` keeps a human reason in history and retains an agent reason in the detection. `tests/test_run_metadata.py:551` covers the API and CLI, saved reasons, and validation. Without the fix, the first proposal raises during save. No adjacent breakage found. The added agent check covers a later suggestion, so preservation of an initial agent detection’s reason is confirmed by code inspection, not that test.

Tests were inspected only. No tests or code were run, and no files were edited.