approve

No High, Medium or Low findings.

The fix is correct and complete. `src/aftershock_logger/session.py` now replays split edits whenever recorded ranges cover every session frame exactly once, regardless of `ended_at`. Incomplete timelines retain reduced checks.

The regression in `tests/test_session.py` checks both a rejected gap split and an accepted run split. `SPEC.md` matches the behavior. No adjacent regressions found by inspection.

Tests and code were not run.