# Logger L9, round 1 confirmation decisions

Codex confirmation round 1 (`round1_confirm_codex.md`): request changes, 2 High, 2 Medium on the round 1 fixes; fixes 4, 6, 7 and 8 confirmed.

| # | Finding | Decision | Where |
| --- | --- | --- | --- |
| 1 | High: failed-run tick counts unchecked | Changed. Once the digests match, `validate` reads the quarantined ticks' `frame_index` and requires frames 0 to n-1 of the run's range. | `session.py` (`_check_failed_run`), test `test_failed_run_ticks_must_cover_its_range` |
| 2 | High: close can still lose an open episode after a terminal tick | Changed. On the close path the last tick's terminal flags are cleared before the `system_error` seal, since a fault seals at any length. | `session.py` (`close`), test `test_close_after_a_terminal_tick_keeps_the_episode` |
| 3 | Medium: failed finalization cannot be retried | Changed. The session keeps the gap dataset until its finalize succeeds and is no longer marked broken; it refuses new frames while ending, and a second `finalize` call retries with the end decided the first time. | `session.py` (`finalize`, `_finish`, `_close_gap_dataset`), test `test_a_gap_dataset_that_cannot_finalize_fails_the_session` |
| 5 | Medium: nested type errors crash validation | Changed. A gap index that is not an int and a run entry without a string `run_id` skip their dependent checks, and each session is checked inside its own guard that reports "could not be checked" instead of raising. | `session.py` (`_check_one_session`, `session_problems`), mutation cases |
