# Gate review decisions (logger: video encoder backpressure)

The lead's gate Codex review of pull request #19 at 0a1b741 found three High findings (on the pull request).

| Finding | Decision | Change |
| --- | --- | --- |
| 1 High: a stopped Drive episode can lose its recorded prefix when staging fails, then the logger's close discards it | Accepted. `run.end` on an encoder-stopped run quarantines the recorded frames when the seal or the staged write fails (kept in memory for `retry_quarantine()` and the spill if that write fails too). `RunLogger.close` saves an encoder-stopped run left open instead of discarding it. The session records such an episode as a failed write (`write_status` failed, `persistence_error` row), so coverage holds and the session ends as `fault_stop`. | `src/aftershock_logger/logger.py` (`end`, `_save_after_encoder_stop`, `RunLogger.close`), `src/aftershock_logger/session.py` (`seal_episode`); tests `test_a_drive_episode_stopped_by_the_encoder_that_cannot_be_staged_is_quarantined`, `test_closing_the_logger_keeps_a_run_left_open_after_an_encoder_stop` |
| 2 High: a camera with no chunk_index is treated as deferred, so it drops out of save, validate and repair | Accepted. The expected cameras come from the dataset's video features. A camera is left out only when deferral is established (LeRobot encodes in batches and does not stream); otherwise a missing video or missing columns count as unreadable. validate reads the episode table for every episode, also ones it could not load. | `src/aftershock_logger/adapter.py` (`episode_video_frames`, `video_keys`, `saved_video_frames`, `episode_metadata`), `src/aftershock_logger/validate.py`; tests `test_a_camera_missing_from_the_episode_record_is_reported_not_skipped` (columns removed, and set to null), `test_at_save_a_camera_the_record_leaves_out_stops_the_commit`, `test_a_count_the_episode_record_does_not_give_is_a_problem` |
| 3 High: after EncoderBacklog, seal_episode("operator_abort") on a short episode takes the short stop branch and discards the run | Accepted. `seal_episode` checks the run's stored stop before the short stop rule and refuses any termination but `system_error`. | `src/aftershock_logger/session.py` (`seal_episode`); test `test_a_short_drive_episode_that_hits_the_budget_is_no_short_stop` |

Tests: `python -m pytest -q`: 626 passed, 82 skipped.

## Confirmation round (gate_confirm_codex.md)

| Finding | Decision | Change |
| --- | --- | --- |
| 1 High: a cleanup failure of the staged folder skips the protected quarantine write | Accepted. Removing a half written staged folder can no longer stop the quarantine: a failure is added to the stop's detail, the frames are written to quarantine (or kept in memory for the retry and the spill), and validate reports the folder left in runs/. A run block that exits normally after such a quarantine raises the stop instead of reporting a run with no end. | `src/aftershock_logger/logger.py` (`_quarantine_direct`, `RunLogger.run`); test `test_a_staged_folder_that_cannot_be_removed_does_not_lose_the_frames` |
| 2, 3: resolved | No change. | |

Tests: `python -m pytest -q`: 627 passed, 82 skipped.

## Second confirmation round (gate_confirm2_codex.md)

| Finding | Decision | Change |
| --- | --- | --- |
| 1 High: an interrupt during the cleanup still skips the protected write | Accepted. The run is registered for the retry and the spill before anything else, and leaves that list only once its quarantine is written. A failure of any kind while removing the staged folder (an interrupt included) is recorded in the detail; the frames are written; an interrupt is raised again after that. | `src/aftershock_logger/logger.py` (`_quarantine_direct`); `test_a_staged_folder_that_cannot_be_removed_does_not_lose_the_frames` now runs with an OSError and with a KeyboardInterrupt |

Tests: `python -m pytest -q`: 628 passed, 82 skipped.
