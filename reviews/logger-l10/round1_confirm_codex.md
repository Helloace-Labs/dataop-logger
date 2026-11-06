request changes.

1. **Low — Startup warnings still run on the caller.** Fix 1 is mostly correct. In `src/aftershock_logger/live.py`, `_fail` no longer warns, and `_warn` catches warning exceptions. However, `LiveTap.__init__` calls `_warn` directly after setup fails. Warning output can still block the caller. Emit that warning from a background thread too. No other nearby regression found.

2. **Fix 2 is correct and complete.** In `src/aftershock_logger/live.py`, `read_events` detects replacement, retries the reads, deduplicates sequences and reports interior holes. No nearby regression found.

3. **Fix 3 is correct and complete.** In `src/aftershock_logger/logger.py`, `RunLogger.close` retains the lock until both writer and tap threads exit. Quarantine retries and spills happen before release. No nearby regression found.

4. **Medium — REC counts remain outside the coherent snapshot.** Fix 4 is incomplete. In `src/aftershock_logger/live.py`, `frame` updates `_rec` before publishing `_snap`, and `_status` reads them separately. Status can show two flagged frames while reporting only one accepted frame. Counts can also cross a run boundary. Publish REC counts with the frame and run identity in one snapshot. Add a test paused between counter updates and snapshot publication.

   **Medium — Camera image keys disappeared.** The same fix introduces a regression in `_cameras` and `_status`. Camera metadata now contains dimensions but loses the documented `key` field. Readers cannot reliably map cleaned camera identifiers back to image features. Restore the image key in the immutable camera metadata and assert its presence in the status test.

5. **High — Camera names are sanitized before privacy cleaning.** Fix 5 is incomplete. In `src/aftershock_logger/live.py`, `_setup` builds `cam_given` through `camera_name` before scrubbing it. That replaces path separators, so dataset-root and home-folder strings no longer match the cleaner. Private path components can survive in camera identifiers and preview filenames. Scrub the original camera name first, then make it filename-safe and unique. Add camera-name tests containing root and home strings. Dictionary-key cleaning itself is correct.

6. **Medium — Preview contention still escapes whole-call measurement.** Fix 6 is incomplete. In `tests/test_live.py`, `_replay_full_calls` measures complete calls only without previews. Preview-enabled measurements still time just the hook. Their tap health is also not asserted; the health assertion covers a separate replay. Measure whole calls with previews enabled against the matched baseline, and assert health for every measured tap. The clock selection is implemented as described.

7. **Fix 7 is correct and complete for the reported crash.** In `src/aftershock_logger/live.py`, `_write_previews` safely skips slots removed during encoding through `pop(cam, None)`. No nearby regression found.

8. **Medium — Automatic cleanup bypasses early stop publication.** Fix 8 is incomplete. In `src/aftershock_logger/logger.py`, the `RunLogger.run` exit paths call `_quarantine_stopped` directly when the queue has failed. If failure happens after the last `put`, `_tell` never publishes `stop_seen` before cleanup waits. Status can still show an active run without a stop. Publish detection before cleanup through a shared, once-only notification path. Test normal and exceptional block exits after a background write failure. Retry and spill updates are correctly connected.

Tests and scripts were not run.