## Review by Claude (fallback, Codex at limit), confirmation round

Reviewed commit: `bb917a25663c6b08038d6389fe227ca569a7e491` (diff `6d77bae..bb917a2`, against round 1 at `6d77bae`)

**Verdict: approve.** 0 High, 0 Medium, 3 Low (new, none blocking). All round 1 findings are resolved.

**Round 1 status**
- H1 resolved. `put()` now copies, checks and appends the first frame after a write failure before it raises; `_stop_told` (set by `put()`, `check()` and `drain()`) stops any further frames, so memory stays bounded at about `capacity + 2`. Covered by `test_first_put_after_a_write_failure_keeps_its_frame` (asserts the last `sent_action`) and `test_check_counts_as_telling_the_caller`.
- H2 resolved. `ControlledStop._settle()` puts either "N frames quarantined at ..." or "N frames were NOT written (...)" into the message; `RunLogger.close()` retries each unsaved run, then spills to `spill_dir` and raises a `RunError` naming the path, or saying the frames were not written anywhere. The README example no longer re-raises silently.
- M1 resolved. `drain()` times out only after `stall_timeout` with no growth in `_written`; `close()` joins with 0 after `drain()` found the writer hung, so the 20 s double wait is gone. Both timeouts are validated arguments of `run.queue()`.
- M2 resolved. The new tests do reach the intended paths: the hung writer at `run.end()` with a 5 s `stop_timeout` asserted under 3 s, LeRobot failing after the frame reached its buffer (row counts, buffer reset, next run commits), `put()` returning while the writer is blocked in `add_frame` below capacity, and the stop leaving the `RunLogger` block with the quarantine still failing (spilled, and not written anywhere).
- L1 to L5 resolved as described in the decisions file. On L5: the writer exits on its next loop once a stop or abandon is set, so the unbounded `join()` after `_busy` is cleared can only wait for one append.

**Default `spill_dir` in the system temp directory.** Acceptable as a last resort, since the alternative is losing the frames when the process exits and the error names the path. Two limits should be stated, though (L6): temp folders are cleared on reboot on macOS and on many Linux systems (where /tmp is often tmpfs or aged out by tmpfiles), and on a single-disk robot computer the temp folder is usually on the same disk that just filled up, so the default spill fails in the most common case.

### Low

**L6. The spill message and docs do not say the default folder is temporary.** `src/aftershock_logger/logger.py`, `RunLogger.close`; README; SPEC
- Why it matters: the message says "move that folder ... when there is space", but nothing warns that a reboot may delete it first. A shared /tmp folder created by one user can also be unwritable for another, which turns into "NOT written anywhere".
- Fix: when the spill path came from the default, add "this is a temporary folder that may be cleared on reboot; move it before restarting" to the message, and recommend in README and SPEC setting `spill_dir` to a persistent folder on a different disk than the dataset for robot use. A per-user default (for example including the user name in the folder name) avoids the shared-folder case.

**L7. After a spill, the `ControlledStop` still says the frames were not written, and the close error replaces the stop.** `src/aftershock_logger/logger.py`, `RunLogger.close`
- Why it matters: the spill branch sets `run.quarantine_path` but does not update `run._stop`, so a caller that kept the stop object still reads "NOT written ... close the RunLogger". When the stop is still propagating as `RunLogger.__exit__` runs, the `RunError` from `close()` replaces it, so an `except ControlledStop` placed outside the `RunLogger` block (for example around a robot-hold) does not fire. The README example catches inside the block, so the documented flow is fine.
- Fix: in the spill branch, set `run._stop.quarantine_path` and call `_settle()` with a "spilled to <path>" tail. Mention in SPEC that `close()` can raise a `RunError` chained to the stop, so the hold-the-robot handler belongs inside the `RunLogger` block (or have `__exit__` warn instead of raising while a `ControlledStop` is propagating).

**L8. Small test gaps for L1, L2 and L5.** `tests/test_frame_queue.py`
- Why it matters: `test_quarantine_record_survives_a_failed_folder_sync` never reaches the new "destination already holds this run's record" branch in `write_partial`, because the first write succeeds and `retry_quarantine()` returns early. L2 (interrupt during the quarantine write) and L5 (`close()` after the writer just finished) have no tests.
- Fix: one test that pre-creates the destination with a matching manifest and calls `write_partial` (and one with a mismatched manifest that must raise); one that raises `KeyboardInterrupt` from `write_partial` and asserts `logger.broken` is set and the run is in `_unsaved_runs`.

