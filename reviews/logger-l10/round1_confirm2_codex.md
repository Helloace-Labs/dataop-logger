request changes.

1. **Fix 1 is correct and complete.** In `src/aftershock_logger/live.py`, `LiveTap.__init__` emits the setup failure warning from a daemon thread. Warning exceptions remain contained. No nearby regression found.

4. **Fix 4 is correct and complete.** In `src/aftershock_logger/live.py`, `frame` publishes REC counts inside `_snap`, and `_status` reads those counts with the matching frame and run identity. `_cameras` restores the cleaned image `key`. No nearby regression found.

5. **Fix 5 is correct and complete.** In `src/aftershock_logger/live.py`, `_setup` cleans the original camera name before making it filename-safe and unique. Camera timestamps can use the original name or full image key. No nearby regression found.

6. **Fix 6 is correct and complete.** In `tests/test_live.py`, `_replay` measures whole calls for matched tap-off, tap-on and preview-enabled runs. Each measured tap receives a health assertion. Whole-call figures are reported in benchmark mode, as the decisions specify. No nearby regression found.

8. **Medium — Stop publication can race with quarantine completion.** Fix 8 is partly correct but incomplete. Sections: `src/aftershock_logger/live.py`, `stop_seen`, `_stop_update` and `_loop`; `src/aftershock_logger/logger.py`, `Run._quarantine_stopped`.

   The cleanup hook now precedes the writer wait, and the tap polls for stopped queues. However, both threads can publish stop state. The tap can set `_stop_seen_for` and pause before publishing `detected`. Cleanup then skips detection and publishes `quarantined`. When the tap resumes, it can overwrite that final state with `detected`, clearing the quarantine location and appending events out of order. This is a nearby regression from adding tap-thread detection.

   Serialize per-run deduplication, status assignment and event append with final-phase publication. Keep filesystem work outside that critical section. Add a barrier test that pauses detection after claiming the run while cleanup proceeds. The current test waits for detection before leaving the block, so it misses this race.

Tests and scripts were not run.