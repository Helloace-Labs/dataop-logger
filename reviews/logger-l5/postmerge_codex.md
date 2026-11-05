# Codex post-merge confirmation, pull request #5

Reviewed squash commit `a051fa0`, judged against main. Posted on the pull request as "Review by Codex (post-merge)".

request changes

These findings remain on current main.

1. **High — Hold the robot before quarantine work.**  
   Files: `README.md:177`, bounded queue example; `src/aftershock_logger/logger.py:521`, `_quarantine_stopped`.  
   The example catches `ControlledStop` outside the run block. Before `hold_the_robot()` executes, cleanup can wait 10 seconds for the writer and then perform synchronous disk writes. The robot can keep executing its last command during this delay. Catch the stop inside the run block, hold the robot immediately, then re-raise. Keep an outer handler for the completed quarantine report.

2. **High — A timed-out writer can outlive its dataset lock.**  
   Files: `src/aftershock_logger/frame_queue.py:199`, `FrameQueue.close`; `src/aftershock_logger/logger.py:103`, `RunLogger.close`.  
   A timeout leaves the writer thread running inside `add_frame`. Closing the logger then releases the dataset lock. Reopening the dataset permits another writer while the old operation can still modify files. `_abandoned` only prevents the later sidecar append. Retain dataset ownership until that worker exits, and prevent reopening meanwhile. Add a blocked-writer test that attempts to acquire a second logger.

3. **High — Retry can accept an incomplete quarantine record.**  
   Files: `src/aftershock_logger/frame_queue.py:274`, `write_partial`; `src/aftershock_logger/logger.py:572`, `retry_quarantine`.  
   The earlier L1 retry fix checks only the existing manifest’s run ID, length and status. A destination missing its parquet files still counts as success. Retry then clears the retained frames and unblocks recording, losing the recovery copy. Verify required files, checksums and correspondence with the retained record before returning success. On failure, retain the frames and allow spilling. Add a manifest-only destination test.

4. **High — An unreadable quarantine manifest passes validation.**  
   File: `src/aftershock_logger/validate.py:326`, `quarantine_entries` at line 431, and `repair` at line 490.  
   Deleting or corrupting a partial run’s manifest classifies it as `kind="run"` with problems. Both callers ignore those problems. Validation can return `ok=True`, and repair omits the damaged run entirely. Report unreadable quarantine entries as validation errors and as untouched records in repair. Add missing-manifest and invalid-JSON cases.

No tests or scripts were run. `RULES.md`, `docs/WORKFLOW.md` and `docs/CODEX.md` are absent from this checkout.