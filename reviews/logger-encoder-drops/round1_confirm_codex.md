request changes

1. **High — Fix 1 is incomplete.** Prediction filtering is correct. But `src/aftershock_logger/logger.py:775` resets the buffer before an unprotected quarantine write. If that write fails, its error replaces the controlled stop. The run is never registered for retry or spill, so closing cannot preserve its frames. Reuse the queued quarantine failure handling.

2. **Fix 2 is correct and complete.** `src/aftershock_logger/validate.py:516` checks camera counts before committing a unique match. A mismatch quarantines the run with its reason. No adjacent regression found.

3. **High — Fix 3 is incomplete.** Labels, fault evidence, and final termination are correct. But `src/aftershock_logger/session.py:310` never checks the session’s stored stop. After `seal_episode("system_error")`, another `add_frame` opens a gap and records more frames. Add the stop check before either recording branch. The new test only checks rejection before sealing.

4. **Fix 4 is correct and complete.** `src/aftershock_logger/adapter.py:208` reports unreadable counts separately for each encoded camera and explicitly excludes deferred cameras. Saving and validation reject unknown counts. No adjacent regression found.

5. **Medium — Fix 5 is improved but incomplete.** `tests/test_encoder_backpressure.py:85` increases the delay and checks queue fullness. It still relies on scheduling and does not establish that writing stays blocked until the consumer is released. Use synchronization and assert that blocked write. Timing flakiness remains; no other adjacent regression found.

Tests and code were not run.