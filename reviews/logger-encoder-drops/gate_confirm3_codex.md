approve

**High — Finding 1 is resolved.** The fix in `68ab906` is correct and complete. `src/aftershock_logger/logger.py:810` registers the frames for recovery before staged-folder cleanup. Cleanup interrupts no longer skip quarantine. A failed write keeps the recovery entry for retry and spill. A successful write removes it, then re-raises the interrupt.

No adjacent regression found. `tests/test_encoder_backpressure.py:604` now covers ordinary cleanup failure and `KeyboardInterrupt`.

Read-only review. No tests or code were run.