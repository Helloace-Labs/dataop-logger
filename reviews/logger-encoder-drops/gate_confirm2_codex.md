request changes

1. **High — Finding 1 is still incomplete.** `src/aftershock_logger/logger.py:813` catches ordinary cleanup errors, but a `KeyboardInterrupt` during `shutil.rmtree` still skips the protected quarantine write. The run is already `quarantined` and absent from `_unsaved_runs`, so `RunLogger.close()` skips recovery and spill. The recorded prefix can still be lost on exit. Register the frames for recovery before cleanup, and preserve interrupt propagation after securing them. Add coverage for interrupted cleanup.

The fix is correct for ordinary filesystem errors. The new test covers that case. The block-exit change correctly raises the stored stop; no adjacent regression found.

Read-only review. No tests or code were run.