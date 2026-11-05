request changes

1. **Medium — Synchronize the existing stalled-writer test before closing.**  
   Files: `tests/test_frame_queue.py:329`, `test_memory_stays_bounded_under_a_stalled_writer`; `src/aftershock_logger/logger.py:118`, `RunLogger.close`.  
   The test calls `gate.set()` and immediately leaves the logger block. Opening the gate does not mean the worker has exited. The new `close()` can still see it alive and raise an uncaught `RunError`, making this existing test fail depending on scheduling. After opening the gate, join the queue’s worker with a bounded timeout and assert it exited before leaving the logger block. Release the gate in `finally`.

No tests or scripts were run. `RULES.md`, `docs/WORKFLOW.md`, and `docs/CODEX.md` are absent.