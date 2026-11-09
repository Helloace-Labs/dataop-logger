approve

- **Finding 1 (High): Correct and complete.** `src/aftershock_logger/logger.py` preserves the controlled stop, retains frames after a quarantine write failure, and registers the run for retry and spill. No adjacent regression found.
- **Finding 3 (High): Correct and complete.** `src/aftershock_logger/session.py` checks the stored stop before either recording branch. Sealing and writing remain available. No adjacent regression found.
- **Finding 5 (Medium): Correct and complete.** `tests/test_encoder_backpressure.py` holds the encoder with an event, checks that writing waits without dropping frames, then releases it and verifies every frame was saved. Both recording modes are covered. No adjacent regression found.

Only these three fixes in `fd25435` were reviewed. Tests and code were not run.