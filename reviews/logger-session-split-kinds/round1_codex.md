request changes

- **Medium — Replay complete open sessions.** `src/aftershock_logger/session.py:817–821`, `_check_one_session`.

  The `ended` condition skips replay whenever `ended_at` is null. However, `close()` can save a complete timeline without setting `ended_at`. An accepted `episode_split` inside a recorded gap then passes `validate` with only an open-session warning, although `check_against_session` rejects it.

  Replay whenever the recorded ranges cover every frame once, regardless of `ended_at`. Keep the reduced checks for incomplete timelines. Add a regression in `tests/test_session.py` using a session closed without finalization and a split inside a gap.

Tests and code were not run.