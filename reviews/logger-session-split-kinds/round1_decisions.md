# Round 1 decisions (logger: session split edits, Aftershock issue #447)

| Finding | Decision | Change |
| --- | --- | --- |
| Medium: replay complete open sessions | Accepted. A session closed without finalize keeps a whole timeline with `ended_at` null, so `validate` now replays the split edits whenever the recorded runs and gaps cover every session frame once, ended or not. A record still recording (or a broken one) gets only the id, kind and frame checks. | `src/aftershock_logger/session.py` (`_check_one_session`), `SPEC.md` (validate), regression in `tests/test_session.py` (`test_close_keeps_an_open_episode_and_the_active_gap`: a split inside a gap of a closed session is refused, one inside its run passes) |
