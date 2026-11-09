# Round 1 confirmation decisions (logger: video encoder backpressure)

| Finding | Decision | Change |
| --- | --- | --- |
| 1 High: the direct quarantine write is unprotected | Accepted. `_quarantine_direct` handles a failed write as a queued stop does: the stop keeps `quarantine_error`, the frames stay in memory for `run.retry_quarantine()`, the run is registered with the logger, and closing the RunLogger retries and then spills them. | `src/aftershock_logger/logger.py` (`_quarantine_direct`); test `test_a_direct_quarantine_that_cannot_be_written_keeps_the_frames_for_the_spill` |
| 2, 4: correct | No change. | |
| 3 High: after sealing, add_frame opens a gap and records more | Accepted. `SessionLogger.add_frame` raises the stored stop before either branch, so nothing more is recorded into an episode or a gap. | `src/aftershock_logger/session.py` (`add_frame`); the Drive test now also adds a frame after the seal and the write |
| 5 Medium: the backpressure test relies on scheduling | Accepted. A new test holds the encoder threads on an event: with the camera queue full, the write (direct) or the writer (queued) stays blocked for 0.5 s, past LeRobot's 0.1 s put timeout, with no frame dropped and the frame not in LeRobot; releasing the encoder lets it finish, and the video holds every frame. The timed test stays as a realistic slow encoder. | `tests/test_encoder_backpressure.py` (`held_encoder`, `test_a_frame_waits_while_the_encoder_queue_is_full_and_goes_in_when_it_has_room`) |
