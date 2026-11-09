request changes

1. **High — A direct stop can lose the recorded prefix.**  
   Files and sections: `src/aftershock_logger/logger.py`, `_save_after_encoder_stop`, `_check_rows`, and `RunLogger.close`.  
   A prediction may refer to the next frame before `add_frame` records it. If that frame hits encoder backlog, automatic sealing rejects the prediction. This replaces `EncoderBacklog` with another error and leaves the run active. Closing the logger then discards all recorded frames. Separate predictions for the rejected frame, preserve the prefix if sealing fails, and keep the controlled stop visible. Add a prediction-enabled stop test.

2. **High — Repair can commit a known video mismatch.**  
   Files and sections: `src/aftershock_logger/logger.py`, `_save_and_commit`; `src/aftershock_logger/validate.py`, `repair` and `_matches`.  
   The new post-save guard leaves mismatched episodes staged. Repair matches only rows, actions, and task, then commits the run and removes `save_error`. It can therefore commit the episode the guard rejected. Check camera counts before repair commits a match. Add a repair regression test for the post-save mismatch case.

3. **High — Drive gaps can erase the backlog reason.**  
   Files and sections: `src/aftershock_logger/session.py`, `add_frame` and `_close_gap`; `src/aftershock_logger/logger.py`, `end` and `_seal`.  
   A gap that raises `EncoderBacklog` remains active. `_close_gap` calls `_seal("unknown")`, bypassing the stop checks and label added by `end`. The gap can be committed without any persisted backlog reason, and the session can continue. Handle the stop in the session, record fault evidence, preserve its reason when saving the gap, and end as `fault_stop`. Add a streaming-gap test.

4. **High — Missing camera counts silently pass both checks.**  
   Files and sections: `src/aftershock_logger/logger.py`, `_saved_video_frames`; `src/aftershock_logger/adapter.py`, `episode_video_frames`; `src/aftershock_logger/validate.py`, `video_frame_problems`.  
   Missing or null timestamps omit a camera. A conversion error makes `_saved_video_frames` return an empty result, hiding other camera counts too. Saving and validation check only returned counts, so incomplete metadata can pass without checking every camera. Require a valid count for every completed video camera. Handle deferred encoding explicitly and report unreadable counts. Keep the metadata access in `adapter.py`.

5. **High — The successful backpressure test can pass without backpressure.**  
   File and section: `tests/test_encoder_backpressure.py`, `test_a_slow_encoder_gets_backpressure_and_loses_no_frame`.  
   The simulated delay is 0.1 seconds, equal to the encoder’s existing queue-put timeout. Its existing wait can avoid drops even if the logger’s wait is removed. Timing jitter can also make this test flaky. Hold the consumer longer than that timeout but less than `encoder_wait_s`, using synchronization, and assert that writing waits and every frame survives.

Tests and code were not run.