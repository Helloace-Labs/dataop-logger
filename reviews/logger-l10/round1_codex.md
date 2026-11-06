request changes.

1. **High — Tap warnings can abort recording.**  
   Files: `src/aftershock_logger/live.py`, `LiveTap._fail`; `tests/test_live.py`, failure tests.  
   `warnings.warn()` raises when warnings are configured as errors. A caller-side tap failure then escapes from its exception handler into `add_frame` or `put`, and the run can be discarded. Warning output also performs I/O on the caller’s thread. Move warning emission to the tap thread and protect it against exceptions. Add failure tests with warnings treated as errors and assert that recording still commits.

2. **High — Rotation can silently skip unacknowledged events.**  
   File: `src/aftershock_logger/live.py`, `read_events` and `_flush_events`.  
   The reader reads the archive before the current file. Rotation between those reads can replace the current file before it is read. The reader then returns newer events while skipping records now held in the archive. The gap check misses this because it checks only the earliest retained sequence. Detect file changes and retry, deduplicate records, and check continuity after `after_seq`. Add a test that forces rotation between the two reads.

3. **High — A timed-out tap can overwrite a new writer’s files.**  
   Files: `src/aftershock_logger/live.py`, `close` and `_loop`; `src/aftershock_logger/logger.py`, `RunLogger.close`.  
   After the two-second join timeout, the logger releases its writer lock even if the tap is still encoding or writing. A new logger can start before the old tap resumes. The old tap can then overwrite current status with an older sequence or `closed: true`, and append conflicting events. Keep ownership until the old thread exits, using deferred lock release while keeping close bounded. Test reopening while the old tap is blocked.

4. **High — Status can combine different frames or runs.**  
   File: `src/aftershock_logger/live.py`, `frame` and `_status`.  
   Frame metadata, joint state and camera fields are updated and read separately. A status can identify one frame or run while carrying another’s joint values. Non-array state values also retain caller-owned storage, so reusing a tensor can change the snapshot without another accepted frame. Publish one coherent snapshot with owned state storage. Add tests that pause snapshot construction across a frame update and reuse a state tensor.

5. **High — Privacy filtering misses JSON keys and preview filenames.**  
   File: `src/aftershock_logger/live.py`, `_Scrub.__call__`, `camera_name` and `_write_previews`.  
   Dictionary keys are never scrubbed. A camera named after the host exposes that hostname in status keys and preview filenames, even though string values are cleaned. Scrub keys and derive collision-safe preview identifiers that contain no private names. Extend the privacy test to camera names and filenames.

6. **High — The performance test measures only the frame hook.**  
   File: `tests/test_live.py`, `_hook_costs` and `test_tap_cost_per_frame_stays_small`.  
   Timing starts inside `LiveTap.frame`. It excludes tap-induced contention during the rest of `add_frame` or `put`, including queue copying and dataset work. The test can pass while the required API latency exceeds its limit. Measure complete API calls with the tap off and on during matched 2,000-frame replays, assert that the tap remains healthy, and report those measurements.

7. **Medium — Ending a run can break preview processing.**  
   File: `src/aftershock_logger/live.py`, `run_state` and `_write_previews`.  
   The tap snapshots camera keys, then encodes each image. While it encodes the first camera, `run_state` can clear the remaining slots. The next `pop(cam)` raises and disables the tap. Transfer slots safely to a private batch and handle run changes explicitly. Add a two-camera test that ends the run while the first encoder call is paused.

8. **Medium — Controlled stops remain invisible during cleanup.**  
   File: `src/aftershock_logger/logger.py`, `_quarantine_stopped` and `retry_quarantine`.  
   The stop reaches the tap only after waiting for the frame writer and attempting quarantine writes. During that wait, status keeps advancing while showing an active run without a stop. A successful quarantine retry also leaves the reported path unchanged. Publish the stop when first detected, then update its quarantine result after writes, retries and spills.

Tests and scripts were not run.