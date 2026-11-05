## Review by Claude (fallback reviewer)

Reviewed commit: `6d77bae845c99f76039643975c92e216e875e47c` (base `5fed0cf`)

**Verdict: changes requested.** 2 High, 2 Medium, 5 Low. The queue bound, the off-thread writes and the repair handling look right. The gaps are two paths where accepted or sent frames can still disappear, and the tests that would catch them are missing.

### High

**H1. The tick that first sees a write failure is never recorded.** `src/aftershock_logger/frame_queue.py`, `FrameQueue.put`
- Why it matters: `put()` raises at the top (`if self._stop is not None: raise self._stop`) before it calls `_prepare` or appends anything. A write failure in the writer thread only sets `_stop`. So the next `put()` is the first time the loop hears about it, and that frame is dropped. In the documented usage (README, SPEC) `put()` gets `sent_ns`, so the action has already gone to the robot. The quarantine then has N frames for N+1 executed actions, and nothing records the missing one. The docstring says "every tick sent to the robot is recorded somewhere", and the race path a few lines below does keep the frame, so the two paths behave differently.
- Fix: keep the frame on the first `put()` after a stop of any reason. Run `_copy_frame`, `_observation` and `_prepare` first, then append under the lock and raise. Put a cap on frames accepted after a stop (for example one, or up to `capacity + 1` in total) so a caller that keeps calling `put()` cannot grow memory. Add a test where the writer fails, the loop then calls `put()` once, and the quarantine has that frame's `sent_action` as its last row.

**H2. Frames "kept in memory" after a failed quarantine write are lost when the documented flow ends.** `src/aftershock_logger/logger.py` (`Run._quarantine_stopped`, `RunLogger.close`), `src/aftershock_logger/frame_queue.py` (`ControlledStop`)
- Why it matters: the README example re-raises `ControlledStop` out of the run block. In a script that usually means leaving `with RunLogger(...)` and the process exits. `RunLogger.close()` does not look at `_unsaved_quarantine`, so the frames go away with the process. The error message does not show that they were never written either: `ControlledStop.__init__` builds the message before `quarantine_error` is set, so a traceback shows only `write_failed (OSError: No space left on device)`. The only test of `retry_quarantine()` retries inside the same `with` block, which real callers will not do.
- Fix: (a) Rebuild the exception message (`self.args`) after quarantine, so `str(stop)` says either "N frames quarantined at quarantine/<run_id>" or "N frames NOT written: <error>; call run.retry_quarantine()". (b) In `RunLogger.close()`, if any run still has `_unsaved_quarantine`, call `retry_quarantine()` once. If that fails, write the rows somewhere other than the dataset root (a spill directory the caller sets, falling back to the system temp dir) and raise a `RunError` that names where they went. (c) Add a test that lets the `ControlledStop` leave the `RunLogger` block while the disk is still full, and assert the frames end up somewhere or that the error says they did not.

### Medium

**M1. The drain and stop timeouts are a fixed 10 s, so a slow but healthy run gets quarantined.** `src/aftershock_logger/frame_queue.py` (`drain`, `close`, `WRITER_TIMEOUT_S`)
- Why it matters: `run.end()` can have up to `capacity + 1` frames waiting (2 s of frames by default). Image encoding is often slower than real time, so draining that many frames can take more than 10 s even though every write succeeds. The run is then quarantined, and all of its image features are dropped (`dropped_features`), although nothing was wrong. After a drain timeout, `_quarantine_stopped` calls `close()` and waits up to another 10 s, so the robot is held for up to 20 s.
- Fix: measure the timeout as time without progress. Restart the wait whenever `_written` grows, and treat only "no frame finished in `WRITER_TIMEOUT_S`" as a hung writer. You could also scale the limit with `depth()` at the start of the drain. Let the `close()` that follows a drain timeout use a short or zero join, since the writer was already judged stuck.

**M2. Tests are missing for the hardest cases.** `tests/test_frame_queue.py`
- Why it matters: the claims about full disks and hung writers are only partly tested.
  - Nothing checks the frame passed to the `put()` that raises after a write failure (H1).
  - The hung-writer test only covers the stop that `put()` triggers. Nothing covers `run.end()` → `drain()` timing out on a hung writer, including the double wait.
  - Nothing covers a quarantine failure when the `ControlledStop` leaves the `RunLogger` block (H2).
  - The full-disk test raises before calling the real `add_frame`, so LeRobot never fails halfway through a frame (for example an image file written but the buffer not appended). That is the case where `reset_episode_buffer` and the `buffer_size` check in `_write_frame` matter.
  - The "never blocks on disk" check is a single timing bound against a writer that sleeps. Nothing checks that `put()` returns while the writer is blocked inside `add_frame` with the queue below capacity.
- Fix: add one test for each item above. For the partial LeRobot failure, wrap `add_frame` so it calls the real one and then raises (or raises after the image is written), and assert the quarantine row count and `lerobot_buffer_reset`. For blocking, block the writer on an `Event` and assert that `capacity` calls to `put()` all return within a few ms.

### Low

**L1. If the directory fsync after the rename fails, every retry fails.** `src/aftershock_logger/frame_queue.py`, `write_partial`
- Why it matters: `os.rename(tmp, dest)` succeeds, then `_fsync_dir(qdir)` raises. The `except` removes a `tmp` that no longer exists and re-raises. The run is reported as not quarantined, and `retry_quarantine()` then always fails with `already exists`, so the logger stays refused.
- Fix: treat the rename as the commit point and move the fsync after it, outside the cleanup `except`. In `write_partial`, if `dest` already exists with a manifest whose `run_id` and `length` match, return it instead of raising.

**L2. An interrupt during the quarantine write leaves frames in memory without blocking the logger.** `src/aftershock_logger/logger.py`, `_quarantine_stopped`
- Why it matters: it catches only `Exception`. A `KeyboardInterrupt` during `write_partial` leaves `state="quarantined"` and `_unsaved_quarantine` set, but `logger.broken` is never set, so new runs start while the frames are only in memory.
- Fix: catch `BaseException`, set `quarantine_error` and `broken` the same way, then re-raise.

**L3. `validate` fails on any dot entry in `quarantine/`.** `src/aftershock_logger/validate.py`, `quarantine_entries`
- Why it matters: any name that starts with `.` (for example a `.DS_Store` that macOS Finder creates) is reported as an "unfinished quarantine write" error, so `validate` exits nonzero on a healthy dataset.
- Fix: count only directories that match `.<run_id>.*.tmp`, the pattern `write_partial` creates, as unfinished, and ignore other hidden files.

**L4. `_copy_frame` copies only arrays and tensors.** `src/aftershock_logger/frame_queue.py`
- Why it matters: lists, nested dicts and PIL images are queued by reference, so a caller that reuses them on the next tick changes frames that are already queued. Torch tensors are cloned on their own device, so a CUDA camera tensor keeps up to `capacity + 1` frames in GPU memory.
- Fix: deep-copy any other mutable value (`copy.deepcopy`, or `np.array(v)` for lists of numbers), and clone tensors to the CPU (`v.detach().to("cpu", copy=True)`).

**L5. `close()` can report a timeout for a writer that has just finished.** `src/aftershock_logger/frame_queue.py`, `FrameQueue.close`
- Why it matters: between `join(timeout)` returning and `close()` taking the lock, the writer can finish its frame and exit. `close()` still returns `False`, so the LeRobot buffer is not reset and the logger is marked broken for no reason.
- Fix: after taking the lock, set `_abandoned` and return `False` only if `self._inflight is not None` or the thread is still alive. Otherwise return `True`.

