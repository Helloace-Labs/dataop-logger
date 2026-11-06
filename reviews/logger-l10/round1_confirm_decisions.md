# Round 1 confirmation decisions (logger L10, live tap)

The confirmation round (`round1_confirm_codex.md`) confirmed fixes 2, 3 and 7 and found the rest incomplete.

| Item | Severity | Decision | What changed |
| --- | --- | --- | --- |
| 1 | Low | Changed | When the tap cannot be set up, its one warning is emitted from a short daemon thread, never on the caller's thread (`LiveTap.__init__`). |
| 4 (REC counts) | Medium | Changed | The REC counts are part of each frame's snapshot (`_snap["rec"]`), so the status shows the counts as of the frame it shows. Test: `test_rec_counts_are_read_with_the_frame_they_belong_to` (a status taken after the counts changed and before the snapshot was published). |
| 4 (camera key) | Medium | Changed | Each camera's entry carries its image key again (cleaned). The status test checks it. |
| 5 | High | Changed | Camera ids come from the camera name cleaned first (root, home and host replaced), and only then made safe for a file name and unique; the image keys in the status are cleaned the same way. A caller's `camera_ns` may name a camera by its name or its full image key. Test: `test_camera_names_holding_the_root_or_home_are_cleaned_before_they_become_ids`. |
| 6 | Medium | Changed | The performance test runs three matched real time replays per path (tap off, tap on with no preview request, tap on with a preview every second), each timing the whole call and the tap's hook, and asserts tap health for every tap that was measured. The hook limits are asserted; whole call times are printed in the full run and reported in the pull request. With `add_frame` the whole call includes LeRobot's own synchronous image writes, whose p99 moved between 14 and 57 ms from run to run with the tap off, so it is reported rather than asserted. |
| 8 | Medium | Changed | The run block publishes the stop (once per run) before it waits for the writer and writes the quarantine, whatever way the block is left, and the tap thread also publishes a stop it sees on the run's queue, so a write failure after the last `put` shows while the loop is still in the block. Test: `test_a_write_failure_after_the_last_put_is_shown_before_the_block_ends`, for a normal and an exceptional exit. |

## Second confirmation (`round1_confirm2_codex.md`)

Items 1, 4, 5 and 6 confirmed.

| Item | Severity | Decision | What changed |
| --- | --- | --- | --- |
| 8 (race) | Medium | Changed | Stop publication is serialized by one lock (`_stop_lock`, memory only, no file work inside): the detection claim, the status assignment and the event append happen together, and publishing where the frames went also marks the run as seen, so a late detection is never published after or over the final state. Test: `test_a_late_detection_never_lands_after_or_over_the_final_stop` (the detection paused after its claim while the final state is published). |

## Third confirmation (`round1_confirm3_codex.md`)

| Item | Severity | Decision | What changed |
| --- | --- | --- | --- |
| 8 (lock) | Medium | Changed | The quarantine path is made relative to the root before the lock is taken; `_stop_update` under `_stop_lock` only builds the status dict and appends the event. |

## Fourth confirmation (`round1_confirm4_codex.md`)

| Item | Severity | Decision | What changed |
| --- | --- | --- | --- |
| 8 (errors) | Medium | Changed | `controlled_stop` contains every error again: the path work catches any exception (a link loop raises `RuntimeError`) and shows no path, and the whole hook goes through `_fail` on any other error, so nothing reaches the run block. Test: `test_a_quarantine_path_that_cannot_be_resolved_never_reaches_the_run_block`. |
