# Round 5 decisions (Claude, on the final Codex code review)

Codex verdict: **no high-severity issues remain.** It confirmed 7 of the 11 round 4 fixes complete and 4 partly fixed, and found 4 medium and 3 low issues. All fixed. Tests: 28 -> 32, all passing against lerobot 0.6.2, including a camera (video) dataset.

| Finding | Fix | Test |
| --- | --- | --- |
| M1 missing video accepted | An episode counts only if every video file its metadata references exists and is non-empty, and its global `index` rows match the registered span; otherwise it is an incomplete save | `test_missing_video_is_incomplete` |
| M2 reader reordered or duplicated rows | Reader index follows the wrapped LeRobot view's own row order | `test_reader_follows_lerobot_order` (episodes=[2, 0, 2]) |
| M3 future obs_frame_index committed | `end()` requires every prediction's `obs_frame_index` to be in the final run | `test_prediction_future_frame_must_exist` |
| M4 malformed integrity crashed repair | Integrity structure checked; repair's per-run error boundary covers it | `test_repair_survives_malformed_integrity` |
| L1 saved length not checked | After save, the registered episode length and span must equal the run; mismatch stops the logger and keeps the run for repair | |
| L2 remap identity | `remap --repo-id --revision` records the destination identity | |
| L3 prediction column types | Integer Arrow types required | |

Deliberately not done in v0: decoding video frames during validation (file existence and size only), and reopening the dataset with LeRobot inside every save. Both are slow on real camera datasets; `validate` on a finalized dataset is the authoritative check.
