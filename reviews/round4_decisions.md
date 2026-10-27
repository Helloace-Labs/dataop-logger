# Round 4 decisions (Claude, on the Codex code review)

Codex confirmed 10 of the 15 round 3 fixes complete and 5 partly fixed, and found 2 high, 6 medium and 3 low issues. All accepted. Tests: 22 -> 28, all passing.

| Finding | Fix | Test |
| --- | --- | --- |
| H1 repair could commit an episode LeRobot never registered | Only episodes present and consistent in `meta/episodes` (length, from/to span, task) are candidates; data rows outside it are reported as an incomplete save | `test_incomplete_save_is_not_repaired` (metadata save fails after data is written) |
| H2 finite tampering of expert/base labels undetected | Manifest `integrity` holds SHA-256 of `ticks.parquet`, `predictions.parquet` and the events; checked before semantic checks and before repair commits. Note: this detects corruption and accidental edits, not a deliberate attacker who also rewrites the manifest | `test_validate_reports_corruption_without_crashing` (isolated finite expert edit), `test_events_tamper_detected` |
| M1 reused caller action buffer | `coerce()` always makes an owned copy | `test_reused_action_buffer_is_safe` |
| M2 blend without components | `blend` requires base and expert actions; any `expert_action` requires `expert_source`; same rules on disk | `test_blend_and_expert_source_rules` |
| M3 prediction validation | Disk checks for negative ids and times, `obs_frame_index` range, NaN chunks | covered by validate |
| M4 malformed buffer bypassed quarantine | `buffer_snapshot` canonicalizes or raises `EpisodeBufferError`; task count checked; any pre-save check exception quarantines without saving | `test_malformed_buffer_blocks_save` |
| M5 post-save count mismatch did not stop the logger | Marks the logger broken and keeps the staged run for repair | `test_unexpected_episode_count_stops_logger` |
| M6 remap kept old dataset identity | Destination manifest names the new root; old identity kept under `remapped_from` | |
| L1 schema strings | Units, frame and normalization note must be non-empty strings | |
| L2 timestamp types | Integer column types and non-negative values checked on disk; event timestamps checked | |
| L3 bound_episodes scope | `bound_episodes` is the view; `all_bound_episodes` is root-wide | reader test |
