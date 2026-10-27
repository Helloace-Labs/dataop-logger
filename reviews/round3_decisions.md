# Round 3 decisions (Claude, on the Codex code review)

All 15 findings accepted. Tests went from 12 to 22, all passing against lerobot 0.6.2.

| Finding | Fix | Test |
| --- | --- | --- |
| H1 reader used absolute indices on filtered views | Reader walks the view's selected episodes and maps rows through `absolute_to_relative_idx` | `test_reader_filtered_views_and_missing_policy` (episodes=[1], [0, 3]) |
| H2 buffer not frozen after `end()` | `_check_buffer()` compares episode index, size, every buffered action byte-for-byte and tasks, at `end()` and again right before save; mismatch quarantines without saving | `test_mutation_after_end_blocks_save` (direct add, in-place action edit) |
| H3 save failure hid the run from repair | Staged run stays in `runs/` with `save_error`; logger marked broken; repair commits the exact match | `test_save_failure_keeps_run_for_repair` |
| H4 outcome and flags could disagree | `end()` sets the final flag from the reason and rejects contradictions; validate checks the same mapping; `rl_ready` in manifest | `test_end_sets_terminal_flag_from_reason`, `test_contradicting_flag_rejected` |
| H5 task not tied to frames | Frame task must equal run task; validate, repair and remap use the task read from LeRobot `task_index` | `test_task_mismatch_rejected` |
| H6 shallow validation | Validate rebuilds all invariants: columns, shapes, finiteness, masks zero-filled, control-mode rules, blend weights, inference ids, clocks, terminal flags, takeovers recomputed, events, predictions | `test_validate_reports_corruption_without_crashing` |
| M1 1-DoF | `ActionSchema.canon_rows()` used everywhere | `test_one_dof_round_trip_and_repair` |
| M2 fabricated timestamps | `obs_ns` and `sent_ns` are now required; `obs_ns <= sent_ns`; no backwards time | all tests pass explicit times |
| M3 partial add_frame | Any add_frame exception discards the run and resets the buffer unconditionally | covered by discard path |
| M4 prediction rules | `inference_id` given exactly when `base_action` is; non-negative ids and offsets; offset < valid_len | `test_prediction_rules` |
| M5 reader fields | Reader returns every tick field, rejects duplicate bindings, per-run dims, fallback dim from LeRobot features, validation summary | reader test |
| M6 schema metadata | Unique names, positive rate, both-or-no bounds, low <= high, finite actions; bounds serialized as `{low, high}` per SPEC | `test_schema_validation` |
| M7 crashes on corrupt files | `iter_run_dirs` classifies missing or unreadable manifests; per-run errors caught; repair leaves unreadable runs untouched | `test_validate_reports_corruption_without_crashing`, repair test |
| L1 has_predictions | `predictions_count`, empty table always written when enabled | `test_prediction_rules` |
| L2 pre-frame events | `frame_index = -1` before the first frame | validate checks the range |

Schema version stays at 1: nothing has been recorded with the earlier serialization.
