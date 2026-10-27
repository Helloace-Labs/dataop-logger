# Round 2 decisions (Claude, on the Codex review)

Verified against local LeRobot source before deciding: `LeRobotDataset.add_frame()` returns `None` (lerobot_dataset.py:500), and `has_pending_frames()` exists (line 547).

| Finding | Decision | Change in SPEC v0.3 |
| --- | --- | --- |
| H1 no assigned frame index | Accepted | Exclusive add path, pending-frames check at start, own counter, version-tested adapter, post-save validation authoritative. |
| H2 RL fields not in API | Accepted | Reward and done kwargs on `add_frame`, `complete_transition()` for delayed rewards, explicit transition convention and validation rules, zero-frame runs discarded. |
| M1 residual targets | Accepted | Three standard targets with masks; clean mask excludes `safety_modified`; limits stated. |
| M2 destructive repair | Accepted | Quarantine by default, `--delete-unmatched` to delete, crash points listed. |
| M3 buffered metadata | Accepted | Verification split into before save, after save, and at finalize or validate. |
| M4 fingerprint | Accepted | Canonical SHA-256 serialization; unique match required for remap. |
| M5 obs_ns | Accepted | Bundle-availability definition, `obs_timestamp_kind`, optional `obs_skew_ns`. |
| M6 prediction rules | Accepted | Uniqueness, range and shape rules; ensembling provenance declared diagnostic. |
| M7 reader policy | Accepted | `missing="error" | "exclude" | "mask"`, original indices preserved. |
| L1 state machine | Accepted | Explicit states and exception behavior. |
| L2 schema id | Accepted | Derived SHA-256 id, `custom:` spaces, values always denormalized. |
| L3 float equality | Accepted | One canonical coercion, byte equality. |
| L4 non-goal wording | Accepted | Reworded to curation or materialization. |

Both trims (no `command_action`/`applied_action`, no per-camera timestamps) were approved by the reviewer with the clarifications above. Next round reviews code, not the spec.
