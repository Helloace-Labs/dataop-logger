# Round 1 decisions (Claude, on the Codex review)

| Finding | Decision | Change in SPEC v0.2 |
| --- | --- | --- |
| H1 residual RL not reconstructible | Accepted | Per-tick `reward`/`reward_valid`, `terminated`, `truncated`; v0 claims DAgger and supervised residual only, RL-ready only with user reward. Behavior log-prob deferred. |
| H2 commit ordering | Accepted | Context manager owns stage, `save_episode`, verify, commit; per-run manifests; `index.jsonl` derived; `validate --repair`. |
| H3 frame alignment | Accepted | `run.add_frame()` calls LeRobot first and records the assigned index; monotonic `obs_ns`, `sent_ns`. |
| H4 action provenance | Accepted, trimmed | `base_action`, `expert_action`, `sent_action`, `control_mode` with `blend`, `blend_weight`, validity flags, `safety_modified`. Separate `command_action`/`applied_action` left out of v0: `safety_modified` plus `sent_action` covers our two setups; can add later. |
| H5 action schema | Accepted | Per-run `action_schema`; all three actions in robot command space; shape checks. |
| M1 chunks | Accepted | Selected action mandatory; optional `predictions.parquet` keyed by `inference_id`. |
| M2 identity | Accepted | `run_id`, fingerprint, `remap` command. |
| M3 sidecar vs features | Accepted | Sidecar canonical; `DataopDataset` reader in v0; export in v0.3. |
| M4 takeovers | Accepted | Half-open ranges derived from per-tick state; typed events. |
| M5 timing | Accepted, trimmed | `obs_ns`, `sent_ns` per tick; inference start/end in predictions table. Per-camera timestamps deferred. |
| M6 provenance | Accepted | Structured `policy` and `env` records. |
| L1 outcomes | Accepted | `termination_reason`, `task_success`, namespaced `labels`. |
| L2 null/NaN | Accepted | Fixed arrays plus validity flags. |
| L3 validate | Accepted | `validate` in v0 with `--json` and nonzero exit. |
| L4 concurrency | Accepted | One local writer, lock, same-dir temp files, fsync; remote URIs rejected. |
