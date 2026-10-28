# dataop-logger: design spec (v0.3, after review rounds 1 and 2)

## Purpose

A small open-source Python package that records the provenance post-training needs alongside a normal LeRobot recording, without changing the LeRobot dataset. Most stacks record what the robot did. They drop what the base policy proposed, what the expert would have done, who was in control, how long each step took, and how each run ended. Those are what DAgger and residual-network datasets need, and they are hard to recover later.

v0 claims: **DAgger and supervised residual datasets**, plus **RL-ready transitions** when the user supplies per-tick reward. It does not claim residual RL datasets from episode outcomes alone.

## Users and first test setups

- Teams running a LeRobot policy (ACT, SmolVLA, pi0.5) on a real arm, with a human or a scripted skill sometimes taking over or blending in.
- First setups: our SO-101 tube pick (SmolVLA, rescue skill, base policy kept running in shadow during rescues) and a second low-cost arm doing cup stacking (ACT or SmolVLA, camera-guided skill).

## Core model

- **Run:** one LeRobot episode attempt. Gets a `run_id` (UUID4) at `start_run`, never reused.
- **Tick:** one LeRobot frame. `dataset.add_frame` returns nothing, so the logger is the exclusive add path during a run: it requires `dataset.has_pending_frames() == False` at run start, calls `dataset.add_frame` first, and assigns the sidecar `frame_index` from its own successful-add counter starting at 0. Before save it checks that counter against the episode buffer size through a small, version-tested adapter (the only place private LeRobot state is touched). Post-save validation is authoritative.
- **Prediction:** one policy inference. Chunked policies produce an action chunk that several ticks draw from.

## Per-run action schema

Declared once per run, stored in the run manifest, checked on every tick:

`action_schema = {names[n], units[n], space: "joint_position" | "joint_velocity" | "ee_delta" | "custom:<name>", frame, bounds: {low[n], high[n]} or null (descriptive, not enforced), dtype, rate_hz, normalization_note}`

Values are always logged denormalized, in robot command space. `id` is the SHA-256 of the schema's canonical JSON (sorted keys, no whitespace), derived by the logger, never supplied by the caller.

`base_action`, `expert_action` and `sent_action` must all be in this space, the robot command space, before they reach the logger. Policy-space outputs are converted by the user's postprocessor first. A shape mismatch raises immediately. A run may use a different schema from other runs; files are per run, so shapes never mix in one table.

## Per-tick table: `dataop/runs/<run_id>/ticks.parquet`

| Field | Type | Meaning |
| --- | --- | --- |
| `frame_index` | int64 | Exact index assigned by LeRobot in this episode |
| `control_mode` | enum | `base`, `expert`, `blend`, `none` |
| `expert_source` | string or null | `human`, `skill`, or a namespaced string such as `skill:rescue_v3` |
| `blend_weight` | float32 | Expert weight in `[0,1]` when `blend`, else 0 or 1 |
| `base_action` | float32[n] | Base policy's selected action for this tick, before safety filters |
| `base_valid` | bool | False when the base policy did not run this tick |
| `expert_action` | float32[n] | Expert's action, even when not executed |
| `expert_valid` | bool | False when no expert label exists |
| `sent_action` | float32[n] | Final command after arbitration and all software safety transforms, as passed to the robot API. The logger coerces it once to the schema dtype and passes that same array into the LeRobot frame, so the two are byte-identical |
| `safety_modified` | bool | True when a clip, rate limit or guard changed the command |
| `inference_id` | int64 or -1 | Prediction that produced `base_action` |
| `chunk_offset` | int32 or -1 | Step within that prediction's chunk |
| `obs_ns` | int64 | Required. Monotonic time the complete observation bundle used for this action became available. `clock.obs_timestamp_kind` says whether it is a capture time or a bundle-completion time. Per-camera alignment is out of scope in v0 |
| `sent_ns` | int64 | Required. Monotonic time the command was sent |
| `obs_skew_ns` | int64 or -1 | Optional: max minus min camera time in the bundle, when known |
| `reward` | float32 | User-supplied reward for applying `sent_action` at this tick; `reward_valid` false when absent |
| `reward_valid` | bool | |
| `terminated` | bool | True only on the last tick of a run that reached a true terminal state |
| `truncated` | bool | True only on the last tick of a run cut off (timeout, abort) |

Arrays are fixed-size with explicit validity flags, never null lists. NaN is not used as "missing".

**Transition convention.** Row `t` is `(observation_t, sent_action_t, reward_t, terminated_t, truncated_t)`, where `reward_t` results from applying `sent_action_t`. When an integration only learns the reward on the next loop, it calls `run.complete_transition(reward, terminated, truncated)`, which fills the previous row; the logger keeps one pending row for this. Validation: only the last row may have `terminated` or `truncated`; not both; they must agree with `termination_reason`; an RL-ready run has every `reward_valid` true. Zero-frame runs are discarded, never committed.

**Standard targets the reader exposes** (each with its mask):
- DAgger label: `expert_action` where `expert_valid`.
- Expert residual: `expert_action - base_action` where both valid.
- Executed residual: `sent_action - base_action` where `base_valid`; the default clean mask also requires `not safety_modified`.

v0 cannot separate the arbitration output from a safety-filter correction, and does not model actuator execution error.

## Predictions table (optional): `dataop/runs/<run_id>/predictions.parquet`

`inference_id`, `obs_frame_index`, `infer_start_ns`, `infer_end_ns`, `action_chunk[horizon, n]`, `valid_len`. Recommended for chunked policies, off by default to save space. Rules: `inference_id` unique within a run; a tick with `inference_id = -1` has `base_valid = False`; `0 <= chunk_offset < valid_len`; chunk shape matches the action schema. With temporal ensembling, v0 stores only the final selected `base_action`, and prediction provenance is diagnostic, not exactly reconstructible.

## Per-run manifest: `dataop/runs/<run_id>/manifest.json`

- `schema_version`, `run_id`, `status` (`staged` or `committed`), `episode_index`, `length`
- `fingerprint`: SHA-256 over a canonical byte serialization of schema version, action schema id, task (UTF-8), length, and `sent_action` as little-endian contiguous float32. Remapping evidence only: a remap requires a unique match and refuses ambiguous ones. `run_id` is the durable identity, and `dataset_id`/`revision` are stored separately
- `task`, `started_at`/`ended_at` (wall clock, for humans only)
- `termination_reason`: `success`, `failure`, `timeout`, `operator_abort`, `system_error`; `task_success`: bool or null; `labels`: list of namespaced strings (`failure:dropped`, `safety:collision`)
- `events`: typed list `{event_id, ns, frame_index, kind, source, reason, text}` for takeover requests, handbacks, drops
- `subgoals`: derived summary of the task's steps from `subgoal` events, `{name, start_frame, end_frame, status, reason}` with `end_frame` exclusive and status `completed`, `failed` or `unfinished`
- `failure_mode`: free-text name for how the run failed (`dropped_after_lift`, `no_grasp`), or null; groups failures across runs
- `takeovers`: derived summary of half-open ranges `[start_frame, end_frame)` per `control_mode` and `expert_source`
- `policy`: family, hub repo and revision or artifact sha256, code revision, config digest, chunk size, temporal ensembling, device and precision
- `env`: typed `robot_id`, `calibration_digest`, `operator` (pseudonym), `software` (lerobot and dataop-logger versions), plus free-form `extras` (lighting, etc.)
- `action_schema` as above, `clock`: monotonic source description

`dataop/index.jsonl` is a derived cache rebuilt from manifests.

## Review layer: `dataop/runs/<run_id>/review.json`

Mutable and separate from the manifest and its digests: `{version, rev, seq, reviewed, reviewed_at, edits[]}`. Each edit is `{id, kind, payload, source (agent|human), reason, status (proposed|accepted|rejected), accepted_seq}`. Kinds: `outcome`, `failure_start`, `exclude`, `step_boundary` (by step index), `reject_run`, `note`, and `label` `{start, end, text, driver (policy|rescue|null), segment}`: a person's plain-language description of a span. An agent records its reading as `interpretation` `{train: keep|exclude, meaning, by, at}`; rewriting a label creates a new edit without the old interpretation. Human edits are accepted at once. Agent edits wait as proposals, and a new proposal reopens a reviewed run. A run cannot be marked reviewed with open proposals. For single-valued kinds the latest accepted edit is in effect, and undoing it restores the previous one. All writes hold an `fcntl` lock and bump `rev`; the viewer API rejects stale revisions with 409. `train_mask` is False for rejected runs, excluded spans, labels interpreted as `exclude`, and on failed runs `[failure_start, next expert/blend frame)`.

## API

The logger owns ordering so it cannot drift:

```python
from dataop_logger import RunLogger

log = RunLogger(dataset, action_schema=schema)          # one writer per dataset root, lock enforced
with log.run(task="pick up the tube", policy=policy_info, env=env) as run:
    for each tick:
        run.add_frame(frame,                              # calls dataset.add_frame(frame) first
                      sent_action=a_sent, base_action=a_base, expert_action=a_exp,
                      control_mode="base", expert_source=None,
                      obs_ns=t_obs, sent_ns=t_sent, inference_id=k, chunk_offset=j)
                      # optional RL fields:
                      reward=r, reward_valid=True, terminated=False, truncated=False)
    run.end(termination_reason="success")                 # stages the run
# leaving the block after run.end(): dataset.save_episode(), then commit
```

**State machine:** `active -> staged -> committed`, with terminal `discarded`, `save_failed` and `quarantined`. `run.end()` sets the final tick's flag from the reason (`success`/`failure` -> `terminated`; `timeout`/`operator_abort`/`system_error` -> `truncated`) and rejects a contradicting flag. Any exception from `dataset.add_frame` discards the run and resets the LeRobot buffer. If `save_episode` raises, the staged run stays in `runs/` with a `save_error` note for `validate --repair`, and the logger refuses further runs. An exception before `run.end()` clears the LeRobot episode buffer and discards the sidecar state without saving. Leaving the block without `run.end()` is an error, never treated as success. A second `end` or an `add_frame` after `end` raises.

**What is verified when:** before save, the frame counter, buffer size, action shapes, sidecar indices, and byte equality of each `sent_action` against the logger's own copy of the frame actions; after save, that `save_episode` completed and the writer's episode index and length; at `finalize()` or `validate`, disk-level metadata, parquet rows and action equality. Supported LeRobot versions are listed in the README and tested in CI.

- `run.discard()` clears both the LeRobot episode buffer and the staged run (re-record).
- Crash recovery with `validate --repair` is non-destructive: an exact unique fingerprint match is committed; no match or several matches move the run to `dataop/quarantine/<run_id>/` and are reported; deletion only with `--delete-unmatched`.
- Crash points: before stage, nothing written; after stage and before or during save, a staged run with no matching episode goes to quarantine; after save and before commit, the staged run matches and is committed; after commit and before the index rebuild, the index is rebuilt from manifests.
- `run.event(kind, source, reason, text)` adds a typed event. `run.subgoal(name, status, reason)` adds a `subgoal` event (`started`, `completed` or `failed`); a step with no `started` begins where the previous one ended. `run.end(..., failure_mode=...)` names the failure. `run.predictions(inference_id, chunk, ...)` logs a chunk.
- Local filesystem only in v0; temp files in the destination directory, fsync, then rename. Remote URIs raise.

## CLI

- `dataop-logger validate <root>`: episode binding, frame counts and indices, monotonic times, `sent_action == LeRobot action`, schema checks, takeover ranges, staged or orphaned runs. Nonzero exit on any error; `--json` report.
- `dataop-logger inspect <root>`: runs, outcomes, minutes per control mode, takeovers.
- `dataop-logger remap <old_root> <new_root>`: rebinds runs to a filtered or renumbered dataset by fingerprint.

## Python reader

`DataopDataset(root, missing="error" | "exclude" | "mask")`: a PyTorch dataset that joins LeRobot frames with sidecar fields by `(episode_index, frame_index)`, returns tensors plus validity masks and the standard targets above, and keeps LeRobot's delta-timestamp sampling. Default `missing="error"`. Original LeRobot episode and frame indices are preserved even when episodes are excluded. It exposes the bound episode ids and a validation summary. It does not rebalance, relabel, split or publish a dataset. An `export --into-lerobot` command that copies selected fields into a derived dataset is planned for v0.3.

## Non-goals for v0

Uploading, cloud storage, multiple concurrent writers, real-time takeover control, and curation or materialization of derived training datasets. Those belong to the hosted product.

## Review decisions

See `reviews/round1_decisions.md` and `reviews/round2_decisions.md`.
