# Senior robotics ML review: dataop-logger v0

## Bottom line

The core idea is useful, and a sidecar is the right default for a v0. However, the proposed fields are not yet sufficient to reliably build all three claimed dataset types.

- **DAgger:** sufficient only in the narrow case where every human/skill-driven `sent_action` is an expert label for the observation at that exact frame. It does not capture expert queries that are observed but not executed, corrections under blending, or action/observation latency.
- **Residual supervised learning:** nearly sufficient for one-step policies if `residual = sent_action - base_action` is meaningful in a shared action representation. The schema must preserve pre-safety base action, post-safety command, action semantics, clipping, timing, and validity.
- **Residual RL:** insufficient. Episode outcome alone is not transition-level reward. A usable RL export needs at least reward/cost, terminated versus truncated, and unambiguous transition alignment; serious offline RL use may also need behavior-policy metadata such as log probability or exploration/noise parameters.

I would not advertise “residual RL datasets” until those transition fields exist. “Captures intervention rollouts that can later be labeled for RL” would be accurate for v0.

## High-severity findings

### H1. Residual RL cannot be reconstructed from the proposed data

`outcome` supplies one terminal label per episode, but residual RL normally consumes transitions `(observation_t, action_t, reward_t, next_observation_t, terminated_t, truncated_t)`. A terminal success/failure can be converted into a sparse reward only by an undocumented convention, and `aborted`/`timeout` must not be conflated with an MDP terminal state. Safety costs and intervention costs are also common objectives in residual robot RL.

**Concrete fix:** add per-tick nullable fields:

- `reward: float32 | null`
- `cost: float32 | null` (or a named dictionary/feature namespace)
- `terminated: bool`
- `truncated: bool`
- optionally `discount: float32 | null`

Define transition semantics explicitly: row `t` contains `observation_t` in LeRobot, the command applied for that transition, and the reward/done information resulting from that command. Provide an exporter that joins the sidecar to LeRobot and constructs `next_observation` without crossing episode boundaries. For offline RL, add optional `behavior_log_prob`, exploration/noise metadata, and policy distribution parameters when the policy exposes them.

### H2. The logger commits before LeRobot does, creating orphaned or falsely committed runs

The example calls `log.end_run()` before `dataset.save_episode()`. Thus the sidecar can be atomically written while video encoding, parquet writing, or metadata update later fails. Atomic rename protects a single file from partial writes; it does not make the sidecar and LeRobot episode one transaction. Conversely, if the process crashes after LeRobot saves but before the sidecar commits, the episode has no sidecar. `runs.jsonl` also cannot be transactionally coordinated with the tick parquet by merely appending a line.

**Concrete fix:** use a two-phase lifecycle:

1. `prepare_run()` writes a staged tick file and staged run manifest with a generated `run_id` and expected episode index/length.
2. Call `dataset.save_episode()`.
3. Verify the saved episode metadata and then `commit_run()` by atomically publishing a per-run manifest.

Prefer one atomic manifest per run (for example `dataop/runs/<run_id>.json`) over an append-only `runs.jsonl`; generate `runs.jsonl` as an index/cache. On startup and in `inspect --repair`, reconcile staged files, committed manifests, and actual LeRobot episode metadata. A higher-level context manager or wrapper should own this ordering so users cannot accidentally invert it.

### H3. Episode index and frame count alignment are too fragile

LeRobot assigns `frame_index` and `timestamp` inside `add_frame`; callers should not provide them. In the proposed loop, `log.tick()` happens first. If `dataset.add_frame()` rejects a frame, image writing fails, a frame is intentionally skipped, or calls occur on different code paths, the sidecar is already one row ahead. Counting frames at `end_run` detects the problem late but cannot identify or repair it. In current LeRobot, timestamp is normally synthesized as `frame_index / fps`, so “same clock as the dataset” is not necessarily the real capture/control clock.

**Concrete fix:** make the primary API a coordinated call such as `log.add_frame(frame, sent_action=..., ...)` that calls LeRobot first and appends the sidecar row only after success, using the writer buffer's actual episode/frame identity. If monkey-patching/wrapping is undesirable, reverse the documented order and have `tick()` accept or verify the just-added frame index. Store both:

- `frame_index`: exact zero-based index within the saved LeRobot episode
- `control_timestamp_ns`: monotonic timestamp at command generation/send

Also record `observation_timestamp_ns` where available and define all alignment semantics. Do not use wall-clock time for control alignment.

### H4. The action provenance is not rich enough for DAgger or residual targets

`driver` plus `base_action` and `sent_action` assumes mutually exclusive control and equates the executed command with the expert label. Real intervention stacks commonly blend policy and human/skill actions, pass commands through clipping/rate limits/collision filters, or query an expert without executing its answer. In those cases `sent_action - base_action` is not the intended residual, and a DAgger label cannot be identified. `base_action = null when no policy ran` also prevents residual examples during some takeover implementations unless the base policy is deliberately evaluated in shadow mode.

**Concrete fix:** separate proposal, supervision, arbitration, and actuator command:

- `base_action_raw`: base policy proposal before safety/post-processing
- `expert_action`: nullable human/skill label even when it was not executed
- `expert_source`: `human`, `skill`, or a namespaced string
- `command_action`: command after arbitration but before the robot transport
- `sent_action`: command actually passed to the robot API
- `applied_action`: optional measured/acknowledged action when available
- `control_mode`: `base`, `expert`, `blend`, `residual`, `none`
- `blend_weight` or arbitration metadata
- `label_valid` / `base_valid`, plus takeover reason

Keep evaluating the base policy in shadow mode during takeover when feasible; otherwise those frames are not supervised residual pairs. Explicitly define which field must equal LeRobot's `action`. For broad compatibility, that should usually be `sent_action`/behavior action, not an unexecuted expert label.

### H5. Action schema and semantics are underspecified

`float32[n]` is not enough to safely subtract actions. LeRobot robots/policies may use joint position, joint velocity, Cartesian delta, normalized policy space, or heterogeneous components including a gripper. Safety transforms may change scale or dimension. Different robots or policy revisions may have different action sizes. A parquet fixed-size list cannot silently vary `n` between rows; a full action chunk adds another time dimension and may have a different horizon on different policies.

**Concrete fix:** make action representation a versioned per-run schema containing shape, ordered names, units, coordinate frame, normalization, bounds, dtype, rate, and semantic type. Require `base_action_raw`, expert/command actions, and LeRobot `action` to declare their spaces. Store transforms needed to map policy space to robot-command space, or store comparable actions directly in both spaces. Reject incompatible dimensions at runtime. Permit different action schemas between runs via `action_schema_id`; do not mix incompatible shapes in one parquet column/file without an explicit union/ragged encoding.

## Medium-severity findings

### M1. Full policy chunks should be preserved, but not in place of the executed-step proposal

For ACT, SmolVLA, pi-family, and other chunking/receding-horizon policies, storing only the selected action loses replanning behavior, temporal ensembling inputs, uncertainty across overlapping predictions, and targets for residual sequence models. Storing only the chunk, however, makes the actual decision ambiguous because selection, temporal ensembling, latency, and clipping occur afterward.

**Concrete fix:** store both:

- per tick, `base_action_selected` in the command-comparable space;
- optionally, a chunk record keyed by `inference_id`, with `prediction_observation_frame`, `inference_timestamp_ns`, `action_chunk[horizon, action_dim]`, valid length/mask, and policy/checkpoint ID;
- per tick, the `inference_id`, chunk offset(s), temporal-ensemble weights or selector metadata used to produce `base_action_selected`.

Put chunks in a separate table (`dataop/predictions/...`) to avoid repeating an entire chunk on every tick. This directly answers open question 1: **store both; the selected tick action is mandatory, and the full chunk is optional but strongly recommended for chunked policies.**

### M2. Re-recording, deletion, splitting, and merging invalidate index-only identity

`episode_index` is positional, not a durable identity. LeRobot's re-record path clears the current episode buffer before save, which the proposed `discard_run()` can mirror. Post-hoc dataset tools commonly create a new dataset when deleting/filtering/splitting and may renumber episodes. Copying the sidecar unchanged then silently joins records to the wrong episode.

**Concrete fix:** generate a stable `run_id` UUID at `start_run` and persist it in sidecar metadata. After save, bind it to an episode fingerprint containing at minimum episode length, task, capture start, and a hash over stable frame data (for example timestamps plus action bytes), as well as dataset identity/revision. Provide explicit `dataop remap`, `filter`, and `validate` commands that consume an old-to-new episode map or fingerprints. For an unsaved re-record, discard the staged run and reuse neither its `run_id` nor its committed status. Never silently infer by current episode index alone.

This answers open question 3: **handle unsaved re-records transactionally; handle post-save deletion/reordering through stable run IDs, fingerprints, and explicit remapping/validation.**

### M3. Sidecar is the right canonical format, but adoption needs join/export support

Adding fixed extra LeRobot features is convenient for training but forces every frame to carry nullable/logging-specific fields and can require dataset schema migrations. A sidecar preserves loading by ordinary/older LeRobot consumers and accommodates chunk tables and run metadata better. Its costs are discoverability, upload behavior, synchronization, and the lack of direct access from normal LeRobot training transforms.

**Concrete fix:** keep the sidecar canonical for v0, but ship:

- a validated `DataOpDataset`/PyTorch wrapper that joins by episode and frame and preserves LeRobot sampling/delta-timestamp behavior;
- an `export --into-lerobot` command that materializes selected scalar/fixed-shape fields as ordinary features in a derived dataset;
- Hub upload/download documentation and a verification that `dataop/` files are actually included by the supported push workflow;
- a manifest with sidecar schema version, compatible LeRobot dataset format/version, hashes, and feature/action schemas.

This answers open question 2: **use a sidecar as canonical storage and offer a materialized LeRobot export for training convenience.**

### M4. Takeovers should not exist only as a lossy derived list

Driver changes can derive intervals, but edge semantics are unspecified: are `start_frame` and `end_frame` inclusive, and what happens at episode end? Driver changes also do not capture a requested takeover that failed, operator reaction time, takeover reason, or arbitration/blend state. Free-text `event` is not dependable for training or metrics.

**Concrete fix:** define half-open ranges `[start_frame, end_frame)` and generate them from typed per-frame control state. Add typed event records with event ID, monotonic timestamp, requested/effective frame, source, reason code, and optional text. Treat the per-frame state as canonical and `takeovers` as a derived summary/cache.

### M5. Multiple-camera and asynchronous sensor timing is not addressed

The sidecar need not duplicate camera data, and multiple camera keys themselves are not a layout problem. The problem is that a LeRobot frame can contain observations captured at different times, while the action is inferred and sent later. Frame-index equality alone hides camera skew, inference latency, and control latency; these are material for DAgger and residual learning.

**Concrete fix:** record a monotonic timestamp for each observation/camera when available, plus inference start/end and command-send timestamps. At minimum record `observation_timestamp_ns`, `policy_inference_started_ns`, `policy_inference_ended_ns`, and `action_sent_ns`, with clock-domain metadata. The action should be associated with the observation/inference that produced it, not merely whichever dataset frame was appended nearby.

### M6. Policy identity and environment provenance are too weak for reproducibility

A “short hash,” checkpoint path, and free-form `env` dictionary are easy to collide, become invalid off-machine, and cannot support filtering without conventions.

**Concrete fix:** use a structured policy record with model family, immutable Hub repo/revision or full artifact digest, code revision, policy config digest, action schema ID, inference settings (chunk size, temporal ensemble, device/precision), and deterministic/non-deterministic mode. Give common environment fields typed names (`robot_id`, calibration digest, operator pseudonym, software revision) and leave a separate `extras` dictionary for extension.

## Low-severity findings

### L1. Outcome and failure taxonomy needs extensibility without losing consistency

One optional `failure_kind` string will quickly fragment into spelling variants, and an episode can succeed at the task while violating safety or requiring intervention.

**Concrete fix:** define `task_success: bool | null`, `termination_reason` (`success`, `failure`, `timeout`, `operator_abort`, `system_error`), zero or more namespaced failure/safety labels, and optional scalar metrics. Keep `outcome` as a derived convenience value if desired.

### L2. Null and NaN semantics need to be explicit

For arrays, parquet null, an all-NaN vector, and “policy did not run” are not interchangeable. Many tensor data loaders handle null list columns poorly.

**Concrete fix:** store fixed-shape numeric values plus an explicit validity mask/boolean; reserve null for fields genuinely absent from a schema. Document NaN policy and ensure the PyTorch wrapper produces tensors plus masks.

### L3. The CLI is too narrow to establish trust

An `inspect` summary does not catch the failure modes most likely to corrupt training data.

**Concrete fix:** include `validate` from v0. It should verify episode existence, unique run binding, frame counts and exact indices, timestamp monotonicity/tolerance, action equality with the LeRobot `action`, action schema compatibility, takeover interval consistency, hashes, and staged/orphaned records. Exit nonzero on corruption and offer a machine-readable report.

### L4. Concurrent writers and remote/object-store roots are unspecified

Local temp-file rename is atomic only within the same filesystem. Multiple recording processes can race on `runs.jsonl`; object-store roots do not provide ordinary filesystem rename semantics.

**Concrete fix:** state that v0 supports one local writer per dataset root, enforce a lock, create temp files in the destination directory, fsync file and directory where durability matters, and reject unsupported URI/object-store roots with a clear error. Add a storage abstraction later rather than implying atomicity everywhere.

## Recommended v0 acceptance criteria

Before release, I would require:

1. A coordinated add-frame API that cannot silently drift from LeRobot frame indices.
2. Two-phase commit/recovery plus `validate` for sidecar/episode consistency.
3. Stable run identity and an explicit remapping story for derived datasets.
4. Versioned action-space metadata and strict shape/semantic checks.
5. Separate base proposal, expert label, arbitration result, and sent action, with validity masks.
6. Typed monotonic timing sufficient to diagnose observation/inference/action latency.
7. Per-transition reward/termination fields, or removal of the residual-RL dataset claim from v0.
8. A documented LeRobot join wrapper and a materialization/export path.

With those changes, this becomes a credible, low-friction provenance layer for intervention learning rather than merely an auxiliary action log.
