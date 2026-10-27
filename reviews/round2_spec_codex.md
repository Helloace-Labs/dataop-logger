# Round 2 review: dataop-logger spec v0.2

## Verdict

v0.2 is a large improvement and resolves the intent of all five round 1 high-severity findings. The data model is now credible for DAgger and one-step supervised residual learning, and it can support RL transitions when the user supplies correctly aligned rewards.

I would approve both proposed trims for v0:

- **No separate `command_action` / `applied_action`: acceptable**, provided v0 explicitly defines `sent_action` as the final command after arbitration and all software safety transforms, excludes `safety_modified` rows by default from expert-residual targets (or defines their target separately), and does not claim to model actuator execution error.
- **No per-camera timestamps: acceptable**, provided `obs_ns` has a precise aggregation rule and is presented as an approximate frame-level timing signal. It is adequate for the two initial synchronous/near-synchronous setups, but not for sensor-latency research or strongly asynchronous multi-camera systems.

I found two new high-severity specification gaps in the revised API. They are fixable without expanding scope: the spec assumes a public LeRobot frame index return that does not exist, and it declares RL fields without showing how they enter or align through the API.

## Round 1 high-finding audit

### H1 — Residual RL not reconstructible: **fixed in scope and schema, API clarification still required**

The spec now has `reward`, `reward_valid`, `terminated`, and `truncated`, and correctly narrows the product claim. This resolves the original overclaim. The remaining problem is operational: the example `add_frame` call has no reward/done parameters and `run.end()` occurs after the final row has already been appended. See new H2 below.

### H2 — Sidecar committed before LeRobot: **fixed conceptually**

The context manager now stages the sidecar, calls `dataset.save_episode()`, verifies, and commits. Per-run manifests replace an authoritative append-only JSONL index, and repair is specified. This is the correct architecture.

The crash-state rules still need slightly safer wording: repair should quarantine ambiguous or corrupt staged runs rather than deleting them automatically. See M2.

### H3 — Frame alignment fragile: **mostly fixed, implementation contract incomplete**

Owning `add_frame` ordering and appending sidecar data only after LeRobot accepts the frame resolves the drift hazard. However, current LeRobot `LeRobotDataset.add_frame()` returns `None`; it does not return the assigned frame index. The spec's phrase “using the frame index the dataset actually assigned” therefore depends on private writer state or an inference that is not stated. See new H1.

### H4 — Action provenance insufficient: **fixed for the stated v0 use cases**

Separate base, expert, and sent actions; validity flags; source; blend state; weight; and shadow-policy operation are sufficient to construct exact DAgger labels and supervised residual targets for the initial systems. The trimmed omission of command/applied actions is acceptable under a tighter `sent_action` contract. See M1.

### H5 — Action schema underspecified: **fixed, with small hardening needed**

The per-run schema now captures names, units, semantic space, frame, normalization, bounds, dtype, and rate; all logged actions share that space; and mismatches fail immediately. This handles different robot/action sizes because each run has its own parquet table. The schema ID should be a canonical digest rather than a caller-chosen label. See L2.

## High-severity findings

### H1. The API cannot obtain an assigned LeRobot frame index as written

In current LeRobot, `dataset.add_frame(frame)` computes and appends `frame_index` internally but returns `None`. A wrapper can know that the call succeeded, but it cannot obtain the assigned index through the public return value. Reading `dataset.writer.episode_buffer` is private-version coupling; using a local count is safe only if this logger is the sole path for adding frames and the episode buffer was empty when the run began.

This matters because the spec promises exact alignment and adoption by ordinary LeRobot users. An implementation that reaches into writer internals may break across LeRobot versions; an implementation that merely counts without enforcing exclusivity can silently drift if user code calls `dataset.add_frame()` directly.

**Concrete fix:** define and enforce one supported v0 strategy:

1. At `run` entry, require `dataset.has_pending_frames() == False`.
2. Make `run.add_frame()` the exclusive frame-add path while the run is active (document direct calls as unsupported and, if practical, detect writer buffer-size changes).
3. Assign the sidecar index from the logger's successful-add counter, starting at zero.
4. Before save, compare that counter and the LeRobot episode buffer size through a small version-adapter layer; after save, validate against saved episode length and indices.

Isolate any private LeRobot access in a compatibility adapter with tested supported version ranges. Do not imply `add_frame()` returns an index. If avoiding private APIs is a hard requirement, call the locally counted value the expected frame index and make post-save validation authoritative.

### H2. Reward and terminal fields are in the table but absent from the write API and transition contract

The table declares `reward`, `reward_valid`, `terminated`, and `truncated`, but the example `run.add_frame(...)` accepts none of them. `run.end(termination_reason=...)` happens after the last tick is logged, so the spec does not say whether it mutates the last buffered row, whether callers pass termination flags on `add_frame`, or which action a delayed reward belongs to.

Robotics loops commonly observe reward/result for action `t` at the next control iteration. Without an explicit convention, one integration will log reward alongside `observation_t/action_t`, while another will log it alongside `observation_{t+1}`. Both files validate structurally but create different RL transitions. A zero-length run also has no row on which to place termination.

**Concrete fix:** add reward and done fields to the API and define one transition convention. Recommended:

```python
run.add_frame(..., reward=r_t, reward_valid=True,
              terminated=terminated_t, truncated=truncated_t)
```

Define row `t` as `(observation_t, action_t, reward_t, terminated_t, truncated_t)`, where `reward_t` is the result of applying `action_t`; if the environment reports it on the next loop, the integration must retain and attach it to the prior pending tick. A simpler implementation is a one-row delayed buffer or `run.complete_transition(reward, terminated, truncated)` API.

Alternatively, allow `run.end()` to patch only the final `terminated`/`truncated` flags, but still specify how rewards align. Validate that only the last row may be terminal/truncated, the two flags are not both true unless explicitly allowed, `termination_reason` agrees with them, and an RL-ready run has every required reward valid. Define behavior for zero-frame runs (discard, never commit as an episode).

## Medium-severity findings

### M1. Residual target semantics remain ambiguous around blending and safety modification

The three actions now make the raw data expressive, but “residual dataset” can mean at least:

- expert correction: `expert_action - base_action`;
- behavior correction: `sent_action - base_action`;
- arbitration residual before safety filtering.

For `control_mode="blend"`, `sent_action` may equal the weighted blend, while the desired learned residual might target the full expert action. If `safety_modified=True`, `sent_action - base_action` includes safety-filter intervention and is not necessarily an expert residual. Since `command_action` is intentionally omitted, the pre-safety arbitration output cannot be recovered.

**Concrete fix:** explicitly define standard reader targets and masks:

- DAgger label: `expert_action` where `expert_valid`.
- Expert residual: `expert_action - base_action` where both valid.
- Executed residual: `sent_action - base_action` where both valid.
- Default clean executed-residual mask: both valid and `not safety_modified`.

Document that v0 cannot separate arbitration residual from safety-filter correction and cannot learn applied actuator error. This makes the no-`command_action`/`applied_action` trim acceptable rather than silently lossy.

### M2. `validate --repair` must not automatically delete unmatched staged data

The spec says repair commits a staged run if a matching episode exists, “otherwise deletes it.” A missing match may mean the episode was never saved, but it may also mean a partial LeRobot save, a temporarily incomplete metadata parquet, an unsupported LeRobot version, or a bug in fingerprint calculation. Automatic deletion destroys the best forensic record of a failed real-robot collection attempt.

**Concrete fix:** make default repair non-destructive:

- exact unique match: commit;
- no match: move to `dataop/quarantine/<run_id>/` or retain as staged and report it;
- multiple/conflicting matches: quarantine and require explicit resolution;
- deletion only with `--delete-unmatched` (or explicit `discard`) and a clear report.

Also specify the crash points and recovery result: before stage, after stage/before save, during save, after save/before commit, and after commit/before index rebuild.

### M3. Post-save verification may depend on LeRobot metadata that is still buffered

Current LeRobot v3 buffers some metadata and requires `finalize()` to flush/close parquet writers. Immediately after `save_episode()`, not every verification path based on reloading the dataset from disk is guaranteed to see finalized metadata. Conversely, `save_episode()` mutates/resets the in-memory episode buffer, so verification cannot assume the pre-save buffer remains available.

**Concrete fix:** define what is verified before versus after save:

- before save: frame count, action shape/schema, sidecar indices, and equality against a canonical copy of the frame actions retained by the logger;
- after save: LeRobot writer-reported episode index/length and successful save completion;
- at `finalize()` or `validate`: disk-level metadata, parquet rows, and action equality.

State the supported LeRobot format/version range and test this lifecycle against those versions. Do not require reopening an unfinalized dataset inside every run context.

### M4. The episode fingerprint is useful but not sufficiently collision-resistant as specified

Length plus task, first `sent_ns`, and `sent_action` hash is strong within one capture directory, but monotonic timestamps are process/boot-local and are not portable identity. Hashing only actions can collide for repeated scripted/static runs, especially after copying or concatenating datasets. It is also unclear whether the hash is over canonical little-endian float32 bytes, the user input, or values read back from LeRobot.

**Concrete fix:** define a canonical fingerprint serialization and include dataset provenance. For example: schema version, action schema digest, task UTF-8 bytes, length, canonical little-endian contiguous float action bytes, and optional stable observation sample hashes. Store the original dataset ID/revision separately. Treat fingerprints as remapping evidence, not universal identity; require a unique match and refuse ambiguous remaps. `run_id` remains the durable sidecar identity.

### M5. Frame-level `obs_ns` is acceptable, but “latest camera” needs a precise and honest contract

Using the latest camera timestamp is conservative for “all observations available by” timing, but it is not the timestamp of a single coherent observation if cameras are asynchronous. It can also hide large skew and may be unavailable if LeRobot integrations do not expose capture times.

**Concrete fix:** define `obs_ns` as the monotonic time at which the complete observation bundle used for inference became available. If actual camera capture timestamps are unavailable, allow it to be the bundle acquisition/completion timestamp and record `clock.obs_timestamp_kind` accordingly. Optionally add `obs_skew_ns` (max minus min sensor time) as one scalar when integrations can provide it; this preserves v0 simplicity without requiring per-camera columns. State that per-camera alignment is out of scope and that strongly asynchronous camera users need an extension.

### M6. Prediction-table referential and shape constraints need validation rules

Predictions are optional, yet each tick has an `inference_id`. The spec does not say whether IDs must be unique per run, whether `chunk_offset < valid_len`, how temporal ensembling references multiple overlapping predictions, or how a tick can refer to more than one contributing chunk. A single `inference_id` plus a free-form selector metadata field may be insufficient for temporal ensembling.

**Concrete fix:** specify that inference IDs are unique within a run; `-1` requires `base_valid=False` unless a non-chunked/unlogged prediction convention is explicitly allowed; offsets must be in range; and logged prediction shapes must match the action schema. For temporal ensembling, either use a separate contribution table `(frame_index, inference_id, chunk_offset, weight)` or clearly state that v0 stores only the selected/final base action and prediction provenance is best-effort diagnostic metadata, not exactly reconstructible.

### M7. The reader promise needs a missing-sidecar policy

Real datasets will contain a mixture of logged and legacy episodes, deleted runs, or quarantined runs. `DataopDataset(root)` does not define whether it rejects the entire dataset, filters unbound episodes, or returns invalid masks. Silent filtering would change LeRobot episode indices and sampling behavior.

**Concrete fix:** add an explicit policy such as `missing="error" | "exclude" | "mask"`, defaulting to `error`. Expose bound episode IDs and a validation summary. Preserve original LeRobot episode/frame indices even when excluding episodes.

## Low-severity findings

### L1. Context-manager end and exception behavior should be explicit

The example requires `run.end(...)` inside a `with` block and then performs saving on block exit. It is unclear what happens if the loop raises, `run.end()` is omitted, `run.end()` is called twice, or an exception occurs after staging.

**Concrete fix:** define a small state machine (`active -> staged -> committed`, with `discarded`/`quarantined` terminal states). On an exception before a successful explicit `end`, clear the LeRobot episode buffer and quarantine or discard temporary sidecar state without saving. Reject double end/add-after-end with clear errors. Do not infer success merely from clean block exit.

### L2. `action_schema.id` should be derived and normalization must be canonical

A caller-supplied ID can accidentally reuse a name for a changed schema. Free-form normalization objects and ellipsis in the `space` enum undermine equality checks.

**Concrete fix:** define a canonical JSON representation and derive `id` from its SHA-256 digest. Namespaced custom spaces should be allowed (for example `custom:robot_xyz`) while known spaces receive validation. Specify bounds shape and whether action values are logged normalized or denormalized; the current text implies denormalized robot-command space, so `normalization` should describe provenance/inversion rather than the current value domain.

### L3. Exact floating-point equality should be defined as canonical equality or tolerance

“Must equal LeRobot `action`” and validator equality are ambiguous across NumPy/Torch conversion and float dtype coercion. Exact comparison may fail for representational reasons; loose tolerance may conceal wrong actions.

**Concrete fix:** coerce `sent_action` once to the declared dtype/shape and pass that same canonical array into both the LeRobot frame and sidecar. Then byte equality is appropriate. If the logger only receives two independently converted values, use configurable `atol`/`rtol` and report the maximum error. Prefer the first design.

### L4. “Building training datasets” conflicts slightly with the product claim

The purpose promises DAgger and residual datasets, while non-goals say “dataset building,” and the reader joins data for training. The likely intended distinction is that v0 records and reads provenance but does not generate curated/materialized training repositories.

**Concrete fix:** change the non-goal wording to “curation/materialization of derived training datasets” and state that `DataopDataset` exposes labels/masks but does not rebalance, relabel, split, or publish a new dataset.

## Recommendation

Proceed with v0 after resolving H1 and H2 in the spec. Neither requires adding new logged modalities; both require making the existing lifecycle implementable and unambiguous.

The two scope trims are reasonable:

- Keep only `base_action`, `expert_action`, and final post-safety `sent_action` in v0; document the residual masks and the inability to recover pre-safety arbitration or physical applied action.
- Keep one frame-level `obs_ns`; define it as observation-bundle availability time, record its timestamp kind, and state the limitation for asynchronous multi-camera systems.

With those clarifications, v0.2 is small enough to ship while still being technically honest and useful to LeRobot teams.
