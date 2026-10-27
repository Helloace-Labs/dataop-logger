# Round 4 code review: dataop-logger

## Verdict

The latest commit is a substantial improvement. The happy path, filtered reader mapping, buffer recheck, terminal-state mapping, task binding, 1-DoF handling, and corrupt-file reporting are all materially better. The requested test suite passes:

```text
python -m pytest -q tests
22 passed in 3.28s
```

High-severity issues remain, so I cannot say the implementation is clear of high-severity problems. In particular:

1. Repair treats raw data parquet as authoritative and can commit an episode that LeRobot never registered in episode metadata.
2. The sidecar has no integrity digest over expert/base labels or the rest of the tick table, so finite supervision tampering still passes validation.

Both were reproduced with temporary LeRobot 0.6 datasets outside the repository.

## Round 3 finding audit

### H1 — Reader used absolute indices on filtered views: **fixed**

**Evidence:** `src/dataop_logger/reader.py:49-65` restricts construction to `selected_episodes()`, obtains LeRobot's absolute-to-relative map, and maps each dataset-wide frame index to its relative position in the selected view. `tests/test_logger.py:333-359` covers a single selected episode and a noncontiguous `[0, 3]` selection. The reproduced Round 3 length/index failure no longer occurs.

### H2 — Buffer was mutable after `end()`: **fixed**

**Evidence:** `src/dataop_logger/logger.py:291` checks the buffer while ending, and `src/dataop_logger/logger.py:383-388` repeats the check immediately before save. `_check_buffer()` at lines 328-342 verifies episode index, size, action count, exact action bytes, and task values. A mismatch is cleared and quarantined without calling `save_episode()`. `tests/test_logger.py:167-181` covers an extra direct frame and replacement of a buffered action after `end()`.

There is a separate aliasing problem for reusable caller action arrays; see M1 below.

### H3 — Save failure hid the staged run from repair: **partly fixed**

**Evidence:** `src/dataop_logger/logger.py:390-399` now leaves the run staged under `dataop/runs`, records `save_error`, and marks the logger broken. `repair()` sees staged runs, and `tests/test_logger.py:184-204` verifies recovery when a complete successful save is followed by an injected exception.

**Incomplete:** The test fails only after all LeRobot save work completed. In the actual LeRobot 0.6 ordering, data parquet is written before `meta.save_episode()`. `load_lerobot_episodes()` and repair treat those raw rows as a complete episode even if metadata never registered it. This can turn an incomplete save into a falsely committed run; see new H1.

### H4 — Outcome and terminal flags could disagree: **fixed**

**Evidence:** `src/dataop_logger/schema.py:14-15` defines the reason-to-flag mapping. `Run.end()` sets the implied final flag and rejects the opposite flag at `src/dataop_logger/logger.py:284-290`. Disk validation checks the same mapping at `src/dataop_logger/validate.py:131-138`, and `rl_ready` is derived from all reward-valid flags. Tests cover all five reasons and contradictory flags at `tests/test_logger.py:87-110`.

### H5 — Run task was not tied to LeRobot task/fingerprint: **fixed**

**Evidence:** `src/dataop_logger/logger.py:179-181` rejects a mismatched frame task and lines 192-194 force the canonical run task into the copied LeRobot frame. `src/dataop_logger/validate.py:25-54` loads actual `task_index` values and resolves the dataset task. Validation, repair, and remap compare the actual task at lines 177-178 and 225-228. The writer mismatch is covered at `tests/test_logger.py:148-154`.

### H6 — Disk validation omitted semantic invariants: **partly fixed**

**Evidence:** `src/dataop_logger/validate.py:64-186` now validates required tick columns, action shapes and finiteness, invalid-row zero fill, control modes, blend weights, inference fields, clocks, terminal flags, reward readiness, takeover derivation, event frame ranges, prediction count/references, fingerprint, task, and exact LeRobot action bytes. Per-run failures are isolated by `validate()` at lines 199-210.

**Incomplete:** Arbitrary finite changes to `base_action` or `expert_action` remain undetectable because the manifest authenticates only the fingerprint over `sent_action`. I reproduced changing an expert label from `[1, 1]` to `[999, -999]`; `validate()` still returned `ok=True`. Prediction and event validation also omit several writer invariants. See new H2 and M3.

### M1 — 1-DoF actions could not validate/repair/remap: **fixed**

**Evidence:** `ActionSchema.canon_rows()` at `src/dataop_logger/schema.py:94-99` canonicalizes scalar-on-disk LeRobot actions to `[rows, dim]`. Validation and fingerprint matching both use it at `src/dataop_logger/validate.py:47-54` and 179-185. `tests/test_logger.py:318-330` covers 1-DoF round trip and repair.

### M2 — Timestamps were silently fabricated: **fixed**

**Evidence:** `obs_ns` and `sent_ns` are required arguments at `src/dataop_logger/logger.py:133-141`; the writer checks nonnegative integer values, `obs_ns <= sent_ns`, and monotonicity at lines 164-170. The current tests explicitly supply timestamps.

Disk validation still permits negative but ordered timestamps; see L2.

### M3 — Partial `add_frame()` failure could leave a dirty buffer: **fixed**

**Evidence:** Any exception from LeRobot's `add_frame()` calls `discard()` at `src/dataop_logger/logger.py:195-199`, and discard now uses unconditional `adapter.reset_episode_buffer()` at lines 299-306. The adapter calls LeRobot clear even when buffer size is zero and has a version-specific reset fallback at `src/dataop_logger/adapter.py:48-57`.

The implementation is correct, though there is still no dedicated test that injects a failure after LeRobot partially mutates its first frame.

### M4 — Prediction invariants depended on optional-table mode: **partly fixed**

**Evidence:** `src/dataop_logger/logger.py:158-163` enforces the table-independent ID/base-action/offset rules. Prediction writes enforce nonnegative unique IDs, ordered inference times, observation-frame range, finite chunk values, shape, and valid length at lines 246-269. `_check_rows()` verifies references and valid offsets when the table is enabled at lines 313-326. Tests cover missing IDs, bad offsets, and an enabled empty table at `tests/test_logger.py:226-253`.

**Incomplete:** Disk validation does not repeat all writer rules: it does not check prediction IDs are nonnegative, `obs_frame_index` is in range, times are nonnegative, or chunks are finite. See M3.

### M5 — Reader omitted fields/summary and mishandled duplicate/different schemas: **fixed**

**Evidence:** `src/dataop_logger/reader.py:30` exposes a validation summary; lines 32-45 reject duplicate bindings and retain dimension per run; lines 38-44 load every tick column; lines 79-89 provide complete bound/masked sidecar records. The fallback dimension comes from the LeRobot action feature at lines 46-47. The reader test verifies filtering, missing modes, expert source, targets, and action equality.

`bound_episodes` is not scoped to the selected reader view; see L3.

### M6 — Action schema accepted impossible metadata and serialized bounds incorrectly: **partly fixed**

**Evidence:** `src/dataop_logger/schema.py:31-51` now validates names, action space, positive finite rate, complete/finite/ordered bounds, and dtype. `coerce()` rejects non-finite actions at lines 82-92. `to_dict()` emits the SPEC v0.3 nested `bounds` shape at lines 57-62 and documents bounds as descriptive.

**Incomplete:** Units, coordinate frame, and normalization note still accept empty or non-string values even though they participate in schema identity and action semantics. This is low severity compared with the original failures, but the fix is not literally complete for all schema metadata.

### M7 — Validator/repair crashed on corrupt or incomplete sidecars: **fixed**

**Evidence:** `src/dataop_logger/store.py:72-99` enumerates every run directory and classifies missing, malformed, incomplete, and unknown-status manifests. `validate()` reports per-run failures and continues at `src/dataop_logger/validate.py:199-210`. `repair()` leaves unreadable runs untouched at lines 237-247. Tests cover malformed JSON, a missing manifest, and unreadable parquet at `tests/test_logger.py:263-315`.

### L1 — `has_predictions` could disagree with file presence: **fixed**

**Evidence:** The manifest now stores `predictions_count` at `src/dataop_logger/logger.py:372`, and `_write_files()` always writes a schema-correct table when logging is enabled, including zero rows, at lines 376-381. Validation checks count/file consistency at `src/dataop_logger/validate.py:146-167`; the empty-table case is tested.

### L2 — Pre-first-frame events were attached to frame 0: **fixed**

**Evidence:** `src/dataop_logger/logger.py:239-244` writes `frame_index=-1` before the first frame. Validation permits exactly `[-1, length-1]` at `src/dataop_logger/validate.py:143-145`.

## New and remaining findings

## High severity

### H1. Repair can commit a raw parquet episode that LeRobot never registered

**File:** `src/dataop_logger/validate.py:25-44`, `src/dataop_logger/validate.py:170-185`, `src/dataop_logger/validate.py:225-260`

**Failure scenario:** LeRobot 0.6 writes episode data parquet before calling `meta.save_episode()`. If metadata save or video work fails after `_save_episode_data()`, finalizing the parquet writer makes those raw rows readable, but `meta.total_episodes`, `meta.total_frames`, and `meta.episodes` still exclude the episode. `load_lerobot_episodes()` scans the raw `data/` tree only. Repair sees a unique action/task fingerprint and commits it; validation then reports `ok=True`, even though reopening `LeRobotDataset` yields length 0 and zero episodes. I reproduced exactly this by raising from `writer._meta.save_episode`: repair committed episode 0 while LeRobot metadata still reported no episodes.

This is a false recovery and can hide missing videos, stats, episode metadata, or other incomplete save artifacts.

**Concrete fix:** Load the authoritative LeRobot episode table and admit only episode indices present there, with matching `dataset_from_index`/`dataset_to_index`, length, task metadata, and referenced video/data artifacts. Treat raw parquet rows not covered by authoritative episode metadata as partial/orphaned data and an error, never a repair candidate. For a save-failed run, repair should commit only after the entire LeRobot episode is finalized and loadable through a newly opened `LeRobotDataset`. Add fault injection before metadata save, during video encoding, and after metadata save; assert that only the truly complete case commits.

### H2. Validation still cannot detect finite corruption of expert/base supervision

**File:** `src/dataop_logger/logger.py:358-381`, `src/dataop_logger/validate.py:100-105`, `src/dataop_logger/validate.py:168-169`

**Failure scenario:** The manifest fingerprint covers task, schema, length, and `sent_action`, but no digest covers `expert_action`, `base_action`, validity masks, rewards, timing, control modes, predictions, or events. Semantic checks detect malformed values, but a finite plausible value is indistinguishable from the original. I reproduced changing a committed expert label to finite values `[999, -999]`; the sidecar still passed every check with `ok=True`. That silently changes DAgger labels and expert-residual targets, which are the package's primary outputs.

The new corruption test does not catch this: it tampers with an expert action but simultaneously creates unrelated unreadable/missing files, then restores the expert action without ever asserting that the expert-label change itself was reported.

**Concrete fix:** Keep the episode fingerprint for remapping, but add a separate integrity digest to the staged manifest covering a canonical serialization of every tick column, prediction rows, and relevant manifest-derived event data (or store SHA-256 hashes of the atomically written parquet files plus an events/manifest payload digest). Validate those digests before semantic checks and before repair commits a staged run. Add isolated tamper tests for finite base/expert actions, masks, reward, timing, control mode, predictions, and events.

## Medium severity

### M1. Reusing a caller-owned NumPy action buffer causes the run to be discarded at `end()`

**File:** `src/dataop_logger/schema.py:82-92`, `src/dataop_logger/logger.py:183-210`, `src/dataop_logger/logger.py:328-340`

**Failure scenario:** `coerce()` uses `np.asarray(...).reshape(-1)` followed by `ascontiguousarray`; for an already contiguous float32 vector this can remain a view sharing the caller's storage. Marking that view read-only does not make its base array immutable. LeRobot stores the view by reference. A normal control loop that reuses one NumPy action array and overwrites it for the next tick mutates prior buffered actions. The sidecar correctly retains copies, so `_check_buffer()` detects the mismatch only at `end()` and discards the entire real-world run. I reproduced a two-frame loop with one reused action array ending in `RunError: buffered action at frame 0 differs`.

**Concrete fix:** Make the canonical command an owned copy before passing it to LeRobot, for example `np.array(value, dtype=np.float32, order="C", copy=True).reshape(-1)`, then mark that owned array read-only. The sidecar and LeRobot may share this owned canonical array; neither should share caller storage. Add NumPy-buffer-reuse and CPU-torch-storage-reuse tests.

### M2. `blend` mode can be committed with neither a base nor expert action

**File:** `src/dataop_logger/logger.py:143-163`, `src/dataop_logger/validate.py:106-121`

**Failure scenario:** The writer requires an expert source and weight for `control_mode="blend"`, but it does not require `base_action` or `expert_action`. A tick with `blend_weight=0.5`, both validity flags false, and only a sent action commits and validates successfully. I reproduced such a run. It claims blended control while preserving neither component, so neither the stated expert residual nor the arbitration can be interpreted. Similarly, an expert label recorded while base control remains active need not identify `expert_source`.

**Concrete fix:** Define the semantic invariant explicitly and enforce it in writer and validator: blend requires both `base_valid` and `expert_valid`, and every `expert_valid` row requires a nonempty `expert_source` even when the expert was queried but not executed. If partial/unknown blends are required, add a distinct control mode or explicit component-validity semantics rather than calling them blend. Add writer and disk-tamper tests.

### M3. Prediction-table validation does not reproduce writer validation

**File:** `src/dataop_logger/validate.py:146-165`

**Failure scenario:** Disk validation checks count, uniqueness, flattened length, valid length, time ordering, and tick references, but it accepts unreferenced negative inference IDs, negative inference timestamps, out-of-range `obs_frame_index`, and NaN/Inf action chunks. All are rejected by `Run.predictions()` at `src/dataop_logger/logger.py:251-264`. A damaged prediction table can therefore pass validation and expose false diagnostic provenance.

**Concrete fix:** Define and validate the exact prediction parquet schema; require nonnegative IDs and times, `-1 <= obs_frame_index < run_length` (or the precisely documented pre-frame convention), finite float32 chunks reshaped to `[horizon, dim]`, and all writer invariants. Add one isolated disk-tamper test for each field.

### M4. Some malformed buffer snapshots bypass the intended quarantine/reset path

**File:** `src/dataop_logger/adapter.py:38-45`, `src/dataop_logger/logger.py:328-342`, `src/dataop_logger/logger.py:383-388`

**Failure scenario:** `_check_buffer()` verifies action count but not task count. Removing a task from the buffer can pass `any(t != self.task ...)`, then fail destructively inside `save_episode()`. Also, `_save_and_commit()` catches only `RunError` from `_check_buffer`; a malformed `episode_index` or action value can raise `ValueError`/`TypeError`, leaving a staged run and dirty buffer without the documented quarantine/reset behavior. These require direct/private buffer mutation, but guarding that mutation is the purpose of this check.

**Concrete fix:** Have `buffer_snapshot()` return validated canonical types or raise one adapter-specific error. Verify task count equals row count and each action has exactly the schema shape. Catch all snapshot/validation exceptions before save, reset/quarantine consistently, and do not call `save_episode()` after an unverifiable snapshot. Add task-list truncation, malformed episode index, and malformed action-shape tests.

### M5. An unexpected post-save episode count does not mark the writer unusable

**File:** `src/dataop_logger/logger.py:400-405`

**Failure scenario:** If `save_episode()` returns but total episodes or the expected index is wrong, the run is quarantined, yet `logger.broken` remains unset. Because the active run is now `quarantined`, the same logger permits another run against a writer whose save accounting has already proven inconsistent. That can cascade corruption or bind subsequent runs to the wrong index.

**Concrete fix:** Treat every failed post-save verification like a mid-save exception: set `logger.broken`, retain enough staged evidence for repair, and prohibit further runs until the dataset is reopened and repaired. Add a fake writer test where save advances by zero or two episodes.

### M6. Remap leaves the manifest's dataset identity pointing at the old dataset

**File:** `src/dataop_logger/validate.py:280-295`

**Failure scenario:** `remap()` copies the run and changes `episode_index`, but leaves `manifest["dataset"]` unchanged. A run under the new root therefore still claims the old root/repo/revision as its bound dataset, while `remapped_from` separately records the old location. This is incorrect provenance and makes later audits ambiguous.

**Concrete fix:** Load the new dataset identity and replace `manifest["dataset"]` when writing the destination manifest; keep the full old identity and episode index under `remapped_from`. Validate that committed manifests name the dataset root/revision they reside under. Add a remap provenance test.

## Low severity

### L1. Schema semantic strings are not validated

**File:** `src/dataop_logger/schema.py:31-51`

**Failure scenario:** Empty/non-string units, frame, or normalization note are accepted and included in the schema digest. A run can therefore have a valid schema ID but unusable action semantics.

**Concrete fix:** Require one nonempty string unit per action and nonempty string `frame`/`normalization_note` values. Add invalid-schema tests.

### L2. Disk validation accepts negative or non-integer timing values

**File:** `src/dataop_logger/validate.py:126-130`

**Failure scenario:** Writer-side code requires nonnegative integer monotonic timestamps, but validation casts values to `int64` and only checks relative ordering. Ordered negative timestamps, and some floating values that truncate cleanly, can pass.

**Concrete fix:** Validate parquet types and require every `obs_ns`, `sent_ns`, event timestamp, and prediction timestamp to be an integer in the allowed nonnegative range before converting. Add isolated tamper tests.

### L3. `bound_episodes` reports runs outside the selected LeRobot view

**File:** `src/dataop_logger/reader.py:31-45`, `src/dataop_logger/reader.py:67-69`

**Failure scenario:** The reader loads every committed run under the root, and `bound_episodes` returns all of them even when the wrapped LeRobot dataset selected only one episode. Iteration is correctly scoped, but the advertised binding summary is misleading for the view.

**Concrete fix:** Return the intersection of committed bindings and `view_eps`, while optionally exposing a separate `all_bound_episodes` property for root-wide inspection. Add an assertion to the filtered-reader test.

## Test coverage to add next

1. Partial save after data parquet but before LeRobot episode metadata, with and without video.
2. Isolated finite tampering of every training-relevant sidecar field, backed by integrity digests.
3. Reused NumPy and CPU torch action storage across ticks.
4. Blend without base/expert components and expert labels without a source.
5. Prediction parquet NaN, negative ID/time, and invalid observation-frame references.
6. Malformed private buffer task/action/index values before save.
7. Post-save episode-count mismatch and prohibition of further runs.
8. Remap identity and selected-view `bound_episodes`.
