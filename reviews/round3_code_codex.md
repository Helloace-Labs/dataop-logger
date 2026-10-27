# Round 3 code review: dataop-logger implementation

## Verdict

The basic write/save/commit path is compact and the existing integration tests pass, but the implementation is not yet safe to ship as the provenance layer described by SPEC v0.3. The most important gaps are in the exact places where the sidecar is supposed to add trust: a filtered LeRobot dataset is joined with the wrong index space, `end()` does not freeze or revalidate the LeRobot buffer before saving, save failures can strand a real episode with its recoverable sidecar hidden in quarantine, and terminal flags can disagree with the run outcome while still looking RL-ready.

I ran:

```text
python -m pytest -q tests
12 passed in 3.88s
```

I also reproduced the filtered-reader, post-`end()` mutation, and 1-DoF validation failures with temporary LeRobot 0.6 datasets outside the repository.

## High severity

### H1. `DataopDataset` uses dataset-wide absolute indices as positions in an episode-filtered LeRobot view

**File:** `src/dataop_logger/reader.py:39-48`, `src/dataop_logger/reader.py:57-60`

**Failure scenario:** LeRobot 0.6's `__getitem__` takes a relative row position in the loaded view. When it is opened with `episodes=[2]`, its valid positions are `0..len(view)-1`, while `meta.episodes[*].dataset_from_index` and `dataset_to_index` remain dataset-wide absolute indices. `DataopDataset` builds `self.index` from every metadata episode, not from `lerobot_dataset.episodes`, and passes those absolute values to `self.ds`. In a reproduced three-episode dataset, wrapping a LeRobot view containing only episode 2 reported length 6 instead of 2; positions after the first two are out of range. Other episode selections can return the wrong episode before failing. This breaks the promised preservation of LeRobot episode selection and delta-timestamp sampling.

**Concrete fix:** Build the wrapper over the underlying dataset's selected episode list and relative row positions. Intersect `lerobot_dataset.episodes` (or all episodes when it is `None`) with the missing-sidecar policy, then create relative spans in the order of the filtered HF dataset. Alternatively, map each selected episode's absolute indices through `lerobot_dataset.absolute_to_relative_idx`. Never pass dataset-wide absolute indices to `LeRobotDataset.__getitem__`. Add tests for `episodes=[2]`, noncontiguous selections such as `[0, 2]`, `episode_filter`, all three missing policies, and delta-timestamp windows.

### H2. `run.end()` does not freeze the episode buffer, and `_save_and_commit()` does not repeat the promised pre-save checks

**File:** `src/dataop_logger/logger.py:228-235`, `src/dataop_logger/logger.py:324-338`

**Failure scenario:** `end()` checks only the buffer size and writes the staged sidecar. User code still runs inside the `with` block before `_save_and_commit()` is invoked. A direct `dataset.add_frame(...)` in that window adds an extra LeRobot frame; `_save_and_commit()` saves it and marks the one-row sidecar committed without rechecking length or actions. I reproduced a committed run of sidecar length 1 bound to a two-frame LeRobot episode. The same problem occurs if code mutates `writer.episode_buffer["action"]` after `end()`. This violates the exclusive-add contract and SPEC's byte-equality check before save; validation only discovers the corruption later.

**Concrete fix:** Immediately before `save_episode()`, recheck the buffer episode index and size and compare every buffered action byte-for-byte against immutable copies retained by the run. Abort without saving on any mismatch. The adapter should expose a version-tested snapshot function for this check. Make `end()` transition the recording API into a genuinely sealed state; if practical, guard or proxy direct dataset writes for the entire run context. Add tests for a direct add, action mutation, and buffer replacement after `end()`.

### H3. A `save_episode()` exception makes potentially recoverable data invisible to repair

**File:** `src/dataop_logger/logger.py:324-330`, `src/dataop_logger/logger.py:249-253`, `src/dataop_logger/validate.py:105-129`

**Failure scenario:** LeRobot 0.6 mutates the episode buffer near the beginning of `save_episode()` and writes episode parquet before all video and metadata operations finish. It can therefore raise after some or all of an episode is on disk; with batched video encoding it can even raise after metadata increments `total_episodes`. The logger unconditionally moves the staged run to `dataop/quarantine/`. `repair()` scans only staged manifests under `dataop/runs`, so it can never match that sidecar to the episode. A valid episode can be left unbound, and the only exact provenance record is excluded from the documented recovery mechanism.

**Concrete fix:** Preserve the staged run in the repair scan after any save exception. Either leave it under `runs/` with a recorded failure note, or make repair inspect staged manifests in quarantine as well. Mark the writer/logger unusable after a mid-save exception rather than attempting another run against LeRobot's destructively mutated buffer. Repair should commit an exact unique on-disk match, quarantine only after reconciliation proves there is no match, and retain the original staged files in either case. Add fault-injection tests at `_save_episode_data`, video encoding, metadata save, and post-metadata batch encoding.

### H4. Successful/failed and truncated outcomes can be committed with contradictory transition flags

**File:** `src/dataop_logger/logger.py:255-266`, `src/dataop_logger/validate.py:60-63`

**Failure scenario:** `_check_rows()` rejects a present flag with the wrong reason, but it never requires the expected flag. A run with every `reward_valid=True` can end with `termination_reason="success"` and a final row where both `terminated` and `truncated` are false; a timeout can likewise have `truncated=False`. Such a run meets the spec's stated reward criterion for RL readiness but produces an incorrect final transition. Disk validation also fails to check mutual exclusion or agreement with `termination_reason` at all.

**Concrete fix:** Define and enforce the complete mapping: `success`/`failure` require final `terminated=True`; `timeout`/`operator_abort` require final `truncated=True`; explicitly decide and document `system_error` (normally truncated). Either have `end()` set the final flag from the reason or require the caller to supply it and reject omissions. Apply the same checks in disk validation and report whether a run is RL-ready. Add tests for every termination reason, missing flags, both flags, and all-reward-valid runs.

### H5. The run task is not tied to LeRobot's per-frame task, so the fingerprint does not actually validate task identity

**File:** `src/dataop_logger/logger.py:122-156`, `src/dataop_logger/validate.py:17-31`, `src/dataop_logger/validate.py:111-115`, `src/dataop_logger/validate.py:147-156`

**Failure scenario:** `run(task="A")` accepts frames whose LeRobot `frame["task"]` is `"B"` (several current tests already do this). The manifest and fingerprint say A while the dataset records B. `load_lerobot_episodes()` does not load `task_index`, and repair/remap recompute each candidate fingerprint using the manifest task rather than the candidate episode's task. Consequently the task component provides no matching evidence, validation cannot detect semantic disagreement, and a unique action sequence can be rebound to an episode with a different task.

**Concrete fix:** For v0's one-task-per-run model, require every frame's task to equal `Run.task` before calling LeRobot. On disk, load `task_index`, resolve it through LeRobot metadata, require one consistent task per episode, and use that actual task in validation, repair, and remap fingerprint calculations. If multi-task episodes are intended, change the fingerprint and manifest to contain the canonical per-frame task sequence. Add mismatch and remap tests where two episodes share actions but have different tasks.

### H6. Disk validation omits most semantic invariants and can approve silently corrupted supervision

**File:** `src/dataop_logger/validate.py:34-76`

**Failure scenario:** `_check_run()` verifies only sent-action shape/hash/equality, sequential indices, monotonic individual clocks, and takeover bounds. It does not validate required columns/types; `base_action` or `expert_action` shapes; validity-mask semantics; control-mode/expert-source/blend rules; finite values; `obs_ns <= sent_ns`; terminal-reason consistency; prediction references and chunk shape/ranges; manifest status/required fields; or that `takeovers` exactly equal the ranges derived from ticks. A damaged or manually edited expert label, mask, prediction table, or takeover summary can therefore pass `validate` and silently change training targets or metrics.

**Concrete fix:** Make validation reconstruct the complete per-run invariant set independently of the writer: validate the parquet schema and all action arrays, run `_check_rows`-equivalent rules on disk, validate prediction referential integrity, recompute takeovers, check event frame ranges, check all required manifest fields/status, and reject non-finite timestamps/actions/rewards. Do not trust hashes alone; the fingerprint intentionally covers only sent actions and binding fields. Add one tamper test per invariant group, including expert/base fields that are not fingerprinted.

## Medium severity

### M1. One-dimensional LeRobot actions cannot validate, repair, or remap

**File:** `src/dataop_logger/validate.py:22-30`, `src/dataop_logger/validate.py:50-55`, `src/dataop_logger/fingerprint.py:21`

**Failure scenario:** LeRobot 0.6 serializes a numeric feature declared with `shape=(1,)` as a scalar column. `load_lerobot_episodes()` therefore returns action shape `(N,)`, while the sidecar fixed-size list has shape `(N, 1)`. Validation always reports an action-shape mismatch, and repair/remap compute a fingerprint whose encoded second dimension is zero rather than one, so no staged 1-DoF run can match. I reproduced this with a one-action dataset.

**Concrete fix:** Canonicalize loaded actions with the manifest schema before shape checking or fingerprinting: if total size is `N * schema.dim`, reshape to `(N, schema.dim)` and require the row count to agree. Use one shared canonicalization helper in validate, repair, and remap. Add round-trip, crash-repair, and remap tests for `dim == 1`.

### M2. Missing timestamps are silently fabricated after LeRobot frame I/O

**File:** `src/dataop_logger/logger.py:122-126`, `src/dataop_logger/logger.py:156-178`

**Failure scenario:** `obs_ns` and `sent_ns` are optional and both default to a `monotonic_ns()` reading taken after `dataset.add_frame()`. Camera image writes or encoder backpressure may make that substantially later than both observation availability and command send. The resulting rows look fully valid and can understate or erase observation-to-command latency; there are no validity flags to distinguish supplied measurements from fabricated defaults.

**Concrete fix:** Require explicit `obs_ns` and `sent_ns` for the v0 API, as the SPEC example does, or add validity/source fields and clearly label logger-generated timestamps. Validate `obs_ns <= sent_ns`, nonnegative `obs_skew_ns` when present, and cross-row monotonicity. Add tests with delayed frame I/O and invalid clock ordering.

### M3. A partial LeRobot `add_frame()` failure can leave an undetected dirty buffer

**File:** `src/dataop_logger/logger.py:149-179`, `src/dataop_logger/logger.py:238-247`, `src/dataop_logger/adapter.py:7-8`

**Failure scenario:** LeRobot appends `frame_index`, `timestamp`, and task before writing all feature/image values and increments `episode_buffer["size"]` only at the end. If the first frame raises during image/feature handling, `has_pending_frames()` is false even though lists or temporary images may have been mutated. `discard()` therefore does not clear the buffer. If an integration catches an `add_frame` exception and continues, the local count can still equal LeRobot's `size` while internal feature lists are misaligned, causing a later save failure or malformed partial state.

**Concrete fix:** Treat any exception from `dataset.add_frame()` as fatal to the run: unconditionally clear/reset the LeRobot episode buffer (not conditional on `has_pending_frames()`), remove partial image artifacts through LeRobot's clear API, mark the run discarded, and require a new run. Add a fault-injecting dataset that raises after partially mutating the first and a later frame.

### M4. Prediction invariants depend incorrectly on whether the optional table is enabled

**File:** `src/dataop_logger/logger.py:141-147`, `src/dataop_logger/logger.py:199-216`, `src/dataop_logger/logger.py:267-278`

**Failure scenario:** With `log_predictions=False`, a tick may have `inference_id=-1` and `base_valid=True`, directly violating SPEC's rule that `-1` means the base did not run. Negative IDs other than `-1` are accepted, and positive `chunk_offset` values have no range check. The `elif` at lines 277-278 only rejects offsets below `-1`, and only if some inference ID exists. These rows are structurally valid but have ambiguous or false policy provenance.

**Concrete fix:** Enforce table-independent rules on every tick: `inference_id == -1` iff no base prediction is claimed; otherwise require a nonnegative ID and a nonnegative offset (or explicitly support non-chunked predictions with a separate convention). When predictions are logged, require a referenced unique row and `offset < valid_len`; validate `obs_frame_index`, inference time ordering, finite chunks, and schema shape. Repeat these rules in disk validation. Add tests with predictions both enabled and disabled.

### M5. The reader does not expose the sidecar contract or a validation summary promised by SPEC

**File:** `src/dataop_logger/reader.py:11-13`, `src/dataop_logger/reader.py:28-38`, `src/dataop_logger/reader.py:63-83`

**Failure scenario:** The wrapper returns only the three actions, six flags, reward, and control mode. It drops `expert_source`, `blend_weight`, inference/chunk IDs, observation/send timing, and observation skew, so callers cannot perform provenance filtering or latency analysis through the advertised Python reader. It also has no validation summary and silently overwrites one committed run with another if two manifests bind the same episode. A single global `self.dim` is taken from the last manifest, which makes masked legacy rows wrong or zero-width when action schemas differ or no run is bound.

**Concrete fix:** Load and return all per-tick fields with suitable tensor/string handling, expose a validation summary, reject duplicate bindings, and retain schema/dimension per run. For an unbound masked episode, infer the action dimension from the LeRobot action feature or require a declared fallback schema. Add reader tests for all fields, duplicate bindings, heterogeneous schemas, and datasets with zero committed runs.

### M6. Action schemas can encode impossible metadata and do not match the specified manifest shape

**File:** `src/dataop_logger/schema.py:28-40`, `src/dataop_logger/schema.py:46-55`, `src/dataop_logger/schema.py:65-72`

**Failure scenario:** The implementation accepts zero/negative/non-finite `rate_hz`, duplicate or empty names, an empty `custom:` suffix, one-sided bounds, `low > high`, and NaN/Inf action values. `inspect()` divides by `rate_hz`, while invalid bounds and non-finite actions undermine target semantics. The manifest serializes top-level `low`/`high`, whereas SPEC v0.3 specifies `bounds: {low, high}`; external consumers implementing the spec will compute a different canonical schema ID.

**Concrete fix:** Validate all schema metadata and action finiteness, decide whether bounds are enforced or descriptive and document that decision, and serialize exactly the v0.3 schema shape. Because changing canonical JSON changes IDs and fingerprints, bump the schema version and provide an explicit migration/compatibility path. Add invalid-schema and non-finite-action tests.

### M7. Validator/repair can crash instead of reporting corrupt or incomplete sidecar files

**File:** `src/dataop_logger/store.py:68-78`, `src/dataop_logger/validate.py:34-55`, `src/dataop_logger/validate.py:105-129`

**Failure scenario:** `iter_manifests()` ignores a directory with no manifest and catches only `FileNotFoundError`; malformed JSON aborts the command. Missing/corrupt parquet, malformed manifest keys, or a bad staged schema similarly raise uncaught exceptions in validate/repair. A crash while staging can leave `ticks.parquet` without a manifest, and `validate` reports neither an error nor an orphan. This contradicts the CLI promise of a machine-readable report and nonzero status for corruption.

**Concrete fix:** Enumerate all run directories, classify missing/malformed manifests and parquet as explicit errors, and catch per-run parse/read failures so remaining runs are still checked. Repair must never delete or overwrite an unreadable run automatically. Ensure `--json` still emits valid JSON on corruption. Add truncated JSON/parquet, missing manifest/ticks, unknown status, and missing-column tests.

## Low severity

### L1. `has_predictions` can be true while `predictions.parquet` does not exist

**File:** `src/dataop_logger/logger.py:315`, `src/dataop_logger/logger.py:318-322`

**Failure scenario:** Starting with `log_predictions=True` but recording no predictions sets `has_predictions=true`; `_write_files()` writes the table only when the list is nonempty. Consumers cannot tell whether the file is missing by design, was lost, or was corrupted.

**Concrete fix:** Define the manifest field as `predictions_count` and always write an empty table with an explicit schema when logging is enabled, or set `has_predictions` from file presence/nonzero count. Validate the manifest/file relationship and add an empty-prediction test.

### L2. Events emitted before the first frame are falsely attached to frame 0

**File:** `src/dataop_logger/logger.py:193-197`

**Failure scenario:** `max(len(rows) - 1, 0)` records a pre-first-frame takeover request as effective at frame 0. That loses the distinction between an event before observation/action 0 and one occurring during frame 0, despite the event timestamp being intended to capture takeover reaction timing.

**Concrete fix:** Use `frame_index=-1` (and document it) for pre-run events, or store separate requested/effective frame fields. Validate event frame indices in `[-1, length-1]` and test pre-first-frame and post-final-frame behavior.

## Missing test coverage to prioritize

The fixes above should be protected first by these end-to-end cases:

1. Filtered/noncontiguous LeRobot episode views with delta timestamps and each missing policy.
2. Direct dataset mutation after `end()` and exact pre-save action-byte verification.
3. Fault injection at multiple points inside `add_frame()` and `save_episode()`, followed by repair.
4. Full terminal-reason/reward-validity matrix.
5. Task mismatch and same-actions/different-task fingerprint matching.
6. Tampering of every sidecar field not covered by the current fingerprint, including predictions and takeovers.
7. 1-DoF validation, repair, and remap.
8. Malformed/missing manifests and parquet with `validate --json`.
