# Round 5 final code review: dataop-logger

## Verdict

The requested test suite passes:

```text
python -m pytest -q tests
28 passed in 2.46s
```

**No high-severity issues remain.** The two Round 4 high-severity problems are materially fixed: repair no longer treats unregistered raw parquet as an episode, and the sidecar files/events are covered by integrity digests. I found four medium-severity gaps and three low-severity gaps. The most important remaining one for the stated camera-based first deployments is that validation and repair still accept an episode whose video artifact is missing.

## Round 4 finding audit

### H1 — Repair could commit an episode LeRobot never registered: **partly fixed**

**Evidence:** `src/dataop_logger/adapter.py:95-106` reads the authoritative `meta/episodes` table. `src/dataop_logger/validate.py:26-56` admits raw rows only when an episode metadata row exists and its length/span/task are consistent, and reports other rows as an incomplete save. `tests/test_logger.py:326-346` covers failure before metadata registration. This correctly fixes the reproduced Round 4 raw-parquet false commit.

**Incomplete:** The Round 4 fix also called for referenced data/video artifacts and a loadable reopened episode. Neither is checked. I created and finalized a one-frame LeRobot 0.6 video episode, deleted its sole MP4, and `validate()` still returned `ok=True`. The same metadata/data-only candidate is eligible for repair. See M1.

### H2 — Finite base/expert corruption was undetectable: **fixed**

**Evidence:** `src/dataop_logger/logger.py:386-395` hashes the exact serialized ticks and predictions parquet bytes and a canonical serialization of events before writing the manifest. `src/dataop_logger/validate.py:80-84,261-280` checks those digests before semantic validation, and `repair()` checks them before matching. `tests/test_logger.py:294-324` isolates finite expert-label and event tampering. This meets the accepted threat model of accidental corruption rather than an attacker who also rewrites the manifest.

### M1 — Reused caller action storage mutated the buffered episode: **fixed**

**Evidence:** `src/dataop_logger/schema.py:88-98` now uses `np.array(..., copy=True)` and passes an owned canonical array to LeRobot. `tests/test_logger.py:348-360` covers reuse of one NumPy buffer across ticks.

### M2 — `blend` could omit its component actions: **fixed**

**Evidence:** `src/dataop_logger/logger.py:146-162` requires both base and expert actions for blend and requires a source whenever an expert action exists. `src/dataop_logger/validate.py:123-139` repeats the rules on disk. `tests/test_logger.py:363-376` covers both writer failures.

### M3 — Prediction disk validation did not reproduce writer validation: **partly fixed**

**Evidence:** `src/dataop_logger/validate.py:174-198` now checks nonnegative IDs/times, observation-frame bounds, finite chunks, shape/valid length, uniqueness, and tick references.

**Incomplete:** The writer itself permits `obs_frame_index == len(self._rows)` and does not recheck it against the final run length, so it can create a committed file that this validator rejects. Prediction integer column types also are not checked. See M3 and L3.

### M4 — Malformed buffer snapshots bypassed quarantine/reset: **fixed**

**Evidence:** `src/dataop_logger/adapter.py:38-59` canonicalizes a snapshot or raises `EpisodeBufferError`; `src/dataop_logger/logger.py:333-351` checks the frame, action, and task counts plus exact shape/bytes/task; and `src/dataop_logger/logger.py:397-402` quarantines and resets on any pre-save check exception. `tests/test_logger.py:450-469` covers truncated tasks and a malformed action.

### M5 — Post-save count mismatch did not stop the writer: **fixed**

**Evidence:** `src/dataop_logger/logger.py:414-420` retains the run as staged, records the failure, sets `logger.broken`, and raises. `tests/test_logger.py:472-482` verifies both the failure and refusal of another run.

### M6 — Remap retained the old dataset identity: **partly fixed**

**Evidence:** `src/dataop_logger/validate.py:351-355` now changes the destination root and preserves the complete old identity under `remapped_from`.

**Incomplete:** It hard-codes the new `repo_id` and `revision` to `None` instead of loading the destination dataset identity. A remapped manifest therefore still does not contain the promised destination dataset ID/revision. See L2.

### L1 — Schema semantic strings were not validated: **fixed**

**Evidence:** `src/dataop_logger/schema.py:37-44` requires every unit, `frame`, and `normalization_note` to be a nonempty string. The parameterized schema test covers these cases.

### L2 — Disk validation accepted bad timestamp types/values: **partly fixed**

**Evidence:** `src/dataop_logger/validate.py:147-156` checks tick integer types and nonnegative/ordered timestamps, and lines 169-173 check event timestamp type/value.

**Incomplete:** `predictions.parquet` is value-checked after conversion to Python values, but its timestamp/index columns are not required to have integer Arrow types. See L3.

### L3 — `bound_episodes` ignored the selected view: **fixed**

**Evidence:** `src/dataop_logger/reader.py:68-75` distinguishes view-scoped `bound_episodes` from root-wide `all_bound_episodes`, and `tests/test_logger.py:423-430` covers the selected-view result.

## New and remaining findings

## High severity

None.

## Medium severity

### M1. Validation and repair accept metadata-backed episodes with missing video artifacts

**File:** `src/dataop_logger/adapter.py:95-106`, `src/dataop_logger/validate.py:26-56`, `src/dataop_logger/validate.py:289-324`

**Failure scenario:** `episode_metadata()` reads only the episode index, length, data span, and tasks. `load_lerobot_episodes()` reads only data parquet and never verifies the video fields/files or attempts to load the episode. A video can be deleted or encoding can fail after metadata registration, yet validation considers the episode complete and repair can commit a staged run against it. I reproduced the direct case with a finalized LeRobot 0.6 camera dataset: after deleting `videos/observation.images.cam/chunk-000/file-000.mp4`, validation still returned `ok=True`. The dataset is then unusable for the camera-based deployments named in the spec.

**Concrete fix:** Read the destination metadata's video keys and each episode's referenced video records, require every referenced artifact to exist, and verify that a newly opened LeRobot dataset can load the selected episode (including a frame from every video key) before treating it as a validation/repair candidate. Also check that the raw global `index` values equal the episode metadata's exact `[dataset_from_index, dataset_to_index)` range, rather than checking only that the span has the right width. Add missing, truncated, and post-metadata encoding-failure tests.

### M2. The wrapper changes the underlying LeRobot view's order and can duplicate samples

**File:** `src/dataop_logger/reader.py:49-66`

**Failure scenario:** The wrapper iterates the facade's raw requested `dataset.episodes` order and maps each absolute row to a relative position. LeRobot 0.6 resolves/filter-loads the data in its own row order. For a view requested as `episodes=[2, 0]`, I reproduced the underlying LeRobot sequence as episode/frame `[(0,0), (0,1), (2,0)]`, while `DataopDataset` returned `[(2,0), (0,0), (0,1)]`. Repeated requested episode IDs likewise make the wrapper repeat rows that the wrapped dataset contains only once. Out-of-range requested IDs are ignored by LeRobot but cause `missing="error"` to fail and `missing="mask"` to raise `KeyError` at `spans[e]`.

This violates the promise to wrap LeRobot while keeping its sampling behavior and can silently change training order or sample weighting.

**Concrete fix:** Build the wrapper index in the wrapped reader's actual relative-row order. Use the reader-resolved episode list (or intersect/deduplicate the facade list with metadata), map absolute rows to relative rows, then sort/deduplicate by relative position before applying the missing-run policy. Assert that, absent `missing="exclude"`, `DataopDataset[i]` has the same episode/frame identity as `lerobot_dataset[i]`. Add reversed, duplicate, and invalid episode-selection tests.

### M3. The prediction writer can commit an observation reference its validator rejects

**File:** `src/dataop_logger/logger.py:251-274`, `src/dataop_logger/logger.py:318-331`, `src/dataop_logger/validate.py:174-198`

**Failure scenario:** `predictions()` accepts `obs_frame_index <= len(self._rows)`. This is useful temporarily when a prediction precedes the frame it will control, but `_check_rows()` never verifies that the referenced frame eventually exists. I logged one frame, then logged a prediction with `obs_frame_index=1`, ended the run, and the writer committed it. `validate()` then reported `prediction 0 obs_frame_index out of range` because the final length was one. The package can therefore reject an untouched file it created itself.

**Concrete fix:** Keep the permissive recording-time check if future-frame logging is intended, but at `end()` require every prediction to satisfy `-1 <= obs_frame_index < len(rows)`. Otherwise change the recording-time upper bound to `< len(self._rows)`. Add tests where an equal-to-current-length reference is followed by a frame (valid) and where the run ends first (rejected before staging).

### M4. A malformed `integrity` value crashes repair instead of being left untouched

**File:** `src/dataop_logger/store.py:81-105`, `src/dataop_logger/validate.py:261-280`, `src/dataop_logger/validate.py:289-304`

**Failure scenario:** `iter_run_dirs()` checks only that the `integrity` key exists. `integrity_problems()` assumes it is a mapping and calls `.get()`. In validation the outer per-run catch converts this into an error, but repair calls `integrity_problems()` outside its `try`. I changed a staged manifest to `"integrity": "bad"`; `repair()` raised `AttributeError: 'str' object has no attribute 'get'` and stopped processing all runs. This regresses the Round 4 requirement that corrupt runs not crash repair.

**Concrete fix:** Validate the manifest/integrity structure in `iter_run_dirs()` (mapping with string hashes or `None`) and/or make `integrity_problems()` return a descriptive problem for malformed values. Put the integrity call inside repair's per-run exception boundary. Add a staged malformed-type test alongside the existing missing/malformed-manifest tests.

## Low severity

### L1. Post-save verification still does not check the saved episode length

**File:** `src/dataop_logger/logger.py:397-423`

**Failure scenario:** The spec promises post-save verification of both episode index and length. `_save_and_commit()` checks only the change in `total_episodes` and expected index. If LeRobot advances the count but records an incomplete/wrong-length episode, the sidecar is immediately committed; only a later explicit validation can discover it.

**Concrete fix:** After save, read the newly registered episode metadata and require its `length` and exact data span to equal `len(self._rows)` before committing. On mismatch, retain staged state and break the logger as for the count mismatch. Add a fake-writer test that advances the episode count while registering the wrong length.

### L2. Remap discards the destination repository and revision identity

**File:** `src/dataop_logger/validate.py:342-355`

**Failure scenario:** Remap writes the correct destination root but sets `repo_id` and `revision` to `None`, even when the new LeRobot metadata has both. Provenance cannot identify which destination dataset revision received the copied run.

**Concrete fix:** Load the new dataset metadata/facade and populate the same identity fields used by `adapter.dataset_identity()`. Keep the old identity under `remapped_from`. Add an assertion for all three destination identity fields.

### L3. Prediction parquet integer types are not validated

**File:** `src/dataop_logger/validate.py:174-198`

**Failure scenario:** Unlike tick timestamps/indices, prediction IDs, observation indices, timestamps, horizon, and valid length are never checked against the writer's integer Arrow schema. A matching-digest file from a buggy writer can use positive floating values and pass the numerical checks, even though it violates the on-disk format and writer contract.

**Concrete fix:** Define the expected prediction schema in one shared location and validate each column type before reading values. Add a float-typed prediction timestamp/index test whose digest matches the file.

## Important missing tests

1. Camera episode with missing/corrupt MP4, plus video encoding failure after episode metadata registration.
2. Reader views requested in reversed order, with duplicate IDs, and with IDs LeRobot resolves away; compare every returned episode/frame identity to the wrapped dataset.
3. Prediction `obs_frame_index == current length`, both with and without the later frame.
4. Staged manifests with non-object/malformed integrity values passed to `repair()`.
5. Post-save episode-count success combined with a wrong registered length.
6. Remap provenance for destination repo ID and revision, and prediction parquet schema types.
