# Round 17 design review: batches, import and export

## Verdict

The batch-as-round model matches the SO-101 DAgger loop and is a useful abstraction, but v0.4 is not implementable as written. Import conflicts with the current sidecar schema and misstates some LeRobot conventions; export omits the rules needed to keep cut datasets temporally and semantically valid; and the residual recipe does not reproduce the cited correction-head build. Keep the design, but narrow the first release and fix the high findings below.

## High

### 1. Generic import cannot produce a currently valid dataop run

The draft introduces `control_mode="unknown"`, `termination_reason/outcome="unknown"`, and `clock.obs_timestamp_kind="from_dataset_timestamp"` (lines 101-119). Existing code accepts only `base|expert|blend|none`, five known termination reasons, and `capture|bundle_completion`; validation also requires final terminal flags to agree with the reason. Imported `base_action` additionally needs nonnegative inference/chunk IDs. A plain LeRobot dataset therefore cannot be represented honestly without failing validation or inventing provenance.

**Fix:** define an import schema migration before the importer: add explicit unknown control/end provenance (and validity flags), allow the imported timestamp kind, and make inference provenance independently optional when a base action is known. Update validator, reader, viewer, takeover derivation and manifest schema together. Do not encode unknown control as `none`.

Use LeRobot's actual conventions precisely: `action` is the executed action; HIL replay-buffer datasets use `complementary_info.is_intervention`; reward/done/success are `next.reward`, `next.done`, and `next.success` (shape `(1,)`), with optional `next.truncated`. For HIL, map non-intervention `action` to `base_action` and intervention `action` to `expert_action`; a shadow base action during intervention needs an explicit mapped feature and is not a LeRobot standard. Derive episode success with `any(next.success)`, not only the last row, and do not equate `next.done` with `terminated` when truncation is unknown.

### 2. The `watcher` preset does not match the files from which it was derived

The draft requires a per-episode base-action file of length `T` and refuses a length mismatch (lines 109, 121-123). In `backfill_dataop.py`, `epNNN.npy` has only `policy_frames` rows; the remaining `skill_frames` are rescue rows. Results JSON supplies that boundary. The backfill also derives grasp/lift segments and failure types with tube kinematics, not from generic Watcher files.

**Fix:** make the preset an explicit prefix mapping: require `len(raw) == policy_frames`, place it on `[0, policy_frames)`, set the intervention suffix from `policy_frames`, and map suffix `action` to expert action. Keep the kinematic stage/failure inference out of the preset; expose it as an optional annotation hook and leave those fields unknown otherwise.

### 3. Export does not yet define a valid derived time series

Splitting only at excluded spans is insufficient (lines 146-150). DAgger label filtering, segment filtering, residual validity, and relabel hooks can all create additional gaps. Concatenating frames across any such gap makes temporal sampling cross discontinuities. New fragment ends also invalidate inherited `next.done`, `next.truncated`, and `next.success`. `--val` does not say whether it creates two dataset roots or a usable LeRobot split.

**Fix:** partition every run into maximal contiguous kept source-frame spans after all selectors and hooks. Materialize each span as a new episode through `LeRobotDataset.create/add_frame/save_episode/finalize`, which resets reserved indices/timestamps, re-encodes video and recomputes episode/global stats. Rewrite terminal features at artificial ends as truncations (or remove transition-only features from BC exports), and preserve success only on the true successful terminal span. Specify `--val` as two concrete LeRobot datasets, or define and test an episode ordering plus split metadata that the intended trainer actually consumes.

Before combining roots or `--base`, require equal fps, robot type and compatible feature schemas: LeRobot `merge_datasets` rejects differences. Also reject or explicitly map action dimensions/names. Preserve arbitrary non-action features, but remove or recompute action-derived features that would become stale when `action` is replaced.

### 4. The residual recipe is not the cited residual-network target pipeline

The generic subtraction is mathematically valid, but `build_correction_data.py` does much more: it recomputes the frozen policy's chunk-aged base proposal on rescue observations, selects only the grasp phase, subsamples rescue frames, emits only four action dimensions, creates handcrafted input features, weights rescue examples 2:1, and adds zero targets only from unguarded raw-policy successes. The proposed relabel hook can return only replacement actions, so it cannot express selection, features, dimensions or weights.

**Fix:** call the built-in recipe “full-action supervised residual” and require recorded, action-space-compatible `base_action`. Restrict `--zero-on-success` to successful base-controlled frames where `base_action` is valid, `sent_action == base_action`, and `safety_modified` is false. Either expand the hook result to `{keep_mask, target, extra_features, sample_weight, split_points}` or state that the SO-101 correction dataset remains a user script. Do not claim the current recipe replaces it.

### 5. Locked-review digests do not make export deterministic

The output also depends on mutable LeRobot observations/videos, ticks, manifests, dataset lists, effective drive-boundary edits, root ordering and arbitrary hook dependencies. Current `train_mask` uses recorded control modes; accepted `drive_boundary` edits are not applied by `reader.py`, so review can show one takeover boundary while export uses another. A file digest of a hook does not make execution deterministic.

**Fix:** create one canonical effective-run snapshot used by viewer, mask and export, including reviewed drive modes. At lock/export record the lists digest, full policy/action schema, manifest and ticks integrity, source dataset metadata plus immutable Hub revision or local data/video digests, canonical source/run order, exact split-hash algorithm and seed, hook digest/config, and final output digest. Validate sources immediately before materialization. For hooks, also record the returned selection/target digest.

## Medium

### 1. Batch membership and lineage need one atomic authority

Separate batch JSON files cannot, under per-file locks, enforce “one batch per run” or “one collecting batch per root.” Automatic grouping by policy can also merge separate experiments that reused a checkpoint. The stored policy `{family, source, revision}` drops artifact/config digests already present in run manifests. `link <batch> --model` is ambiguous when a batch has several exports.

**Fix:** use one revisioned `dataop/batches.json` registry (or one dataset-wide `batches.lock`) containing the run-to-batch map and collecting batch ID. Make the round/batch ID explicit in collection; policy-based naming may be a convenience, not identity. Store the complete policy identity digest. Link a model to a specific export ID, then let a later batch reference that model/export; support multiple input exports if needed.

Enforce locked-run edits centrally in `review.transaction`, not separately in every viewer/CLI/API caller, and either refuse locking with unreviewed runs/open proposals or require an explicit override recorded in history.

### 2. The proposed copy implementation is invalid

LeRobot `delete_episodes(dataset, [])` raises `ValueError("No episodes to delete")`; it cannot be used as a copy primitive. Merging sources also requires identical fps, robot type and features.

**Fix:** byte-copy a complete local/Hub snapshot to `--into` and attach there. Do not merge during import; keep session datasets separate and combine compatible materialized exports later. This is simpler and preserves the source exactly.

### 3. DAgger selection needs one more target rule

The recipe is correct only if every selected observation has exactly one authoritative target. For `blend`, `expert_action` is appropriate; for expert control it may fall back to `sent_action` only when the importer/validator proves the sent command is the expert command. Accepted drive-boundary edits must not manufacture expert labels where none exist.

**Fix:** make target validity explicit after effective-control edits: use `expert_action` whenever valid; allow `sent_action` fallback only on provenance-certified expert frames; otherwise drop the frame and report the reason.

## Low / simplification

- Reduce the first release to: one atomic batch registry with `open|closed`, attach-only import plus `plain` and correctly specified HIL/Watcher mappings, and one span-materialization engine with `filtered` and `dagger`. Add residuals, copy/push/pull and viewer export after real-data tests.
- A hard review lock may be unnecessary. A closed batch plus an immutable export snapshot of each effective review gives reproducibility without blocking later corrections; newer reviews simply mark prior exports stale.
- The tube-pick hover pose, FK/table thresholds, gripper rules, “never let go,” squeeze adjustment, grasp-window selection and correction features are all task-specific. Keeping them outside the package is right, but the hook contract must be rich enough to host them before the design says the old scripts are replaced.
