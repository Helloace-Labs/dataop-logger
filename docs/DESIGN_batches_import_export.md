# Batches, import and training-data export (design, v0.4 draft)

Three features turn the logger from "record and review one dataset" into the post-training loop a team actually runs:
collect a batch of runs with one policy, review it, generate a training set from it with a script, train the next
policy, and repeat. Everything here is generic. Nothing depends on the task, the robot, the joints, the cameras or the
frame rate; task-specific logic (for example our tube-pick hover relabel) stays in user code behind a hook.

## What we learned from the SO-101 loop

Our DAgger rounds on the SO-101 worked like this (see `~/so101/build_smolvla100_dagger.py`, `build_correction_data.py`):

- One policy version (SmolVLA v7b, then SmolVLA-100) ran 30 to 100 runs, spread over several sessions
  (`vla_..._plain100a`, `_b`, `_c`). A rescue skill took over when a run failed. That set of runs is one DAgger round.
- After the round, a script built the next training set: the original demos, plus each rescued run's expert frames,
  with the policy's frames just before the takeover relabelled toward where the rescue lined up. Failed rescues and
  edge cases were dropped. The policy's own successes were not added. Then the model was fine-tuned and the next round
  started with the new checkpoint.
- The correction head (residual network) used a different training set from the same runs: on the rescue's grasp
  phase, target = rescue command minus what the frozen base model proposed in that recorded state; on the policy's own
  successful grasps, target = zero, so it learns to leave good behaviour alone.
- Each session was its own LeRobot dataset, plus `<session>_raw/epNNN.npy` (base model output per frame) and
  `<session>_results_*.json` (per run: outcome, frame where the rescue took over, failure reason). The example dataset
  in the viewer was re-recorded from one such session by a one-off script (`~/so101/backfill_dataop.py`).

So a batch is a round, export is the build script, and import is what the backfill script did by hand.

## 1. Batches

**Definition.** A batch is a set of runs collected with one policy version and reviewed together, so that one
training set can be generated from it. In DAgger it is one round.

**Storage.** `dataop/batches/<batch_id>.json`, one file per batch, under the same lock rules as `lists.json`:

```
{version: 1, batch_id, name, purpose: free text, status, created_at,
 policy: {family, source, revision} | null,     # the policy the runs were collected with (from run manifests)
 parent_export: {batch_id, export_id} | null,     # the training set whose model collected this batch (lineage)
 runs: [run_id],                                  # membership; manifests stay immutable
 locked: {at, by, runs: {run_id: {review_rev, review_digest}}} | null,
 exports: [export record, see section 3],
 history: [{at, by, what}]}
```

**Lifecycle.** `collecting -> reviewing -> locked`. Export is allowed only from `locked`.

- `collecting`: new committed runs join it. At most one collecting batch per dataset root at a time.
- `reviewing`: closed to new runs; review edits continue.
- `locked`: review edits on its runs are refused (viewer, CLI and agent API all return a clear error naming the
  batch). Locking records each run's review `rev` and a digest of its effective review, so an export can prove which
  review it was built from. `unlock` is allowed, needs a reason, is written to `history`, and marks earlier exports
  as `stale` (still listed, flagged as built from an older review).

**How runs join.**

- Logger: `RunLogger(dataset, ..., batch="round 3")` opens or continues that batch; each committed run is added at
  commit. Without `batch=`, a run joins the collecting batch, or a new one is opened and named after the run's
  policy (`smolvla @ <revision>`). A run whose policy differs from the batch's policy is still added but flagged
  `mixed_policy` on the batch (shown as a data warning), because mixing policies in one DAgger round is usually a
  mistake but sometimes intended.
- Import: imported runs join the batch named with `--batch` (section 2).
- Existing datasets with no batches: on first read, every committed run with no batch goes into one batch per policy
  (source + revision), status `reviewing`. Nothing is moved; this is just grouping.
- CLI: `dataop-logger batches <root> list | show <b> | new <name> | add <b> <run>... | move <run> <b> | close <b> |
  lock <b> | unlock <b> --reason ... | rename <b> <name> | link <b> --model <source@revision>`. All print JSON with
  `--json` for agents.

A run belongs to exactly one batch. A batch lives in one dataset root. Export can combine several batches, including
batches from several roots (section 3), which covers teams that record each session into its own dataset.

**Lineage.** `link <batch> --model <source@revision>` records which model was trained from one of the batch's
exports. When a later batch's policy matches that model, its `parent_export` is set, giving the chain
round 1 -> export -> model v2 -> round 2. The viewer shows it as one line above the run list.

**Viewer.** A batch picker sits above the run list (default: the newest batch). The summary block shows that batch's
success rate, rescues, interventions and review progress (`18 of 30 runs reviewed`). Buttons: Close batch, Lock
batch, Export (section 3). A locked batch shows a lock chip and disables editing with a tooltip that says why.

## 2. Import

**Where the current example came from.** From Watcher session files: a LeRobot v3.0 dataset per session, a `.npy` of
the base model's output per episode, and a results JSON per session. `backfill_dataop.py` replayed each episode
through `RunLogger` into a new dataset. That is a script for one format; import makes it a product feature.

**What import does.** It binds dataop runs to episodes of an existing LeRobot dataset and fills the sidecar from what
the dataset (and optional side files) already contain. It never rewrites LeRobot files.

**Sources (channels).**

| Source | How |
| --- | --- |
| Local LeRobot v3.0 folder | read in place |
| LeRobot v2.1 folder | refused with the exact LeRobot command to convert it first (`convert_dataset_v21_to_v30`), so we never ship a second converter |
| Hugging Face Hub repo id | `LeRobotDataset(repo_id)` downloads it; import then requires `--into` (copy mode), since the HF cache is not ours to modify |
| Watcher session | preset `watcher` (reads `<session>_raw/` and `<session>_results_*.json`) |

**Modes.** `attach` (default): write `dataop/` into the dataset root, LeRobot files untouched. `copy` (`--into
<new_root>`): copy the dataset with LeRobot's own tools first (`delete_episodes` with an empty list, or
`merge_datasets` when several sources are combined into one root), then attach to the copy. Copy is the default
for Hub sources and for a source the user cannot write.

**Field mapping.** A small mapping (JSON) says where each sidecar field comes from. Every field is optional; anything
unmapped is recorded as unknown, never invented.

| Sidecar field | From |
| --- | --- |
| `sent_action` | the dataset's `action` feature (always) |
| `control_mode` | a per-frame bool column (LeRobot's HIL convention `complementary_info.is_intervention`: true = expert), a per-frame string column, or a per-episode takeover frame from a side file; otherwise `unknown` (new enum value for imported data) |
| `expert_source` | constant string or column |
| `base_action` | a per-frame feature (for example `complementary_info.policy_action`) or a per-episode file (`.npy` T x n, or `.parquet`/`.csv` with `frame_index`) via a path pattern with `{episode_index}` |
| `expert_action` | a feature or file as above; with an intervention flag, `action` on intervention frames |
| outcome | `next.success` on the last frame (LeRobot constant), a per-episode results file field, or `unknown` |
| `failure_mode`, task segments | per-episode results file fields, or a per-frame string column for segments |
| `reward`, `terminated` | `next.reward`, `next.done` |
| times | from LeRobot `timestamp`; `clock.obs_timestamp_kind = "from_dataset_timestamp"` so latency stats say they are not measured |

Presets bundle common mappings: `lerobot-hil` (HIL-SERL recordings), `watcher`, `plain` (actions only). Every run
records `origin: {kind: "import", source, preset, mapping digest, fields: {field: where it came from | "unknown"}}`,
and the viewer's run card shows unknown fields as data warnings ("Base action not recorded: residual targets
unavailable for this run").

**Safety.** `--dry-run` (default for agents) prints, per field, the mapped source and how many episodes have it, plus
the episodes that will be bound, skipped (already bound, matched by fingerprint) or refused (length mismatch with a
side file). Import is idempotent and resumable: an episode already bound is skipped. Validation runs at the end.

**CLI.** `dataop-logger import <source> [--into <root>] [--preset p | --map map.json] [--batch name] [--episodes
0-29] [--dry-run]`. Python: `dataop_logger.importer.import_dataset(...)` with the same arguments.

**Sharing.** `dataop-logger push <root> <repo_id>` uploads only the `dataop/` folder into the same Hub dataset repo
(`path_in_repo="dataop"`); LeRobot ignores it, and `pull` brings it back. Moving or copying the dataset folder
carries the sidecar with it because it lives inside the root.

## 3. Generating training data (export)

**No AI here.** AI helps during review (filling the schema, proposing cuts, reading notes). Export is a plain,
deterministic script: the same locked batch and the same recipe parameters always produce the same training set.
An agent may start an export, but only through the same CLI a person uses.

**Command.** `dataop-logger export --batch <root>:<batch> [--batch ...] --recipe <name> [recipe options]
--into <out_root> [--dry-run]`. Refuses unlocked batches unless `--allow-unlocked`, which is recorded.

**Output.** A new LeRobot v3.0 dataset written with LeRobot's own writer (so any LeRobot trainer reads it), plus
`<out_root>/dataop_export.json`: source batches and roots, each run's locked review rev and digest, recipe name and
parameters, logger and LeRobot versions, and per-run counts (frames kept, dropped and why). Stats are computed by
LeRobot. The export record is also appended to each source batch's `exports`.

**Frame selection is shared by every recipe.** A frame is usable only if the run's `train_mask` is true (rejected
runs, excluded spans, cuts, labels read as exclude, and the failure window on failed runs are already off). An
excluded span inside a run splits it into separate output episodes, so LeRobot's temporal sampling never jumps over a
cut. Options common to every recipe: `--segments a,b` keeps only frames inside those task segments (names from the
dataset's list), `--outcomes success,failure`, `--val 0.1` splits by run with a stable hash of `run_id`.

**Recipes.**

| Recipe | Frames | Action written |
| --- | --- | --- |
| `dagger` | expert-driven frames (`expert`/`blend`) and policy frames that have an expert label; policy frames without a label are dropped; optional `--with-success` adds the policy's own successful runs; optional `--base <root>` merges a base demo dataset in (LeRobot `merge_datasets`) | `expert_action` where valid, else `sent_action` on expert-driven frames |
| `residual` | frames with both `base_action` and `expert_action` valid; optional `--zero-on-success` adds the policy's own successful frames with target zero | `expert_action - base_action`, plus feature `observation.base_action` so the head sees what it corrects |
| `filtered` | successful runs only (plain behaviour cloning) | `sent_action` |

**Hook for task-specific relabels.** `--relabel module:function` calls user code per run with the run's frames,
sidecar and effective review, and gets back replacement actions and a reason; the export record names the hook and
its file digest. Our tube-pick hover relabel lives there, not in the package.

**Dry run and viewer.** `--dry-run` prints episodes, frames and per-run reasons without writing. The viewer's Export
button opens a short panel: recipe, the counts from a dry run, output folder, Export. The finished export appears in
the batch's export list with its path and frame count.

## Build order

1. Batches (storage, lifecycle, lock checks in every edit path, CLI, viewer picker and summary).
2. Import (mapping engine, presets `plain`, `lerobot-hil`, `watcher`, attach and copy modes, dry run, push/pull).
3. Export (shared frame selection, the three recipes, the relabel hook, export record, viewer panel).

Each step: build, test on a copy of real data, Codex review, fix, commit.
