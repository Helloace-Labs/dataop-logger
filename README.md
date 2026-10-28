# Aftershock

The post-training data workspace for robotics.

`dataop-logger` is Aftershock's open-source logger and local viewer. The package, the `dataop-logger` command and the
`dataop/` sidecar folder keep their current names for now, so existing datasets keep working.

Record what post-training needs, next to a normal [LeRobot](https://github.com/huggingface/lerobot) dataset.

LeRobot saves what the robot did. To build DAgger, residual-network or RL datasets from real deployment runs you also need, for every frame:

- what the **base policy** proposed (even while a human or rescue skill was driving),
- what the **expert** did or would have done,
- the action **actually sent**, and **who was in control**,
- timing, an optional **reward**, and **how the run ended**.

dataop-logger records all of that in a small sidecar folder. Your dataset stays a plain LeRobot dataset that loads and trains anywhere.

## Install

```bash
pip install "git+https://github.com/Helloace-Labs/dataop-logger.git"
```

Requires Python 3.10+ and `lerobot` 0.6.x (tested with 0.6.2). PyTorch is only needed for the reader.

## Set it up with your coding agent

Give Claude Code, Codex, Cursor or any coding agent this prompt from inside your robot project:

```text
Read https://raw.githubusercontent.com/Helloace-Labs/dataop-logger/main/skills/dataop-logger/SKILL.md
and add dataop-logger to my recording loop. Validate it when done.
```

The agent checks your LeRobot version, finds your recording loop, wires in the logger and runs `dataop-logger validate`. Claude Code users can also install it as a skill by copying `skills/dataop-logger/` into `~/.claude/skills/`.

## Quick start

Three changes to an existing LeRobot recording loop: describe your action once, record through `run.add_frame`, and let the `run` block save the episode.

```python
import time
from dataop_logger import ActionSchema, EnvInfo, PolicyInfo, RunLogger

schema = ActionSchema(
    names=("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
    units=("deg",) * 5 + ("pct",),
    space="joint_position",          # or joint_velocity, ee_delta, ee_pose, custom:<name>
    rate_hz=30,
)

with RunLogger(dataset, schema) as log:              # dataset: a LeRobotDataset opened for writing
    with log.run(task="pick up the tube",
                 policy=PolicyInfo(family="smolvla", source="me/smolvla_tube"),
                 env=EnvInfo(robot_id="arm-1", extras={"light": "daylight"})) as run:
        for step in range(max_steps):
            obs = robot.get_observation()
            t_obs = time.monotonic_ns()
            action = policy.select_action(obs)        # in robot command space, after postprocessing
            robot.send_action(action)
            t_sent = time.monotonic_ns()
            run.add_frame(frame, sent_action=action, base_action=action, control_mode="base",
                          inference_id=step, chunk_offset=0, obs_ns=t_obs, sent_ns=t_sent)
        run.end(termination_reason="success")         # success, failure, timeout, operator_abort, system_error
```

`run.add_frame` replaces `dataset.add_frame`, and leaving the `run` block replaces `dataset.save_episode`. Do not call those two LeRobot methods yourself while a run is open; the logger checks and refuses to save if the buffer was changed behind its back.

## Recording takeovers

When a person or a scripted skill takes over, keep running the base policy in shadow if you can, and log both actions:

```python
run.add_frame(frame,
              sent_action=expert_action,      # what the motors got
              expert_action=expert_action,    # the expert's label
              base_action=shadow_action,      # what the policy would have done (optional, but needed for residuals)
              inference_id=k, chunk_offset=j, # only when base_action is given
              control_mode="expert", expert_source="human",   # or "skill:rescue_v3"
              obs_ns=t_obs, sent_ns=t_sent)
```

| `control_mode` | Meaning | Required |
| --- | --- | --- |
| `base` | The policy drove | `base_action` |
| `expert` | A human or skill drove | `expert_action`, `expert_source` |
| `blend` | A mix of both | `base_action`, `expert_action`, `expert_source`, `blend_weight` in [0, 1] |
| `none` | Nobody (idle, resetting) | nothing |

You can also log an expert label without executing it (`control_mode="base"` with an `expert_action` and `expert_source`), which is what DAgger queries look like.

Typed events mark moments worth finding later:

```python
run.event("takeover", source="human", reason="tube slipping")
```

## Task steps and failure modes

Pass-fail hides why a run failed. Mark the task's steps as they happen, and name the failure when the run ends:

```python
run.subgoal("grasp", "started")          # just before the step's first frame
...
run.subgoal("grasp", "completed")        # just after its last frame; the next step starts here
run.subgoal("lift", "failed", reason="tube slipped")
run.end(termination_reason="failure", failure_mode="dropped_during_lift")
```

The manifest gets a `subgoals` list of frame spans with their status. `inspect` counts failure modes and failed steps across runs, and the viewer groups failed runs by failure mode and draws each run's steps above its timeline. Steps can come from anything your stack already knows: a success detector, gripper and height thresholds, or a person's label.

## Rewards for RL

Pass `reward=` on `add_frame`, or, if your reward arrives one loop late, attach it to the previous tick:

```python
run.complete_transition(reward=r, terminated=False, truncated=False)
```

Row `t` is `(observation_t, sent_action_t, reward_t)`. `run.end()` sets the final tick's `terminated` (success, failure) or `truncated` (timeout, operator_abort, system_error) flag and rejects contradictions. A run with a reward on every tick is marked `rl_ready`.

## Action chunks (optional)

For chunking policies such as ACT or SmolVLA, log each inference's full chunk:

```python
with log.run(task="...", log_predictions=True) as run:
    run.predictions(inference_id=k, action_chunk=chunk, obs_frame_index=i,
                    infer_start_ns=t0, infer_end_ns=t1)
```

## Re-recording and errors

- `run.discard()` drops the current episode, both LeRobot's buffer and the sidecar.
- An exception inside the `run` block discards the episode. Leaving the block without `run.end()` is an error, never a silent save.
- If `save_episode` fails, the run is kept for repair and the logger refuses further runs until you reopen the dataset.

## Command line

```bash
dataop-logger validate <dataset_root>            # nonzero exit on any problem; --json for a report
dataop-logger validate <dataset_root> --repair   # commit runs that match a saved episode, quarantine the rest
dataop-logger inspect  <dataset_root>            # runs, outcomes, takeovers, minutes per control mode
dataop-logger remap <old_root> <new_root> --repo-id ID --revision REV   # after filtering or renumbering episodes
```

`validate` checks that every run matches its LeRobot episode frame for frame and byte for byte, that the episode was fully saved (metadata and video files present), that the sidecar files match their recorded checksums, and every rule above.

## Viewer

```bash
dataop-logger view <dataset_root>      # opens http://127.0.0.1:8765/dataop/view/
```

A local page: every run with its outcome and a strip showing who drove each frame, then the selected run's camera videos in sync with a control-mode timeline. Click the timeline or an event to jump to that moment. Nothing leaves your machine.

The recorded data is never changed. Labels and cuts you make in the page are saved to a separate file next to each run (see below). Use `--read-only` for a page that cannot save anything, and `--build-only` to write `dataop/view/index.html` without starting a server.

![viewer](docs/viewer.png)

When runs carry `failure_mode` and `subgoals`, the run list opens with failed runs grouped by failure mode. Click a group to filter the list.

## Editing runs

The run view has three rows: task segments, drive mode (who drove), and the playhead row. Only the playhead row moves the playhead. Press **Edit** (or E) to correct the first two:

- **Task segments:** drag the handles on the segment lane. The line between two segments moves both sides together, and an outer start or end can move up to its neighbour.
- **Drive mode:** drag the handle where control switches (for example policy to rescue) to where the switch really happened.
- **Policy fails:** drag the red marker to where the policy actually started to fail. The two seconds before it are underlined.

While you drag, tables below the bar update live. They show each segment's and drive span's start, end, length and frames, the change from the recording, and when the policy fails. Cut frames stay in the recording but are left out of training; click a cut to undo it.

**Lists (the dataset's schema).** Each dataset keeps two agreed lists in `dataop/lists.json`: **failure types** and **task segments**. People and agents pick from them, so counts stay comparable across runs. The first time the lists are read, they are seeded from the names the dataset already uses. In the viewer you never see the lists as such; in Edit you use and grow them in place:
- click a task segment to change its type, add a new type, split it at the playhead or remove it;
- click an empty part of the segment lane to add a segment there;
- pick or add a failure type next to the failure row.

The × on a type removes it from the dataset when no run uses it. A rename (from the command line) keeps the old name as an alias, so recorded data still matches.

Agents use the same lists:

```bash
dataop-logger lists <dataset_root>                                       # JSON
dataop-logger lists <dataset_root> add failure_types collision --description "hit something"
dataop-logger lists <dataset_root> rename task_segments lift raise
```

**Agents fill in the same schema.** `dataop-logger runs <dataset_root>` returns each run's segments, drive mode, failure moment and type, and video files as JSON. An agent that watched the video writes its reading back with `dataop-logger annotate <dataset_root> <run_id> --segments ... --failure-at ... --failure-type ... --takeover-at ... --by my-agent`. Names must come from the lists. Each change shows who set it and is one undoable step in the viewer.

**AI feature: find idle frames.** In Edit, **✦ Find idle frames** runs the built-in idle finder on the run. It marks stretches where the arm barely moves for over a second in light grey, each with a short description. You can cut them all, or cut or keep them one by one. Your own agent can make the same kind of suggestion:

```bash
dataop-logger find-idle <dataset_root> [--run RUN_ID]                     # built-in finder, every run
dataop-logger suggest-cut <dataset_root> <run_id> <start> <end> --reason "..." --by my-agent
```

**Task segments and the failure moment.** The lane above the run bar shows the task segments, each in its own shade of one color. In Edit:
- drag the line between two segments to move where one ends and the next starts;
- click a segment to select its frames for a label or a cut;
- drag the red "fails" marker to where the policy actually started to fail. It is dashed and reads "fails?" while it is only estimated.

The two seconds before the failure are shaded, since those frames matter most for training data. Agents can list a run's drive-mode changes, segment starts and failure moment with `dataop-logger events`, and add a detail to any of them with `dataop-logger note`.

**Agents read the labels.** An agent records what each plain-language label means for training, and the viewer shows that reading under the label:

```bash
dataop-logger labels <dataset_root> --uninterpreted      # JSON: each label with what was recorded in its span
dataop-logger interpret <dataset_root> <run_id> <label_id> --train exclude --meaning "camera blocked by a hand" --by my-agent
```

An existing reading is never overwritten silently: pass `--replace`, and the old one is kept in history. Changing a label's text or span clears its reading until an agent reads it again.

**Where it is stored.** Labels, cuts and readings live in `dataop/runs/<run_id>/review.json`. That file travels with the dataset and never touches the recorded frames, manifest or checksums. The reader sets `item["dataop.train_mask"]` to False for cut frames and for labels read as `exclude`, and every target mask is gated on it. `inspect` reports labels, cut frames and open suggestions. Scripts can also record structured corrections in the same file through `dataop_logger.review` (outcome, failure start, step boundary, reject run); the viewer lists them with an Undo button.

## Batches: one round of collection

A batch is the set of runs collected with one policy version and reviewed together. In DAgger it is one round:
collect with model v3, review, generate the training set, train v4, open the next batch.

```python
RunLogger(dataset, action_schema=schema, batch="round 3")     # new runs join "round 3"
```

Without `batch=`, runs join the batch that is collecting, or a new batch named after the policy. Runs recorded before
batches existed are grouped by policy the first time the dataset is read.

```bash
dataop-logger batches <root> list                      # every batch with runs, success, rescues, review progress
dataop-logger batches <root> new "round 4"              # new runs go here; the old batch closes for review
dataop-logger batches <root> close "round 3"            # no new runs; review continues
dataop-logger batches <root> lock "round 3"             # freeze the review so a training set can be made from it
dataop-logger batches <root> unlock "round 3" --reason "fix a label"   # earlier exports are marked out of date
dataop-logger batches <root> edit "round 3" --model me/smolvla_v4      # the model trained from it (lineage)
```

Locking needs every run marked reviewed (the "Mark reviewed" button in the viewer) and no open suggestions, or
`--force`, which is recorded. While a batch is locked, its runs cannot be edited from the viewer, the CLI or an agent.
The viewer has a batch picker above the run list with the batch's progress and these buttons.

## Import an existing LeRobot dataset

Bring data recorded without the logger, so it can be reviewed and exported the same way:

```bash
dataop-logger import data/my_runs --dry-run                     # what would be bound, field by field
dataop-logger import data/my_runs --preset lerobot-hil --batch "hil 1"
dataop-logger import me/hub_dataset --into data/hub_copy         # a Hub dataset is copied first
```

Import reads LeRobot v3.0 (convert v2.1 with LeRobot's own `convert_dataset_v21_to_v30` first) and never rewrites
LeRobot files; the sidecar goes into `dataop/` inside the dataset folder, so moving or copying the folder carries it.
A mapping says where each field comes from; anything without a source is stored as unknown, never guessed:

| Preset | Who drove | Policy action | Outcome |
| --- | --- | --- | --- |
| `demos` | a person, every frame | none | success |
| `plain` | unknown | unknown | unknown |
| `lerobot-hil` | `complementary_info.is_intervention` | the sent action on policy frames | `next.success` on any frame |
| `watcher` | the `intervention` column | `<session>_raw/epNNN.npy`, one row per policy frame | `<session>_results_*.json` |

The default `--preset auto` picks one from the dataset's columns and side files. The Watcher preset also takes each
episode's time from its raw file, marked as estimated. Pass `--map map.json` to map your own columns and side files (format in `src/dataop_logger/importer.py`). Import is
idempotent: episodes that already have a run are skipped.

In the viewer, **Import** lists the LeRobot datasets on this machine (next to the ones the viewer knows and in
LeRobot's download cache), takes a Hub id too, shows the detected type and counts, and imports in the background.
The dataset picker in the header switches between imported datasets.

## Generate training data

Training sets are made by a plain script. No AI runs here, and the same runs with the same options always write the
same data. In the viewer, **Create training set** asks for the type, then the runs (any dataset, any batch, or single
runs), shows the counts, and writes the set:

| Type | What goes in |
| --- | --- |
| Teleop | runs a person drove from start to finish |
| DAgger | the rescues inside policy runs, labelled with the expert's action |
| LoRA | Teleop and DAgger together, for a LoRA fine-tune of the base model |

Columns that only some of the chosen datasets have (an `intervention` flag, say) are left out so they combine. A set
made from a locked batch stays current while the lock holds; one made from unlocked batches shows "out of date" as
soon as a run in it is reviewed differently. From the command line:

```bash
dataop-logger export --batch data/round3:"round 3" --recipe dagger --into data/round3_dagger
dataop-logger export --batch data/round3:"round 3" --recipe residual --zero-on-success --into data/round3_residual
dataop-logger export --batch data/round3:"round 3" --recipe filtered --val 0.1 --into data/round3_bc
```

| Recipe | Frames | Action written |
| --- | --- | --- |
| `teleop` | successful runs with no policy frames | the sent action |
| `lora` | `teleop` runs plus `dagger` frames from policy runs | as those recipes |
| `dagger` | frames with an expert label in runs the policy drove (`--with-success` adds the policy's successful runs, `--base` adds a demo dataset) | the expert's action |
| `residual` | frames with both the policy's and the expert's action (`--zero-on-success` adds good policy frames with target zero) | expert minus policy, plus `observation.base_action` |
| `filtered` | successful runs | the sent action |

Every recipe skips what the review left out (cuts, rejected runs, the failure window), can be limited with
`--segments` and `--outcomes`, and cuts each run into contiguous stretches that become separate LeRobot episodes, so
training never samples across a gap. The output is a normal LeRobot v3.0 dataset written by LeRobot itself, with
`dataop_export.json` recording the batches, review digests, source file digests and options it came from. Use
`--dry-run` to see per-run counts and reasons first, and `--hook file.py:function` for task-specific selection.
`--runs id1,id2` limits a batch to chosen runs.

## Training with the extra fields

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from dataop_logger.reader import DataopDataset

ds = DataopDataset(LeRobotDataset("me/tube_runs", root="data/tube_runs"), missing="error")
item = ds[0]
item["action"]                        # the usual LeRobot fields are unchanged
item["dataop.dagger_label"], item["dataop.dagger_mask"]                        # expert action
item["dataop.expert_residual"], item["dataop.expert_residual_mask"]            # expert - base
item["dataop.executed_residual"], item["dataop.executed_residual_mask"]        # sent - base, safety-modified ticks masked
```

Every per-tick field is also available as `item["dataop.<field>"]`. `missing="exclude"` skips episodes recorded without the logger, and `missing="mask"` keeps them with all masks false. Episode-filtered LeRobot views are supported and keep LeRobot's order.

## What gets written

```
<dataset_root>/dataop/
  runs/<run_id>/manifest.json       # outcome, takeovers, events, policy, environment, action schema, checksums
  runs/<run_id>/ticks.parquet       # one row per frame
  runs/<run_id>/predictions.parquet # only with log_predictions=True
  index.jsonl                       # summary of committed runs
  quarantine/                       # runs repair could not match, kept for inspection
```

The full field list and design are in [SPEC.md](SPEC.md).

## Limits of v0.1

- One writer process per dataset, local filesystem only.
- One task per run.
- `sent_action` is the final command after your safety filters; the logger does not record a separate pre-filter command or the actuator's measured action.
- One observation timestamp per frame, not per camera.
- Validation checks video files exist; it does not decode them.

## How this was built

The design and code went through five rounds of review between Claude (Anthropic) and Codex (OpenAI). Every review and the decision taken on each finding are in [reviews/](reviews/).

## License

Apache 2.0
