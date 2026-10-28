---
name: dataop-logger
description: Add dataop-logger to a LeRobot recording or deployment loop so each frame records the base policy's action, the expert's action, the action actually sent, who was in control and how the run ended. Use when the user asks to log deployment runs, interventions, takeovers or rescues, or to prepare DAgger, residual or RL data from a LeRobot robot.
---

# Add dataop-logger to a robot project

You are working in the user's robot project, not in the dataop-logger repo. Your goal is a recording loop that writes a normal LeRobot dataset plus the dataop sidecar, and a passing `dataop-logger validate`. Read the whole procedure before editing anything.

## 1. Check the project fits

- Find the LeRobot version with the project's own interpreter (its venv or conda env): `python -c "import lerobot, importlib.metadata as m; print(m.version('lerobot'))"`. The logger needs 0.6.x. On another version, stop and tell the user.
- Find where frames are written. Search for `add_frame(` and `save_episode(`. Typical places:
  - `lerobot-record` (`lerobot/scripts/lerobot_record.py`, function `record_loop`) when the user records with the stock CLI.
  - A custom loop that calls `policy.select_action(...)` and `robot.send_action(...)`.
- If frames are written by ROS, a custom logger or anything other than `LeRobotDataset.add_frame`, stop and tell the user the logger does not cover that stack yet.
- The search may need the project's own variable name (`ds.add_frame(` rather than `dataset.add_frame(`).
- Appending to an existing dataset works. Episodes recorded before the logger have no run, and `validate` lists them as a warning, not an error.
- Never edit files inside the installed `lerobot` package. If the user records with the stock `lerobot-record` CLI, copy its `record_loop` into the user's project as a new script and edit the copy.

## 2. Install

```bash
pip install "git+https://github.com/Helloace-Labs/dataop-logger.git"
```

Use the same environment that runs the recording loop. Add it to the project's dependency file if there is one. The CLI is `dataop-logger`; `python -m dataop_logger.cli` is the same thing.

## 3. Describe the action once

Do this after the `LeRobotDataset` is created or opened and before the episode loop. Build the `ActionSchema` from the dataset's own action feature so names and order match exactly:

```python
from dataop_logger import ActionSchema
names = tuple(dataset.features["action"]["names"])
units = tuple("pct" if "gripper" in n else "deg" for n in names)
schema = ActionSchema(names=names, units=units, space="joint_position", rate_hz=dataset.fps)
```

Ask the user about units and space only if the names do not make them clear (SO-100/SO-101 and Koch arms: degrees for joints, percent for the gripper, `joint_position`). Use `space="custom:<name>"` for anything else.

## 4. Change the loop

Wrap recording in one `RunLogger` and each episode in `log.run(...)`. Keep `dataset.finalize()` after the `RunLogger` block.

```python
import time
from dataop_logger import RunLogger, PolicyInfo, EnvInfo

with RunLogger(dataset, schema) as log:
    for episode in range(num_episodes):
        with log.run(task=task, policy=PolicyInfo(family="smolvla", source=policy_path),
                     env=EnvInfo(robot_id=robot.name)) as run:
            ...  # the existing per-frame loop, changed as below
            run.end(termination_reason="success")   # see step 6
```

`PolicyInfo.family` and `source` are free text: the model type (`smolvla`, `act`, `pi05`, or `scripted` for a hand-written controller) and where it came from (a hub repo id, a checkpoint path, or a file and function name).

Inside the per-frame loop, replace `dataset.add_frame(frame)` with `run.add_frame(...)`, and take two clock readings:

```python
obs = robot.get_observation()
t_obs = time.monotonic_ns()
...
sent = robot.send_action(action)            # LeRobot robots return the action dict actually sent (after clipping)
t_sent = time.monotonic_ns()
sent_arr = [sent[n] for n in names]         # dict -> array in the schema's order
run.add_frame(frame, sent_action=sent_arr, base_action=policy_arr, control_mode="base",
              inference_id=k, chunk_offset=j, obs_ns=t_obs, sent_ns=t_sent)
```

What each mode needs (`sent_action`, `obs_ns` and `sent_ns` are always required):

| `control_mode` | Also required | Optional |
| --- | --- | --- |
| `base` | `base_action`, `inference_id`, `chunk_offset` | `expert_action` + `expert_source` (an expert label that was not executed) |
| `expert` | `expert_action`, `expert_source` | `base_action` + `inference_id` + `chunk_offset` (shadow policy) |
| `blend` | `base_action`, `expert_action`, `expert_source`, `blend_weight` in [0, 1], `inference_id`, `chunk_offset` | |
| `none` | nothing | |

`inference_id` and `chunk_offset` are given exactly when `base_action` is given.

Rules the logger enforces, so get them right the first time:

- `frame` is the same dict the loop used to pass to `dataset.add_frame`: the observation keys and `"task"`. Leave the action out; the logger writes `sent_action` into the dataset's action column itself.
- Actions are 1-D arrays (or lists) of floats in the order of `names`. A LeRobot action dict such as `{"shoulder_pan.pos": ...}` must be converted to that order first.
- `sent_action` is the command that went to the motors.
- `base_action` is the policy's output for this frame after postprocessing, even when someone else is driving (run the policy in shadow if the loop allows it). When there is a `base_action`, also pass `inference_id` (which policy call produced it) and `chunk_offset` (its position in that call's action chunk). `inference_id` counts policy calls from 0 within the run. For a policy that returns one action per call, it is the frame index within the episode and `chunk_offset` is always 0.
- `control_mode`: `"base"` when the policy drove; `"expert"` with `expert_action` and `expert_source` (`"human"` or `"skill:<name>"`) when a person or script drove; `"none"` while idle or resetting. Read the project's own takeover signal (a teleop button, an intervention flag, a rescue state) to choose it. If the project has no such signal, every frame is `"base"`.
- `obs_ns <= sent_ns`, and both never go backwards. Reading the clock right after `get_observation()` returns matches the logger's default `obs_timestamp_kind="bundle_completion"`. Pass `obs_timestamp_kind="capture"` to `RunLogger` only if `obs_ns` comes from the camera's own capture time.
- Delete the old `dataset.save_episode()` call. Leaving the `run` block saves the episode. Replace `dataset.clear_episode_buffer()` (re-record) with `run.discard()`.
- Do not call `dataset.add_frame`, `save_episode` or `clear_episode_buffer` anywhere while a run is open.
- Reset phases between episodes stay outside `log.run(...)`, or are recorded with `control_mode="none"`.

## 5. Mark moments worth finding

Where the project detects a takeover, failure or anything unusual, add an event:

```python
run.event("takeover", source="human", reason="tube slipping")
```

Call it just before adding the first frame it describes; it is attached to the last frame already added. `kind`, `source` and `reason` are free text. Takeovers are also derived from `control_mode` on their own, so the event is for search and context.

If the project already detects task steps (grasped, lifted, placed), mark them with `run.subgoal(name, "started" | "completed" | "failed", reason=None)`: "started" just before a step's first frame, "completed" or "failed" just after its last. Do not invent step detection the project does not have.

## 6. End each run truthfully

Call `run.end(termination_reason=...)` with one of `success`, `failure`, `timeout`, `operator_abort` or `system_error`, from whatever the project already knows (a success detector, a timeout, the operator's key). Pass `task_success=`, `labels=[...]` and `failure_mode="<how it failed>"` when known. If the project has no success signal, ask the user how a run's outcome is decided. If you cannot ask, pick the closest rule the code supports (for example, a run with a takeover is a `failure`), write it as a comment next to `run.end`, and list it in your report as an assumption.

If an exception escapes the `run` block, or the block ends without `run.end()`, the episode is discarded from both LeRobot and the sidecar, never half-saved.

Rewards are optional. Pass `reward=` on `add_frame`, or call `run.complete_transition(reward=...)` when the reward arrives one loop late.

## 7. Verify

1. Record two or three short episodes, or ask the user to, if the robot is not under your control.
2. Run `dataop-logger validate <dataset_root>`. It must print `OK`. Fix any error it reports before going further.
3. Run `dataop-logger inspect <dataset_root>` and check that the outcomes, takeovers and minutes per control mode match what happened.
4. Optionally run `dataop-logger find-idle <dataset_root>` so the viewer opens with idle stretches suggested as cuts.
5. Tell the user they can browse the runs with `dataop-logger view <dataset_root>`: it starts a local web server and opens a page with each run's outcome, who drove each frame, and the camera videos. Do not run it yourself unless the user asks; it keeps running until stopped.

## Reading people's labels

People label spans in the viewer in plain language. When the user asks you to read them, or before you build a training set:

1. Run `dataop-logger labels <dataset_root> --uninterpreted`. Each entry has the label's text, driver and segment, and what was recorded in that span (control modes, task steps, the run's outcome).
2. Decide for each label whether its frames should train the model. Read `exclude` into anything that says the frames are wrong, unsafe, blocked, a mistake or not to be used. Read `keep` into descriptions of normal or good behaviour, segment names and driver corrections.
3. Record it: `dataop-logger interpret <dataset_root> <run_id> <label_id> --train keep|exclude --meaning "<one short sentence>"`.
4. If a label is ambiguous, ask the user instead of guessing, and quote it.

## Suggesting cuts (AI features)

When the user asks you to find frames that should not be trained on (idle stretches, resets, camera problems), suggest them instead of cutting them: `dataop-logger suggest-cut <dataset_root> <run_id> <start> <end> --reason "<short description>" --by <your name>`. Frames are half-open `[start, end)`. Suggestions show in the viewer in light grey for the person to cut or keep. `dataop-logger find-idle` runs the built-in idle finder if you only need that.

## Report back

Tell the user which file you changed, where the takeover signal comes from, how outcomes are decided, and the `validate` result. Mention anything you could not map (for example no shadow policy during takeovers, so `base_action` is missing on expert frames).

Full reference: https://github.com/Helloace-Labs/dataop-logger (README.md and SPEC.md).
