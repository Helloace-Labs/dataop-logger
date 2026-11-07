"""RunLogger: records provenance next to a LeRobot dataset, owning the add/save ordering so the two cannot drift.

State machine per run: active -> staged -> committed, with terminal discarded, save_failed and quarantined.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
import shutil
import threading
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np
import pyarrow as pa

from . import adapter, batches, store
from .fingerprint import fingerprint
from .schema import (LIVE_CONTROL_MODES as CONTROL_MODES, SCHEMA_VERSION, SUBGOAL_STATUSES, TERMINAL_FLAG, LIVE_TERMINATION_REASONS as TERMINATION_REASONS,
                     UNKNOWN, ActionSchema, EnvInfo, PlannerInfo, PolicyInfo, planner_value, task_id_value,
                     task_rev_value, TickSequence, tick_values)

from importlib.metadata import PackageNotFoundError, version as _pkg_version  # noqa: E402  (also used by _lerobot_version)


def _own_version() -> str:
    """Recorded in manifests (under the key `dataop_logger`, which keeps its name so old readers still find it)."""
    for dist in ("aftershock-logger", "dataop-logger"):  # new distribution name, then the old one
        try:
            return _pkg_version(dist)
        except PackageNotFoundError:
            continue
        except Exception:
            break
    return "0.1.0+src"                                   # running from a source checkout


_DATAOP_VERSION = _own_version()

_FROM_LOGGER = object()            # run(planner=...) not given: use the RunLogger's planner
_AUTO = object()                   # add_frame(subtask_cmd=...) not given: None without a planner, else unknown

PRED_SCHEMA = pa.schema([("inference_id", pa.int64()), ("obs_frame_index", pa.int64()),
                         ("infer_start_ns", pa.int64()), ("infer_end_ns", pa.int64()), ("horizon", pa.int64()),
                         ("valid_len", pa.int64()), ("action_chunk", pa.list_(pa.float32()))])


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _lerobot_version() -> str | None:
    try:
        return _pkg_version("lerobot")
    except Exception:
        return None


class RunError(RuntimeError):
    pass


def default_spill_dir() -> Path:
    """A persistent per-user folder: $XDG_DATA_HOME/aftershock-logger/spill, else ~/.local/share/aftershock-logger/spill.
    It survives reboots, but is usually on the same disk as the dataset; set RunLogger(spill_dir=...) on a robot."""
    import os
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base) / "aftershock-logger" / "spill"


class RunLogger:
    """One per dataset root. Holds the single-writer lock until close()."""

    def __init__(self, dataset, action_schema: ActionSchema, obs_timestamp_kind: str = "bundle_completion",
                 action_key: str = "action", batch: str | None = None,
                 planner: PlannerInfo | dict | None | str = UNKNOWN, spill_dir: str | None = None,
                 live: bool = False):
        """batch: the batch new runs join (opened if it does not exist). Without it, runs join the batch that is
        collecting, or a new batch named after the policy.

        planner: the planner that picks subtasks in every run of this writer: PlannerInfo(name, id), None when no
        planner runs, or "unknown" (the default) when the caller does not say. A run can override it.

        spill_dir: where close() writes a stopped run's frames when they still cannot be written to the dataset's
        quarantine folder. On a robot, set it to a persistent folder on a different disk from the dataset. The
        default is a per-user folder (default_spill_dir()), which is usually on the same disk as the dataset.

        live: turn on the live tap (logger task L10): a background thread writes the run's live status to
        dataop/live/ for a local reader such as a station, and camera previews only while that reader asks for them.
        Off by default; it never changes what is recorded and its errors never reach the caller. See live.py."""
        if obs_timestamp_kind not in ("capture", "bundle_completion"):
            raise ValueError("obs_timestamp_kind must be 'capture' or 'bundle_completion'")
        self.dataset = dataset
        self.schema = action_schema
        self.root = dataset.root
        self.action_key = action_key
        self.obs_timestamp_kind = obs_timestamp_kind
        self.batch = batch
        self.planner = planner_value(planner)
        self.broken: str | None = None                   # set after a mid-save failure; no more runs
        self._lock = store.WriterLock(self.root)
        self._active: Run | None = None
        self.spill_dir = spill_dir
        self._unsaved_runs: list[Run] = []                # stopped runs whose frames are only in memory
        self._stuck_writers: list = []                    # frame writer threads still inside LeRobot's add_frame
        self._lock_handed_off = False
        self.live_tap = None
        if live:
            from .live import LiveTap
            self.live_tap = LiveTap(self)

    def close(self) -> None:
        """Release the writer lock. A stopped run whose frames are still only in memory is written to quarantine
        once more, then to spill_dir; a RunError names where they went, or that they were not written at all.

        A frame writer that timed out may still be inside LeRobot's add_frame and can still change the dataset's
        files. The lock is then kept until that thread returns (a background thread releases it), so no other
        writer can open the dataset meanwhile, and the RunError says so. A live tap thread that has not finished
        within its short close wait keeps the lock the same way, so it can never write over a newer writer's live
        files; that case raises nothing."""
        problems = []
        try:
            if self._active is not None and self._active.state == "active":
                self._active.discard()
        finally:
            stuck = [t for t in self._stuck_writers if t.is_alive()]
            if stuck:
                problems.append(f"a frame writer is still inside LeRobot's add_frame, so {self._lock.path} stays "
                                "locked until it returns (or this process ends); no other writer can open the "
                                "dataset meanwhile")
            problems += self._spill_unsaved()
            tap_thread = None
            if self.live_tap is not None:
                self.live_tap.close()                  # the last status says closed, before the lock goes
                t = self.live_tap._thread
                tap_thread = t if t is not None and t.is_alive() else None
            waiting = stuck + ([tap_thread] if tap_thread is not None else [])
            if not waiting:
                if not self._lock_handed_off:
                    self._lock.release()
            elif not self._lock_handed_off:
                self._lock_handed_off = True
                lock = self._lock

                def release_when_done():
                    for t in waiting:
                        t.join()
                    lock.release()
                threading.Thread(target=release_when_done, name="dataop-lock-release", daemon=True).start()
        if problems:
            raise RunError("; ".join(problems))

    def _spill_unsaved(self) -> list[str]:
        problems = []
        for run in list(self._unsaved_runs):
            try:
                run.retry_quarantine()
                continue
            except Exception as e:
                first = f"{type(e).__name__}: {e}"
            from .frame_queue import write_partial
            spill = Path(self.spill_dir) if self.spill_dir else default_spill_dir()
            rows, observations, info = run._unsaved_quarantine
            try:
                spill.mkdir(parents=True, exist_ok=True)
                path = write_partial(run, rows, observations, info, qdir=spill)
            except Exception as e:
                problems.append(f"run {run.run_id}: {len(rows)} frames were NOT written anywhere (quarantine: {first}; "
                                f"spill folder {spill}: {type(e).__name__}: {e}); they are lost when this process ends")
                continue
            run.quarantine_path, run._unsaved_quarantine = path, None
            self._unsaved_runs.remove(run)
            if run._stop is not None:
                run._stop.quarantine_path, run._stop.spilled = path, True
                run._stop._settle()
                run._live("controlled_stop", run._stop)
            hint = "" if self.spill_dir else (" (the default spill folder, usually on the same disk as the dataset; "
                                              "set spill_dir to a persistent folder on a different disk)")
            problems.append(f"run {run.run_id}: {len(rows)} frames could not be written to the dataset's quarantine "
                            f"({first}) and were spilled to {path}{hint}; move that folder to "
                            f"{store.quarantine_dir(self.root)} when there is space")
        return problems

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        """A close() error never hides the exception that is leaving the block: a ControlledStop gets it attached
        (close_error, and in its message) and a warning repeats it; the original exception keeps propagating."""
        try:
            self.close()
        except RunError as e:
            if exc is None:
                raise
            from .frame_queue import ControlledStop
            if isinstance(exc, ControlledStop):
                exc.close_error = str(e)
                exc._settle()
            warnings.warn(f"closing the logger: {e}")
        return False

    @contextmanager
    def run(self, task: str, policy: PolicyInfo | None | str = UNKNOWN, env: EnvInfo | None = None,
            log_predictions: bool = False, task_id: str = UNKNOWN, task_rev: int | str = UNKNOWN,
            planner=_FROM_LOGGER):
        """Record one episode. Call run.end(...) inside the block; saving and commit happen on exit.

        policy: the PolicyInfo that drove, None when no policy ran (a teleop demo; then no frame may carry a
        base_action), or "unknown" (the default) when the caller does not say.
        task_id and task_rev name the Aftershock task and its revision (its subtask list); "unknown" when not given.
        planner overrides the RunLogger's planner for this run."""
        if self.broken:
            raise RunError(f"this logger stopped after a failed save ({self.broken}); "
                           "run `aftershock-logger validate --repair` and open a new dataset writer")
        if self._active is not None and self._active.state in ("active", "staged"):
            raise RunError("a run is already in progress on this dataset")
        if not isinstance(task, str) or not task:
            raise ValueError("task must be a non-empty string")
        if not (policy is None or isinstance(policy, PolicyInfo) or (isinstance(policy, str) and policy == UNKNOWN)):
            raise ValueError("policy must be a PolicyInfo, None (no policy ran) or 'unknown'")
        planned = {"task_id": task_id_value(task_id), "task_rev": task_rev_value(task_rev),
                   "planner": self.planner if planner is _FROM_LOGGER else planner_value(planner)}
        try:
            batches.target_for(self.root, self.batch)        # refuse before recording into a closed batch
        except batches.BatchError as e:
            raise RunError(str(e)) from None
        if adapter.has_pending_frames(self.dataset):
            raise RunError("the dataset has unsaved frames; save or clear the episode buffer before starting a run")
        r = Run(self, task, policy, env, log_predictions, planned)
        self._active = r
        if self.live_tap is not None:
            self.live_tap.run_started(r)
        try:
            yield r
        except BaseException as exc:
            if r.state == "active" and r._queue is not None and r._queue.stopped is not None:
                stop = r._quarantine_stopped()                 # keep every accepted frame, never discard them
                if exc is not stop:
                    warnings.warn(f"{stop}; frames: {stop.quarantine_path or stop.quarantine_error}")
                raise
            if r.state == "active":
                r.discard()
            elif r.state == "staged":
                r._abandon_staged("exception after run.end(); the episode was not saved")
            raise
        if r.state == "discarded":
            return
        if r.state == "active" and r._queue is not None and r._queue.stopped is not None:
            raise r._quarantine_stopped()                      # the caller caught ControlledStop and left the block
        if r.state != "staged":
            r.discard()
            raise RunError("the run block ended without run.end(); the episode was discarded, not saved")
        r._save_and_commit()


class Run:
    def __init__(self, logger: RunLogger, task: str, policy: PolicyInfo | None | str, env: EnvInfo | None,
                 log_predictions: bool, planned: dict | None = None):
        self.logger = logger
        planned = planned or {"task_id": UNKNOWN, "task_rev": UNKNOWN, "planner": logger.planner}
        self.task_id, self.task_rev, self.planner = planned["task_id"], planned["task_rev"], planned["planner"]
        self._seq = TickSequence(self.planner)
        self.dataset = logger.dataset
        self.schema = logger.schema
        self.task = task
        self.policy = policy
        self.env = env or EnvInfo()
        self.run_id = str(uuid.uuid4())
        self.state = "active"
        self.started_at = _now_iso()
        self.ended_at = None
        self.episode_index = adapter.pending_episode_index(self.dataset)
        self.run_path = store.runs_dir(logger.root) / self.run_id
        self._rows: list[dict] = []
        self._events: list[dict] = []
        self._predictions: list[dict] | None = [] if log_predictions else None
        self._prediction_ids: set[int] = set()
        self._end: dict | None = None
        self._integrity: dict | None = None
        self._count = 0                                    # frames accepted (queued or written)
        self._last: dict | None = None                     # the last accepted tick row
        self._queue = None                                 # a FrameQueue when frames go through run.queue()
        self.quarantine_path = None                        # set when this run's partial record is quarantined
        self._unsaved_quarantine = None                    # a partial record that could not be written yet
        self._blocks_logger = False                        # True while that record stops the logger
        self._stop = None                                  # the ControlledStop that ended this run
        self._session: dict | None = None                  # manifest fields set by a SessionLogger (logger L9)

    def _live(self, hook: str, *args) -> None:
        """Tell the live tap, if any (its hooks never raise)."""
        tap = self.logger.live_tap
        if tap is not None:
            getattr(tap, hook)(self, *args)

    # ── recording ────────────────────────────────────────────────────────────

    def _require(self, *states):
        if self.state not in states:
            raise RunError(f"run is {self.state}; this call needs it to be {' or '.join(states)}")

    def add_frame(self, frame: dict, *, sent_action, control_mode: str, obs_ns: int, sent_ns: int,
                  base_action=None, expert_action=None, expert_source: str | None = None,
                  blend_weight: float | None = None, safety_modified: bool = False, obs_skew_ns: int = -1,
                  inference_id: int = -1, chunk_offset: int = -1, reward: float | None = None,
                  terminated: bool = False, truncated: bool = False, prompt: str = UNKNOWN,
                  prompt_rev: int | str = UNKNOWN, subtask_cmd=_AUTO, control_gen: int | str = UNKNOWN,
                  camera_ns=None) -> int:
        """Add one frame to LeRobot first, then the sidecar row. Returns the frame index.

        obs_ns and sent_ns are required monotonic-clock readings (time.monotonic_ns()) taken by the caller.
        Optional, "unknown" when not given: prompt (the text the policy received on this frame), prompt_rev (counts
        prompt changes), control_gen (the command generation counter) and subtask_cmd (the subtask id the planner
        commanded, or None when no planner commanded one).
        camera_ns: optional {camera: capture time from time.monotonic_ns()}, used only by the live tap (REC-04 while
        recording: a camera whose capture time stops moving has a gap). It is not recorded in the sidecar.
        """
        self._require("active")
        if self._queue is not None:
            raise RunError("this run records through run.queue(); add frames with queue.put()")
        frame, row = self._prepare(frame, sent_action=sent_action, control_mode=control_mode, obs_ns=obs_ns,
                                   sent_ns=sent_ns, base_action=base_action, expert_action=expert_action,
                                   expert_source=expert_source, blend_weight=blend_weight,
                                   safety_modified=safety_modified, obs_skew_ns=obs_skew_ns,
                                   inference_id=inference_id, chunk_offset=chunk_offset, reward=reward,
                                   terminated=terminated, truncated=truncated, prompt=prompt, prompt_rev=prompt_rev,
                                   subtask_cmd=subtask_cmd, control_gen=control_gen, camera_ns=camera_ns)
        try:
            self._write_frame(frame, row)
        except BaseException:
            self.discard()                                     # a partial add can leave LeRobot's buffer misaligned
            raise
        self._rows.append(row)
        return row["frame_index"]

    def _prepare(self, frame: dict, *, sent_action, control_mode: str, obs_ns: int, sent_ns: int,
                 base_action=None, expert_action=None, expert_source: str | None = None,
                 blend_weight: float | None = None, safety_modified: bool = False, obs_skew_ns: int = -1,
                 inference_id: int = -1, chunk_offset: int = -1, reward: float | None = None,
                 terminated: bool = False, truncated: bool = False, prompt: str = UNKNOWN,
                 prompt_rev: int | str = UNKNOWN, subtask_cmd=_AUTO,
                 control_gen: int | str = UNKNOWN, camera_ns=None) -> tuple[dict, dict]:
        """Check one frame against the rules and the frames accepted before it. Returns the frame for LeRobot and
        the sidecar row, and counts the frame as accepted. Touches neither LeRobot nor the disk."""
        if subtask_cmd is _AUTO:
            subtask_cmd = UNKNOWN
        if self.policy is None and (base_action is not None or control_mode in ("base", "blend")):
            raise ValueError("this run was started with policy=None (no policy ran), so no frame can carry a "
                             "base_action or be driven by the policy; pass the PolicyInfo, or 'unknown'")
        planned = tick_values(prompt, prompt_rev, subtask_cmd, control_gen)
        bad = self._seq.problem(planned)
        if bad:
            raise ValueError(bad)
        if control_mode not in CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {CONTROL_MODES}")
        if control_mode in ("expert", "blend") and not expert_source:
            raise ValueError("expert_source is required when an expert is in control")
        if control_mode == "blend":
            if blend_weight is None or not 0.0 <= blend_weight <= 1.0:
                raise ValueError("blend needs blend_weight in [0, 1]")
        elif blend_weight is not None:
            raise ValueError("blend_weight is only used with control_mode='blend'")
        else:
            blend_weight = 1.0 if control_mode == "expert" else 0.0
        if control_mode == "base" and base_action is None:
            raise ValueError("control_mode='base' needs base_action")
        if control_mode == "expert" and expert_action is None:
            raise ValueError("control_mode='expert' needs expert_action")
        if control_mode == "blend" and (base_action is None or expert_action is None):
            raise ValueError("control_mode='blend' needs both base_action and expert_action")
        if expert_action is not None and not expert_source:
            raise ValueError("expert_action needs expert_source, even when the expert was queried but not executed")
        if (inference_id == -1) != (base_action is None):
            raise ValueError("inference_id must be given (>= 0) exactly when base_action is given")
        if inference_id != -1 and (inference_id < 0 or chunk_offset < 0):
            raise ValueError("inference_id and chunk_offset must be >= 0 when base_action is given")
        if inference_id == -1 and chunk_offset != -1:
            raise ValueError("chunk_offset must be -1 when there is no base_action")
        for name, v in (("obs_ns", obs_ns), ("sent_ns", sent_ns)):
            if not isinstance(v, (int, np.integer)) or v < 0:
                raise ValueError(f"{name} must be a non-negative int from time.monotonic_ns()")
        if obs_ns > sent_ns:
            raise ValueError("obs_ns is after sent_ns")
        if self._last and (obs_ns < self._last["obs_ns"] or sent_ns < self._last["sent_ns"]):
            raise ValueError("timestamps went backwards")
        if obs_skew_ns < -1:
            raise ValueError("obs_skew_ns must be >= 0, or -1 when unknown")
        if reward is not None and not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if terminated and truncated:
            raise ValueError("a tick cannot be both terminated and truncated")
        if self._last and (self._last["terminated"] or self._last["truncated"]):
            raise RunError("the previous tick ended the episode (terminated or truncated)")
        task = frame.get("task", self.task)
        if task != self.task:
            raise ValueError(f"frame task {task!r} differs from the run task {self.task!r}")

        sent = self.schema.coerce(sent_action, "sent_action")
        base = self.schema.coerce(base_action, "base_action") if base_action is not None else None
        expert = self.schema.coerce(expert_action, "expert_action") if expert_action is not None else None

        frame = dict(frame)
        frame["task"] = self.task
        frame[self.logger.action_key] = sent                  # the same canonical (read-only) array goes to LeRobot
        n = self.schema.dim
        row = {
            "frame_index": self._count,
            "control_mode": control_mode,
            "expert_source": expert_source,
            "blend_weight": float(blend_weight),
            "base_action": base.copy() if base is not None else np.zeros(n, np.float32),
            "base_valid": base is not None,
            "expert_action": expert.copy() if expert is not None else np.zeros(n, np.float32),
            "expert_valid": expert is not None,
            "sent_action": sent.copy(),
            "safety_modified": bool(safety_modified),
            "inference_id": int(inference_id),
            "chunk_offset": int(chunk_offset),
            "obs_ns": int(obs_ns),
            "obs_skew_ns": int(obs_skew_ns),
            "sent_ns": int(sent_ns),
            "reward": float(reward) if reward is not None else 0.0,
            "reward_valid": reward is not None,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            **planned,
        }
        self._seq.accept(planned)
        self._last = row
        self._count += 1
        if self.logger.live_tap is not None:
            self.logger.live_tap.frame(self, frame, row, camera_ns)
        return frame, row

    def _write_frame(self, frame: dict, row: dict) -> None:
        """Hand one prepared frame to LeRobot. The caller appends the row once this returns."""
        expected = row["frame_index"]
        if expected != len(self._rows):
            raise RunError(f"frame {expected} is out of order; the run has written {len(self._rows)}")
        before = adapter.buffer_size(self.dataset)
        if before is not None and before != expected:
            raise RunError(f"episode buffer has {before} frames but this run added {expected}; "
                           "add frames only through run.add_frame()")
        self.dataset.add_frame(frame)

    def queue(self, max_seconds: float = 2.0, max_frames: int | None = None, stall_timeout: float | None = None,
              stop_timeout: float | None = None):
        """Record through a bounded queue: queue.put() takes a frame without waiting for LeRobot, and a writer thread
        adds it. If the queue is full or a write fails, put() raises ControlledStop and the run is quarantined when
        the run block exits. stall_timeout: how long run.end() waits with no frame finishing before it calls the
        writer hung; stop_timeout: how long a stop waits for the frame being written (both 10 s by default).
        See frame_queue.FrameQueue."""
        self._require("active")
        if self._queue is not None:
            raise RunError("this run already has a queue")
        if self._count:
            raise RunError("start the queue before the first frame")
        from .frame_queue import FrameQueue
        self._queue = FrameQueue(self, max_seconds=max_seconds, max_frames=max_frames, stall_timeout=stall_timeout,
                                 stop_timeout=stop_timeout)
        return self._queue

    def complete_transition(self, reward: float, terminated: bool = False, truncated: bool = False) -> None:
        """Attach a reward (and done flags) that arrived one loop late to the previous tick."""
        self._require("active")
        if self._last is None:
            raise RunError("no tick to complete")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        row = self._last                                       # the same dict, queued or written
        if row["reward_valid"]:
            raise RunError("the previous tick already has a reward")
        term, trunc = row["terminated"] or bool(terminated), row["truncated"] or bool(truncated)
        if term and trunc:
            raise ValueError("a tick cannot be both terminated and truncated")
        row.update(reward=float(reward), reward_valid=True, terminated=term, truncated=trunc)

    def event(self, kind: str, source: str | None = None, reason: str | None = None, text: str | None = None) -> None:
        """A typed event at the current tick; frame_index is -1 before the first frame."""
        self._require("active")
        self._events.append({"event_id": len(self._events), "ns": time.monotonic_ns(),
                             "frame_index": self._count - 1, "kind": kind, "source": source,
                             "reason": reason, "text": text})

    def subgoal(self, name: str, status: str, reason: str | None = None) -> None:
        """Mark a step of the task. Call "started" just before its first frame, "completed" or "failed" just after its
        last. A step with no "started" begins where the previous step ended. Summarised as manifest["subgoals"]."""
        if not isinstance(name, str) or not name:
            raise ValueError("subgoal name must be a non-empty string")
        if status not in SUBGOAL_STATUSES:
            raise ValueError(f"subgoal status must be one of {SUBGOAL_STATUSES}")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason must be a string or None")
        problem = subgoal_problem(self._events + [{"kind": "subgoal", "source": name, "text": status}])
        if problem:
            raise ValueError(problem)
        self.event("subgoal", source=name, reason=reason, text=status)

    def predictions(self, inference_id: int, action_chunk, obs_frame_index: int, infer_start_ns: int,
                    infer_end_ns: int, valid_len: int | None = None) -> None:
        self._require("active")
        if self._predictions is None:
            raise RunError("start the run with log_predictions=True to log action chunks")
        if inference_id < 0 or inference_id in self._prediction_ids:
            raise ValueError(f"inference_id {inference_id} must be >= 0 and unique in the run")
        if not 0 <= infer_start_ns <= infer_end_ns:
            raise ValueError("need 0 <= infer_start_ns <= infer_end_ns")
        if not -1 <= obs_frame_index <= self._count:
            raise ValueError("obs_frame_index must be -1 (before the first frame) or a frame of this run")
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != self.schema.dim:
            raise ValueError(f"action_chunk must be [horizon, {self.schema.dim}], got {list(chunk.shape)}")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("action_chunk has NaN or Inf values")
        valid_len = chunk.shape[0] if valid_len is None else int(valid_len)
        if not 0 < valid_len <= chunk.shape[0]:
            raise ValueError("valid_len must be in [1, horizon]")
        self._prediction_ids.add(inference_id)
        self._predictions.append({"inference_id": int(inference_id), "obs_frame_index": int(obs_frame_index),
                                  "infer_start_ns": int(infer_start_ns), "infer_end_ns": int(infer_end_ns),
                                  "horizon": int(chunk.shape[0]), "valid_len": valid_len,
                                  "action_chunk": chunk.reshape(-1).tolist()})

    # ── ending ───────────────────────────────────────────────────────────────

    def end(self, termination_reason: str, task_success: bool | None = None, labels: list[str] | None = None,
            failure_mode: str | None = None) -> None:
        """Seal and stage the run. The final tick's terminated/truncated flag is set from the reason.

        Saving to LeRobot and commit happen when the run block exits; do not touch the dataset in between.
        """
        self._require("active")
        if termination_reason not in TERMINATION_REASONS:
            raise ValueError(f"termination_reason must be one of {TERMINATION_REASONS}")
        self._seal(termination_reason, task_success, labels, failure_mode)

    def _seal(self, termination_reason: str, task_success: bool | None = None, labels: list[str] | None = None,
              failure_mode: str | None = None) -> None:
        """end() after its reason check. A session's gap (logger L9, session.py) is sealed with "unknown": it is not
        an episode and has no outcome, so its last tick gets neither terminated nor truncated."""
        if self._queue is not None:
            self._queue.drain()                                # every queued frame written, or ControlledStop
            self._queue = None
        if not self._rows:
            self.discard()
            raise RunError("a run with no frames is discarded, never saved")
        last = self._rows[-1]
        want = TERMINAL_FLAG.get(termination_reason)
        if want is None:
            if last["terminated"] or last["truncated"]:
                raise RunError("a gap's last tick is neither terminated nor truncated")
        else:
            other = "truncated" if want == "terminated" else "terminated"
            if last[other]:
                raise RunError(f"the last tick is {other}, which contradicts termination_reason={termination_reason!r}")
            last[want] = True
        self._check_rows()
        self._check_buffer()
        self.ended_at = _now_iso()
        if failure_mode is not None and (not isinstance(failure_mode, str) or not failure_mode):
            raise ValueError("failure_mode must be a non-empty string or None")
        if failure_mode is not None and termination_reason == "success":
            raise ValueError("a successful run has no failure_mode")
        self._end = {"termination_reason": termination_reason, "task_success": task_success,
                     "failure_mode": failure_mode,
                     "labels": sorted(set(labels or [])),
                     "rl_ready": all(r["reward_valid"] for r in self._rows)}
        self._write_files("staged")
        self.state = "staged"
        self._live("run_ended")

    def discard(self) -> None:
        """Re-record: drop the LeRobot episode buffer and everything this run logged."""
        if self.state in ("committed", "discarded", "quarantined", "save_failed"):
            return
        if self._queue is not None:
            if self._queue.stopped is not None:
                raise RunError("the queue stopped (ControlledStop); the run is quarantined when its block exits, "
                               "not discarded")
            if not self._queue.close():                        # the writer may still be inside LeRobot
                self.logger.broken = "a frame writer did not finish within its timeout"
                self.state = "discarded"
                self._live("run_state", "discarded")
                raise RunError(f"run discarded, but {self.logger.broken}; open a new dataset writer")
            self._queue = None
        adapter.reset_episode_buffer(self.dataset)
        if self.run_path.exists():
            shutil.rmtree(self.run_path)
        self.state = "discarded"
        self._live("run_state", "discarded")

    def _quarantine_stopped(self):
        """After a ControlledStop: wait for the writer, reset the LeRobot buffer, write every accepted frame to
        dataop/quarantine/<run_id>/ and return the ControlledStop to raise, carrying where the frames went."""
        from .frame_queue import write_partial
        q = self._queue
        stop = q.stopped
        self._live("stop_seen", stop)                          # once per run; before the writer wait
        joined = q.close()
        written, pending, observations, dropped = q.snapshot()
        reset_error = None
        if joined:
            try:
                adapter.reset_episode_buffer(self.dataset)
            except Exception as e:
                reset_error = f"{type(e).__name__}: {e}"
        else:
            reset_error = "the frame writer did not finish within its timeout, so the LeRobot buffer was left alone"
        if reset_error:
            self.logger.broken = f"run {self.run_id} stopped and its LeRobot buffer could not be reset ({reset_error})"
        rows = written + pending
        self._end = {"termination_reason": "system_error", "task_success": None, "failure_mode": None,
                     "labels": [], "rl_ready": all(r["reward_valid"] for r in rows)}
        self.ended_at = _now_iso()
        info = {"reason": stop.reason, "detail": stop.detail, "frames": len(rows),
                "frames_written_to_lerobot": len(written), "frames_queued": len(pending),
                "queue_capacity": q.capacity, "dropped_features": dropped,
                "lerobot_buffer_reset": reset_error is None, "lerobot_buffer_error": reset_error,
                "stopped_at": self.ended_at}
        self._unsaved_quarantine = (rows, observations, info)
        self._queue = None
        self.state = "quarantined"
        stop.frames = len(rows)
        self._stop = stop
        try:
            self.quarantine_path = write_partial(self, rows, observations, info)
        except BaseException as e:                     # an interrupt must not leave the frames unreported either
            stop.quarantine_error = f"{type(e).__name__}: {e}"
            self._blocks_logger = reset_error is None
            self.logger._unsaved_runs.append(self)
            unsaved = (f"run {self.run_id} stopped and its {len(rows)} frames were NOT written to quarantine "
                       f"({stop.quarantine_error}); they are only in memory: free space and call "
                       "run.retry_quarantine(), or close the RunLogger, which retries and then spills them")
            self.logger.broken = f"{self.logger.broken}; {unsaved}" if reset_error else unsaved
            stop._settle()
            self._live("controlled_stop", stop)
            self._live("run_state", "quarantined")
            if not isinstance(e, Exception):
                raise
        else:
            self._unsaved_quarantine = None
            stop.quarantine_path = self.quarantine_path
            stop._settle()
            self._live("controlled_stop", stop)
            self._live("run_state", "quarantined")
        return stop

    def retry_quarantine(self):
        """Write a stopped run's frames to quarantine again, after a failed first try (for example a full disk).
        Returns the quarantine path; raises if the write fails again, keeping the frames in memory."""
        if self._unsaved_quarantine is None:
            if self.quarantine_path is not None:
                return self.quarantine_path
            raise RunError("this run has no frames waiting for quarantine")
        from .frame_queue import write_partial
        rows, observations, info = self._unsaved_quarantine
        self.quarantine_path = write_partial(self, rows, observations, info)
        self._unsaved_quarantine = None
        if self in self.logger._unsaved_runs:
            self.logger._unsaved_runs.remove(self)
        if self._stop is not None:
            self._stop.quarantine_path, self._stop.quarantine_error = self.quarantine_path, None
            self._stop._settle()
            self._live("controlled_stop", self._stop)
        if self._blocks_logger:
            self.logger.broken = None
            self._blocks_logger = False
        return self.quarantine_path

    def _abandon_staged(self, reason: str) -> None:
        adapter.reset_episode_buffer(self.dataset)
        store.quarantine(self.logger.root, self.run_path, reason)
        self.state = "quarantined"
        self._live("run_state", "quarantined")

    def _check_rows(self) -> None:
        rows = self._rows
        for r in rows[:-1]:
            if r["terminated"] or r["truncated"]:
                raise RunError(f"tick {r['frame_index']} is terminated/truncated but is not the last tick")
        if self._predictions is not None:
            for p in self._predictions:
                if not -1 <= p["obs_frame_index"] < len(rows):
                    raise RunError(f"prediction {p['inference_id']} refers to frame {p['obs_frame_index']}, "
                                   f"but the run has {len(rows)} frames")
            known = {p["inference_id"]: p["valid_len"] for p in self._predictions}
            for r in rows:
                if r["inference_id"] == -1:
                    continue
                if r["inference_id"] not in known:
                    raise RunError(f"tick {r['frame_index']} refers to unlogged inference {r['inference_id']}")
                if r["chunk_offset"] >= known[r["inference_id"]]:
                    raise RunError(f"tick {r['frame_index']} chunk_offset is past the chunk's valid_len")

    def _check_buffer(self) -> None:
        """Exact pre-save check: LeRobot's buffered episode equals what this run logged, byte for byte."""
        try:
            snap = adapter.buffer_snapshot(self.dataset, self.logger.action_key)
        except adapter.EpisodeBufferError as e:
            raise RunError(str(e)) from e
        if snap is None:
            return                                             # unsupported LeRobot: validate is authoritative
        n = len(self._rows)
        if snap["episode_index"] is not None and snap["episode_index"] != self.episode_index:
            raise RunError(f"buffer is for episode {snap['episode_index']}, run expected {self.episode_index}")
        if snap["size"] != n or len(snap["actions"]) != n or len(snap["tasks"]) != n:
            raise RunError(f"episode buffer has {snap['size']} frames, {len(snap['actions'])} actions and "
                           f"{len(snap['tasks'])} tasks; the run logged {n}")
        for i, (a, r) in enumerate(zip(snap["actions"], self._rows)):
            if a.shape != (self.schema.dim,) or a.tobytes() != r["sent_action"].tobytes():
                raise RunError(f"buffered action at frame {i} differs from the logged sent_action")
        if any(t != self.task for t in snap["tasks"]):
            raise RunError("a buffered frame has a different task from the run")

    def _ticks_table(self, rows: list[dict] | None = None) -> pa.Table:
        rows = self._rows if rows is None else rows
        n = self.schema.dim
        cols = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            if k.endswith("_action"):
                flat = np.concatenate(vals).astype(np.float32)
                cols[k] = pa.FixedSizeListArray.from_arrays(pa.array(flat, pa.float32()), n)
            elif k in ("expert_source", "prompt", "subtask_cmd"):
                cols[k] = pa.array(vals, pa.string())
            elif k in ("prompt_rev", "control_gen"):
                cols[k] = pa.array(vals, pa.int64())
            else:
                cols[k] = pa.array(vals)
        return pa.table(cols)

    def _manifest(self, status: str, rows: list[dict] | None = None, **extra) -> dict:
        rows = self._rows if rows is None else rows
        sent = np.stack([r["sent_action"] for r in rows])
        return {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id, "status": status,
            "episode_index": self.episode_index, "length": len(rows), "task": self.task,
            "task_id": self.task_id, "task_rev": self.task_rev, "planner": self.planner,
            "started_at": self.started_at, "ended_at": self.ended_at,
            **self._end,
            "fingerprint": fingerprint(self.schema.id, self.task, sent),
            "events": self._events, "takeovers": derive_takeovers(rows),
            "subgoals": derive_subgoals(self._events, len(rows)),
            "policy": asdict(self.policy) if isinstance(self.policy, PolicyInfo) else self.policy,
            "env": {**asdict(self.env), "software": {"lerobot": _lerobot_version(), "dataop_logger": _DATAOP_VERSION}},
            "action_schema": {**self.schema.to_dict(), "id": self.schema.id},
            "clock": {"source": "time.monotonic_ns", "obs_timestamp_kind": self.logger.obs_timestamp_kind},
            "dataset": adapter.dataset_identity(self.dataset),
            "predictions_count": None if self._predictions is None else len(self._predictions),
            "integrity": self._integrity,
            "batch": {"name": self.logger.batch} if self.logger.batch else None,
            **(self._session or {}),
            **extra,
        }

    def _write_files(self, status: str) -> None:
        ticks = store.table_bytes(self._ticks_table())
        store.atomic_write_bytes(self.run_path / "ticks.parquet", ticks)
        self._integrity = {"ticks_sha256": store.sha256(ticks), "events_sha256": store.json_sha256(self._events),
                           "predictions_sha256": None}
        if self._predictions is not None:                      # written even when empty, so absence means loss
            preds = store.table_bytes(pa.Table.from_pylist(self._predictions, schema=PRED_SCHEMA))
            store.atomic_write_bytes(self.run_path / "predictions.parquet", preds)
            self._integrity["predictions_sha256"] = store.sha256(preds)
        store.atomic_write_json(self.run_path / "manifest.json", self._manifest(status))   # manifest last

    def _save_and_commit(self) -> None:
        try:
            self._check_buffer()                               # nothing may change between end() and save
        except Exception as e:
            self._abandon_staged(f"the LeRobot buffer changed after run.end(): {e}")
            raise RunError(f"the LeRobot buffer changed after run.end(); nothing was saved ({e})") from e
        before = adapter.total_episodes(self.dataset)
        try:
            self.dataset.save_episode()
        except BaseException as e:
            # LeRobot may have written part or all of the episode. Keep the staged run where repair looks for it,
            # and stop this logger: the LeRobot buffer was mutated by the failed save.
            store.atomic_write_json(self.run_path / "manifest.json",
                                    self._manifest("staged", save_error=f"{type(e).__name__}: {e}"))
            self.state = "save_failed"
            self.logger.broken = f"{type(e).__name__}: {e}"
            self._live("run_state", "save_failed")
            raise
        after = adapter.total_episodes(self.dataset)
        if after != before + 1 or self.episode_index != before:
            msg = f"after save, total_episodes went {before}->{after}, run expected episode {self.episode_index}"
            store.atomic_write_json(self.run_path / "manifest.json", self._manifest("staged", save_error=msg))
            self.state = "save_failed"
            self.logger.broken = msg
            self._live("run_state", "save_failed")
            raise RunError(f"LeRobot saved an unexpected episode index ({msg}); run validate --repair")
        try:
            em = adapter.saved_episode(self.dataset, self.episode_index)
        except Exception as e:                          # malformed in-memory record
            em, bad = None, f"{type(e).__name__}: {e}"
        else:
            bad = None
        if em is None or em["length"] != len(self._rows) or em["to"] - em["from"] != len(self._rows):
            msg = (f"could not confirm LeRobot saved episode {self.episode_index} ({bad or 'no saved-episode record'})"
                   if em is None else
                   f"LeRobot registered episode {self.episode_index} with length {em['length']}, run has {len(self._rows)}")
            store.atomic_write_json(self.run_path / "manifest.json", self._manifest("staged", save_error=msg))
            self.state = "save_failed"
            self.logger.broken = msg
            self._live("run_state", "save_failed")
            raise RunError(msg)
        store.atomic_write_json(self.run_path / "manifest.json", self._manifest("committed"))
        self.state = "committed"
        self._live("run_state", "committed")
        store.rebuild_index(self.logger.root)
        try:
            batches.join(self.logger.root, self.run_id, self.logger.batch,
                         asdict(self.policy) if isinstance(self.policy, PolicyInfo) else None)
        except Exception as e:                           # the run is saved; batches place it on the next read
            warnings.warn(f"run {self.run_id} was saved but not added to a batch yet ({e})")


def derive_takeovers(rows) -> list[dict]:
    """Half-open [start_frame, end_frame) ranges where an expert was in control or blended in."""
    out, start = [], 0
    key = lambda r: (r["control_mode"], r["expert_source"])
    for i in range(1, len(rows) + 1):
        if i == len(rows) or key(rows[i]) != key(rows[start]):
            r0 = rows[start]
            if r0["control_mode"] in ("expert", "blend"):
                out.append({"start_frame": start, "end_frame": i, "control_mode": r0["control_mode"],
                            "expert_source": r0["expert_source"]})
            start = i
    return out


def subgoal_problem(events) -> str | None:
    """Steps run one at a time: "started" needs no open step; "completed"/"failed" closes the open step of that name,
    or, with no step open, a step that began where the previous one ended. Returns the first violation."""
    open_ = None
    for e in events:
        if e.get("kind") != "subgoal":
            continue
        name, status = e.get("source"), e.get("text")
        if not isinstance(name, str) or not name or status not in SUBGOAL_STATUSES:
            return f"subgoal event {e.get('event_id')} has no name or an invalid status"
        if status == "started":
            if open_ is not None:
                return f"step {name!r} started while step {open_!r} is still open"
            open_ = name
        elif open_ is not None and open_ != name:
            return f"step {name!r} {status} while step {open_!r} is open"
        else:
            open_ = None
    return None


def derive_subgoals(events, length: int) -> list[dict]:
    """Spans of the task's steps from subgoal events: [{name, start_frame, end_frame, status, reason}], end exclusive.
    A step still open when the run ends is "unfinished"."""
    out, open_, prev_end = [], {}, 0
    for e in events:
        if e.get("kind") != "subgoal":
            continue
        name, status, f = e["source"], e["text"], e["frame_index"]
        if status == "started":
            open_[name] = f + 1
            continue
        start = open_.pop(name, prev_end)
        end = max(start, f + 1)
        out.append({"name": name, "start_frame": start, "end_frame": end, "status": status, "reason": e.get("reason")})
        prev_end = end
    for name, start in open_.items():
        out.append({"name": name, "start_frame": start, "end_frame": length, "status": "unfinished", "reason": None})
    return sorted(out, key=lambda s: (s["start_frame"], s["end_frame"]))
