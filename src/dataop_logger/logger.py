"""RunLogger: records provenance next to a LeRobot dataset, owning the add/save ordering so the two cannot drift.

State machine per run: active -> staged -> committed, with terminal discarded, save_failed and quarantined.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict

import numpy as np
import pyarrow as pa

from . import adapter, store
from .fingerprint import fingerprint
from .schema import (CONTROL_MODES, SCHEMA_VERSION, SUBGOAL_STATUSES, TERMINAL_FLAG, TERMINATION_REASONS,
                     ActionSchema, EnvInfo, PolicyInfo)

try:                                                     # recorded in manifests
    from importlib.metadata import version as _pkg_version
    _DATAOP_VERSION = _pkg_version("dataop-logger")
except Exception:                                        # running from a source checkout
    _DATAOP_VERSION = "0.1.0+src"

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


class RunLogger:
    """One per dataset root. Holds the single-writer lock until close()."""

    def __init__(self, dataset, action_schema: ActionSchema, obs_timestamp_kind: str = "bundle_completion",
                 action_key: str = "action"):
        if obs_timestamp_kind not in ("capture", "bundle_completion"):
            raise ValueError("obs_timestamp_kind must be 'capture' or 'bundle_completion'")
        self.dataset = dataset
        self.schema = action_schema
        self.root = dataset.root
        self.action_key = action_key
        self.obs_timestamp_kind = obs_timestamp_kind
        self.broken: str | None = None                   # set after a mid-save failure; no more runs
        self._lock = store.WriterLock(self.root)
        self._active: Run | None = None

    def close(self) -> None:
        if self._active is not None and self._active.state == "active":
            self._active.discard()
        self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def run(self, task: str, policy: PolicyInfo | None = None, env: EnvInfo | None = None,
            log_predictions: bool = False):
        """Record one episode. Call run.end(...) inside the block; saving and commit happen on exit."""
        if self.broken:
            raise RunError(f"this logger stopped after a failed save ({self.broken}); "
                           "run `dataop-logger validate --repair` and open a new dataset writer")
        if self._active is not None and self._active.state in ("active", "staged"):
            raise RunError("a run is already in progress on this dataset")
        if not isinstance(task, str) or not task:
            raise ValueError("task must be a non-empty string")
        if adapter.has_pending_frames(self.dataset):
            raise RunError("the dataset has unsaved frames; save or clear the episode buffer before starting a run")
        r = Run(self, task, policy, env, log_predictions)
        self._active = r
        try:
            yield r
        except BaseException:
            if r.state == "active":
                r.discard()
            elif r.state == "staged":
                r._abandon_staged("exception after run.end(); the episode was not saved")
            raise
        if r.state == "discarded":
            return
        if r.state != "staged":
            r.discard()
            raise RunError("the run block ended without run.end(); the episode was discarded, not saved")
        r._save_and_commit()


class Run:
    def __init__(self, logger: RunLogger, task: str, policy: PolicyInfo | None, env: EnvInfo | None,
                 log_predictions: bool):
        self.logger = logger
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

    # ── recording ────────────────────────────────────────────────────────────

    def _require(self, *states):
        if self.state not in states:
            raise RunError(f"run is {self.state}; this call needs it to be {' or '.join(states)}")

    def add_frame(self, frame: dict, *, sent_action, control_mode: str, obs_ns: int, sent_ns: int,
                  base_action=None, expert_action=None, expert_source: str | None = None,
                  blend_weight: float | None = None, safety_modified: bool = False, obs_skew_ns: int = -1,
                  inference_id: int = -1, chunk_offset: int = -1, reward: float | None = None,
                  terminated: bool = False, truncated: bool = False) -> int:
        """Add one frame to LeRobot first, then the sidecar row. Returns the frame index.

        obs_ns and sent_ns are required monotonic-clock readings (time.monotonic_ns()) taken by the caller.
        """
        self._require("active")
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
        if self._rows and (obs_ns < self._rows[-1]["obs_ns"] or sent_ns < self._rows[-1]["sent_ns"]):
            raise ValueError("timestamps went backwards")
        if obs_skew_ns < -1:
            raise ValueError("obs_skew_ns must be >= 0, or -1 when unknown")
        if reward is not None and not np.isfinite(reward):
            raise ValueError("reward must be finite")
        if terminated and truncated:
            raise ValueError("a tick cannot be both terminated and truncated")
        if self._rows and (self._rows[-1]["terminated"] or self._rows[-1]["truncated"]):
            raise RunError("the previous tick ended the episode (terminated or truncated)")
        task = frame.get("task", self.task)
        if task != self.task:
            raise ValueError(f"frame task {task!r} differs from the run task {self.task!r}")

        sent = self.schema.coerce(sent_action, "sent_action")
        base = self.schema.coerce(base_action, "base_action") if base_action is not None else None
        expert = self.schema.coerce(expert_action, "expert_action") if expert_action is not None else None

        expected = len(self._rows)
        before = adapter.buffer_size(self.dataset)
        if before is not None and before != expected:
            raise RunError(f"episode buffer has {before} frames but this run added {expected}; "
                           "add frames only through run.add_frame()")
        frame = dict(frame)
        frame["task"] = self.task
        frame[self.logger.action_key] = sent                  # the same canonical (read-only) array goes to LeRobot
        try:
            self.dataset.add_frame(frame)
        except BaseException:
            self.discard()                                     # a partial add can leave LeRobot's buffer misaligned
            raise
        n = self.schema.dim
        self._rows.append({
            "frame_index": expected,
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
        })
        return expected

    def complete_transition(self, reward: float, terminated: bool = False, truncated: bool = False) -> None:
        """Attach a reward (and done flags) that arrived one loop late to the previous tick."""
        self._require("active")
        if not self._rows:
            raise RunError("no tick to complete")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        row = self._rows[-1]
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
                             "frame_index": len(self._rows) - 1, "kind": kind, "source": source,
                             "reason": reason, "text": text})

    def subgoal(self, name: str, status: str, reason: str | None = None) -> None:
        """Mark a step of the task. Call "started" just before its first frame, "completed" or "failed" just after its
        last. A step with no "started" begins where the previous step ended. Summarised as manifest["subgoals"]."""
        if not isinstance(name, str) or not name:
            raise ValueError("subgoal name must be a non-empty string")
        if status not in SUBGOAL_STATUSES:
            raise ValueError(f"subgoal status must be one of {SUBGOAL_STATUSES}")
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
        if not -1 <= obs_frame_index <= len(self._rows):
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
        if not self._rows:
            self.discard()
            raise RunError("a run with no frames is discarded, never saved")
        last = self._rows[-1]
        want = TERMINAL_FLAG[termination_reason]
        other = "truncated" if want == "terminated" else "terminated"
        if last[other]:
            raise RunError(f"the last tick is {other}, which contradicts termination_reason={termination_reason!r}")
        last[want] = True
        self._check_rows()
        self._check_buffer()
        self.ended_at = _now_iso()
        if failure_mode is not None and (not isinstance(failure_mode, str) or not failure_mode):
            raise ValueError("failure_mode must be a non-empty string or None")
        self._end = {"termination_reason": termination_reason, "task_success": task_success,
                     "failure_mode": failure_mode,
                     "labels": sorted(set(labels or [])),
                     "rl_ready": all(r["reward_valid"] for r in self._rows)}
        self._write_files("staged")
        self.state = "staged"

    def discard(self) -> None:
        """Re-record: drop the LeRobot episode buffer and everything this run logged."""
        if self.state in ("committed", "discarded", "quarantined", "save_failed"):
            return
        adapter.reset_episode_buffer(self.dataset)
        if self.run_path.exists():
            shutil.rmtree(self.run_path)
        self.state = "discarded"

    def _abandon_staged(self, reason: str) -> None:
        adapter.reset_episode_buffer(self.dataset)
        store.quarantine(self.logger.root, self.run_path, reason)
        self.state = "quarantined"

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

    def _ticks_table(self) -> pa.Table:
        n = self.schema.dim
        cols = {}
        for k in self._rows[0]:
            vals = [r[k] for r in self._rows]
            if k.endswith("_action"):
                flat = np.concatenate(vals).astype(np.float32)
                cols[k] = pa.FixedSizeListArray.from_arrays(pa.array(flat, pa.float32()), n)
            elif k == "expert_source":
                cols[k] = pa.array(vals, pa.string())
            else:
                cols[k] = pa.array(vals)
        return pa.table(cols)

    def _manifest(self, status: str, **extra) -> dict:
        sent = np.stack([r["sent_action"] for r in self._rows])
        return {
            "schema_version": SCHEMA_VERSION, "run_id": self.run_id, "status": status,
            "episode_index": self.episode_index, "length": len(self._rows), "task": self.task,
            "started_at": self.started_at, "ended_at": self.ended_at,
            **self._end,
            "fingerprint": fingerprint(self.schema.id, self.task, sent),
            "events": self._events, "takeovers": derive_takeovers(self._rows),
            "subgoals": derive_subgoals(self._events, len(self._rows)),
            "policy": asdict(self.policy) if self.policy else None,
            "env": {**asdict(self.env), "software": {"lerobot": _lerobot_version(), "dataop_logger": _DATAOP_VERSION}},
            "action_schema": {**self.schema.to_dict(), "id": self.schema.id},
            "clock": {"source": "time.monotonic_ns", "obs_timestamp_kind": self.logger.obs_timestamp_kind},
            "dataset": adapter.dataset_identity(self.dataset),
            "predictions_count": None if self._predictions is None else len(self._predictions),
            "integrity": self._integrity,
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
            raise
        after = adapter.total_episodes(self.dataset)
        if after != before + 1 or self.episode_index != before:
            msg = f"after save, total_episodes went {before}->{after}, run expected episode {self.episode_index}"
            store.atomic_write_json(self.run_path / "manifest.json", self._manifest("staged", save_error=msg))
            self.state = "save_failed"
            self.logger.broken = msg
            raise RunError(f"LeRobot saved an unexpected episode index ({msg}); run validate --repair")
        em = adapter.saved_episode(self.dataset, self.episode_index)
        if em is not None and (em["length"] != len(self._rows) or em["to"] - em["from"] != len(self._rows)):
            msg = f"LeRobot registered episode {self.episode_index} with length {em['length']}, run has {len(self._rows)}"
            store.atomic_write_json(self.run_path / "manifest.json", self._manifest("staged", save_error=msg))
            self.state = "save_failed"
            self.logger.broken = msg
            raise RunError(msg)
        store.atomic_write_json(self.run_path / "manifest.json", self._manifest("committed"))
        self.state = "committed"
        store.rebuild_index(self.logger.root)


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
