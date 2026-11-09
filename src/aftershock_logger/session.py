"""Drive sessions (logger task L9): one continuous recording from Start to Stop, split into runs and gaps.

In a Drive session the station records every tick from Start until the session ends. Each episode is saved as its
own run of the dataset, as any run is. The ticks between episodes (the reset, an episode being written, the time
after Stop) are gaps. A gap is saved as one episode of a second LeRobot dataset in the session folder, with the same
features as the runs (every camera, `observation.state`, `action`) and the same sidecar ticks, so it can become a
trainable episode later. Every session frame is in exactly one run or one gap.

    <root>/dataop/sessions/<session_id>/session.json      the session record (runs, gaps, write status, ended_by)
    <root>/dataop/sessions/<session_id>/evidence.parquet  the evidence log: one row per record, {session_frame, kind,
                                                          payload}, covering gap time as well as runs
    <root>/dataop/sessions/<session_id>/review.json       the session's review (format 2, starts empty at rev 0)
    <root>/dataop/sessions/<session_id>/gaps/              the gap dataset (LeRobot), with its own dataop/ sidecar

The logger records what the station's session runner decides; it does not run the session state machine. One
SessionLogger is used from one thread. Every tick goes through `add_frame`, which writes it to the open episode or to
the active gap, so session frames cannot drift. See SPEC.md, Drive sessions.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import adapter, review_format, store
from .logger import Run, RunError, RunLogger
from .schema import UNKNOWN, EnvInfo, PolicyInfo, task_id_value

SESSIONS = "sessions"
SHORT_STOP_S = 1.0          # a stopped episode this long or shorter keeps its frames in the gap, no run
EVIDENCE_KINDS = ("state", "criteria", "reset_check", "write", "fault", "persistence_error")
WRITE_STATUSES = ("pending", "committed", "failed")
MOVED_BY = ("person", "reset_skill", "task", "none", "unknown")
RESET_CHECKS = ("home", "zone", "expert_active", "torque", "cameras")
RESCUE_OUTCOMES = ("success", "failure", "unknown")
ENDED_BY = ("user_stop", "fault_stop")
EVIDENCE_SCHEMA = pa.schema([("session_frame", pa.int64()), ("kind", pa.string()), ("payload", pa.string())])
SESSION_KEYS = ("session_id", "station_id", "go", "model_id", "planner", "task_id", "task_rev", "fps", "started_at",
                "ended_at", "ended_by", "fault", "session_frames", "runs", "gaps")
_UUID4 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_SLUG = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class SessionError(RunError):
    pass


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def sessions_dir(root) -> Path:
    return store.dataop_dir(Path(root)) / SESSIONS


def session_dir(root, session_id: str) -> Path:
    return sessions_dir(root) / session_id


def utc_now() -> str:
    """UTC wall clock in the form Aftershock's records use (`...Z`), for people only."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _copy(v):
    """An owned copy of a frame or tick value, so a caller that reuses its buffers cannot change a frame the session
    may still have to move into a gap (the short stop rule)."""
    if hasattr(v, "detach"):                                  # torch tensor
        return v.detach().clone()
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, dict):
        return {k: _copy(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_copy(x) for x in v)
    return v


_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]{1,6})?Z$")
GO_SCOPE_KEYS = ("model_id", "planner", "task_id", "task_rev", "station_rev")


def _str(v) -> bool:
    return isinstance(v, str) and bool(v)


def go_problems(go) -> list[str]:
    """The go as the cloud session models take it: {by, at (UTC, ending in Z), scope {model_id, planner (null in
    this version), task_id, task_rev >= 1, station_rev >= 0}}."""
    if not isinstance(go, dict) or set(go) != {"by", "at", "scope"}:
        return ["go must be {by, at, scope}: the station owner's go for this session"]
    out = []
    if not _str(go["by"]):
        out.append("go.by must be a non-empty string")
    if not isinstance(go["at"], str) or not _UTC.match(go["at"]):
        out.append("go.at must be a UTC time ending in Z")
    sc = go["scope"]
    if not isinstance(sc, dict) or set(sc) != set(GO_SCOPE_KEYS):
        out.append(f"go.scope must have exactly {GO_SCOPE_KEYS}")
    else:
        if not (_str(sc["model_id"]) and _str(sc["task_id"])):
            out.append("go.scope names its model_id and task_id")
        if sc["planner"] is not None:
            out.append("go.scope.planner is null (no planner in this version)")
        if not (_int(sc["task_rev"]) and sc["task_rev"] >= 1 and _int(sc["station_rev"]) and sc["station_rev"] >= 0):
            out.append("go.scope.task_rev is an int >= 1 and station_rev an int >= 0")
    return out


def _check_go(go) -> dict:
    bad = go_problems(go)
    if bad:
        raise ValueError("; ".join(bad))
    return json.loads(json.dumps(go))


def _check_reset_checks(rc) -> dict:
    rc = {} if rc is None else rc
    if not isinstance(rc, dict) or set(rc) - set(RESET_CHECKS):
        raise ValueError(f"reset_checks takes only {RESET_CHECKS}")
    out = {k: rc.get(k) for k in RESET_CHECKS}
    if any(v is not None and not isinstance(v, bool) for v in out.values()):
        raise ValueError("each reset check is true, false or None (did not run)")
    return out


def coverage_problems(runs: list[tuple[int, int]], gaps: list[tuple[int, int]], session_frames: int) -> list[str]:
    """Every session frame belongs to exactly one run or one gap. Ranges are half-open [start, end)."""
    spans = sorted([(s, e, "run") for s, e in runs] + [(s, e, "gap") for s, e in gaps])
    out, at = [], 0
    for s, e, kind in spans:
        if e <= s:
            out.append(f"the {kind} at {s} has no frames")
        if s > at:
            out.append(f"frames {at} to {s} are in no run or gap")
        if s < at:
            out.append(f"the {kind} at {s} overlaps frames before {at}")
        at = max(at, e)
    if at < session_frames:
        out.append(f"frames {at} to {session_frames} are in no run or gap")
    if at > session_frames:
        out.append(f"ranges run past the session's {session_frames} frames")
    return out


class SessionLogger:
    """Records one Drive session on a RunLogger's dataset. Use it as a context manager, or call close().

        with RunLogger(dataset, schema, planner=None) as log, \\
             SessionLogger(log, session_id=sid, station_id="st_1", go=go, model_id="m_1", task="pick the tube",
                           task_id="tube-pick", task_rev=3, policy=policy) as sess:
            sess.add_frame(frame, sent_action=hold, control_mode="none", obs_ns=t0, sent_ns=t1)   # gap (reset)
            run = sess.start_episode(moved_by="person", reset_checks={"home": True, "zone": True})
            sess.add_frame(frame, sent_action=a, base_action=a, control_mode="base", inference_id=0, chunk_offset=0,
                           obs_ns=t0, sent_ns=t1)                                                  # the run
            sess.seal_episode("success")       # the run's write is pending; the next frames go to a new gap
            sess.write_pending()               # "committed" or "failed"
            ...
            sess.finalize("user_stop")

    `start_episode` closes the active gap (the gap closes on the reset handoff's tick) and opens a run on the next
    frame. `seal_episode` fixes the termination; a stopped episode of `SHORT_STOP_S` or less is not saved as a run:
    its frames move to a new gap with a `short_stop` mark. `write_pending` saves the sealed run; a failure
    quarantines it, records `persistence_error`, and the session ends as `fault_stop`. `finalize` closes the active
    gap and writes the final session record once no write is pending."""

    def __init__(self, logger: RunLogger, *, session_id: str, station_id: str, go: dict, model_id: str, task: str,
                 task_id: str, task_rev: int, policy: PolicyInfo | str = UNKNOWN, env: EnvInfo | None = None,
                 gap_dataset=None):
        """logger: the RunLogger of the dataset the runs go to; it must be built with planner=None (Drive has no
        planner in this version). go: the station owner's go, {by, at, scope}, copied into every run as go_ref.
        policy: the PolicyInfo of the model the session runs (Drive always runs one), or "unknown".
        gap_dataset: an empty LeRobot dataset at <session folder>/gaps with the runs' features; made with the
        dataset's own features when not given."""
        if not isinstance(session_id, str) or not _UUID4.match(session_id):
            raise ValueError("session_id must be a lowercase uuid4")
        for name, v in (("station_id", station_id), ("model_id", model_id), ("task", task)):
            if not isinstance(v, str) or not v:
                raise ValueError(f"{name} must be a non-empty string")
        if task_id_value(task_id) == UNKNOWN:
            raise ValueError("a Drive session names its task_id")
        if isinstance(task_rev, bool) or not isinstance(task_rev, int) or task_rev < 1:
            raise ValueError("task_rev must be an int >= 1")
        if logger.planner is not None:
            raise ValueError("Drive sessions have no planner in this version: build the RunLogger with planner=None")
        if not (isinstance(policy, PolicyInfo) or policy == UNKNOWN):
            raise ValueError("a Drive session runs a model: policy must be a PolicyInfo, or 'unknown'")
        self.logger = logger
        self.root = Path(logger.root)
        self.session_id = session_id
        self.station_id, self.model_id, self.task = station_id, model_id, task
        self.task_id, self.task_rev = task_id, task_rev
        self.go = _check_go(go)
        self.policy, self.env = policy, env
        self.fps = adapter.dataset_fps(logger.dataset)
        self.dir = session_dir(self.root, session_id)
        if self.dir.exists():
            raise SessionError(f"session {session_id} already has a folder; a session id is never reused")
        self.dir.mkdir(parents=True)
        self.started_at = utc_now()
        self.ended_at = self.ended_by = self.fault_kind = None
        self.frames = 0                                       # session frames recorded so far
        self.runs: list[dict] = []                            # session.json runs[], in order
        self.gaps: list[dict] = []                            # session.json gaps[], in order
        self.broken: str | None = None
        self._encoder_stop = None                             # an EncoderBacklog from a run (#479)
        self._evidence: list[tuple[int, str, str]] = []
        self._episode = None                                  # {run, cm, start, buffer} while an episode is open
        self._pending = None                                  # {run, cm, entry} while a sealed run is not written
        self._gap = None                                      # {run, cm, start, short_stop} while a gap is open
        self._last_fault = None
        store.atomic_write_bytes(self.dir / "review.json",
                                 review_format.dump_review(review_format.empty_review(session_id)).encode())
        if gap_dataset is None:
            gap_dataset = adapter.create_like(logger.dataset, self.dir / "gaps", f"local/gaps-{session_id}")
        if Path(gap_dataset.root).resolve() != (self.dir / "gaps").resolve():
            raise ValueError(f"the gap dataset must be at {self.dir / 'gaps'}")
        self.gap_dataset = gap_dataset
        self._gap_writer_closed = False
        self._finalizing = False                              # finalize() failed after the gap dataset closed
        self._gap_logger = RunLogger(gap_dataset, logger.schema, obs_timestamp_kind=logger.obs_timestamp_kind,
                                     action_key=logger.action_key, planner=None)
        self._write_record()
        self._write_evidence()

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        """A close() error never hides the exception leaving the block: it is warned instead."""
        try:
            self.close()
        except SessionError as e:
            if exc is None:
                raise
            warnings.warn(f"closing the session logger: {e}")
        return False

    def close(self) -> None:
        """Release the gap dataset. A session that was not finalized keeps every frame it recorded: an open episode
        is sealed as `system_error` (with a `fault` row, `session_closed`) and written, and the active gap is saved.
        The session then stays open on disk (ended_at null), and `validate` reports it. Raises SessionError naming
        anything that could not be kept or closed."""
        if self._gap_logger is None:
            return
        problems = []
        if self.ended_at is None and not self.broken and not self._finalizing:
            try:
                if self._episode is not None:
                    run = self._episode["run"]
                    took = any(r["control_mode"] in ("expert", "blend") for r in run._rows)
                    self._add_evidence(max(self.frames - 1, 0), "fault",
                                       {"fault": "session_closed",
                                        "detail": "the session logger closed with an episode open"})
                    if run._rows and run.state == "active":       # a fault seals at any length, whatever was flagged
                        run._rows[-1]["terminated"] = run._rows[-1]["truncated"] = False
                    self.seal_episode("system_error", rescue_outcome="unknown" if took else None)
                if self._pending is not None:
                    self.write_pending()
                self._close_gap("unknown", None, reset_passed=False)
            except Exception as e:
                problems.append(f"{type(e).__name__}: {e}")
            try:
                self._flush()
            except Exception as e:
                problems.append(f"the session record could not be written ({type(e).__name__}: {e})")
        problems += self._close_gap_dataset()
        self._gap_logger = None                               # closed for good, finalized or not
        if problems:
            raise SessionError("; ".join(problems))

    def _close_gap_dataset(self) -> list[str]:
        """Close the gap writer (once) and finalize the gap dataset. The session keeps the gap dataset until its
        finalize succeeds, so a failed finalize can be retried."""
        problems = []
        if not self._gap_writer_closed:
            self._gap_writer_closed = True
            try:
                self._gap_logger.close()
            except Exception as e:
                problems.append(f"the gap dataset's writer did not close cleanly ({type(e).__name__}: {e})")
        try:
            self.gap_dataset.finalize()
        except Exception as e:
            problems.append(f"the gap dataset could not be finalized ({type(e).__name__}: {e})")
        else:
            self._gap_logger = None
        return problems

    # ── recording ──────────────────────────────────────────────────────────

    def _require_open(self) -> None:
        if self.ended_at is not None:
            raise SessionError("the session is finalized")
        if self.broken:
            raise SessionError(f"the session stopped recording ({self.broken})")
        if self._gap_logger is None or self._gap_writer_closed:
            raise SessionError("the session logger is closed" if not self._finalizing else
                               "the session is ending (finalize failed); call finalize() again")

    def add_frame(self, frame: dict, **tick) -> int:
        """Record one session tick: into the open episode's run, or else into the active gap (opened on the first
        frame after a seal, and at session frame 0). Takes the same arguments as Run.add_frame. Returns the frame's
        session frame (its address in the session, counted from 0 at Start and never reset)."""
        self._require_open()
        ep = self._episode
        if ep is not None:
            keep = ep["buffer"] is not None and ep["run"]._count + 1 <= SHORT_STOP_S * self.fps
            saved = _copy(frame) if keep else None
            try:
                ep["run"].add_frame(frame, **tick)
            except BaseException as e:
                self._encoder_stopped(e)
                if ep["run"].state != "active":
                    self.broken = "the open episode was discarded after a failed frame"
                raise
            ep["n"] += 1
            if keep:
                ep["buffer"].append(saved)
            else:
                ep["buffer"] = None                           # longer than a short stop: no longer needed
        else:
            if self._gap is None:
                self._open_gap(self.frames)
            try:
                self._gap["run"].add_frame(frame, **tick)
            except BaseException as e:
                self._encoder_stopped(e)
                if self._gap["run"].state != "active":
                    self.broken = "the active gap was discarded after a failed frame"
                raise
        self.frames += 1
        return self.frames - 1

    def _encoder_stopped(self, e: BaseException) -> None:
        """LeRobot's video encoder fell behind past the frame budget (#479): that frame was not recorded, so the
        session records no more frames and starts no episode. A `fault` row (`logger_failed`) names the stop, and the
        session ends as `fault_stop` with that fault. The open episode (or gap) keeps every frame it recorded: the
        station seals it as system_error, and a gap is saved with the label encoder_backlog."""
        from .frame_queue import EncoderBacklog
        if not isinstance(e, EncoderBacklog) or self._encoder_stop is not None:
            return
        self._encoder_stop = e
        self._add_evidence(max(self.frames - 1, 0), "fault", {"fault": "logger_failed", "detail": str(e)})

    def _open_gap(self, start: int, short_stop: tuple[int, int] | None = None) -> None:
        cm = self._gap_logger.run(task=self.task, policy=self.policy, env=self.env, task_id=self.task_id,
                                  task_rev=self.task_rev)
        run = cm.__enter__()
        self._gap = {"run": run, "cm": cm, "start": start, "short_stop": short_stop}

    def _close_gap(self, moved_by: str, reset_checks, reset_passed: bool) -> None:
        if moved_by not in MOVED_BY:
            raise ValueError(f"moved_by must be one of {MOVED_BY}")
        checks = _check_reset_checks(reset_checks)
        gap = self._gap
        if gap is None:
            return
        run: Run = gap["run"]
        n = run._count
        if n != self.frames - gap["start"]:
            raise SessionError("the gap's frames do not match the session's frame count")
        start, end = gap["start"], gap["start"] + n
        gap_id = f"g_{run.episode_index}"
        short = gap["short_stop"]
        if reset_passed:
            run.event("reset_pass", source="station")
        run._session = {"session_gap": {"session_id": self.session_id, "gap_id": gap_id,
                                        "session_frame_start": start, "session_frame_end": end,
                                        "short_stop": None if short is None else {"start": short[0], "end": short[1]},
                                        "moved_by": moved_by, "reset_checks": checks}}
        self._gap = None
        try:
            run._seal("unknown")
            gap["cm"].__exit__(None, None, None)              # saves the gap episode and commits its sidecar
        except BaseException as e:
            self.broken = f"gap {gap_id} could not be saved ({type(e).__name__}: {e})"
            self._add_evidence(max(end - 1, 0), "persistence_error",
                               {"gap_id": gap_id, "error": f"{type(e).__name__}: {e}"})
            self._flush()
            raise SessionError(self.broken) from e
        self.gaps.append({"gap_id": gap_id, "range": {"start": start, "end": end},
                          "gap_episode_index": run.episode_index, "moved_by": moved_by, "reset_checks": checks,
                          "short_stop": None if short is None else {"start": short[0], "end": short[1]}})
        self._flush()

    def start_episode(self, *, moved_by: str = "unknown", reset_checks: dict | None = None,
                      log_predictions: bool = False) -> Run:
        """The reset handoff passed on the last recorded tick: close the active gap (with who moved the arm and the
        reset checks on its last frame, and a `reset_pass` event) and open the next episode's run, which takes the
        next frame. Returns the Run, for events, predictions, subgoals and complete_transition; add its frames
        with `add_frame`, never with run.add_frame."""
        self._require_open()
        if self._encoder_stop is not None:
            raise self._encoder_stop
        if self._episode is not None:
            raise SessionError("an episode is already open")
        if self._pending is not None:
            raise SessionError("the last sealed run is still being written; call write_pending() first")
        self._close_gap(moved_by, reset_checks, reset_passed=True)
        cm = self.logger.run(task=self.task, policy=self.policy, env=self.env, log_predictions=log_predictions,
                             task_id=self.task_id, task_rev=self.task_rev)
        run = cm.__enter__()
        run._session = {"session_id": self.session_id, "episode_in_session": len(self.runs) + 1,
                        "session_frame_start": self.frames, "session_frame_end": None, "go_ref": self.go,
                        "rescue_outcome": None}
        run.event("episode_start", source="station")
        self._episode = {"run": run, "cm": cm, "start": self.frames, "buffer": [], "n": 0}
        return run

    def rescue_success(self) -> None:
        """The success criteria held in `expert` on the last recorded tick: a `rescue_success` event (it does not
        latch and seals nothing)."""
        self._require_open()
        if self._episode is None:
            raise SessionError("no episode is open")
        self._episode["run"].event("rescue_success", source="station")

    def seal_episode(self, termination_reason: str, *, rescue_outcome: str | None = None,
                     task_success: bool | None = None, labels: list[str] | None = None,
                     failure_mode: str | None = None) -> str:
        """Seal the open episode on the last recorded tick, with its termination fixed now. Returns:

        - "pending": the run is staged and its write is pending (`write_pending` saves it); the next frame opens a
          new gap. An `operator_abort` run is tagged `stopped`.
        - "short_stop": a Stop (`operator_abort`) ended an episode of `SHORT_STOP_S` or less, so it is not a run:
          its frames move to a new gap with a `short_stop` mark, and the recording goes on in that gap.
        - "empty": the episode had no frames, so there is nothing to keep.

        rescue_outcome: null when there was no takeover; with a takeover, `success` or `failure` for Rescue done
        (termination `failure`), `unknown` for a Stop or a fault (`operator_abort`, `system_error`)."""
        self._require_open()
        ep = self._episode
        if ep is None:
            raise SessionError("no episode is open")
        run: Run = ep["run"]
        n, start = run._count, ep["start"]
        if n != ep["n"]:
            raise SessionError("frames were added to the episode's run directly; add every session frame with "
                               "SessionLogger.add_frame so session frames cannot drift")
        took_over = any(r["control_mode"] in ("expert", "blend") for r in run._rows)
        if termination_reason not in ("success", "failure", "timeout", "operator_abort", "system_error"):
            raise ValueError("termination_reason must be success, failure, timeout, operator_abort or system_error")
        if not took_over and rescue_outcome is not None:
            raise ValueError("an episode with no takeover has rescue_outcome None")
        if took_over:
            want = {"failure": ("success", "failure"), "operator_abort": ("unknown",),
                    "system_error": ("unknown",)}.get(termination_reason)
            if want is None:
                raise ValueError("an episode with a takeover is sealed only by Rescue done (failure), Stop "
                                 "(operator_abort) or a fault (system_error)")
            if rescue_outcome not in want:
                raise ValueError(f"with a takeover and termination {termination_reason!r}, rescue_outcome is one of "
                                 f"{want}")
        if n == 0:
            self._episode = None
            run.discard()
            ep["cm"].__exit__(None, None, None)
            return "empty"
        if termination_reason == "operator_abort" and n <= SHORT_STOP_S * self.fps:
            frames = ep["buffer"]
            if frames is None or len(frames) != n:
                raise SessionError("the short stop's frames were not kept")    # cannot happen: the buffer covers it
            self._episode = None
            run.discard()
            ep["cm"].__exit__(None, None, None)
            rows, events = list(run._rows), [e for e in run._events if e["kind"] != "episode_start"]
            self._open_gap(start, short_stop=(start, start + n))
            try:
                self._replay(frames, rows, events)
            except BaseException as e:
                self.broken = f"the short stop's frames could not be moved to the gap ({type(e).__name__}: {e})"
                raise SessionError(self.broken) from e
            return "short_stop"
        tags = sorted(set(labels or []) | ({"stopped"} if termination_reason == "operator_abort" else set()))
        run._session["session_frame_end"] = start + n
        run._session["rescue_outcome"] = rescue_outcome
        run.event("episode_end", source="station", reason=termination_reason)
        try:
            run.end(termination_reason, task_success=task_success, labels=tags, failure_mode=failure_mode)
        except BaseException:
            if run.state != "active":
                self._episode = None
                self.broken = "the open episode could not be sealed and was discarded"
            raise
        self._episode = None
        entry = {"run_id": run.run_id, "episode_in_session": run._session["episode_in_session"],
                 "range": {"start": start, "end": start + n}, "write_status": "pending",
                 "termination": termination_reason, "rescue_outcome": rescue_outcome,
                 "stopped": termination_reason == "operator_abort"}
        self.runs.append(entry)
        self._pending = {"run": run, "cm": ep["cm"], "entry": entry}
        self._add_evidence(start + n - 1, "write", {"run_id": run.run_id, "write_status": "pending"})
        self._flush()
        return "pending"

    def _replay(self, frames: list[dict], rows: list[dict], events: list[dict]) -> None:
        """Move a short stop's frames into the active gap as they were recorded: the observations kept by
        add_frame, and each tick from the run's own record (so a reward added later by complete_transition is
        kept), with terminal flags cleared because a gap is not an episode. The run's events move with their
        frames (the gap starts at the episode's first frame, so frame indices stay the same)."""
        gap: Run = self._gap["run"]
        by_frame: dict[int, list[dict]] = {}
        for e in events:
            by_frame.setdefault(e["frame_index"], []).append(e)

        def move(f):
            for e in by_frame.get(f, []):
                gap._events.append({**e, "event_id": len(gap._events)})
        move(-1)
        for i, (obs, r) in enumerate(zip(frames, rows)):
            tick = {"sent_action": r["sent_action"], "control_mode": r["control_mode"], "obs_ns": r["obs_ns"],
                    "sent_ns": r["sent_ns"], "expert_source": r["expert_source"],
                    "safety_modified": r["safety_modified"], "obs_skew_ns": r["obs_skew_ns"],
                    "inference_id": r["inference_id"], "chunk_offset": r["chunk_offset"], "prompt": r["prompt"],
                    "prompt_rev": UNKNOWN if r["prompt_rev"] == -1 else r["prompt_rev"],
                    "subtask_cmd": r["subtask_cmd"],
                    "control_gen": UNKNOWN if r["control_gen"] == -1 else r["control_gen"]}
            if r["base_valid"]:
                tick["base_action"] = r["base_action"]
            if r["expert_valid"]:
                tick["expert_action"] = r["expert_action"]
            if r["control_mode"] == "blend":
                tick["blend_weight"] = r["blend_weight"]
            if r["reward_valid"]:
                tick["reward"] = r["reward"]
            gap.add_frame(obs, **tick)
            move(i)

    def write_pending(self) -> str | None:
        """Save the sealed run to the dataset and commit it. Returns "committed", "failed", or None when no write is
        pending. A failed write quarantines the run with its sealed termination (dataop/quarantine/<run_id>),
        records a `persistence_error` row, and makes the session end as `fault_stop`. A failed save also stops the
        RunLogger, so no further episode can start (the session faults, as the state machine says)."""
        if self._pending is None:
            return None
        if self.ended_at is not None:
            raise SessionError("the session is finalized")
        p, self._pending = self._pending, None
        run, entry = p["run"], p["entry"]
        error = None
        try:
            p["cm"].__exit__(None, None, None)
        except BaseException as e:                            # an interrupt is recorded too, then raised
            error = e
        frame = max(self.frames - 1, 0)
        if run.state == "committed":                         # the manifest committed: the run is saved
            entry["write_status"] = "committed"
            if error is not None:
                warnings.warn(f"run {run.run_id} was committed, but a step after its commit failed "
                              f"({type(error).__name__}: {error})")
        else:
            detail = f"{type(error).__name__}: {error}" if error is not None else f"run ended as {run.state}"
            if run.state in ("save_failed", "staged") and run.run_path.exists():
                try:
                    store.quarantine(self.root, run.run_path, f"Drive session {self.session_id}: the write of a "
                                                              f"sealed run failed ({detail})")
                    run.state = "quarantined"
                except Exception as qe:
                    detail += f"; quarantine failed ({type(qe).__name__}: {qe})"
            entry["write_status"] = "failed"
            self._add_evidence(frame, "persistence_error", {"run_id": run.run_id, "error": detail})
        self._add_evidence(frame, "write", {"run_id": run.run_id, "write_status": entry["write_status"]})
        self._flush()
        if error is not None and not isinstance(error, Exception):
            raise error
        return entry["write_status"]

    @property
    def pending_writes(self) -> int:
        return 0 if self._pending is None else 1

    # ── the evidence log ───────────────────────────────────────────────────

    def _add_evidence(self, session_frame: int, kind: str, payload: dict) -> None:
        self._evidence.append((int(session_frame), kind, json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                                                    allow_nan=False)))

    def evidence(self, kind: str, payload: dict, session_frame: int | None = None) -> None:
        """One row of the evidence log, at `session_frame` (default: the last recorded frame). Kinds: `state` (from,
        to, event), `criteria` (each success criterion's value and whether all hold, per frame in policy or
        expert), `reset_check` (per frame in reset_wait), `fault`; `write` and `persistence_error` are written by
        the session itself. The payload is a JSON object."""
        self._require_open()
        if kind not in ("state", "criteria", "reset_check", "fault"):
            raise ValueError("evidence kind must be state, criteria, reset_check or fault (the session writes write "
                             "and persistence_error rows itself)")
        if not isinstance(payload, dict):
            raise ValueError("an evidence payload is a JSON object")
        if session_frame is None:
            session_frame = max(self.frames - 1, 0)
        if isinstance(session_frame, bool) or not isinstance(session_frame, int) or \
                not 0 <= session_frame < max(self.frames, 1):
            raise ValueError("session_frame must be a recorded session frame")
        self._add_evidence(session_frame, kind, payload)
        if kind == "fault":
            self._last_fault = payload.get("fault")

    def state(self, frm: str, to: str, event: str | None) -> None:
        """A `state` row: the session runner moved from one state to another on the last recorded tick."""
        self.evidence("state", {"from": frm, "to": to, "event": event})

    def fault(self, kind: str, detail: str | None = None) -> None:
        """A `fault` row (watchdog, camera_lost, bus_lost, lease_lost, ...). The station seals an open episode with
        system_error itself; the fault named last is the session's fault when it ends as fault_stop."""
        if not isinstance(kind, str) or not _SLUG.match(kind):
            raise ValueError("a fault kind is a short name such as camera_lost")
        self.evidence("fault", {"fault": kind, "detail": detail})

    # ── ending ─────────────────────────────────────────────────────────────

    def finalize(self, ended_by: str, *, fault: str | None = None, moved_by: str = "unknown",
                 reset_checks: dict | None = None) -> dict:
        """Every pending write is resolved: close the active gap and write the final session record. A failed write
        makes the session `fault_stop` with the fault `persistence_error` (unless a fault was named), even after a
        user's Stop. Returns the record (session.json). If the gap dataset cannot be finalized it raises and
        records persistence_error; call finalize again (with the same ended_by) to retry."""
        if self._finalizing:
            return self._finish(*self._final)                 # the end decided on the first call
        self._require_open()
        if ended_by not in ENDED_BY:
            raise ValueError(f"ended_by must be one of {ENDED_BY}")
        if self._episode is not None:
            raise SessionError("seal the open episode before the session ends")
        if self._pending is not None:
            raise SessionError("a session cannot end while a write is pending")
        if fault is not None and (not isinstance(fault, str) or not _SLUG.match(fault)):
            raise ValueError("a fault kind is a short name such as camera_lost")
        self._close_gap(moved_by, reset_checks, reset_passed=False)
        failed = any(r["write_status"] == "failed" for r in self.runs)
        if (failed or self._encoder_stop is not None) and ended_by == "user_stop":
            ended_by = "fault_stop"
        if ended_by == "fault_stop":
            fault = fault or ("persistence_error" if failed else
                              "logger_failed" if self._encoder_stop is not None else self._last_fault)
        elif fault is not None:
            raise ValueError("a session ended by the user's Stop names no fault")
        problems = coverage_problems([(r["range"]["start"], r["range"]["end"]) for r in self.runs],
                                     [(g["range"]["start"], g["range"]["end"]) for g in self.gaps], self.frames)
        if problems:
            self.broken = "; ".join(problems)
            raise SessionError(f"the session's frames are not covered exactly once: {self.broken}")
        self._finalizing = True
        self._final = (ended_by, fault)
        return self._finish(ended_by, fault)

    def _finish(self, ended_by: str, fault: str | None) -> dict:
        problems = self._close_gap_dataset()
        if problems:
            detail = "; ".join(problems)
            self._add_evidence(max(self.frames - 1, 0), "persistence_error", {"error": detail})
            self._flush()
            raise SessionError(f"the session could not be finalized: {detail}; call finalize() again to retry")
        self._finalizing = False
        self.ended_at, self.ended_by, self.fault_kind = utc_now(), ended_by, fault
        self._flush()
        return self.record()

    # ── files ──────────────────────────────────────────────────────────────

    def record(self) -> dict:
        return {"session_id": self.session_id, "station_id": self.station_id, "go": self.go,
                "model_id": self.model_id, "planner": None, "task_id": self.task_id, "task_rev": self.task_rev,
                "fps": self.fps, "started_at": self.started_at, "ended_at": self.ended_at, "ended_by": self.ended_by,
                "fault": self.fault_kind, "session_frames": self.frames,
                "runs": [dict(r) for r in self.runs], "gaps": [dict(g) for g in self.gaps]}

    def _write_record(self) -> None:
        store.atomic_write_json(self.dir / "session.json", self.record())

    def _write_evidence(self) -> None:
        cols = list(zip(*self._evidence)) if self._evidence else [[], [], []]
        table = pa.table({"session_frame": pa.array(cols[0], pa.int64()), "kind": pa.array(cols[1], pa.string()),
                          "payload": pa.array(cols[2], pa.string())}, schema=EVIDENCE_SCHEMA)
        store.atomic_write_table(self.dir / "evidence.parquet", table)

    def _flush(self) -> None:
        """The evidence log, then the record, at every boundary (a gap closing, a seal, a write result, the end)."""
        self._write_evidence()
        self._write_record()


# ── reading ───────────────────────────────────────────────────────────────


def read_session(root, session_id: str) -> dict:
    return json.loads((session_dir(root, session_id) / "session.json").read_text())


def read_evidence(root, session_id: str) -> list[dict]:
    """The evidence log as rows {session_frame, kind, payload}, with the payload as a dict, in the order written."""
    t = pq.read_table(session_dir(root, session_id) / "evidence.parquet")
    return [{"session_frame": f, "kind": k, "payload": json.loads(p)}
            for f, k, p in zip(t.column("session_frame").to_pylist(), t.column("kind").to_pylist(),
                               t.column("payload").to_pylist())]


def iter_sessions(root):
    d = sessions_dir(root)
    if not d.is_dir():
        return
    for p in sorted(x for x in d.iterdir() if x.is_dir() and not x.name.startswith(".")):
        yield p


def _range(r) -> tuple[int, int] | None:
    if isinstance(r, dict) and set(r) == {"start", "end"} and _int(r["start"]) and _int(r["end"]) \
            and 0 <= r["start"] < r["end"]:
        return r["start"], r["end"]
    return None


def _check_one_session(root: Path, sdir: Path, errors: list, warnings: list, listed: set) -> None:
    from . import validate as V
    sid = sdir.name
    err = lambda s: errors.append(f"session {sid}: {s}")
    try:
        rec = json.loads((sdir / "session.json").read_text())
    except Exception as e:
        err(f"session.json is missing or unreadable ({type(e).__name__}: {e})")
        return
    if not isinstance(rec, dict) or set(rec) != set(SESSION_KEYS):
        err("session.json does not have exactly the session record's keys")
        return
    if rec["session_id"] != sid:
        err("session_id does not match its folder")
    if not _int(rec["session_frames"]) or rec["session_frames"] < 0:
        err("session_frames is not a frame count")
        return
    for p in record_problems(rec):
        err(p)
    ended = rec["ended_at"] is not None
    if ended != (rec["ended_by"] is not None):
        err("ended_at and ended_by are set together")
    if rec["ended_by"] not in ENDED_BY + (None,):
        err(f"unknown ended_by {rec['ended_by']!r}")
    runs, gaps = rec["runs"] if isinstance(rec["runs"], list) else [], \
        rec["gaps"] if isinstance(rec["gaps"], list) else []
    run_ranges, gap_ranges = [], []
    for i, r in enumerate(runs, 1):
        rng = _range(r.get("range")) if isinstance(r, dict) else None
        if rng is not None and not isinstance(r.get("run_id"), str):
            err(f"run entry {i} has no run_id")
            continue
        if rng is None:
            err(f"run entry {i} has no valid range")
            continue
        run_ranges.append(rng)
        if r.get("episode_in_session") != i:
            err(f"run entry {i} has episode_in_session {r.get('episode_in_session')!r}")
        ws = r.get("write_status")
        if ws not in WRITE_STATUSES:
            err(f"run {r.get('run_id')}: unknown write_status {ws!r}")
        elif ws == "pending":
            (err if ended else lambda s: warnings.append(f"session {sid}: {s}"))(
                f"run {r.get('run_id')}: its write is still pending")
        elif ws == "committed":
            listed.add(str(r.get("run_id")))
            _check_session_run(root, sid, rec, r, rng, err)
        else:
            _check_failed_run(root, r, rng, err)
    gap_root = sdir / "gaps"
    gap_runs = {}
    if gaps:
        res = V.validate(gap_root)
        for e in res["errors"]:
            err(f"gap dataset: {e}")
        for _, m in store.iter_manifests(gap_root, "committed"):
            gap_runs[m["episode_index"]] = m
    for g in gaps:
        rng = _range(g.get("range")) if isinstance(g, dict) else None
        if rng is None:
            err("a gap has no valid range")
            continue
        gap_ranges.append(rng)
        gi = g.get("gap_episode_index")
        m = gap_runs.get(gi) if _int(gi) else None
        sg = (m or {}).get("session_gap") or {}
        if m is None:
            err(f"gap {g.get('gap_id')}: no committed episode {g.get('gap_episode_index')} in the gap dataset")
        elif sg.get("gap_id") != g.get("gap_id") or sg.get("session_id") != sid or \
                (sg.get("session_frame_start"), sg.get("session_frame_end")) != rng or \
                m["length"] != rng[1] - rng[0] or sg.get("short_stop") != g.get("short_stop") or \
                sg.get("moved_by") != g.get("moved_by") or sg.get("reset_checks") != g.get("reset_checks"):
            err(f"gap {g.get('gap_id')}: its gap episode does not match the session record")
        if g.get("moved_by") not in MOVED_BY:
            err(f"gap {g.get('gap_id')}: unknown moved_by {g.get('moved_by')!r}")
        ss = g.get("short_stop")
        if ss is not None:
            sr = _range(ss)
            if sr is None or not (rng[0] <= sr[0] and sr[1] <= rng[1]):
                err(f"gap {g.get('gap_id')}: short_stop is not a range inside the gap")
    if ended:
        for p in coverage_problems(run_ranges, gap_ranges, rec["session_frames"]):
            err(p)
        if any(isinstance(r, dict) and r.get("write_status") == "failed" for r in runs) and \
                rec["ended_by"] != "fault_stop":
            err("a session with a failed write ends with fault_stop")
    else:
        warnings.append(f"session {sid} is open (not finalized)")
    try:
        t = pq.read_table(sdir / "evidence.parquet")
        if t.schema.names != EVIDENCE_SCHEMA.names:
            err(f"evidence.parquet has columns {t.schema.names}")
        else:
            for f, k, p in zip(t.column("session_frame").to_pylist(), t.column("kind").to_pylist(),
                               t.column("payload").to_pylist()):
                if k not in EVIDENCE_KINDS:
                    err(f"evidence row of unknown kind {k!r}")
                    break
                if not _int(f) or not 0 <= f < max(rec["session_frames"], 1):
                    err(f"evidence row at session frame {f} is outside the session")
                    break
                if not isinstance(json.loads(p), dict):
                    err("an evidence payload is not a JSON object")
                    break
    except Exception as e:
        err(f"evidence.parquet is missing or unreadable ({type(e).__name__}: {e})")
    try:
        rv = review_format.load_review((sdir / "review.json").read_text())
    except Exception as e:
        err(f"review.json is missing or unreadable ({type(e).__name__}: {e})")
        return
    try:
        # The split edits in effect replay on every record that covers each session frame once, ended or not (a
        # session closed without finalize keeps a whole timeline); a record still recording, or a broken one, is
        # checked for the review's id, kinds and frames only.
        whole = not coverage_problems(run_ranges, gap_ranges, rec["session_frames"]) \
            and len(run_ranges) == len(runs) and len(gap_ranges) == len(gaps)
        check_against_session(rv, rec, replay=whole)
    except (ValueError, KeyError, TypeError) as e:
        err(f"review.json: {e}")


# ---------------------------------------------------------------- the session review (Aftershock issue #447)
#
# A session's review (review.json) is review format 2 keyed by the session id in run_id. It holds only the three split
# edits (review_format.SESSION_KINDS), whose frames are session frames. The split edits in effect apply one after
# another in acceptance order to the recorded timeline (each run an episode, each gap a gap). This is the same rule
# as the Aftershock cloud's session store, so a review the logger accepts the cloud accepts too.


def episode_id_for(session_id: str, edit_id: str, start: int, end: int) -> str:
    """The id of an episode a split edit made: uuid4 shaped, derived from the session, the edit and the episode's
    session frames [start, end), so replaying the same edits gives the same ids."""
    h = hashlib.sha256(f"aftershock.episode:{session_id}:{edit_id}:{start}:{end}".encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-4{h[13:16]}-{'89ab'[int(h[16], 16) & 3]}{h[17:20]}-{h[20:32]}"


def _sources(rec: dict) -> dict[str, tuple[int, int]]:
    """The recorded sources: every run by its id, and every short stop kept in a gap as `short_stop:<gap_id>`."""
    out = {r["run_id"]: (r["range"]["start"], r["range"]["end"]) for r in rec["runs"]}
    out.update({f"short_stop:{g['gap_id']}": (g["short_stop"]["start"], g["short_stop"]["end"])
                for g in rec["gaps"] if g["short_stop"] is not None})
    return out


def _seg(kind: str, seg_id, start: int, end: int, src: dict, confirmed: bool) -> dict:
    runs = () if kind == "gap" else tuple(k for k, (s, e) in sorted(src.items(), key=lambda kv: kv[1])
                                          if s < end and start < e)
    return {"kind": kind, "id": seg_id, "start": start, "end": end, "from_runs": list(runs),
            "confirmed": True if kind == "gap" else confirmed}


def recorded_timeline(rec: dict) -> list[dict]:
    """The timeline before any split edit, in session frame order: each run an episode with the run's id
    (confirmed), each gap a gap. The record must cover every session frame once."""
    problems = coverage_problems([(r["range"]["start"], r["range"]["end"]) for r in rec["runs"]],
                                 [(g["range"]["start"], g["range"]["end"]) for g in rec["gaps"]], rec["session_frames"])
    if problems:
        raise ValueError("; ".join(problems))
    src = _sources(rec)
    segs = [_seg("episode", r["run_id"], r["range"]["start"], r["range"]["end"], src, True) for r in rec["runs"]]
    segs += [_seg("gap", g["gap_id"], g["range"]["start"], g["range"]["end"], src, True) for g in rec["gaps"]]
    return sorted(segs, key=lambda x: x["start"])


def apply_split_edit(segs: list[dict], kind: str, p: dict, session_id: str, edit_id: str, src: dict) -> list[dict]:
    """One accepted split edit on the timeline so far. ValueError when it does not fit (no boundary at `at`, no
    episode starting at `start` and a later one ending at `end`, no episode holding `at`). Every episode the edit
    changes needs review and gets `episode_id_for(session_id, edit_id, start, end)`; a gap keeps its id."""
    def new(kind_, old_id, a, b):
        return _seg(kind_, old_id if kind_ == "gap" else episode_id_for(session_id, edit_id, a, b), a, b, src, False)

    if kind == "episode_boundary":
        i = next((k for k in range(len(segs) - 1) if segs[k]["end"] == p["at"]), None)
        if i is None:
            raise ValueError(f"no boundary at session frame {p['at']}")
        a, b = segs[i], segs[i + 1]
        if a["kind"] == "gap" and b["kind"] == "gap":
            raise ValueError(f"the boundary at session frame {p['at']} is between two gaps")
        if p["to"] == p["at"]:
            return list(segs)
        if not a["start"] < p["to"] < b["end"]:
            raise ValueError("a moved boundary leaves both segments at least one frame")
        return segs[:i] + [new(a["kind"], a["id"], a["start"], p["to"]),
                           new(b["kind"], b["id"], p["to"], b["end"])] + segs[i + 2:]
    if kind == "episode_merge":
        i = next((k for k, x in enumerate(segs) if x["start"] == p["start"] and x["kind"] == "episode"), None)
        j = next((k for k, x in enumerate(segs) if x["end"] == p["end"] and x["kind"] == "episode"), None)
        if i is None or j is None or not i < j:
            raise ValueError(f"no two episodes start at {p['start']} and end at {p['end']}")
        return segs[:i] + [new("episode", None, segs[i]["start"], segs[j]["end"])] + segs[j + 1:]
    if kind == "episode_split":
        i = next((k for k, x in enumerate(segs) if x["start"] < p["at"] < x["end"]), None)
        if i is None or segs[i]["kind"] != "episode":
            raise ValueError(f"no episode holds session frame {p['at']} with frames on both sides")
        e = segs[i]
        return segs[:i] + [new("episode", None, e["start"], p["at"]), new("episode", None, p["at"], e["end"])] \
            + segs[i + 1:]
    raise ValueError(f"{kind} is not a split edit")


def effective_timeline(rec: dict, review: dict) -> list[dict]:
    """The session timeline after the review's split edits in effect, applied in acceptance order: segments
    `{kind (episode or gap), id, start, end, from_runs, confirmed}`. An episode that again matches a committed run
    exactly takes the run's id (its outcome still needs review). ValueError when an edit does not fit, or when an
    edit changes the frames of a run whose write failed: those frames stay one episode of their own."""
    src = _sources(rec)
    failed = {(r["run_id"], r["range"]["start"], r["range"]["end"]) for r in rec["runs"]
              if r["write_status"] == "failed"}
    segs = recorded_timeline(rec)
    for e in review_format.active_edits(review):
        if e["kind"] not in review_format.SESSION_KINDS:
            raise ValueError(f"edit {e['edit_id']} ({e['kind']}): a session review takes only split edits")
        try:
            segs = apply_split_edit(segs, e["kind"], e["payload"], rec["session_id"], e["edit_id"], src)
        except ValueError as err:
            raise ValueError(f"edit {e['edit_id']} ({e['kind']}): {err}") from None
        kept = {(x["id"], x["start"], x["end"]) for x in segs if x["kind"] == "episode"}
        for run_id, _, _ in sorted(failed - kept):
            raise ValueError(f"edit {e['edit_id']} ({e['kind']}): run {run_id} failed to write; no split edit may "
                             "change its frames")
    committed = {(r["range"]["start"], r["range"]["end"]): r["run_id"] for r in rec["runs"]
                 if r["write_status"] == "committed"}
    for x in segs:
        if x["kind"] == "episode" and (x["start"], x["end"]) in committed:
            x["id"] = committed[(x["start"], x["end"])]
    return segs


def check_against_session(review: dict, rec: dict, replay: bool = True) -> None:
    """A session review fits its session record: it names the session, every edit and every history version is a
    split edit inside the session's frames, and (with `replay`, which needs a record that covers every frame) the
    edits in effect apply. ValueError otherwise (fail closed), as review_format.check_against_manifest is for a run."""
    if review["run_id"] != rec["session_id"]:
        raise ValueError("the review is for another session")
    n = rec["session_frames"]
    for e in review_format.all_edit_versions(review):
        if e["kind"] not in review_format.SESSION_KINDS:
            raise ValueError(f"edit {e['edit_id']} ({e['kind']}): a session review takes only split edits")
        for name in ("at", "to", "start", "end"):
            v = e["payload"].get(name)
            if v is not None and v > n:
                raise ValueError(f"edit {e['edit_id']} ({e['kind']}): {name} {v} is past the session's {n} frames")
    if replay:
        effective_timeline(rec, review)


def session_problems(root) -> tuple[list[str], list[str], int]:
    """Check every session folder against the runs, the gap dataset and the evidence log. Returns errors, warnings
    and the number of sessions. A dataset with no sessions folder (every dataset before L9) has nothing to check."""
    from . import validate as V
    root = Path(root)
    errors, warnings, count = [], [], 0
    listed: set[str] = set()
    for sdir in iter_sessions(root):
        count += 1
        try:
            _check_one_session(root, sdir, errors, warnings, listed)
        except Exception as e:                                  # report, keep checking the other sessions
            errors.append(f"session {sdir.name}: could not be checked ({type(e).__name__}: {e})")
    for _, m in store.iter_manifests(root, "committed"):
        if m.get("session_id") is not None and m["run_id"] not in listed:
            errors.append(f"{m['run_id']}: belongs to session {m['session_id']}, whose record does not list it as "
                          "committed")
    return errors, warnings, count


RUN_ENTRY_KEYS = ("run_id", "episode_in_session", "range", "write_status", "termination", "rescue_outcome",
                  "stopped")
GAP_ENTRY_KEYS = ("gap_id", "range", "gap_episode_index", "moved_by", "reset_checks", "short_stop")


def record_problems(rec: dict) -> list[str]:
    """Field and type checks of a session record, as the cloud session models make them."""
    out = []
    for k in ("station_id", "model_id", "task_id"):
        if not _str(rec[k]):
            out.append(f"{k} must be a non-empty string")
    if not _int(rec["task_rev"]) or rec["task_rev"] < 1:
        out.append("task_rev must be an int >= 1")
    if isinstance(rec["fps"], bool) or not isinstance(rec["fps"], (int, float)) or not rec["fps"] > 0:
        out.append("fps must be a positive number")
    if rec["planner"] is not None:
        out.append("planner is null (no planner in this version)")
    for k in ("started_at", "ended_at"):
        if not (rec[k] is None and k == "ended_at") and not (isinstance(rec[k], str) and _UTC.match(rec[k])):
            out.append(f"{k} must be a UTC time ending in Z")
    if rec["fault"] is not None and not (isinstance(rec["fault"], str) and _SLUG.match(rec["fault"])):
        out.append("fault must be a short name or null")
    if rec["ended_by"] == "user_stop" and rec["fault"] is not None:
        out.append("a session ended by the user's Stop names no fault")
    out += go_problems(rec["go"])
    if not isinstance(rec["runs"], list) or not isinstance(rec["gaps"], list):
        return out + ["runs and gaps must be lists"]
    for r in rec["runs"]:
        if not isinstance(r, dict) or set(r) != set(RUN_ENTRY_KEYS):
            out.append(f"a run entry must have exactly {RUN_ENTRY_KEYS}")
            continue
        rid = r["run_id"]
        if not isinstance(rid, str) or not _UUID4.match(rid):
            out.append(f"run entry {rid!r}: run_id must be a uuid4")
        if r["termination"] not in ("success", "failure", "timeout", "operator_abort", "system_error"):
            out.append(f"run {rid}: unknown termination {r['termination']!r}")
        if r["rescue_outcome"] not in RESCUE_OUTCOMES + (None,):
            out.append(f"run {rid}: unknown rescue_outcome {r['rescue_outcome']!r}")
        elif r["rescue_outcome"] is not None and r["termination"] not in ("failure", "operator_abort", "system_error"):
            out.append(f"run {rid}: a rescue outcome goes with failure, operator_abort or system_error")
        if r["stopped"] is not (r["termination"] == "operator_abort"):
            out.append(f"run {rid}: stopped is true exactly for operator_abort")
    for g in rec["gaps"]:
        if not isinstance(g, dict) or set(g) != set(GAP_ENTRY_KEYS):
            out.append(f"a gap entry must have exactly {GAP_ENTRY_KEYS}")
            continue
        if not (isinstance(g["gap_id"], str) and _SLUG.match(g["gap_id"])):
            out.append(f"gap {g['gap_id']!r}: gap_id must be a short name")
        if not _int(g["gap_episode_index"]) or g["gap_episode_index"] < 0:
            out.append(f"gap {g['gap_id']}: gap_episode_index must be an int >= 0")
        rc = g["reset_checks"]
        if not isinstance(rc, dict) or set(rc) != set(RESET_CHECKS) or \
                any(v is not None and not isinstance(v, bool) for v in rc.values()):
            out.append(f"gap {g['gap_id']}: reset_checks must have exactly {RESET_CHECKS}, each true, false or null")
    return out


def _check_failed_run(root: Path, r: dict, rng: tuple[int, int], err) -> None:
    """A failed write is quarantined with its sidecar: the manifest and ticks, whole and of the run's length."""
    from . import validate as V
    rid = str(r.get("run_id"))
    q = store.quarantine_dir(root) / rid
    try:
        m = store.read_manifest(q)
    except Exception:
        err(f"run {rid} failed to write and its record is not in quarantine")
        return
    if m.get("run_id") != rid or m.get("length") != rng[1] - rng[0] or \
            (m.get("session_frame_start"), m.get("session_frame_end")) != rng:
        err(f"run {rid}: its quarantined manifest does not match the session record")
    bad = V.integrity_problems(q, m) if isinstance(m.get("integrity"), dict) else ["no integrity record"]
    for p in bad:
        err(f"run {rid} (quarantined): {p}")
    if not bad:
        fi = pq.read_table(q / "ticks.parquet", columns=["frame_index"]).column("frame_index").to_pylist()
        if fi != list(range(rng[1] - rng[0])):
            err(f"run {rid} (quarantined): its ticks are not frames 0 to {rng[1] - rng[0] - 1} of its range")


def _check_session_run(root: Path, sid: str, rec: dict, r: dict, rng: tuple[int, int], err) -> None:
    rid = str(r.get("run_id"))
    try:
        m = store.read_manifest(store.runs_dir(root) / rid)
    except Exception:
        err(f"run {rid} is listed as committed but has no readable manifest")
        return
    if m.get("status") != "committed":
        err(f"run {rid} is listed as committed but is {m.get('status')!r}")
    want = {"session_id": sid, "episode_in_session": r.get("episode_in_session"), "session_frame_start": rng[0],
            "session_frame_end": rng[1], "go_ref": rec["go"], "termination_reason": r.get("termination"),
            "rescue_outcome": r.get("rescue_outcome")}
    for k, v in want.items():
        if m.get(k) != v:
            err(f"run {rid}: manifest {k} is {m.get(k)!r}, the session record says {v!r}")
    if ("stopped" in (m.get("labels") or [])) != bool(r.get("stopped")):
        err(f"run {rid}: the stopped tag disagrees with the session record")
