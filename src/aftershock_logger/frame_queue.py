"""Bounded frame queue between a robot's control loop and LeRobot (logger task L5).

The control thread calls queue.put() once per tick. put() checks the frame, copies it and returns at once; a writer
thread hands frames to LeRobot in order. The queue holds at most `capacity` frames (2 s of frames by default), so
memory stays bounded however slow the disk is. When the queue is full, or a write fails (for example a full disk),
put() raises ControlledStop: the caller holds the robot and leaves the run block, and the logger writes every frame
it accepted to dataop/quarantine/<run_id>/ with the reason. Such a run is never committed as complete.
"""
from __future__ import annotations

import collections
import copy
import json
import math
import os
import shutil
import tempfile
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa

from . import store
from .logger import PRED_SCHEMA, RunError

QUEUE_FULL = "queue_full"
WRITE_FAILED = "write_failed"
STALL_TIMEOUT_S = 10.0           # run.end() waits this long without any frame finishing before it calls the writer hung
STOP_TIMEOUT_S = 10.0            # a stop or discard waits this long for the frame LeRobot is writing


class ControlledStop(RunError):
    """The logger cannot keep up or cannot write: hold the robot and end the run.

    reason is "queue_full" or "write_failed"; detail says what happened. After the run block exits, frames counts
    every frame the logger accepted, and quarantine_path says where they were written, or quarantine_error says why
    they could not be (they are then kept in memory for run.retry_quarantine())."""

    def __init__(self, reason: str, detail: str, run_id: str):
        super().__init__(f"controlled stop of run {run_id}: {reason} ({detail})")
        self.reason = reason
        self.detail = detail
        self.run_id = run_id
        self.frames: int | None = None
        self.quarantine_path: Path | None = None
        self.quarantine_error: str | None = None
        self.spilled = False                                        # written to the spill folder, not the dataset
        self.close_error: str | None = None                         # what closing the RunLogger reported

    def _settle(self) -> None:
        """Say in the message where the frames went, or that they were never written."""
        head = f"controlled stop of run {self.run_id}: {self.reason} ({self.detail})"
        if self.quarantine_path is not None and self.spilled:
            tail = (f"{self.frames} frames could not be written to the dataset and were spilled to "
                    f"{self.quarantine_path}; move that folder into the dataset's dataop/quarantine/ when there is space")
        elif self.quarantine_path is not None:
            tail = f"{self.frames} frames quarantined at {self.quarantine_path}"
        elif self.quarantine_error is not None:
            tail = (f"{self.frames} frames were NOT written ({self.quarantine_error}); they are only in memory: "
                    "call run.retry_quarantine(), or close the RunLogger, which retries and then spills them")
        else:
            tail = None
        parts = [head] + ([tail] if tail else []) + ([f"closing the logger: {self.close_error}"]
                                                     if self.close_error and not self.spilled else [])
        self.args = ("; ".join(parts),)


def _copy_frame(frame: dict) -> dict:
    """The caller may reuse its values on the next tick, so the queue keeps its own copies. Tensors are copied to
    the CPU so queued frames never hold GPU memory."""
    out = {}
    for k, v in frame.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif hasattr(v, "detach") and hasattr(v, "to"):             # a torch tensor
            out[k] = v.detach().to("cpu", copy=True)
        elif isinstance(v, (str, bytes, int, float, bool, type(None))):
            out[k] = v
        else:                                                       # lists, dicts, PIL images
            out[k] = copy.deepcopy(v)
    return out


def _image_keys(dataset) -> set[str]:
    feats = getattr(dataset, "features", None) or {}
    return {k for k, f in feats.items() if isinstance(f, dict) and f.get("dtype") in ("image", "video")}


class FrameQueue:
    """Create it with run.queue(); see the module docstring."""

    def __init__(self, run, max_seconds: float = 2.0, max_frames: int | None = None,
                 stall_timeout: float | None = None, stop_timeout: float | None = None):
        if max_frames is None:
            if not (isinstance(max_seconds, (int, float)) and math.isfinite(max_seconds) and max_seconds > 0):
                raise ValueError("max_seconds must be a positive number")
            max_frames = max(1, math.ceil(max_seconds * run.schema.rate_hz))
        if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1:
            raise ValueError("max_frames must be an integer >= 1")
        stall_timeout = STALL_TIMEOUT_S if stall_timeout is None else stall_timeout
        stop_timeout = STOP_TIMEOUT_S if stop_timeout is None else stop_timeout
        for name, v in (("stall_timeout", stall_timeout), ("stop_timeout", stop_timeout)):
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be a number of seconds >= 0")
        self.run = run
        self.capacity = max_frames
        self.stall_timeout = float(stall_timeout)
        self.stop_timeout = float(stop_timeout)
        self._busy = False                                         # the writer is inside LeRobot's add_frame
        self._stop_told = False                                    # the caller has been given the ControlledStop
        self._writer_stuck = False
        self._image_keys = _image_keys(run.dataset)
        self._cv = threading.Condition()
        self._items: collections.deque = collections.deque()       # (frame, row, observation), oldest first
        self._inflight = None                                      # the item LeRobot is writing now
        self._written: list[dict] = []                             # rows LeRobot accepted, in order
        self._observations: list[dict] = []                        # low-dimensional frame values of every frame
        self._stop: ControlledStop | None = None
        self._closing = False
        self._abandoned = False
        self.max_depth = 0                                         # the most frames ever waiting at once
        self._thread = threading.Thread(target=self._loop, name=f"frame-writer-{run.run_id[:8]}", daemon=True)
        self._thread.start()

    # ── control thread ───────────────────────────────────────────────────────

    @property
    def stopped(self) -> ControlledStop | None:
        """The ControlledStop once the queue has stopped, else None."""
        return self._stop

    def depth(self) -> int:
        with self._cv:
            return len(self._items)

    def _tell(self):
        self._stop_told = True
        return self._stop

    def check(self) -> None:
        """Raise ControlledStop if a write failed; lets the loop notice a failure without putting a frame."""
        if self._stop is not None:
            raise self._tell()

    def put(self, frame: dict, **kw) -> int:
        """Accept one frame (the same arguments as run.add_frame) without waiting for LeRobot. Returns its index.

        Raises ValueError for a bad frame (nothing is accepted), and ControlledStop when the queue is full or a
        write failed. The frame that finds the queue full is still kept, so every tick sent to the robot is
        recorded somewhere. After a write failure, the first put() still keeps its frame (its action already reached
        the robot) before it raises; once the caller has been told of the stop, put() keeps nothing more, so memory
        stays bounded."""
        self.run._require("active")
        if self._stop is not None and self._stop_told:
            raise self._stop
        try:
            frame = _copy_frame(frame)
            observation = self._observation(frame)
            frame, row = self.run._prepare(frame, **kw)
        except Exception as e:
            if self._stop is not None:                              # still tell the caller to stop
                raise self._tell() from e
            raise
        with self._cv:
            if self._stop is not None:                              # a write failed before or while this was checked
                self._items.append((frame, row, observation))
                self.max_depth = max(self.max_depth, len(self._items))
                raise self._tell()
            full = len(self._items) >= self.capacity
            self._items.append((frame, row, observation))
            self.max_depth = max(self.max_depth, len(self._items))
            if full:
                self._stop = ControlledStop(QUEUE_FULL, f"{self.capacity} frames were waiting to be written",
                                            self.run.run_id)
            self._cv.notify_all()
        if full:
            raise self._tell()
        return row["frame_index"]

    def drain(self) -> None:
        """Wait until every queued frame is written, then stop the writer. A slow writer is fine as long as frames
        keep finishing: only stall_timeout seconds without any frame finishing counts as a hung writer. Raises
        ControlledStop on a failure or a hung writer."""
        done = lambda: self._stop is not None or (not self._items and self._inflight is None)  # noqa: E731
        with self._cv:
            self._closing = True
            self._cv.notify_all()
            while not done():
                seen = len(self._written)
                if not self._cv.wait_for(lambda: done() or len(self._written) != seen, self.stall_timeout):
                    self._writer_stuck = True
                    self._stop = ControlledStop(WRITE_FAILED, f"no frame finished writing in {self.stall_timeout} s "
                                                f"with {len(self._items) + 1} frames left", self.run.run_id)
        if self._stop is not None:
            raise self._tell()
        self._thread.join()

    def close(self) -> bool:
        """Stop the writer after the frame it is writing. Returns False if it did not finish within stop_timeout (no
        wait at all once drain() found it hung); it then never appends another row, so a snapshot taken now stays
        complete."""
        with self._cv:
            self._closing = True
            if self._stop is None:
                self._abandoned = True                              # a discard: queued frames are not written
            self._cv.notify_all()
        self._thread.join(0 if self._writer_stuck else self.stop_timeout)
        with self._cv:
            if self._thread.is_alive() and self._busy:
                self._abandoned = True
                return False
        self._thread.join()                                         # it has finished its frame and is exiting
        return True

    def snapshot(self) -> tuple[list[dict], list[dict], list[dict], list[str]]:
        """Rows LeRobot accepted, rows still waiting (the failed one first), every frame's low-dimensional values
        and the names of the image features that are not kept."""
        with self._cv:
            written = list(self._written)
            pending = ([self._inflight] if self._inflight is not None else []) + list(self._items)
            observations = list(self._observations) + [it[2] for it in pending]
            return written, [it[1] for it in pending], observations, sorted(self._image_keys)

    # ── writer thread ────────────────────────────────────────────────────────

    def _observation(self, frame: dict) -> dict:
        out = {}
        for k, v in frame.items():
            if k in self._image_keys or k == "task":
                continue
            if hasattr(v, "detach") and hasattr(v, "cpu"):
                v = v.detach().cpu().numpy()
            a = np.asarray(v)
            if a.dtype.kind in "biuf":
                out[k] = a.reshape(-1).tolist()
        return out

    def _loop(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._items or self._closing or self._stop is not None)
                if self._stop is not None or self._abandoned or not self._items:
                    self._cv.notify_all()
                    return
                item = self._items.popleft()
                self._inflight = item
                self._busy = True
            try:
                self.run._write_frame(item[0], item[1])
            except BaseException as e:                              # noqa: BLE001 (reported to the control thread)
                self._busy = False
                with self._cv:
                    if self._stop is None:
                        self._stop = ControlledStop(WRITE_FAILED, f"{type(e).__name__}: {e}", self.run.run_id)
                    self._cv.notify_all()
                return
            self._busy = False                                      # finished; close() may wait for the append
            with self._cv:
                if self._abandoned:
                    return
                self._inflight = None
                self._written.append(item[1])
                self._observations.append(item[2])
                self.run._rows.append(item[1])
                self._cv.notify_all()


def write_partial(run, rows: list[dict], observations: list[dict], info: dict, qdir: Path | None = None) -> Path:
    """Write a stopped run's frames to dataop/quarantine/<run_id>/ (or to qdir/<run_id>/): all files go to a temporary
    folder first, which is renamed into place only when complete, so a quarantine folder is never half written."""
    qdir = store.quarantine_dir(run.logger.root) if qdir is None else Path(qdir)
    dest = qdir / run.run_id
    if dest.exists():
        try:                                                        # an earlier try renamed it into place
            m = json.loads((dest / "manifest.json").read_text())
        except Exception:
            m = {}
        if m.get("run_id") == run.run_id and m.get("length") == len(rows) and m.get("status") == "quarantined":
            return dest
        raise RunError(f"{dest} already exists and is not this run's record")
    qdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=qdir, prefix=f".{run.run_id}.", suffix=".tmp"))
    try:
        integrity = {"ticks_sha256": None, "events_sha256": store.json_sha256(run._events),
                     "predictions_sha256": None, "frames_sha256": None}
        if rows:
            ticks = store.table_bytes(run._ticks_table(rows))
            store.atomic_write_bytes(tmp / "ticks.parquet", ticks)
            integrity["ticks_sha256"] = store.sha256(ticks)
        frames = store.table_bytes(_observations_table(observations))
        store.atomic_write_bytes(tmp / "frames.parquet", frames)
        integrity["frames_sha256"] = store.sha256(frames)
        if run._predictions is not None:
            preds = store.table_bytes(pa.Table.from_pylist(run._predictions, schema=PRED_SCHEMA))
            store.atomic_write_bytes(tmp / "predictions.parquet", preds)
            integrity["predictions_sha256"] = store.sha256(preds)
        run._integrity = integrity
        manifest = (run._manifest("quarantined", rows, quarantine=info) if rows else
                    {"schema_version": None, "run_id": run.run_id, "status": "quarantined", "length": 0,
                     "task": run.task, "events": run._events, "integrity": integrity, "quarantine": info})
        store.atomic_write_json(tmp / "manifest.json", manifest)
        store.atomic_write_bytes(tmp / "QUARANTINED.txt",
                                 f"partial run stopped: {info['reason']} ({info['detail']}); {len(rows)} frames\n"
                                 .encode())
        os.rename(tmp, dest)                                       # the commit point
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    try:
        store._fsync_dir(qdir)
    except OSError as e:                                            # the record is in place; say the sync failed
        import warnings
        warnings.warn(f"quarantine record {dest} is written but its folder could not be synced ({e})")
    return dest


def _observations_table(observations: list[dict]) -> pa.Table:
    keys = sorted({k for o in observations for k in o})
    cols = {"frame_index": pa.array(range(len(observations)), pa.int64())}
    for k in keys:
        cols[k] = pa.array([o.get(k) for o in observations], pa.list_(pa.float64()))
    return pa.table(cols)
