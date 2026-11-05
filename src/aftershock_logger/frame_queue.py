"""Bounded frame queue between a robot's control loop and LeRobot (logger task L5).

The control thread calls queue.put() once per tick. put() checks the frame, copies it and returns at once; a writer
thread hands frames to LeRobot in order. The queue holds at most `capacity` frames (2 s of frames by default), so
memory stays bounded however slow the disk is. When the queue is full, or a write fails (for example a full disk),
put() raises ControlledStop: the caller holds the robot and leaves the run block, and the logger writes every frame
it accepted to dataop/quarantine/<run_id>/ with the reason. Such a run is never committed as complete.
"""
from __future__ import annotations

import collections
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
WRITER_TIMEOUT_S = 10.0          # how long a stop or discard waits for a frame LeRobot is still writing


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


def _copy_frame(frame: dict) -> dict:
    """The caller may reuse its arrays on the next tick, so the queue keeps its own copies."""
    out = {}
    for k, v in frame.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif hasattr(v, "detach") and hasattr(v, "clone"):          # a torch tensor
            out[k] = v.detach().clone()
        else:
            out[k] = v
    return out


def _image_keys(dataset) -> set[str]:
    feats = getattr(dataset, "features", None) or {}
    return {k for k, f in feats.items() if isinstance(f, dict) and f.get("dtype") in ("image", "video")}


class FrameQueue:
    """Create it with run.queue(); see the module docstring."""

    def __init__(self, run, max_seconds: float = 2.0, max_frames: int | None = None):
        if max_frames is None:
            if not (isinstance(max_seconds, (int, float)) and math.isfinite(max_seconds) and max_seconds > 0):
                raise ValueError("max_seconds must be a positive number")
            max_frames = max(1, math.ceil(max_seconds * run.schema.rate_hz))
        if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1:
            raise ValueError("max_frames must be an integer >= 1")
        self.run = run
        self.capacity = max_frames
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

    def check(self) -> None:
        """Raise ControlledStop if a write failed; lets the loop notice a failure without putting a frame."""
        if self._stop is not None:
            raise self._stop

    def put(self, frame: dict, **kw) -> int:
        """Accept one frame (the same arguments as run.add_frame) without waiting for LeRobot. Returns its index.

        Raises ValueError for a bad frame (nothing is accepted), and ControlledStop when the queue is full or a
        write failed. The frame that finds the queue full is still kept, so every tick sent to the robot is
        recorded somewhere."""
        self.run._require("active")
        if self._stop is not None:
            raise self._stop
        frame = _copy_frame(frame)
        observation = self._observation(frame)
        frame, row = self.run._prepare(frame, **kw)
        with self._cv:
            if self._stop is not None:                              # a write failed while this frame was checked
                self._items.append((frame, row, observation))
                raise self._stop
            full = len(self._items) >= self.capacity
            self._items.append((frame, row, observation))
            self.max_depth = max(self.max_depth, len(self._items))
            if full:
                self._stop = ControlledStop(QUEUE_FULL, f"{self.capacity} frames were waiting to be written",
                                            self.run.run_id)
            self._cv.notify_all()
        if full:
            raise self._stop
        return row["frame_index"]

    def drain(self, timeout: float | None = None) -> None:
        """Wait until every queued frame is written (at most `timeout` s, WRITER_TIMEOUT_S by default), then stop
        the writer. Raises ControlledStop on a failure or a timeout."""
        timeout = WRITER_TIMEOUT_S if timeout is None else timeout
        with self._cv:
            self._closing = True
            self._cv.notify_all()
            ok = self._cv.wait_for(lambda: self._stop is not None or (not self._items and self._inflight is None),
                                   timeout)
            if self._stop is None and not ok:
                self._stop = ControlledStop(WRITE_FAILED, f"frames were still being written after {timeout} s",
                                            self.run.run_id)
        if self._stop is not None:
            raise self._stop
        self._thread.join()

    def close(self, timeout: float | None = None) -> bool:
        """Stop the writer after the frame it is writing. Returns False if it did not finish in time (WRITER_TIMEOUT_S
        by default); it then never appends another row, so a snapshot taken now stays complete."""
        timeout = WRITER_TIMEOUT_S if timeout is None else timeout
        with self._cv:
            self._closing = True
            if self._stop is None:
                self._abandoned = True                              # a discard: queued frames are not written
            self._cv.notify_all()
        self._thread.join(timeout)
        if self._thread.is_alive():
            with self._cv:
                self._abandoned = True
            return False
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
            try:
                self.run._write_frame(item[0], item[1])
            except BaseException as e:                              # noqa: BLE001 (reported to the control thread)
                with self._cv:
                    if self._stop is None:
                        self._stop = ControlledStop(WRITE_FAILED, f"{type(e).__name__}: {e}", self.run.run_id)
                    self._cv.notify_all()
                return
            with self._cv:
                if self._abandoned:
                    return
                self._inflight = None
                self._written.append(item[1])
                self._observations.append(item[2])
                self.run._rows.append(item[1])
                self._cv.notify_all()


def write_partial(run, rows: list[dict], observations: list[dict], info: dict) -> Path:
    """Write a stopped run's frames to dataop/quarantine/<run_id>/: all files go to a temporary folder first, which is
    renamed into place only when complete, so a quarantine folder is never half written."""
    qdir = store.quarantine_dir(run.logger.root)
    dest = qdir / run.run_id
    if dest.exists():
        raise RunError(f"{dest} already exists")
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
        os.rename(tmp, dest)
        store._fsync_dir(qdir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return dest


def _observations_table(observations: list[dict]) -> pa.Table:
    keys = sorted({k for o in observations for k in o})
    cols = {"frame_index": pa.array(range(len(observations)), pa.int64())}
    for k in keys:
        cols[k] = pa.array([o.get(k) for o in observations], pa.list_(pa.float64()))
    return pa.table(cols)
