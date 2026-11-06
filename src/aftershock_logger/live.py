"""Live tap (logger task L10): a cheap, local view of the run being recorded, for a station process to read.

With RunLogger(..., live=True) the control loop's thread only updates a small in-memory snapshot and appends
transition records to an in-memory queue: O(1) work per frame, no file I/O and no image encoding. A daemon tap thread
writes, under <root>/dataop/live/:

- status.json     the latest snapshot, replaced by a temp file and a rename (a reader never sees half a file), at
                  least once a second while the RunLogger is open and at once on a transition; `closed: true` on close.
- events.jsonl    one record per transition with a `seq` that only goes up, also across writer restarts. It holds at
                  most ROTATE_RECORDS records; past that it is renamed to events.1.jsonl (the previous one is dropped)
                  and the new file starts with a `file_start` line naming its first `seq`.
- preview/<camera>.jpg   only while a reader keeps preview.want fresh (mtime under PREVIEW_WANT_S old).
- error.json      when the tap failed; the tap is then off until the next run starts.

Nothing here is part of the dataset or the sidecar: validate, import, export and upload ignore dataop/live/. A tap
error never reaches the caller: every hook catches its own errors, and the recorded run is the same byte for byte.
Writes are not fsynced (the files describe the present, not a record), but each is crash safe: a process that dies
mid-write leaves the previous whole status.json, and a torn last line of events.jsonl is cut off by the next writer
and skipped by readers.
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import math
import os
import re
import socket
import threading
import time
import warnings
from pathlib import Path

import numpy as np

from . import store

LIVE_FORMAT = 1
LIVE_DIR = "live"
STATUS = "status.json"
EVENTS = "events.jsonl"
EVENTS_OLD = "events.1.jsonl"
ERROR = "error.json"
PREVIEW_WANT = "preview.want"
PREVIEW_DIR = "preview"

WRITE_EVERY_S = 0.5            # the tap thread's period: status at least once a second, at most every 0.5 s otherwise
ROTATE_RECORDS = 10_000        # records per events file before it rotates
PENDING_MAX = 10_000           # transition records waiting in memory for the tap thread; older ones are dropped
PREVIEW_WANT_S = 10.0          # a preview request lapses when preview.want is this old
PREVIEW_DEFAULT_S = 2.0        # one frame per camera every interval_s seconds unless preview.want says otherwise
PREVIEW_MIN_S = 1.0
PREVIEW_WIDTH = 320            # preview frames are at most this wide
PREVIEW_QUALITY = 70           # JPEG quality
JOIN_S = 2.0                   # close() waits at most this long for the tap thread
GAP_PERIODS = 1.5              # REC-04: a camera with no new frame for this many frame periods has a gap (default)
LONG_GAP_S = 1.0               # REC-04: a gap longer than this flags the run for review (default)
IMAGE_PREFIX = "observation.images."
STATE_KEY = "observation.state"


def live_dir(root) -> Path:
    return store.dataop_dir(Path(root)) / LIVE_DIR


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def camera_name(key: str) -> str:
    """observation.images.top -> top; any other image key keeps its name. Safe as a file name."""
    name = key[len(IMAGE_PREFIX):] if key.startswith(IMAGE_PREFIX) else key
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "camera"


class _Scrub:
    """Removes the dataset root, the home folder and the host name from every string written to a live file."""

    def __init__(self, root: Path):
        pairs = []
        for path, repl in ((str(Path(root).resolve()), "<root>"), (str(Path(root)), "<root>"),
                           (os.path.expanduser("~"), "<home>")):
            if path and path not in ("/", "~") and path not in [p for p, _ in pairs]:
                pairs.append((path, repl))
        try:
            host = socket.gethostname()
        except Exception:                                      # noqa: BLE001
            host = ""
        if host and len(host) > 2:
            pairs.append((host, "<host>"))
            short = host.split(".")[0]
            if len(short) > 2 and short != host:
                pairs.append((short, "<host>"))
        self.pairs = sorted(pairs, key=lambda p: -len(p[0]))   # the longest first: the root before the home folder

    def __call__(self, obj):
        if isinstance(obj, str):
            for path, repl in self.pairs:
                if path in obj:
                    obj = obj.replace(path, repl)
            return obj
        if isinstance(obj, dict):
            return {self(k) if isinstance(k, str) else k: self(v) for k, v in obj.items()}   # keys too
        if isinstance(obj, (list, tuple)):
            return [self(v) for v in obj]
        return obj


def _state_names(dataset) -> list[str] | None:
    feats = getattr(dataset, "features", None) or {}
    f = feats.get(STATE_KEY) if isinstance(feats, dict) else None
    names = f.get("names") if isinstance(f, dict) else None
    if isinstance(names, dict):                                # {"motors": [...]} in some LeRobot datasets
        names = next((v for v in names.values() if isinstance(v, (list, tuple))), None)
    if isinstance(names, (list, tuple)) and all(isinstance(n, str) for n in names):
        return list(names)
    return None


def _image_keys(dataset) -> list[str]:
    feats = getattr(dataset, "features", None) or {}
    return sorted(k for k, f in feats.items() if isinstance(f, dict) and f.get("dtype") in ("image", "video"))


def _plain(v):
    """A JSON-ready copy of a policy, planner or env value (dataclasses become dicts)."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(v) and not isinstance(v, type):
        return asdict(v)
    return v


def _vector(v) -> list | None:
    """The latest observation.state as a list of floats (NaN and Inf become null), or None."""
    if v is None:
        return None
    if hasattr(v, "detach") and hasattr(v, "cpu"):
        v = v.detach().cpu().numpy()
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    return [float(x) if math.isfinite(x) else None for x in a.tolist()]


def _own_state(v):
    """An owned copy of observation.state, so a caller that reuses its array or tensor cannot change the snapshot."""
    if v is None:
        return None
    if hasattr(v, "detach") and hasattr(v, "cpu"):
        return v.detach().cpu().numpy().copy()
    return np.array(v, dtype=np.float64, copy=True).reshape(-1)


def _image_hw(v) -> tuple[int, int] | None:
    shape = getattr(v, "shape", None)
    if shape is None or len(shape) != 3:
        return None
    if shape[0] in (1, 3, 4) and shape[2] not in (1, 3, 4):   # channels first
        return int(shape[1]), int(shape[2])
    return int(shape[0]), int(shape[1])


def _copy_image(v):
    if hasattr(v, "detach") and hasattr(v, "cpu"):
        return v.detach().cpu().numpy().copy()
    return np.array(v, copy=True)


def encode_preview(img, width: int = PREVIEW_WIDTH, quality: int = PREVIEW_QUALITY) -> bytes:
    """A camera frame (HWC or CHW, uint8 or float in [0, 1]) as a JPEG at most `width` pixels wide."""
    from PIL import Image                                      # imported only when a preview is asked for
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[2] not in (1, 3, 4):
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.dtype != np.uint8:
        a = np.asarray(a, dtype=np.float64)
        if a.size and np.nanmax(a) <= 1.0:
            a = a * 255.0
        a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
    im = Image.fromarray(a)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))))
    import io
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    """Temp file next to the destination, then a rename: a reader sees the old or the new file, never half of one.
    No fsync: these files describe the present and are rewritten within a second."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class LiveTap:
    """The live tap of one RunLogger. Created by RunLogger(live=True); see the module docstring.

    Caller-thread hooks (frame, run_started, run_ended, run_state, controlled_stop) never raise and never touch the
    disk. The tap thread does every write. Test hooks: `copies` counts preview frames copied on the caller's thread,
    `encodes` counts JPEGs encoded, `writes` counts status writes."""

    def __init__(self, logger):
        self.failed: str | None = None
        self.copies = 0
        self.encodes = 0
        self.writes = 0
        self._warned = False
        self._closing = False
        self._closed = False
        self._thread = None
        self._wake = threading.Event()
        self._pending: collections.deque = collections.deque()
        self._dropped = 0
        self._dead = False
        self._stop_lock = threading.Lock()                     # stop publication: memory only, never file work
        try:
            self._setup(logger)
        except Exception as e:                                 # noqa: BLE001 (never in the way of recording)
            self._dead = True                                  # every hook returns at once from now on
            self._fail(e)
            try:                                               # warned from a thread, never on the caller's
                threading.Thread(target=self._warn, name="aftershock-live-tap-warn", daemon=True).start()
            except BaseException:                              # noqa: BLE001
                pass
            return
        self._thread = threading.Thread(target=self._loop, name="aftershock-live-tap", daemon=True)
        self._thread.start()

    def _prepare_files(self) -> None:
        """Tap thread, before its first write and again after a failure: a clean folder, the seq to continue from."""
        self.dir.mkdir(parents=True, exist_ok=True)
        for p in self.dir.glob(".*.tmp"):                      # temp files left by a writer that died mid-write
            p.unlink(missing_ok=True)
        self._clear_previews()
        self.status_seq = max(self.status_seq, self._last_status_seq())
        last, self.file_records = self._open_events()
        self.event_seq = max(self.event_seq, last)
        self._files_ready = True

    def _setup(self, logger) -> None:
        self.root = Path(logger.root)
        self.dir = live_dir(self.root)
        self.scrub = _Scrub(self.root)
        self.pid = os.getpid()
        schema = logger.schema
        self.schema_info = {"id": schema.id, "names": list(schema.names), "units": list(schema.units),
                            "space": schema.space, "rate_hz": schema.rate_hz}
        self.period_ns = int(1e9 / schema.rate_hz)
        self.low = None if schema.low is None else np.asarray(schema.low, np.float32)
        self.high = None if schema.high is None else np.asarray(schema.high, np.float32)
        self.state_names = _state_names(logger.dataset)
        self.image_keys = _image_keys(logger.dataset)
        # Camera ids for status keys and preview file names: cleaned of private names, unique.
        self.cam_given = {k: k[len(IMAGE_PREFIX):] if k.startswith(IMAGE_PREFIX) else k
                          for k in self.image_keys}                             # what a caller's camera_ns may use
        self.cam_keys = {k: self.scrub(k) for k in self.image_keys}            # cleaned before any character change
        self.cam_names = {}
        for k in self.image_keys:
            cam = camera_name(self.scrub(self.cam_given[k]))
            n, base = 2, cam
            while cam in self.cam_names.values():
                cam, n = f"{base}_{n}", n + 1
            self.cam_names[k] = cam
        # snapshot, written by the caller's thread, read by the tap thread
        self.run = None                     # dict for the run in progress, else None
        self.last_run = None
        self.stop = None                    # the latest controlled stop
        self._snap = None                   # the last frame's values, replaced whole by each frame
        self._rec = self._new_rec()         # REC-04 and REC-05 counts of the run, replaced whole on each change
        self.cameras: dict[str, dict] = {}  # per camera its image size, replaced whole when a camera is first seen
        self.queue_ref = None
        self._mode = None                   # (control_mode, expert_source) of the last frame
        self._cam_state: dict[str, dict] = {}  # REC-04 per camera -> {last_ns, last_frame, open_from}
        self._rec05_open = None             # first frame of the open out-of-bounds span
        # preview: set by the tap thread, read by the caller's thread
        self.preview_interval_ns: int | None = None
        self._preview_due: dict[str, int] = {}
        self._preview_slots: dict[str, tuple] = {}
        self._preview_files: dict[str, dict] = {}
        self.status_seq = 0
        self.event_seq = 0
        self.file_records = 0
        self._files_ready = False
        self._error_written = None
        self._stop_seen_for = None

    # ── caller's thread ──────────────────────────────────────────────────────

    def _fail(self, e: BaseException) -> None:
        """Turn the tap off until the next run starts. Nothing here can raise or write: the tap thread warns once
        (a warning filter set to raise cannot reach the caller) and leaves error.json for the reader."""
        try:
            self.failed = f"{type(e).__name__}: {e}"
            self._wake.set()
        except BaseException:                                  # noqa: BLE001
            pass

    def _warn(self) -> None:
        """Tap thread only: one warning per failure, never raised further."""
        if self._warned:
            return
        self._warned = True
        try:
            warnings.warn(f"aftershock-logger live tap stopped ({self.failed}); recording continues",
                          RuntimeWarning, stacklevel=2)
        except BaseException:                                  # noqa: BLE001 (warnings configured as errors)
            pass

    def _event(self, rec: dict) -> None:
        if len(self._pending) >= PENDING_MAX:                  # the tap thread is not keeping up: drop the oldest
            self._pending.popleft()
            self._dropped += 1
        self._pending.append(rec)

    @staticmethod
    def _new_rec() -> dict:
        return {"REC-04": {"gaps": 0, "frames_missed": 0, "open": []},
                "REC-05": {"frames": 0, "spans": 0, "open": False}}

    def _set_rec(self, check: str, **changes) -> None:
        """Counters are replaced, never changed in place, so the tap thread always reads one whole version."""
        rec = dict(self._rec)
        rec[check] = {**rec[check], **changes}
        self._rec = rec

    def run_started(self, run) -> None:
        try:
            if self._closing or self._dead:
                return
            if self.failed is not None:                        # a failed tap tries again with each new run
                self.failed = None
                self._warned = False
                self._files_ready = False
            now = time.monotonic_ns()
            pol = run.policy
            self._snap = None
            self._rec = self._new_rec()
            self.stop = None
            self.queue_ref = None
            self._mode = None
            self._cam_state = {}
            self._rec05_open = None
            self.run = {"run_id": run.run_id, "task": run.task, "task_id": run.task_id, "task_rev": run.task_rev,
                        "policy": _plain(pol), "planner": _plain(run.planner), "episode_index": run.episode_index,
                        "started_at": run.started_at, "start_ns": now, "state": "active",
                        "termination_reason": None, "task_success": None}
            self._event({"kind": "run_start", "run_id": run.run_id, "task": run.task, "task_id": run.task_id,
                         "policy": _plain(pol), "planner": _plain(run.planner), "episode_index": run.episode_index,
                         "started_at": run.started_at, "ns": now})
            self._wake.set()
        except Exception as e:                                 # noqa: BLE001
            self._fail(e)

    def frame(self, run, frame: dict, row: dict, camera_ns=None) -> None:
        """One accepted frame. O(1) in the run's length: one new snapshot (frame, mode, prompt, an owned copy of the
        joint state, each camera's time), a bounds test on the action and, only when a preview frame is due, one
        image copy. The snapshot is published by one assignment, so the tap thread reads one frame's values."""
        if self.failed is not None or self._closing or self._dead:
            return
        try:
            r = self.run
            if r is None or r["run_id"] != run.run_id:
                return
            i = row["frame_index"]
            obs_ns = row["obs_ns"]
            mode = (row["control_mode"], row["expert_source"])
            if mode != self._mode:
                self._event({"kind": "control_mode", "run_id": run.run_id, "frame": i, "control_mode": mode[0],
                             "expert_source": mode[1],
                             "previous": None if self._mode is None else self._mode[0], "obs_ns": obs_ns})
                self._mode = mode
                self._wake.set()
            if self.queue_ref is None and run._queue is not None:
                self.queue_ref = run._queue
            self._rec05(run, row, i)
            cams = self._cameras(run, frame, i, obs_ns, camera_ns) if self.image_keys else {}
            self._snap = {"run_id": run.run_id, "frame": i, "frames": i + 1, "control_mode": mode[0],
                          "expert_source": mode[1], "prompt": row.get("prompt"), "subtask_cmd": row.get("subtask_cmd"),
                          "state": _own_state(frame.get(STATE_KEY)), "cams": cams, "rec": self._rec}
        except Exception as e:                                 # noqa: BLE001
            self._fail(e)

    def _rec05(self, run, row, i) -> None:
        if self.low is None:
            return
        a = row["sent_action"]
        out = bool((a < self.low).any() or (a > self.high).any())
        rec = self._rec["REC-05"]
        if out:
            if self._rec05_open is None:
                self._rec05_open = i
                self._set_rec("REC-05", frames=rec["frames"] + 1, spans=rec["spans"] + 1, open=True)
                bad = [self.schema_info["names"][k] for k in np.flatnonzero((a < self.low) | (a > self.high))]
                self._event({"kind": "rec_result", "check_id": "REC-05", "result": "warn", "state": "open",
                             "run_id": run.run_id, "frames": [i, None], "dims": bad,
                             "message": f"frame {i} sent a command out of the action schema's bounds "
                                        f"({', '.join(bad)}); kept as sent and flagged"})
                self._wake.set()
            else:
                self._set_rec("REC-05", frames=rec["frames"] + 1)
        elif self._rec05_open is not None:
            start, self._rec05_open = self._rec05_open, None
            self._set_rec("REC-05", open=False)
            self._event({"kind": "rec_result", "check_id": "REC-05", "result": "warn", "state": "closed",
                         "run_id": run.run_id, "frames": [start, i],
                         "message": f"{i - start} frames from frame {start} sent a command out of reach; flagged"})
            self._wake.set()

    def _cameras(self, run, frame, i, obs_ns, camera_ns) -> dict:
        interval = self.preview_interval_ns
        times = {}
        for key in self.image_keys:
            v = frame.get(key)
            if v is None:
                continue
            cam = self.cam_names[key]
            got = None
            if camera_ns:
                got = camera_ns.get(self.cam_given[key], camera_ns.get(key))
            t = int(got) if isinstance(got, (int, np.integer)) and not isinstance(got, bool) else obs_ns
            if cam not in self.cameras:
                hw = _image_hw(v)
                self.cameras = {**self.cameras, cam: {"key": self.cam_keys[key], "height": hw[0] if hw else None,
                                                      "width": hw[1] if hw else None}}
            times[cam] = (t, i)
            if camera_ns is not None:
                self._rec04(run, cam, t, i, obs_ns)
            if interval is not None and obs_ns >= self._preview_due.get(cam, 0):
                self._preview_due[cam] = obs_ns + interval
                self._preview_slots[cam] = (_copy_image(v), i, obs_ns)
                self.copies += 1
                self._wake.set()
        return times

    def _rec04(self, run, cam, t, i, obs_ns) -> None:
        """A camera whose capture time stops moving for more than GAP_PERIODS frame periods has a gap, opened at once
        and closed (with the frames it missed) when a new frame arrives. Needs camera_ns from the caller."""
        st = self._cam_state.get(cam)
        rec = self._rec["REC-04"]
        if st is None:
            self._cam_state[cam] = {"last_ns": t, "last_frame": i, "open_from": None}
            return
        if t > st["last_ns"]:
            if st["open_from"] is not None:
                start = st["open_from"]
                seconds = max(0.0, (t - st["last_ns"] - self.period_ns) / 1e9)   # as REC-04: the step less a period
                self._set_rec("REC-04", frames_missed=rec["frames_missed"] + max(1, i - start),
                              open=[c for c in rec["open"] if c != cam])
                review = seconds > LONG_GAP_S
                self._event({"kind": "rec_result", "check_id": "REC-04", "result": "warn", "state": "closed",
                             "run_id": run.run_id, "camera": cam, "frames": [start, i], "seconds": round(seconds, 3),
                             "review": review,
                             "message": f"{'Review this run: ' if review else ''}{cam} camera missed "
                                        f"{seconds:.2f} s at frame {start}"})
                self._wake.set()
            st.update(last_ns=t, last_frame=i, open_from=None)
        elif st["open_from"] is None and obs_ns - st["last_ns"] > GAP_PERIODS * self.period_ns:
            st["open_from"] = st["last_frame"] + 1
            self._set_rec("REC-04", gaps=rec["gaps"] + 1, open=sorted(set(rec["open"]) | {cam}))
            self._event({"kind": "rec_result", "check_id": "REC-04", "result": "warn", "state": "open",
                         "run_id": run.run_id, "camera": cam, "frames": [st["open_from"], None],
                         "message": f"{cam} camera has sent no new frame since frame {st['last_frame']}"})
            self._wake.set()

    def run_ended(self, run) -> None:
        """run.end() staged the run: its termination is known, saving follows."""
        try:
            r = self.run
            if r is None or r["run_id"] != run.run_id or self._closing or self._dead:
                return
            end = run._end or {}
            self.run = {**r, "state": "staged", "termination_reason": end.get("termination_reason"),
                        "task_success": end.get("task_success")}
            self._event({"kind": "run_end", "run_id": run.run_id, "frames": len(run._rows),
                         "termination_reason": end.get("termination_reason"), "task_success": end.get("task_success"),
                         "failure_mode": end.get("failure_mode"), "ended_at": run.ended_at, "ns": time.monotonic_ns()})
            self._wake.set()
        except Exception as e:                                 # noqa: BLE001
            self._fail(e)

    def run_state(self, run, state: str) -> None:
        """The run reached a final state: committed, discarded, quarantined or save_failed."""
        try:
            r = self.run
            if r is None or r["run_id"] != run.run_id or self._closing or self._dead:
                return
            end = run._end or {}
            snap = self._snap
            frames = snap["frames"] if snap and snap["run_id"] == run.run_id else 0
            if r["state"] == "active":                         # no run_end yet: say how the run ended
                self._event({"kind": "run_end", "run_id": run.run_id, "frames": frames,
                             "termination_reason": end.get("termination_reason"),
                             "task_success": end.get("task_success"), "failure_mode": end.get("failure_mode"),
                             "ended_at": run.ended_at or _now_iso(), "ns": time.monotonic_ns()})
            self._event({"kind": "run_state", "run_id": run.run_id, "state": state, "ns": time.monotonic_ns()})
            self.last_run = {"run_id": run.run_id, "state": state, "frames": frames,
                             "termination_reason": end.get("termination_reason"),
                             "task_success": end.get("task_success"), "failure_mode": end.get("failure_mode"),
                             "started_at": r["started_at"], "ended_at": run.ended_at or _now_iso(),
                             "rec": self._rec}
            self.run = None
            self._snap = None
            self.queue_ref = None
            self._preview_slots.clear()
            self._wake.set()
        except Exception as e:                                 # noqa: BLE001
            self._fail(e)

    def stop_seen(self, run, stop) -> None:
        """A ControlledStop was detected: the caller was told, the run block is leaving with it, or the tap thread
        saw the queue stop. Published once per run, before the writer wait and the quarantine write."""
        with self._stop_lock:                                  # never after, or over, the frames' final place
            if self._stop_seen_for == run.run_id:
                return
            self._stop_seen_for = run.run_id
            self._stop_update(run, stop, "detected")

    def controlled_stop(self, run, stop) -> None:
        """Where the stopped run's frames went: after the quarantine write, a retry or a spill."""
        q = getattr(stop, "quarantine_path", None)
        phase = "spilled" if getattr(stop, "spilled", False) else "quarantined" if q is not None else "not_written"
        with self._stop_lock:
            self._stop_seen_for = run.run_id                   # a late detection is not published after this
            self._stop_update(run, stop, phase)

    def _stop_update(self, run, stop, phase: str) -> None:
        try:
            if self._closing or self._dead:
                return
            q = getattr(stop, "quarantine_path", None) if phase != "detected" else None
            rel = None
            if q is not None and phase == "quarantined":
                try:
                    rel = str(Path(q).resolve().relative_to(self.root.resolve()))
                except ValueError:
                    rel = None                                 # outside the dataset: no path is shown
            self.stop = {"run_id": run.run_id, "reason": stop.reason, "detail": stop.detail, "frames": stop.frames,
                         "phase": phase, "quarantine": rel, "quarantine_written": phase in ("quarantined", "spilled"),
                         "at": _now_iso()}
            self._event({"kind": "controlled_stop", **self.stop, "ns": time.monotonic_ns()})
            self._wake.set()
        except Exception as e:                                 # noqa: BLE001
            self._fail(e)

    def close(self) -> None:
        """Write the last status with closed: true, remove preview files and stop the thread (waits at most JOIN_S)."""
        try:
            self._closing = True
            self._wake.set()
            if self._thread is not None:
                self._thread.join(JOIN_S)
        except Exception as e:                                 # noqa: BLE001
            self._fail(e)

    # ── tap thread ───────────────────────────────────────────────────────────

    def _loop(self) -> None:
        _lower_priority()
        last_write = 0.0
        while True:
            self._wake.wait(WRITE_EVERY_S)
            self._wake.clear()
            closing = self._closing
            if self.failed is not None:
                self._while_failed()
                if closing:
                    return
                continue
            try:
                if not self._files_ready:
                    self._prepare_files()
                q = self.queue_ref
                if q is not None and q._stop is not None:      # a write failed in the background: show it now
                    self.stop_seen(q.run, q._stop)
                self._check_preview()
                wrote_events = self._flush_events()
                now = time.monotonic()
                if wrote_events or closing or now - last_write >= WRITE_EVERY_S or self._preview_slots:
                    self._write_previews()
                    self._write_status(closed=closing)
                    last_write = now
                if closing:
                    self._clear_previews()
                    self._closed = True
                    return
            except Exception as e:                             # noqa: BLE001
                self._fail(e)
                if closing:
                    self._while_failed()
                    return

    def _while_failed(self) -> None:
        """Once per failure: stop previews and write error.json (best effort; the folder may be read only)."""
        failed = self.failed
        if failed is None or self._error_written == failed:
            return
        self._error_written = failed
        self._warn()
        self.preview_interval_ns = None
        self._preview_slots.clear()
        try:
            body = self.scrub({"live_format": LIVE_FORMAT, "pid": self.pid, "at": _now_iso(),
                               "at_ns": time.monotonic_ns(), "error": failed})
            _write_atomic(self.dir / ERROR, (json.dumps(body) + "\n").encode())
        except Exception:                                      # noqa: BLE001
            pass
        try:
            self._clear_previews()
        except Exception:                                      # noqa: BLE001
            pass

    def _last_status_seq(self) -> int:
        try:
            d = json.loads((self.dir / STATUS).read_text())
            return int(d.get("seq", 0)) if isinstance(d, dict) else 0
        except Exception:                                      # noqa: BLE001
            return 0

    def _open_events(self) -> tuple[int, int]:
        """The last seq written by any earlier writer, and the records in the current file. A torn last line (a
        writer died mid-append) is cut off so new records start on a line of their own."""
        path = self.dir / EVENTS
        last, records = 0, 0
        for old in (self.dir / EVENTS_OLD,):
            for rec in _read_lines(old):
                last = max(last, _seq_of(rec))
        if path.exists():
            data = path.read_bytes()
            cut = data.rfind(b"\n") + 1
            if cut != len(data):
                with open(path, "r+b") as f:
                    f.truncate(cut)
                data = data[:cut]
            for line in data.splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    last = max(last, _seq_of(rec))
                    if rec.get("kind") != "file_start":
                        records += 1
        if not path.exists() or path.stat().st_size == 0:
            self._start_file(path, last + 1)
            records = 0
        return last, records

    def _start_file(self, path: Path, first_seq: int) -> None:
        line = json.dumps({"live_format": LIVE_FORMAT, "kind": "file_start", "first_seq": first_seq}) + "\n"
        _write_atomic(path, line.encode())

    def _flush_events(self) -> bool:
        if not self._pending and not self._dropped:
            return False
        path = self.dir / EVENTS
        lines = []
        if self._dropped:
            n, self._dropped = self._dropped, 0
            self.event_seq += 1
            lines.append({"seq": self.event_seq, "kind": "dropped", "records": n, "at": _now_iso(),
                          "message": f"{n} live records were dropped because the tap fell behind"})
        while self._pending:
            rec = self._pending.popleft()
            self.event_seq += 1
            lines.append({"seq": self.event_seq, **rec, "at": _now_iso()})
        out = []
        for rec in lines:
            if self.file_records >= ROTATE_RECORDS:
                if out:
                    _append(path, out)
                    out = []
                os.replace(path, self.dir / EVENTS_OLD)
                self._start_file(path, rec["seq"])
                self.file_records = 0
            out.append(json.dumps(self.scrub(rec), default=str) + "\n")
            self.file_records += 1
        if out:
            _append(path, out)
        return True

    def _status(self, closed: bool) -> dict:
        r, snap = self.run, self._snap                          # each read once: whole objects, never half updated
        run = joint = None
        cams = {}
        if r is not None:
            run = dict(r)
            start = run.pop("start_ns")
            run["elapsed_s"] = round((time.monotonic_ns() - start) / 1e9, 3)
            mine = snap is not None and snap["run_id"] == r["run_id"]
            run["rec"] = snap["rec"] if mine else self._new_rec()     # the counts as of the frame shown
            for k in ("frame", "frames", "control_mode", "expert_source", "prompt", "subtask_cmd"):
                run[k] = snap[k] if mine else ({"frame": -1, "frames": 0}.get(k))
            joint = {"key": STATE_KEY, "names": self.state_names, "values": _vector(snap["state"]) if mine else None}
            sizes = self.cameras
            for cam, (t, i) in (snap["cams"].items() if mine else ()):
                cams[cam] = {**sizes.get(cam, {}), "last_ns": t, "frame": i}
        q = self.queue_ref
        queue = None
        if q is not None:
            queue = {"depth": len(q._items), "capacity": q.capacity, "max_depth": q.max_depth}
        interval = self.preview_interval_ns
        return {
            "live_format": LIVE_FORMAT, "seq": self.status_seq + 1, "pid": self.pid, "at_ns": time.monotonic_ns(),
            "at": _now_iso(), "closed": bool(closed), "writer": "aftershock-logger",
            "last_event_seq": self.event_seq, "run": run, "last_run": self.last_run,
            "joint_state": joint, "action_schema": self.schema_info, "cameras": cams, "queue": queue,
            "controlled_stop": self.stop,
            "preview": {"active": interval is not None,
                        "interval_s": None if interval is None else interval / 1e9,
                        "frames": dict(self._preview_files)},
        }

    def _write_status(self, closed: bool) -> None:
        body = json.dumps(self.scrub(self._status(closed)), default=str) + "\n"
        _write_atomic(self.dir / STATUS, body.encode())
        self.status_seq += 1
        self.writes += 1
        self._error_written = None                             # a later failure writes error.json again
        err = self.dir / ERROR
        if err.exists():                                       # the tap works again
            err.unlink(missing_ok=True)

    def _check_preview(self) -> None:
        """preview.want under PREVIEW_WANT_S old turns preview on; its JSON body may set interval_s (>= 1 s)."""
        want = self.dir / PREVIEW_WANT
        try:
            st = want.stat()
        except FileNotFoundError:
            st = None
        if st is None or time.time() - st.st_mtime >= PREVIEW_WANT_S:
            if self.preview_interval_ns is not None or self._preview_files:
                self.preview_interval_ns = None
                self._preview_slots.clear()
                self._clear_previews()
            return
        interval = PREVIEW_DEFAULT_S
        try:
            body = json.loads(want.read_text() or "{}")
            v = body.get("interval_s") if isinstance(body, dict) else body
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                interval = max(PREVIEW_MIN_S, float(v))
        except (ValueError, OSError):
            pass
        ns = int(interval * 1e9)
        if self.preview_interval_ns != ns:
            self._preview_due.clear()
            self.preview_interval_ns = ns

    def _write_previews(self) -> None:
        if not self._preview_slots:
            return
        pdir = self.dir / PREVIEW_DIR
        pdir.mkdir(exist_ok=True)
        for cam in list(self._preview_slots):
            slot = self._preview_slots.pop(cam, None)          # the run may have ended and cleared the slots
            if slot is None or self.preview_interval_ns is None:
                continue
            img, frame, ns = slot
            data = encode_preview(img)
            self.encodes += 1
            _write_atomic(pdir / f"{cam}.jpg", data)
            self._preview_files[cam] = {"file": f"{PREVIEW_DIR}/{cam}.jpg", "frame": frame, "obs_ns": ns,
                                        "at": _now_iso(), "bytes": len(data)}

    def _clear_previews(self) -> None:
        self._preview_files.clear()
        pdir = self.dir / PREVIEW_DIR
        if pdir.is_dir():
            for p in pdir.iterdir():
                if p.is_file():
                    p.unlink(missing_ok=True)


def _append(path: Path, lines: list[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))


def _seq_of(rec: dict) -> int:
    if rec.get("kind") == "file_start":
        v = rec.get("first_seq")
        return int(v) - 1 if isinstance(v, int) else 0
    v = rec.get("seq")
    return int(v) if isinstance(v, int) else 0


def _read_lines(path: Path) -> list[dict]:
    """Whole JSON lines of an events file; a torn last line (no newline yet) is skipped."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    out = []
    for line in data[:data.rfind(b"\n") + 1].splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _lower_priority() -> None:
    """Best effort: on Linux a thread can lower its own nice value; elsewhere the tap thread keeps the default."""
    try:
        if hasattr(os, "setpriority") and hasattr(threading, "get_native_id") and os.uname().sysname == "Linux":
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    except Exception:                                          # noqa: BLE001
        pass


# ── readers (the station's side) ────────────────────────────────────────────

def read_status(root) -> dict | None:
    """The latest status.json of a dataset root, or None when there is none or it cannot be read."""
    try:
        d = json.loads((live_dir(root) / STATUS).read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) and d.get("live_format") == LIVE_FORMAT else None


def read_error(root) -> dict | None:
    try:
        d = json.loads((live_dir(root) / ERROR).read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _ident(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_dev, st.st_ino


def read_events(root, after_seq: int = 0, limit: int | None = None) -> list[dict]:
    """Event records with seq > after_seq, in seq order, from events.1.jsonl and events.jsonl.

    The two files are read again when events.jsonl was replaced while they were read (a rotation in between), so a
    rotation never hides records. Records are taken once per seq. Wherever records the reader has not seen are gone
    (rotated past), a {"kind": "gap", "from_seq", "to_seq"} record stands in their place. When the files hold no seq
    as high as after_seq (they were removed and a writer started over), the list is [{"kind": "reset", "last_seq"}],
    and the reader should start again from 0. A torn last line is not returned; it is read once it is whole."""
    d = live_dir(root)
    for _ in range(5):
        before = _ident(d / EVENTS)
        parts = (_read_lines(d / EVENTS_OLD), _read_lines(d / EVENTS))
        if _ident(d / EVENTS) == before:
            break
    first, last, by_seq = None, 0, {}
    for part in parts:
        for rec in part:
            if rec.get("kind") == "file_start":
                v = rec.get("first_seq")
                if isinstance(v, int) and not isinstance(v, bool):
                    first = v if first is None else min(first, v)
                    last = max(last, v - 1)
                continue
            v = rec.get("seq")
            if isinstance(v, int) and not isinstance(v, bool):
                first = v if first is None else min(first, v)
                last = max(last, v)
                by_seq.setdefault(v, rec)
    if first is None:
        return []
    if last < after_seq:
        return [{"kind": "reset", "last_seq": last}]
    out, expect = [], after_seq + 1
    for seq in sorted(k for k in by_seq if k > after_seq):
        if seq > expect:
            out.append({"kind": "gap", "from_seq": expect, "to_seq": seq - 1})
        out.append(by_seq[seq])
        expect = seq + 1
    if not out and first > after_seq + 1:
        out.append({"kind": "gap", "from_seq": after_seq + 1, "to_seq": first - 1})
    return out if limit is None else out[:limit]
