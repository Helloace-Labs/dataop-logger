"""Logger L5: the bounded frame queue, the controlled stop and quarantine of partial runs."""
import errno
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aftershock_logger import (ActionSchema, ControlledStop, PolicyInfo, RunError, RunLogger,  # noqa: E402
                               store)
from aftershock_logger import frame_queue as FQ  # noqa: E402
from aftershock_logger import validate as V  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCHEMA = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
TASK = "pick up the part"
POLICY = PolicyInfo("policy", "ckpt/1")


def make_ds(tmp_path, images=False):
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
             "action": {"dtype": "float32", "shape": (6,), "names": list(NAMES)}}
    if images:
        feats["observation.images.top"] = {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channels"]}
    return lerobot_dataset.LeRobotDataset.create("test/q", fps=30, features=feats, root=tmp_path / "ds",
                                                 use_videos=False)


def frame(i, images=False):
    f = {"observation.state": np.full(6, i, np.float32), "task": TASK}
    if images:
        f["observation.images.top"] = np.full((8, 8, 3), i % 255, np.uint8)
    return f


def kw(i):
    t = 1_000_000 + i * 33_000_000
    a = np.arange(6, dtype=np.float32) + i
    return dict(sent_action=a, base_action=a, control_mode="base", inference_id=i, chunk_offset=0,
                obs_ns=t, sent_ns=t + 1000)


def slow_writer(monkeypatch, ds, delay, gate=None):
    real = ds.add_frame

    def slow(f):
        if gate is not None:
            gate.wait()
        time.sleep(delay)
        real(f)
    monkeypatch.setattr(ds, "add_frame", slow)


def failing_writer(monkeypatch, ds, fail_at, exc=None):
    real, n = ds.add_frame, [0]

    def add(f):
        if n[0] == fail_at:
            raise exc or OSError(errno.ENOSPC, "No space left on device")
        n[0] += 1
        real(f)
    monkeypatch.setattr(ds, "add_frame", add)


def quarantined(ds, run):
    path = store.quarantine_dir(ds.root) / run.run_id
    m = json.loads((path / "manifest.json").read_text())
    return path, m


def test_queued_run_commits_like_a_direct_run(tmp_path):
    ds = make_ds(tmp_path, images=True)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue()
            assert q.capacity == 60                               # 2 s at 30 Hz
            for i in range(20):
                assert q.put(frame(i, True), **kw(i)) == i
            run.event("takeover", source="human")
            run.complete_transition(1.0)
            run.end("success")
    ds.finalize()
    assert run.state == "committed"
    m = store.read_manifest(run.run_path)
    assert m["length"] == 20 and m["events"][0]["frame_index"] == 19
    t = pq.read_table(run.run_path / "ticks.parquet")
    assert t.column("frame_index").to_pylist() == list(range(20))
    assert t.column("reward_valid").to_pylist()[-1] is True
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]


def test_put_copies_reused_caller_arrays(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    gate = threading.Event()
    slow_writer(monkeypatch, ds, 0, gate)
    state = np.zeros(6, np.float32)
    act = np.zeros(6, np.float32)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue()
            for i in range(5):
                state[:] = i
                act[:] = i
                q.put({"observation.state": state, "task": TASK}, sent_action=act, base_action=act,
                      control_mode="base", inference_id=i, chunk_offset=0, obs_ns=i, sent_ns=i + 1)
            gate.set()
            run.end("success")
    ds.finalize()
    assert V.validate(ds.root)["ok"]
    t = pq.read_table(run.run_path / "ticks.parquet")
    assert [r[0] for r in t.column("sent_action").to_pylist()] == [0, 1, 2, 3, 4]


def test_slow_writer_fills_the_queue_and_gets_a_controlled_stop(tmp_path, monkeypatch):
    """Acceptance test 1: a slow writer fills the queue; the caller gets the stop signal; partial frames are in
    quarantine with the reason; nothing reaches LeRobot as an episode."""
    ds = make_ds(tmp_path, images=True)
    slow_writer(monkeypatch, ds, 0.05)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop) as ei:
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue(max_frames=5)
                for i in range(100):
                    t0 = time.monotonic()
                    q.put(frame(i, True), **kw(i))
                    assert time.monotonic() - t0 < 0.04           # put never waits for the writer (50 ms each)
                    assert q.depth() <= q.capacity
                run.end("success")
        stop = ei.value
        assert stop.reason == FQ.QUEUE_FULL and run.state == "quarantined"
        assert q.max_depth == q.capacity + 1                      # the frame that found it full is kept too
        assert not ds.has_pending_frames() and ds.meta.total_episodes == 0
        assert stop.quarantine_path == run.quarantine_path and stop.quarantine_error is None
        path, m = quarantined(ds, run)
        info = m["quarantine"]
        assert m["status"] == "quarantined" and m["termination_reason"] == "system_error"
        assert info["reason"] == "queue_full" and stop.frames == m["length"] == info["frames"]
        n = m["length"]
        assert n == info["frames_written_to_lerobot"] + info["frames_queued"] and info["frames_queued"] >= 5
        assert info["dropped_features"] == ["observation.images.top"]
        t = pq.read_table(path / "ticks.parquet")
        assert t.column("frame_index").to_pylist() == list(range(n))
        assert [r[0] for r in t.column("sent_action").to_pylist()] == list(range(n))   # every accepted tick, in order
        obs = pq.read_table(path / "frames.parquet")
        assert [r[0] for r in obs.column("observation.state").to_pylist()] == list(range(n))
        assert "queue_full" in (path / "QUARANTINED.txt").read_text()
        assert not list(store.runs_dir(ds.root).glob("*")) if store.runs_dir(ds.root).exists() else True
        monkeypatch.undo()
        with log.run(task=TASK, policy=POLICY) as r2:             # the logger records the next run normally
            for i in range(3):
                r2.add_frame(frame(i, True), **kw(i))
            r2.end("success")
    ds.finalize()
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert res["quarantined"] == [run.run_id]
    assert any("quarantined partial run" in w for w in res["warnings"])
    # Acceptance test 2: validate --repair handles the quarantined run: it checks it, keeps it, never commits it
    report = V.repair(ds.root)
    assert any(line.startswith(f"kept quarantined partial run {run.run_id}") for line in report)
    assert (store.quarantine_dir(ds.root) / run.run_id / "manifest.json").exists()
    assert not (store.runs_dir(ds.root) / run.run_id).exists()
    assert V.validate(ds.root)["ok"]


def test_disk_full_while_writing_a_frame_quarantines_every_frame(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    failing_writer(monkeypatch, ds, fail_at=7)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop) as ei:
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue()
                for i in range(12):
                    try:
                        q.put(frame(i), **kw(i))
                    except ControlledStop:
                        break                                     # the loop holds the robot here
                    time.sleep(0.005)
                run.end("success")
        stop = ei.value
        assert stop.reason == FQ.WRITE_FAILED and "No space left" in stop.detail
        path, m = quarantined(ds, run)
        assert m["length"] == stop.frames and m["quarantine"]["frames_written_to_lerobot"] == 7
        assert m["quarantine"]["frames_queued"] == m["length"] - 7 >= 1      # the frame that failed is kept
        t = pq.read_table(path / "ticks.parquet")
        assert [r[0] for r in t.column("sent_action").to_pylist()] == list(range(m["length"]))
        assert m["quarantine"]["lerobot_buffer_reset"] and not ds.has_pending_frames()
        assert not log.broken
    report = V.repair(ds.root)
    assert any("kept quarantined partial run" in line for line in report)


def test_write_failure_seen_by_end_and_check(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    failing_writer(monkeypatch, ds, fail_at=2, exc=RuntimeError("encoder died"))
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop, match="encoder died"):
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue()
                for i in range(4):
                    q.put(frame(i), **kw(i))
                deadline = time.monotonic() + 5
                while q.stopped is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                with pytest.raises(ControlledStop):
                    q.check()
                with pytest.raises(RunError, match="quarantined when its block exits"):
                    run.discard()                                 # stopped frames are never discarded
                run.end("success")
        _, m = quarantined(ds, run)
        assert m["length"] == 4 and m["quarantine"]["frames_written_to_lerobot"] == 2


def test_caught_stop_and_normal_exit_still_quarantines(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    slow_writer(monkeypatch, ds, 0.05)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop):
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue(max_frames=2)
                try:
                    for i in range(10):
                        q.put(frame(i), **kw(i))
                except ControlledStop:
                    pass                                          # leaving the block without run.end()
        assert run.state == "quarantined" and run.quarantine_path.exists()


def test_other_exception_after_a_stop_still_quarantines(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    slow_writer(monkeypatch, ds, 0.05)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.warns(UserWarning, match="controlled stop"):
            with pytest.raises(KeyError):
                with log.run(task=TASK, policy=POLICY) as run:
                    q = run.queue(max_frames=2)
                    try:
                        for i in range(10):
                            q.put(frame(i), **kw(i))
                    except ControlledStop:
                        raise KeyError("caller bug while stopping")
        assert run.state == "quarantined" and run.quarantine_path.exists()


def test_disk_full_while_writing_quarantine_keeps_frames_in_memory(tmp_path, monkeypatch):
    """No silent loss when even the quarantine cannot be written: the stop says so, the frames stay in memory
    (bounded by the run), the logger refuses new runs, and a retry after space is freed writes them all."""
    ds = make_ds(tmp_path)
    slow_writer(monkeypatch, ds, 0.05)
    real_write = store.atomic_write_bytes
    full = [True]

    def write(path, data):
        if full[0] and "quarantine" in str(path):
            raise OSError(errno.ENOSPC, "No space left on device")
        real_write(path, data)
    monkeypatch.setattr(store, "atomic_write_bytes", write)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop) as ei:
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue(max_frames=3)
                for i in range(10):
                    q.put(frame(i), **kw(i))
        stop = ei.value
        assert stop.quarantine_path is None and "No space left" in stop.quarantine_error
        assert not (store.quarantine_dir(ds.root) / run.run_id).exists()
        assert not [p for p in store.quarantine_dir(ds.root).iterdir()]       # no half-written folder left
        assert "retry_quarantine" in log.broken
        with pytest.raises(RunError, match="retry_quarantine"):
            with log.run(task=TASK, policy=POLICY):
                pass
        with pytest.raises(OSError):
            run.retry_quarantine()                                # still full: raises, frames still kept
        full[0] = False
        path = run.retry_quarantine()
        assert not log.broken
        m = json.loads((path / "manifest.json").read_text())
        assert m["length"] == stop.frames
        assert [r[0] for r in pq.read_table(path / "ticks.parquet").column("sent_action").to_pylist()] == \
            list(range(stop.frames))
        monkeypatch.undo()
        with log.run(task=TASK, policy=POLICY) as r2:
            r2.add_frame(frame(0), **kw(0))
            r2.end("success")
    ds.finalize()
    assert V.validate(ds.root)["ok"]


def test_a_stuck_writer_does_not_block_the_stop_forever(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    release = threading.Event()
    real = ds.add_frame
    calls = [0]

    def add(f):
        calls[0] += 1
        if calls[0] == 2:
            release.wait(10)                                      # LeRobot hangs on the second frame
        real(f)
    monkeypatch.setattr(ds, "add_frame", add)
    monkeypatch.setattr(FQ, "WRITER_TIMEOUT_S", 0.2)
    log = RunLogger(ds, SCHEMA)
    with pytest.raises(ControlledStop):
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue(max_frames=2)
            for i in range(10):
                q.put(frame(i), **kw(i))
                time.sleep(0.01)
    _, m = quarantined(ds, run)
    info = m["quarantine"]
    assert info["frames_written_to_lerobot"] == 1 and m["length"] == info["frames"] >= 4
    assert info["lerobot_buffer_reset"] is False and "did not finish" in log.broken
    release.set()
    time.sleep(0.2)
    assert len(run._rows) == 1                                    # the late write was not appended afterwards
    with pytest.raises(RunError):
        with log.run(task=TASK, policy=POLICY):
            pass


def test_memory_stays_bounded_under_a_stalled_writer(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    gate = threading.Event()
    slow_writer(monkeypatch, ds, 0, gate)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop):
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue(max_seconds=0.5)                   # 15 frames at 30 Hz
                assert q.capacity == 15
                for i in range(1000):
                    q.put(frame(i), **kw(i))
                    assert q.depth() <= q.capacity + 1
        assert i in (q.capacity, q.capacity + 1)                   # stopped once capacity frames were waiting
        gate.set()


def test_queue_rules(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, policy=POLICY) as run:
            with pytest.raises(ValueError):
                run.queue(max_frames=0)
            with pytest.raises(ValueError):
                run.queue(max_seconds=float("nan"))
            q = run.queue()
            with pytest.raises(RunError, match="already has a queue"):
                run.queue()
            with pytest.raises(RunError, match="queue.put"):
                run.add_frame(frame(0), **kw(0))
            with pytest.raises(ValueError, match="6"):              # a bad frame is refused, nothing accepted
                q.put(frame(0), **{**kw(0), "sent_action": np.zeros(5)})
            q.put(frame(0), **kw(0))
            with pytest.raises(ValueError, match="backwards"):
                q.put(frame(1), **{**kw(1), "obs_ns": 0, "sent_ns": 1})
            q.put(frame(1), **kw(1))
            run.end("success")
        with log.run(task=TASK, policy=POLICY) as run2:
            run2.add_frame(frame(0), **kw(0))
            with pytest.raises(RunError, match="before the first frame"):
                run2.queue()
            run2.discard()
        with log.run(task=TASK, policy=POLICY) as run3:           # discarding a queued run stops its writer
            q3 = run3.queue()
            q3.put(frame(0), **kw(0))
            run3.discard()
        assert not ds.has_pending_frames() and run3.state == "discarded"
    ds.finalize()
    assert V.validate(ds.root)["ok"]


def test_validate_reports_broken_and_unfinished_quarantine(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    slow_writer(monkeypatch, ds, 0.05)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop):
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue(max_frames=2)
                for i in range(10):
                    q.put(frame(i), **kw(i))
    ds.finalize()
    path, m = quarantined(ds, run)
    (path / "frames.parquet").write_bytes(b"tampered")
    unfinished = store.quarantine_dir(ds.root) / f".{run.run_id}.x.tmp"
    unfinished.mkdir()
    res = V.validate(ds.root)
    assert not res["ok"]
    assert any("frames.parquet does not match" in e for e in res["errors"])
    assert any("unfinished quarantine write" in e for e in res["errors"])
    assert res["quarantined"] == [run.run_id]
    report = V.repair(ds.root)
    assert any(line.startswith(f"left untouched quarantined partial run {run.run_id}") for line in report)
    assert any(line.startswith("left untouched quarantine/.") for line in report)
    assert path.exists() and unfinished.exists()                  # repair never deletes them
