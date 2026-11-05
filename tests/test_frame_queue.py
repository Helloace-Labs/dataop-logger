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
    log = RunLogger(ds, SCHEMA)
    with pytest.raises(ControlledStop):
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue(max_frames=2, stop_timeout=0.2)
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


def wait_for_stop(q, limit=5.0):
    deadline = time.monotonic() + limit
    while q.stopped is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert q.stopped is not None


def test_first_put_after_a_write_failure_keeps_its_frame(tmp_path, monkeypatch):
    """Review round 1, H1: the action of the tick that first hears of a failure already reached the robot."""
    ds = make_ds(tmp_path)
    failing_writer(monkeypatch, ds, fail_at=3)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop):
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue()
                for i in range(5):
                    q.put(frame(i), **kw(i))
                wait_for_stop(q)
                with pytest.raises(ControlledStop):
                    q.put(frame(5), **kw(5))                      # kept, then raises
                with pytest.raises(ControlledStop):
                    q.put(frame(6), **kw(6))                      # the caller was told: nothing more is kept
        path, m = quarantined(ds, run)
        assert m["length"] == 6
        sent = pq.read_table(path / "ticks.parquet").column("sent_action").to_pylist()
        assert sent[-1] == list(np.arange(6, dtype=np.float32) + 5)


def test_check_counts_as_telling_the_caller(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    failing_writer(monkeypatch, ds, fail_at=1)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop):
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue()
                for i in range(3):
                    q.put(frame(i), **kw(i))
                wait_for_stop(q)
                with pytest.raises(ControlledStop):
                    q.check()
                with pytest.raises(ControlledStop):
                    q.put(frame(3), **kw(3))
        assert quarantined(ds, run)[1]["length"] == 3


def _full_quarantine(monkeypatch, ds, state):
    real_write = store.atomic_write_bytes
    qroot = str(store.quarantine_dir(ds.root))

    def write(path, data):
        if state["full"] and str(path).startswith(qroot):
            raise OSError(errno.ENOSPC, "No space left on device")
        if state.get("spill_full") and "spill" in Path(path).parts:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_write(path, data)
    monkeypatch.setattr(store, "atomic_write_bytes", write)


def test_unsaved_frames_are_spilled_when_the_stop_leaves_the_logger(tmp_path, monkeypatch):
    """Review round 1, H2: the documented flow lets the stop leave the RunLogger block; the frames must not vanish."""
    ds = make_ds(tmp_path)
    slow_writer(monkeypatch, ds, 0.05)
    _full_quarantine(monkeypatch, ds, {"full": True})
    spill = tmp_path / "spill"
    with pytest.raises(RunError, match="were written to") as ei:
        with RunLogger(ds, SCHEMA, spill_dir=spill) as log:
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue(max_frames=3)
                for i in range(10):
                    q.put(frame(i), **kw(i))
    stop = ei.value.__context__
    assert isinstance(stop, ControlledStop) and "NOT written" in str(stop)
    rec = spill / run.run_id
    m = json.loads((rec / "manifest.json").read_text())
    assert m["length"] == stop.frames and m["quarantine"]["reason"] == "queue_full"
    assert str(rec) in str(ei.value) and run.quarantine_path == rec


def test_close_says_when_frames_could_not_be_written_anywhere(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    slow_writer(monkeypatch, ds, 0.05)
    _full_quarantine(monkeypatch, ds, {"full": True, "spill_full": True})
    with pytest.raises(RunError, match="NOT written anywhere"):
        with RunLogger(ds, SCHEMA, spill_dir=tmp_path / "spill") as log:
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue(max_frames=3)
                for i in range(10):
                    q.put(frame(i), **kw(i))
    assert run._unsaved_quarantine is not None                    # still in memory for a last retry


def test_close_retries_the_dataset_quarantine_first(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    slow_writer(monkeypatch, ds, 0.05)
    state = {"full": True}
    _full_quarantine(monkeypatch, ds, state)
    log = RunLogger(ds, SCHEMA, spill_dir=tmp_path / "spill")
    with pytest.raises(ControlledStop) as ei:
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue(max_frames=3)
            for i in range(10):
                q.put(frame(i), **kw(i))
    state["full"] = False                                         # space was freed
    log.close()
    path, m = quarantined(ds, run)
    assert m["length"] == ei.value.frames and "quarantined at" in str(ei.value)
    assert not (tmp_path / "spill").exists()


def test_end_on_a_hung_writer_stops_without_waiting_twice(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    release = threading.Event()
    real, calls = ds.add_frame, [0]

    def add(f):
        calls[0] += 1
        if calls[0] == 2:
            release.wait(20)
        real(f)
    monkeypatch.setattr(ds, "add_frame", add)
    log = RunLogger(ds, SCHEMA)
    t0 = time.monotonic()
    with pytest.raises(ControlledStop, match="no frame finished"):
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue(stall_timeout=0.3, stop_timeout=5)
            for i in range(6):
                q.put(frame(i), **kw(i))
            run.end("success")
    assert time.monotonic() - t0 < 3                              # the 5 s stop wait is skipped for a hung writer
    _, m = quarantined(ds, run)
    assert m["length"] == 6 and m["quarantine"]["frames_written_to_lerobot"] == 1
    assert m["quarantine"]["lerobot_buffer_reset"] is False and log.broken
    release.set()


def test_a_slow_but_working_writer_drains_without_a_stop(tmp_path, monkeypatch):
    """Review round 1, M1: only time without progress counts, not the total drain time."""
    ds = make_ds(tmp_path)
    slow_writer(monkeypatch, ds, 0.03)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue(stall_timeout=0.25)
            for i in range(30):                                   # about 0.9 s of writing left at run.end()
                q.put(frame(i), **kw(i))
            run.end("success")
    ds.finalize()
    assert run.state == "committed" and V.validate(ds.root)["ok"]


def test_lerobot_failing_halfway_through_a_frame(tmp_path, monkeypatch):
    ds = make_ds(tmp_path, images=True)
    real, n = ds.add_frame, [0]

    def add(f):
        real(f)                                                   # the frame reaches the buffer, then it fails
        n[0] += 1
        if n[0] == 4:
            raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(ds, "add_frame", add)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ControlledStop):
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue()
                for i in range(6):
                    q.put(frame(i, True), **kw(i))
                run.end("success")
        _, m = quarantined(ds, run)
        info = m["quarantine"]
        assert m["length"] == 6 and info["frames_written_to_lerobot"] == 3 and info["frames_queued"] == 3
        assert info["lerobot_buffer_reset"] and not ds.has_pending_frames() and not log.broken
        monkeypatch.undo()
        with log.run(task=TASK, policy=POLICY) as r2:
            for i in range(3):
                r2.add_frame(frame(i, True), **kw(i))
            r2.end("success")
    ds.finalize()
    assert r2.state == "committed" and V.validate(ds.root)["ok"]


def test_put_returns_while_a_write_is_stuck(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    gate = threading.Event()
    slow_writer(monkeypatch, ds, 0, gate)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue(max_frames=10)
            q.put(frame(0), **kw(0))
            time.sleep(0.05)                                      # the writer is now blocked inside add_frame
            for i in range(1, 11):
                t0 = time.monotonic()
                q.put(frame(i), **kw(i))
                assert time.monotonic() - t0 < 0.02
            assert q.depth() == 10 and q.stopped is None
            gate.set()
            run.end("success")
    ds.finalize()
    assert run.state == "committed" and V.validate(ds.root)["ok"]


def test_copy_frame_copies_mutable_values():
    lst, d = [1.0, 2.0], {"a": [1]}
    out = FQ._copy_frame({"x": lst, "y": d, "s": "text"})
    lst.append(3.0)
    d["a"].append(2)
    assert out == {"x": [1.0, 2.0], "y": {"a": [1]}, "s": "text"}


def test_quarantine_record_survives_a_failed_folder_sync(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    slow_writer(monkeypatch, ds, 0.05)
    real_sync, qdir = store._fsync_dir, store.quarantine_dir(ds.root)

    def sync(p):
        if Path(p) == qdir:                                       # only the sync after the rename fails
            raise OSError("sync failed")
        real_sync(p)
    monkeypatch.setattr(store, "_fsync_dir", sync)
    with RunLogger(ds, SCHEMA, spill_dir=tmp_path / "spill") as log:
        with pytest.warns(UserWarning, match="could not be synced"):
            with pytest.raises(ControlledStop) as ei:
                with log.run(task=TASK, policy=POLICY) as run:
                    q = run.queue(max_frames=2)
                    for i in range(10):
                        q.put(frame(i), **kw(i))
        assert ei.value.quarantine_path is not None and not log.broken
        assert run.retry_quarantine() == run.quarantine_path


def test_hidden_files_in_quarantine_are_not_errors(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, policy=POLICY) as run:
            run.add_frame(frame(0), **kw(0))
            run.end("success")
    ds.finalize()
    store.quarantine_dir(ds.root).mkdir(parents=True)
    (store.quarantine_dir(ds.root) / ".DS_Store").write_bytes(b"x")
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
