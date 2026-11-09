"""Aftershock issue #479: LeRobot's streaming video encoder drops a frame, with only a warning, when its per-camera
queue is full, so the video ends with fewer frames than the dataset has rows and every later frame is misaligned.

The logger waits for room in every encoder queue before it hands a frame over (backpressure within the frame budget,
RunLogger(encoder_wait_s)); past the budget the run makes a controlled stop with the reason encoder_backlog. A run whose
encoder dropped frames anyway is never saved or committed, and validate reports a video whose frame count differs from
its episode's rows, also on a dataset written without the logger. A slow encoder is simulated by slowing the encoder
thread's reads from its frame queue (only that module's queues of the test's queue size)."""
import queue
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aftershock_logger import ActionSchema, EncoderBacklog, PolicyInfo, RunError, RunLogger, adapter  # noqa: E402
from aftershock_logger import validate as V  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
video_utils = pytest.importorskip("lerobot.datasets.video_utils")
av = pytest.importorskip("av")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCHEMA = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
TASK = "pick up the part"
CAM = "observation.images.top"
QSIZE = 3                                              # the encoder queue size of these tests


@pytest.fixture
def slow_encoder(monkeypatch):
    """Each encoder read takes `delay[0]` seconds, as on a loaded robot computer."""
    delay = [0.0]

    class SlowQueue(queue.Queue):
        def get(self, *a, **k):
            item = super().get(*a, **k)
            if self.maxsize == QSIZE and item is not None:
                time.sleep(delay[0])
            return item
    monkeypatch.setattr(video_utils.queue, "Queue", SlowQueue)
    return delay


WRIST = "observation.images.wrist"


def make_ds(tmp_path, name="ds", cams=(CAM,)):
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
             "action": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
             **{c: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]} for c in cams}}
    return lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features=feats, root=tmp_path / name,
                                                 use_videos=True, streaming_encoding=True,
                                                 encoder_queue_maxsize=QSIZE)


def frame(i, cams=(CAM,)):
    return {"observation.state": np.full(6, i, np.float32), "task": TASK,
            **{c: np.full((64, 64, 3), i * 9 % 255, np.uint8) for c in cams}}


def kw(i):
    t = 1_000_000 + i * 33_000_000
    return dict(sent_action=np.full(6, i, np.float32), control_mode="none", obs_ns=t, sent_ns=t + 1000)


def decoded_frames(root: Path) -> dict[str, int]:
    out = {}
    for p in sorted((root / "videos").rglob("*.mp4")):
        with av.open(str(p)) as c:
            out[p.parent.parent.name] = out.get(p.parent.parent.name, 0) + sum(1 for _ in c.decode(video=0))
    return out


def record(log, n, queued=False):
    with log.run(TASK, policy=None) as run:
        q = run.queue() if queued else None
        for i in range(n):
            (q.put if q else run.add_frame)(frame(i), **kw(i))
        run.end("success")
    return run


@pytest.mark.parametrize("queued", [False, True])
def test_a_slow_encoder_gets_backpressure_and_loses_no_frame(tmp_path, slow_encoder, queued, monkeypatch):
    # Each encoder read takes longer than LeRobot's own 0.1 s wait to put a frame, so without the logger's wait
    # frames would be dropped; it stays well inside the 2 s budget.
    slow_encoder[0] = 0.3
    waits = []
    real = adapter.encoder_full

    def spy(ds):
        full = real(ds)
        waits.extend(full)
        return full
    monkeypatch.setattr(adapter, "encoder_full", spy)
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        run = record(log, 12, queued)
    ds.finalize()
    assert run.state == "committed"
    assert waits and all(k == CAM and n == m == QSIZE for k, n, m in waits)   # the logger did wait for room
    assert decoded_frames(ds.root) == {CAM: 12}
    assert adapter.episode_metadata(ds.root)[0]["video_frames"] == {CAM: 12}
    assert V.validate(ds.root)["ok"]


def test_past_the_budget_a_direct_run_stops_and_keeps_every_frame_it_recorded(tmp_path, slow_encoder):
    slow_encoder[0] = 0.5
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.2) as log:
        with pytest.raises(EncoderBacklog) as ei:
            with log.run(TASK, policy=None) as run:
                for i in range(20):
                    run.add_frame(frame(i), **kw(i))
    ds.finalize()
    stop = ei.value
    assert stop.reason == "encoder_backlog" and CAM in stop.detail and "within 0.2 s" in stop.detail
    n = stop.frames
    assert 0 < n < 20 and f"its {n} frames were saved as system_error" in str(stop)
    assert run.state == "committed" and run._end["termination_reason"] == "system_error"
    assert "encoder_backlog" in run._end["labels"]
    assert decoded_frames(ds.root) == {CAM: n} and ds.meta.total_frames == n
    assert V.validate(ds.root)["ok"]


def test_after_a_direct_stop_the_run_takes_no_frame_and_ends_only_as_system_error(tmp_path, slow_encoder):
    slow_encoder[0] = 0.5
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.1) as log:
        with log.run(TASK, policy=None) as run:
            stop = None
            for i in range(20):
                try:
                    run.add_frame(frame(i), **kw(i))
                except EncoderBacklog as e:
                    stop = e
                    break
            assert stop is not None
            n = len(run._rows)
            with pytest.raises(EncoderBacklog):
                run.add_frame(frame(n), **kw(n))          # the robot holds: nothing more is recorded
            assert len(run._rows) == n
            with pytest.raises(RunError, match="ends as system_error"):
                run.end("success")
            run.end("system_error")
    ds.finalize()
    assert run.state == "committed" and "encoder_backlog" in run._end["labels"]
    assert decoded_frames(ds.root) == {CAM: n}
    assert V.validate(ds.root)["ok"]


def test_past_the_budget_a_queued_run_is_quarantined_with_the_reason(tmp_path, slow_encoder):
    slow_encoder[0] = 0.5
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.2) as log:
        with pytest.raises(EncoderBacklog) as ei:
            with log.run(TASK, policy=None) as run:
                q = run.queue()
                for i in range(20):
                    q.put(frame(i), **kw(i))
                run.end("success")
    ds.finalize()
    assert ei.value.reason == "encoder_backlog" and run.state == "quarantined"
    assert ei.value.frames == 20 and ei.value.quarantine_path is not None
    res = V.validate(ds.root)
    assert res["ok"] and any("stopped by encoder_backlog" in w for w in res["warnings"]), res
    assert ds.meta.total_frames == 0


def test_a_full_queue_while_the_encoder_is_behind_names_the_encoder(tmp_path, slow_encoder):
    slow_encoder[0] = 0.5
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=30) as log:
        with pytest.raises(EncoderBacklog) as ei:
            with log.run(TASK, policy=None) as run:
                q = run.queue(max_frames=8)
                for i in range(40):
                    q.put(frame(i), **kw(i))
                    time.sleep(1 / 30)                     # a 30 fps loop
                run.end("success")
    assert "8 frames were waiting to be written while LeRobot's video encoder had no room" in ei.value.detail
    assert run.state == "quarantined"


def _no_backpressure(monkeypatch):
    monkeypatch.setattr(adapter, "encoder_full", lambda ds: [])


def test_frames_the_encoder_dropped_are_never_saved(tmp_path, slow_encoder, monkeypatch):
    """Without the backpressure (a LeRobot version whose queues the logger cannot see), LeRobot's own count of dropped
    frames still stops the run: a direct add_frame that lost a video frame raises and the run is discarded."""
    _no_backpressure(monkeypatch)
    slow_encoder[0] = 0.3
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with pytest.raises(RunError, match=r"video encoder dropped frames \(observation.images.top: 1\)"):
            record(log, 20)
    assert ds.meta.total_episodes == 0


def test_a_drop_found_at_save_refuses_to_seal_and_names_the_camera(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    calls = {"n": 0}
    real = adapter.encoder_dropped

    def dropped_at_save(d):
        calls["n"] += 1
        return {CAM: 3} if calls["n"] > 5 else real(d)       # the frames go in; then the save finds drops
    monkeypatch.setattr(adapter, "encoder_dropped", dropped_at_save)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with pytest.raises(RunError, match=r"dropped frames \(observation.images.top: 3\) of the run's 5"):
            run = None
            with log.run(TASK, policy=None) as run:
                for i in range(5):
                    run.add_frame(frame(i), **kw(i))
                run.end("success")
    assert run.state == "quarantined" and ds.meta.total_episodes == 0


def test_a_saved_video_that_differs_from_its_rows_is_not_committed(tmp_path, slow_encoder, monkeypatch):
    _no_backpressure(monkeypatch)
    monkeypatch.setattr(adapter, "encoder_dropped", lambda ds: {})
    slow_encoder[0] = 0.3
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with pytest.raises(RunError, match=r"video frames that differ from its 20 rows or cannot be read \(observation.images.top: \d+\)"):
            record(log, 20)
        assert log.broken
    ds.finalize()
    res = V.validate(ds.root)
    assert not res["ok"] and any("staged runs not committed" in e for e in res["errors"])


def test_validate_reports_a_video_already_misaligned_on_disk(tmp_path, slow_encoder, monkeypatch):
    """A dataset written before this fix, or by LeRobot without the logger: the episode record says how many frames
    each video holds, and validate compares them with the rows (an error for a run, a warning for an episode with
    no run)."""
    _no_backpressure(monkeypatch)
    monkeypatch.setattr(adapter, "encoder_dropped", lambda ds: {})
    monkeypatch.setattr(adapter, "saved_video_frames", lambda ds, e: {})
    slow_encoder[0] = 0.3
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        record(log, 20)
    for i in range(20):                                 # a second episode written by LeRobot alone
        ds.add_frame({**frame(i), "action": np.zeros(6, np.float32)})
    ds.save_episode()
    ds.finalize()
    held = decoded_frames(ds.root)[CAM]
    assert held < 40
    res = V.validate(ds.root)
    bad = [e for e in res["errors"] if "misaligned" in e]
    assert len(bad) == 1 and "episode 0 camera observation.images.top: its video holds" in bad[0], res["errors"]
    assert "the episode has 20 rows" in bad[0]
    assert any("episode 1 (no dataop run) camera observation.images.top" in w for w in res["warnings"]), res


def test_a_direct_stop_keeps_its_frames_when_a_prediction_names_the_frame_not_recorded(tmp_path, slow_encoder):
    """A policy's prediction is logged before its frame; the frame that hit the budget is not recorded, so that
    prediction is left out and the run is still saved with every frame it recorded."""
    slow_encoder[0] = 0.5
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.2) as log:
        with pytest.raises(EncoderBacklog) as ei:
            with log.run(TASK, policy=None, log_predictions=True) as run:
                for i in range(20):
                    run.predictions(i, np.zeros((2, 6), np.float32), obs_frame_index=i, infer_start_ns=1,
                                    infer_end_ns=2)
                    run.add_frame(frame(i), **kw(i))
    ds.finalize()
    n = ei.value.frames
    assert run.state == "committed" and 0 < n < 20
    assert "1 predictions for the frame that was not recorded were left out" in str(ei.value)
    assert len(run._predictions) == n and decoded_frames(ds.root) == {CAM: n}
    assert V.validate(ds.root)["ok"]


def test_a_direct_stop_that_cannot_be_sealed_quarantines_every_frame(tmp_path, slow_encoder, monkeypatch):
    from aftershock_logger.logger import Run
    slow_encoder[0] = 0.5
    ds = make_ds(tmp_path)

    def refuse(self):
        raise RunError("a tick check failed")
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.2) as log:
        with pytest.raises(EncoderBacklog) as ei:
            with log.run(TASK, policy=None) as run:
                for i in range(20):
                    if i == 3:
                        monkeypatch.setattr(Run, "_check_rows", refuse)
                    run.add_frame(frame(i), **kw(i))
    stop = ei.value
    assert run.state == "quarantined" and stop.quarantine_path is not None and stop.frames == len(run._rows)
    assert "could not be sealed" in str(stop) and "frames quarantined at" in str(stop)
    ds.finalize()
    res = V.validate(ds.root)
    assert res["ok"] and any("stopped by encoder_backlog" in w for w in res["warnings"]), res
    assert ds.meta.total_episodes == 0


def test_repair_never_commits_a_run_whose_saved_video_differs(tmp_path, slow_encoder, monkeypatch):
    _no_backpressure(monkeypatch)
    monkeypatch.setattr(adapter, "encoder_dropped", lambda ds: {})
    slow_encoder[0] = 0.3
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with pytest.raises(RunError, match="the run was not committed"):
            run = None
            with log.run(TASK, policy=None) as run:
                for i in range(20):
                    run.add_frame(frame(i), **kw(i))
                run.end("success")
    ds.finalize()
    report = V.repair(ds.root)
    assert any(run.run_id in r and "quarantined" in r and "misaligned" in r for r in report), report
    assert not list(store_runs(ds.root))


def store_runs(root):
    from aftershock_logger import store
    return [m for _, m in store.iter_manifests(root, "committed")]


def test_a_count_the_episode_record_does_not_give_is_a_problem():
    em = {"length": 5, "video_frames": {CAM: None, "observation.images.wrist": 5}}
    assert V.video_frame_problems(em) == [f"camera {CAM}: the episode record does not say how many frames its "
                                          "video holds"]
    rec = {"episode_index": [0], f"videos/{CAM}/chunk_index": [0], f"videos/{CAM}/from_timestamp": [None],
           f"videos/{CAM}/to_timestamp": [1.0], "videos/b/chunk_index": [None]}
    assert adapter.episode_video_frames(rec, 30, [CAM, "b"]) == {CAM: None, "b": None}   # b has no video
    assert adapter.episode_video_frames(rec, 30, [CAM, "b"], deferred=True) == {CAM: None}   # b encodes later
    assert adapter.episode_video_frames({"episode_index": [0]}, 30, [CAM]) == {CAM: None}   # no columns at all
    assert adapter.episode_video_frames({**rec, f"videos/{CAM}/from_timestamp": [0.5]}, 30, [CAM]) == {CAM: 15}
    assert adapter.episode_video_frames({**rec, f"videos/{CAM}/from_timestamp": [0.5]}, None, [CAM]) == {CAM: None}


def test_a_drive_episode_that_hits_the_budget_faults_the_session_and_keeps_its_frames(tmp_path, slow_encoder):
    import uuid
    from aftershock_logger import SessionLogger, session as S
    go = {"by": "owner@example.com", "at": "2026-11-07T09:10:00Z",
          "scope": {"model_id": "m_1", "planner": None, "task_id": "pick", "task_rev": 1, "station_rev": 0}}
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.2) as log:
        with SessionLogger(log, session_id=str(uuid.uuid4()), station_id="st_1", go=go, model_id="m_1", task=TASK,
                           task_id="pick", task_rev=1, policy=PolicyInfo("pi05", "ckpt/1")) as sess:
            for i in range(5):
                sess.add_frame(frame(i), **kw(i))
            slow_encoder[0] = 0.0
            sess.start_episode(moved_by="person")
            slow_encoder[0] = 0.5
            with pytest.raises(EncoderBacklog):
                for i in range(5, 40):
                    sess.add_frame(frame(i), **kw(i))
            n = sess.frames - 5
            with pytest.raises(EncoderBacklog):
                sess.add_frame(frame(99), **kw(99))      # nothing more is recorded
            assert sess.seal_episode("system_error") == "pending"
            assert sess.write_pending() == "committed"
            with pytest.raises(EncoderBacklog):
                sess.add_frame(frame(100), **kw(100))    # not into a gap either
            with pytest.raises(EncoderBacklog):
                sess.start_episode(moved_by="person")    # the station holds: no new episode
            rec = sess.finalize("user_stop")
    ds.finalize()
    assert rec["ended_by"] == "fault_stop" and rec["fault"] == "logger_failed"
    assert rec["runs"][0]["range"] == {"start": 5, "end": 5 + n} and rec["runs"][0]["termination"] == "system_error"
    rows = S.read_evidence(ds.root, rec["session_id"])
    assert [r["payload"]["fault"] for r in rows if r["kind"] == "fault"] == ["logger_failed"]
    assert decoded_frames(ds.root) == {CAM: n}
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]


@pytest.fixture
def held_encoder(monkeypatch):
    """The encoder threads read nothing while `gate` is clear: a camera's queue fills and stays full."""
    import threading
    gate = threading.Event()

    class HeldQueue(queue.Queue):
        def get(self, *a, **k):
            if self.maxsize == QSIZE:
                gate.wait()
            return super().get(*a, **k)
    monkeypatch.setattr(video_utils.queue, "Queue", HeldQueue)
    yield gate
    gate.set()


@pytest.mark.parametrize("queued", [False, True])
def test_a_frame_waits_while_the_encoder_queue_is_full_and_goes_in_when_it_has_room(tmp_path, held_encoder,
                                                                                   queued):
    """The write blocks while the queue is full, longer than LeRobot's own 0.1 s put timeout, drops nothing, and
    finishes once the encoder reads again: every frame is in the video."""
    import threading
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=30) as log:
        with log.run(TASK, policy=None) as run:
            q = run.queue() if queued else None
            put = q.put if q else run.add_frame
            for i in range(QSIZE):                     # these fill the queue the encoder does not read
                put(frame(i), **kw(i))
            done = threading.Event()
            t = threading.Thread(target=lambda: (put(frame(QSIZE), **kw(QSIZE)), done.set()), daemon=True)
            t.start()
            if queued:                                 # put returns at once; the writer waits for room
                assert done.wait(5)
                deadline = time.monotonic() + 5
                while adapter.encoder_full(ds) == [] and time.monotonic() < deadline:
                    time.sleep(0.01)
            assert adapter.encoder_full(ds) == [(CAM, QSIZE, QSIZE)]
            time.sleep(0.5)                            # well past LeRobot's 0.1 s put timeout
            assert adapter.encoder_dropped(ds) == {}
            assert len(run._rows) == QSIZE              # the waiting frame did not reach LeRobot
            if not queued:
                assert not done.is_set()
            held_encoder.set()                         # the encoder reads again
            t.join(10)
            assert done.is_set()
            run.end("success")
    ds.finalize()
    assert run.state == "committed" and decoded_frames(ds.root) == {CAM: QSIZE + 1}


def test_a_direct_quarantine_that_cannot_be_written_keeps_the_frames_for_the_spill(tmp_path, slow_encoder,
                                                                                  monkeypatch):
    from aftershock_logger import frame_queue as FQ
    from aftershock_logger.logger import Run
    slow_encoder[0] = 0.5
    ds = make_ds(tmp_path)
    real = FQ.write_partial
    calls = []

    def full_disk(run, rows, observations, info, qdir=None):
        calls.append(qdir)
        if qdir is None:
            raise OSError(28, "No space left on device")
        return real(run, rows, observations, info, qdir=qdir)
    monkeypatch.setattr(FQ, "write_partial", full_disk)
    monkeypatch.setattr(Run, "_check_rows", lambda self: (_ for _ in ()).throw(RunError("a tick check failed")))
    spill = tmp_path / "spill"
    with pytest.raises(EncoderBacklog) as ei:
        with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.2, spill_dir=str(spill)) as log:
            with log.run(TASK, policy=None) as run:
                for i in range(20):
                    run.add_frame(frame(i), **kw(i))
    stop = ei.value
    assert run.state == "quarantined" and "No space left" in (stop.quarantine_error or "")
    assert stop.spilled and (spill / run.run_id / "manifest.json").exists()
    assert calls[0] is None and calls[-1] is not None



GO = {"by": "owner@example.com", "at": "2026-11-07T09:10:00Z",
      "scope": {"model_id": "m_1", "planner": None, "task_id": "pick", "task_rev": 1, "station_rev": 0}}


def open_session(log):
    import uuid
    from aftershock_logger import SessionLogger
    return SessionLogger(log, session_id=str(uuid.uuid4()), station_id="st_1", go=GO, model_id="m_1", task=TASK,
                         task_id="pick", task_rev=1, policy=PolicyInfo("pi05", "ckpt/1"))


def backlog_episode(sess, slow_encoder, first, upto):
    """Open an episode and feed it until the encoder budget stops it; returns the frames the episode recorded."""
    slow_encoder[0] = 0.0
    sess.start_episode(moved_by="person")
    slow_encoder[0] = 0.5
    with pytest.raises(EncoderBacklog):
        for i in range(first, upto):
            sess.add_frame(frame(i), **kw(i))
    return sess._episode["run"]._count


def test_a_short_drive_episode_that_hits_the_budget_is_no_short_stop(tmp_path, slow_encoder):
    """An episode of a second or less that stopped on the encoder is sealed only as system_error and kept as a run,
    never discarded as a short stop (whose frames would move to a gap without the reason)."""
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.2) as log:
        with open_session(log) as sess:
            for i in range(5):
                sess.add_frame(frame(i), **kw(i))
            n = backlog_episode(sess, slow_encoder, 5, 40)
            assert n <= 30                                 # a second or less at 30 fps
            from aftershock_logger import SessionError
            with pytest.raises(SessionError, match="sealed as system_error"):
                sess.seal_episode("operator_abort")
            assert sess._episode is not None              # still open, nothing moved
            assert sess.seal_episode("system_error") == "pending"
            run = sess._pending["run"]
            assert sess.write_pending() == "committed"
            rec = sess.finalize("user_stop")
    ds.finalize()
    assert "encoder_backlog" in run._end["labels"] and run._end["termination_reason"] == "system_error"
    assert rec["runs"][0]["range"] == {"start": 5, "end": 5 + n} and not rec["gaps"][-1].get("short_stop")
    assert decoded_frames(ds.root) == {CAM: n}
    assert V.validate(ds.root)["ok"]


def test_a_drive_episode_stopped_by_the_encoder_that_cannot_be_staged_is_quarantined(tmp_path, slow_encoder,
                                                                                     monkeypatch):
    from aftershock_logger.logger import Run
    ds = make_ds(tmp_path)
    real = Run._write_files

    def no_disk(self, status, *a, **k):
        if status == "staged":
            raise OSError(28, "No space left on device")
        return real(self, status, *a, **k)
    with RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.2) as log:
        with open_session(log) as sess:
            for i in range(5):
                sess.add_frame(frame(i), **kw(i))
            n = backlog_episode(sess, slow_encoder, 5, 80)
            run = sess._episode["run"]
            monkeypatch.setattr(Run, "_write_files", no_disk)
            with pytest.raises(OSError):
                sess.seal_episode("system_error")
            monkeypatch.setattr(Run, "_write_files", real)
            assert run.state == "quarantined" and run.quarantine_path is not None
            assert sess._episode is None and sess.runs[-1]["write_status"] == "failed"
            rec = sess.finalize("user_stop")
    ds.finalize()
    assert rec["ended_by"] == "fault_stop" and rec["runs"][0]["range"] == {"start": 5, "end": 5 + n}
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]


def test_closing_the_logger_keeps_a_run_left_open_after_an_encoder_stop(tmp_path, slow_encoder):
    ds = make_ds(tmp_path)
    log = RunLogger(ds, SCHEMA, planner=None, encoder_wait_s=0.2)
    cm = log.run(TASK, policy=None)
    run = cm.__enter__()
    slow_encoder[0] = 0.5
    with pytest.raises(EncoderBacklog):
        for i in range(30):
            run.add_frame(frame(i), **kw(i))
    log.close()                                            # the caller never ended the run nor left its block
    ds.finalize()
    assert run.state == "committed" and "encoder_backlog" in run._end["labels"]
    assert decoded_frames(ds.root) == {CAM: len(run._rows)}


def _drop_camera_meta(root: Path, cam: str, null_only: bool) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    for f in (root / "meta" / "episodes").rglob("*.parquet"):
        t = pq.read_table(f)
        names = [c for c in t.column_names if c.startswith(f"videos/{cam}/")]
        if null_only:
            for c in names:
                i = t.column_names.index(c)
                t = t.set_column(i, c, pa.nulls(len(t), t.schema.field(c).type))
        else:
            t = t.drop(names)
        pq.write_table(t, f)


@pytest.mark.parametrize("null_only", [False, True])
def test_a_camera_missing_from_the_episode_record_is_reported_not_skipped(tmp_path, null_only):
    """Two cameras; the wrist camera's columns are removed from meta/episodes (or set to null). The expected cameras
    come from the dataset's features, so validate reports the wrist camera instead of checking only the top one."""
    ds = make_ds(tmp_path, cams=(CAM, WRIST))
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with log.run(TASK, policy=None) as run:
            for i in range(6):
                run.add_frame(frame(i, (CAM, WRIST)), **kw(i))
            run.end("success")
    ds.finalize()
    assert adapter.episode_metadata(ds.root)[0]["video_frames"] == {CAM: 6, WRIST: 6}
    _drop_camera_meta(ds.root, WRIST, null_only)
    assert adapter.episode_metadata(ds.root)[0]["video_frames"] == {CAM: 6, WRIST: None}
    res = V.validate(ds.root)
    assert not res["ok"]
    assert any(f"camera {WRIST}: the episode record does not say how many frames" in e for e in res["errors"]), res


def test_at_save_a_camera_the_record_leaves_out_stops_the_commit(tmp_path, monkeypatch):
    ds = make_ds(tmp_path, cams=(CAM, WRIST))
    real = adapter.saved_video_frames

    def wrist_gone(d, e):
        rec = dict(d.meta.latest_episode)
        for k in [k for k in rec if k.startswith(f"videos/{WRIST}/")]:
            del rec[k]
        monkeypatch.setattr(d.meta, "latest_episode", rec)
        return real(d, e)
    monkeypatch.setattr(adapter, "saved_video_frames", wrist_gone)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with pytest.raises(RunError, match=r"or cannot be read \(observation\.images\.wrist: unknown\)"):
            with log.run(TASK, policy=None) as run:
                for i in range(6):
                    run.add_frame(frame(i, (CAM, WRIST)), **kw(i))
                run.end("success")
    assert run.state == "save_failed"
