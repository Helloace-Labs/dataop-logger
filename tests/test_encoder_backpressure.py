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

from aftershock_logger import ActionSchema, EncoderBacklog, RunError, RunLogger, adapter  # noqa: E402
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


def make_ds(tmp_path, name="ds"):
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
             "action": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
             CAM: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]}}
    return lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features=feats, root=tmp_path / name,
                                                 use_videos=True, streaming_encoding=True,
                                                 encoder_queue_maxsize=QSIZE)


def frame(i):
    return {"observation.state": np.full(6, i, np.float32), CAM: np.full((64, 64, 3), i * 9 % 255, np.uint8),
            "task": TASK}


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
def test_a_slow_encoder_gets_backpressure_and_loses_no_frame(tmp_path, slow_encoder, queued):
    slow_encoder[0] = 0.1                              # 10 frames a second for a 30 fps loop
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        run = record(log, 20, queued)
    ds.finalize()
    assert run.state == "committed"
    assert decoded_frames(ds.root) == {CAM: 20}
    assert adapter.episode_metadata(ds.root)[0]["video_frames"] == {CAM: 20}
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
        with pytest.raises(RunError, match=r"video frames that differ from its 20 rows \(observation.images.top: \d+\)"):
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
    from aftershock_logger.logger import Run
    monkeypatch.setattr(Run, "_saved_video_frames", lambda self: {})
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
