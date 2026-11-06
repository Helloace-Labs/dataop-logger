"""Logger L10: the live tap (status.json, events.jsonl, preview only on request, never in the way of recording)."""
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from aftershock_logger import ActionSchema, ControlledStop, PolicyInfo, RunLogger, store  # noqa: E402
from aftershock_logger import live as L  # noqa: E402
from aftershock_logger import validate as V  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCHEMA = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
BOUNDED = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30,
                       low=(-200.0,) * 6, high=(200.0,) * 6)
TASK = "pick up the part"
POLICY = PolicyInfo("policy", "ckpt/1")
CAMS = ("observation.images.top", "observation.images.wrist")
PERIOD_NS = 33_333_333


def make_ds(tmp_path, name="ds", cams=0, hw=(24, 32)):
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
             "action": {"dtype": "float32", "shape": (6,), "names": list(NAMES)}}
    for k in CAMS[:cams]:
        feats[k] = {"dtype": "image", "shape": (*hw, 3), "names": ["height", "width", "channels"]}
    return lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features=feats, root=tmp_path / name,
                                                 use_videos=False)


def frame(i, cams=0, hw=(24, 32)):
    f = {"observation.state": np.full(6, i, np.float32), "task": TASK}
    for k in CAMS[:cams]:
        f[k] = np.full((*hw, 3), i % 255, np.uint8)
    return f


def kw(i, mode="base", t=None, action=None):
    t = 1_000_000 + i * PERIOD_NS if t is None else t
    a = np.arange(6, dtype=np.float32) + i if action is None else action
    if mode == "expert":
        return dict(sent_action=a, expert_action=a, expert_source="teleop:leader", control_mode="expert",
                    obs_ns=t, sent_ns=t + 1000)
    return dict(sent_action=a, base_action=a, control_mode="base", inference_id=i, chunk_offset=0,
                obs_ns=t, sent_ns=t + 1000)


def record(log, n=10, takeover=(), cams=0, hw=(24, 32), reason="success", realtime=False, queued=False,
           camera_ns=None, action=None):
    with log.run(task=TASK, policy=POLICY) as run:
        add = run.queue().put if queued else run.add_frame
        t0 = time.monotonic_ns()
        for i in range(n):
            t = time.monotonic_ns() if realtime else 1_000_000 + i * PERIOD_NS
            extra = {"camera_ns": camera_ns(i, t)} if camera_ns else {}
            add(frame(i, cams, hw), **kw(i, "expert" if i in takeover else "base", t,
                                          None if action is None else action(i)), **extra)
            if realtime:
                time.sleep(max(0.0, (t0 + (i + 1) * PERIOD_NS - time.monotonic_ns()) / 1e9))
        run.end(termination_reason=reason)
    return run


def live_files(root):
    d = L.live_dir(root)
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()) if d.exists() else None


def wait_for(cond, timeout, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return cond()


def kinds(events, kind):
    return [e for e in events if e.get("kind") == kind]


def sidecar_bytes(root, run_id):
    """What a run recorded, minus what differs between any two recordings (run id, times, writer identity)."""
    p = store.runs_dir(root) / run_id
    m = json.loads((p / "manifest.json").read_text())
    for k in ("run_id", "started_at", "ended_at", "integrity", "env", "dataset"):
        m.pop(k, None)
    data = sorted((root / "data").rglob("*.parquet"))
    return (p / "ticks.parquet").read_bytes(), m, [d.read_bytes() for d in data]


# ── 1. Off by default ───────────────────────────────────────────────────────

def test_tap_off_makes_no_live_folder(tmp_path):
    ds = make_ds(tmp_path, cams=1)
    with RunLogger(ds, SCHEMA) as log:
        assert log.live_tap is None
        record(log, cams=1)
        record(log, cams=1, queued=True)
    assert not L.live_dir(ds.root).exists()


# ── 3. Status, events, transitions and idle heartbeat ───────────────────────

def test_realtime_replay_status_is_always_whole_and_events_miss_no_transition(tmp_path):
    """A replay at 30 fps (two runs, the first with a 3 frame takeover) read by a tight loop reader of status.json
    and by a reader that polls events.jsonl once a second, as the station does."""
    ds = make_ds(tmp_path, cams=2)
    stop, stop_tight = threading.Event(), threading.Event()
    seen = {"reads": 0, "bad": [], "seqs": [], "ats": []}

    def tight_reader():
        path = L.live_dir(ds.root) / L.STATUS
        while not stop_tight.is_set():
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                continue
            seen["reads"] += 1
            try:
                d = json.loads(raw)
            except ValueError as e:
                seen["bad"].append(str(e))
                continue
            if not seen["seqs"] or seen["seqs"][-1] != d["seq"]:
                seen["seqs"].append(d["seq"])
                seen["ats"].append((time.monotonic(), d["at_ns"], d["closed"]))

    polled, acked = [], [0]

    def poller():
        while not stop.is_set():
            for rec in L.read_events(ds.root, acked[0]):
                assert rec.get("kind") not in ("gap", "reset")
                polled.append(rec)
                acked[0] = rec["seq"]
            stop.wait(1.0)

    threads = [threading.Thread(target=tight_reader), threading.Thread(target=poller)]
    for t in threads:
        t.start()
    try:
        with RunLogger(ds, SCHEMA, live=True) as log:
            r1 = record(log, n=150, takeover=(60, 61, 62), cams=2, realtime=True, queued=True)
            time.sleep(0.5)
            r2 = record(log, n=90, cams=2, realtime=True, reason="failure")
            stop_tight.set()
            threads[0].join()
            idle = []                                           # open and idle: sampled the way a station polls
            for _ in range(32):
                time.sleep(0.1)
                d = L.read_status(ds.root)
                if not idle or idle[-1] != d["at_ns"]:
                    idle.append(d["at_ns"])
            st = L.read_status(ds.root)
            assert st["run"] is None and st["last_run"]["run_id"] == r2.run_id
            assert st["last_run"]["state"] == "committed" and st["last_run"]["termination_reason"] == "failure"
            assert st["pid"] == os.getpid() and st["closed"] is False
        time.sleep(1.2)
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert seen["reads"] > 100 and not seen["bad"]
    assert seen["seqs"] == sorted(set(seen["seqs"]))             # status seq only goes up
    gaps = [(b - a) / 1e9 for a, b in zip(idle, idle[1:])]
    assert len(idle) >= 3 and max(gaps) <= 1.0                  # at moves at least once a second while idle
    assert L.read_status(ds.root)["closed"] is True
    for rec in L.read_events(ds.root, acked[0]):                # what came after the last poll
        polled.append(rec)
    assert [e["seq"] for e in polled] == list(range(1, len(polled) + 1))
    assert [e["run_id"] for e in kinds(polled, "run_start")] == [r1.run_id, r2.run_id]
    ends = kinds(polled, "run_end")
    assert [(e["run_id"], e["termination_reason"], e["frames"]) for e in ends] == [
        (r1.run_id, "success", 150), (r2.run_id, "failure", 90)]
    assert [(e["run_id"], e["state"]) for e in kinds(polled, "run_state")] == [
        (r1.run_id, "committed"), (r2.run_id, "committed")]
    modes = [(e["frame"], e["control_mode"]) for e in kinds(polled, "control_mode") if e["run_id"] == r1.run_id]
    assert modes == [(0, "base"), (60, "expert"), (63, "base")]  # the 3 frame takeover is two records
    ds.finalize()
    assert V.validate(ds.root)["ok"]


def test_status_carries_run_joint_state_schema_and_cameras(tmp_path):
    ds = make_ds(tmp_path, cams=2)
    with RunLogger(ds, SCHEMA, live=True) as log:
        with log.run(task=TASK, policy=POLICY, task_id="pick", task_rev=3) as run:
            q = run.queue()
            for i in range(5):
                q.put(frame(i, 2), **kw(i), prompt="pick it", subtask_cmd=None)
            st = wait_for(lambda: (L.read_status(ds.root) or {}).get("run") and
                          L.read_status(ds.root)["run"]["frames"] == 5 and L.read_status(ds.root), 3)
            run.end(termination_reason="success")
    r = st["run"]
    assert r["run_id"] == run.run_id and r["task"] == TASK and r["task_id"] == "pick" and r["task_rev"] == 3
    assert r["policy"]["family"] == "policy" and r["frame"] == 4 and r["control_mode"] == "base"
    assert r["prompt"] == "pick it" and r["subtask_cmd"] is None and r["state"] == "active"
    assert st["joint_state"] == {"key": "observation.state", "names": list(NAMES), "values": [4.0] * 6}
    assert st["action_schema"] == {"id": SCHEMA.id, "names": list(NAMES), "units": list(SCHEMA.units),
                                   "space": "joint_position", "rate_hz": 30}
    assert set(st["cameras"]) == {"top", "wrist"} and st["cameras"]["top"]["width"] == 32
    assert st["queue"]["capacity"] == 60 and st["controlled_stop"] is None
    assert st["preview"] == {"active": False, "interval_s": None, "frames": {}}


# ── 2. Cost on the caller's thread ──────────────────────────────────────────

def _hook_costs(log):
    """Wrap the tap's frame hook to time it: the tap's extra cost per add_frame or put."""
    tap, real, costs = log.live_tap, log.live_tap.frame, []

    def timed(*a, **k):
        t = time.perf_counter_ns()
        real(*a, **k)
        costs.append(time.perf_counter_ns() - t)
    tap.frame = timed
    return costs


def _p99(xs):
    return statistics.quantiles(xs, n=100)[98] / 1e6


@pytest.mark.parametrize("queued", [False, True], ids=["add_frame", "put"])
def test_tap_cost_per_frame_stays_small(tmp_path, queued):
    """Acceptance 2 at reduced length. AFTERSHOCK_LIVE_BENCH=1 runs the full 2,000 frame replay at 30 fps with two
    480x640 cameras and prints the numbers for the pull request."""
    bench = os.environ.get("AFTERSHOCK_LIVE_BENCH") == "1"
    n, hw = (2000, (480, 640)) if bench else (150, (120, 160))
    out = {}
    for preview in (False, True):
        ds = make_ds(tmp_path, f"ds{int(preview)}", cams=2, hw=hw)
        with RunLogger(ds, SCHEMA, live=True) as log:
            if preview:
                L.live_dir(ds.root).mkdir(parents=True, exist_ok=True)
                (L.live_dir(ds.root) / L.PREVIEW_WANT).write_text('{"interval_s": 1}')
                wait_for(lambda: log.live_tap.preview_interval_ns, 2)
            costs = _hook_costs(log)
            stop = threading.Event()
            if preview:                                          # keep the request fresh, as the station does
                def touch():
                    while not stop.wait(1.0):
                        os.utime(L.live_dir(ds.root) / L.PREVIEW_WANT)
                threading.Thread(target=touch, daemon=True).start()
            record(log, n=n, takeover=range(100, 103), cams=2, hw=hw, realtime=True, queued=queued)
            stop.set()
            out[preview] = (_p99(costs), max(costs) / 1e6, log.live_tap.copies, log.live_tap.encodes)
    if bench:
        print(f"\nlive tap cost per {'put' if queued else 'add_frame'}, {n} frames at 30 fps, two {hw} cameras: "
              f"no preview p99 {out[False][0]:.3f} ms (max {out[False][1]:.3f}); "
              f"preview every 1 s p99 {out[True][0]:.3f} ms (max {out[True][1]:.3f}), "
              f"{out[True][2]} frames copied, {out[True][3]} encoded")
    assert out[False][0] < 0.5 and out[False][2] == 0 and out[False][3] == 0
    assert out[True][0] < 1.5 and out[True][2] >= 2 * (n // 30 - 2) and out[True][3] >= 2


# ── 4. Preview only on request ──────────────────────────────────────────────

def test_preview_only_while_requested(tmp_path):
    ds = make_ds(tmp_path, cams=2, hw=(480, 640))
    pdir = L.live_dir(ds.root) / L.PREVIEW_DIR
    want = L.live_dir(ds.root) / L.PREVIEW_WANT
    with RunLogger(ds, SCHEMA, live=True) as log:
        tap = log.live_tap
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue(max_seconds=30)
            i = 0

            def tick(seconds):
                nonlocal i
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    q.put(frame(i, 2, (480, 640)), **kw(i, t=time.monotonic_ns()))
                    i += 1
                    time.sleep(1 / 30)
            tick(2.0)
            assert tap.copies == 0 and tap.encodes == 0
            assert not pdir.exists() or not any(pdir.iterdir())
            want.write_text(json.dumps({"interval_s": 1}))
            asked = time.monotonic()
            while not (pdir.is_dir() and {p.name for p in pdir.iterdir()} == {"top.jpg", "wrist.jpg"}):
                tick(0.05)
                assert time.monotonic() - asked < 1 + 1 + 0.5     # within interval_s + 1 s
            from PIL import Image
            im = Image.open(pdir / "top.jpg")
            assert im.format == "JPEG" and im.width == 320 and im.height == 240
            st = wait_for(lambda: (L.read_status(ds.root) or {}).get("preview", {}).get("frames", {}).get("top")
                          and L.read_status(ds.root), 2)
            assert st["preview"]["active"] and st["preview"]["interval_s"] == 1.0
            tick(2.0)
            copies = tap.copies
            assert 4 <= copies <= 8                               # one per camera per second
            want.unlink()                                         # the request lapses: files go at once
            gone = time.monotonic()
            while pdir.exists() and any(pdir.iterdir()):
                tick(0.05)
                assert time.monotonic() - gone < 11
            tick(1.0)
            copies = tap.copies
            tick(1.5)
            assert tap.copies == copies                           # no frame copied without a request
            stale = want.with_name("stale.tmp")                   # an old request counts as none
            stale.write_text("{}")
            old = time.time() - L.PREVIEW_WANT_S - 1
            os.utime(stale, (old, old))
            os.replace(stale, want)
            tick(1.5)
            assert tap.copies == copies and not any(pdir.iterdir())
            run.end(termination_reason="success")


def test_preview_files_go_when_the_request_ages_out(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "PREVIEW_WANT_S", 2.0)
    ds = make_ds(tmp_path, cams=1)
    want = L.live_dir(ds.root) / L.PREVIEW_WANT
    want.parent.mkdir(parents=True)
    want.write_text('{"interval_s": 1}')
    with RunLogger(ds, SCHEMA, live=True) as log:
        with log.run(task=TASK, policy=POLICY) as run:
            q = run.queue(max_seconds=30)
            for i in range(120):
                q.put(frame(i, 1), **kw(i, t=time.monotonic_ns()))
                time.sleep(1 / 30)
                if i == 20:
                    assert wait_for(lambda: (L.live_dir(ds.root) / "preview" / "top.jpg").exists(), 2)
            run.end(termination_reason="success")
        assert not (L.live_dir(ds.root) / "preview" / "top.jpg").exists()
        assert log.live_tap.preview_interval_ns is None


# ── 5. A tap that fails never touches the recording ─────────────────────────

def _recorded(tmp_path, name, live, prepare=None, cams=1):
    ds = make_ds(tmp_path, name, cams=cams)
    if prepare:
        prepare(ds)
    with pytest.warns(RuntimeWarning, match="live tap stopped") if prepare else _nothing():
        with RunLogger(ds, SCHEMA, live=live) as log:
            run = record(log, n=40, takeover=(10, 11, 12), cams=cams, queued=True)
            tap = log.live_tap
    return ds, run, tap


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_tap_that_cannot_write_leaves_the_run_byte_for_byte_the_same(tmp_path):
    base_ds, base_run, _ = _recorded(tmp_path, "off", live=False)
    ok_ds, ok_run, _ = _recorded(tmp_path, "on", live=True)

    def read_only(ds):
        d = L.live_dir(ds.root)
        d.mkdir(parents=True)
        d.chmod(0o555)
    ro_ds, ro_run, tap = _recorded(tmp_path, "ro", live=True, prepare=read_only)
    L.live_dir(ro_ds.root).chmod(0o755)
    assert tap.failed and "Permission" in tap.failed
    want = sidecar_bytes(base_ds.root, base_run.run_id)
    assert sidecar_bytes(ok_ds.root, ok_run.run_id) == want
    assert sidecar_bytes(ro_ds.root, ro_run.run_id) == want
    for ds in (ok_ds, ro_ds):
        ds.finalize()
        assert V.validate(ds.root)["ok"]
    assert ro_run.state == "committed"


def test_an_encoder_error_turns_the_tap_off_for_the_run_and_writes_error_json(tmp_path, monkeypatch):
    calls = []

    def broken(img, *a, **k):
        calls.append(1)
        raise OSError("encoder missing")
    monkeypatch.setattr(L, "encode_preview", broken)
    base_ds, base_run, _ = _recorded(tmp_path, "off", live=False)

    def ask(ds):
        L.live_dir(ds.root).mkdir(parents=True)
        (L.live_dir(ds.root) / L.PREVIEW_WANT).write_text('{"interval_s": 1}')
    ds = make_ds(tmp_path, "enc", cams=1)
    ask(ds)
    with pytest.warns(RuntimeWarning, match="encoder missing"):
        with RunLogger(ds, SCHEMA, live=True) as log:
            assert wait_for(lambda: log.live_tap.preview_interval_ns, 2)
            with log.run(task=TASK, policy=POLICY) as run:
                q = run.queue()
                for i in range(40):
                    q.put(frame(i, 1), **kw(i, "expert" if i in (10, 11, 12) else "base"))
                    if i == 0:
                        assert wait_for(lambda: log.live_tap.failed, 3)
                run.end(termination_reason="success")
            err = L.read_error(ds.root)
            assert err and "encoder missing" in err["error"]
            writes = log.live_tap.writes
            time.sleep(1.2)
            assert log.live_tap.writes == writes                  # off for the rest of the run
            (L.live_dir(ds.root) / L.PREVIEW_WANT).unlink()      # with the encoder still broken, no preview asked
            record(log, n=5, cams=1)                              # the next run tries again
            assert wait_for(lambda: log.live_tap.writes > writes, 3)
    assert len(calls) >= 1 and run.state == "committed" and L.read_error(ds.root) is None
    assert sidecar_bytes(ds.root, run.run_id)[0] == sidecar_bytes(base_ds.root, base_run.run_id)[0]


def test_hook_errors_never_reach_the_caller(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    with pytest.warns(RuntimeWarning, match="live tap stopped"):
        with RunLogger(ds, SCHEMA, live=True) as log:
            monkeypatch.setattr(log.live_tap, "_rec05", lambda *a: 1 / 0)
            run = record(log, n=10)
            assert run.state == "committed" and "ZeroDivisionError" in log.live_tap.failed
    ds.finalize()
    assert V.validate(ds.root)["ok"]


# ── 6. Not part of the dataset ──────────────────────────────────────────────

def test_validate_and_export_ignore_the_live_folder(tmp_path):
    from aftershock_logger import batches as B
    from aftershock_logger import export as X
    from aftershock_logger import review as rv
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, live=True, batch="r") as log:
        record(log), record(log, takeover=(3, 4))
    ds.finalize()
    for p, _ in store.iter_manifests(ds.root, "committed"):
        rv.set_reviewed(p, True)
    B.set_status(ds.root, "r", "locked")
    assert "status.json" in live_files(ds.root)
    assert V.validate(ds.root)["ok"]
    X.export([(ds.root, "r")], "filtered", str(tmp_path / "with"))
    aside = tmp_path / "live-aside"
    shutil.move(str(L.live_dir(ds.root)), str(aside))
    assert V.validate(ds.root)["ok"]
    X.export([(ds.root, "r")], "filtered", str(tmp_path / "without"))

    def tree(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
                and "dataop" not in p.parts}
    a, b = tree(tmp_path / "with"), tree(tmp_path / "without")
    note = "dataop_export.json"                                  # names its own output folder
    na, nb = (json.loads(x.pop(note).decode().replace(name, "OUT")) for x, name in ((a, "/with"), (b, "/without")))
    for x in (na, nb):
        x.pop("export_id"), x.pop("at")
    assert na == nb
    assert a.keys() == b.keys() and a == b
    assert not any("live" in k for k in a)


# ── crash safety, rotation, privacy ─────────────────────────────────────────

CHILD = textwrap.dedent('''
    import sys, time
    sys.path.insert(0, sys.argv[1])
    import numpy as np
    from aftershock_logger import ActionSchema, PolicyInfo, RunLogger
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names = ("a", "b")
    s = ActionSchema(names=names, units=("deg", "deg"), space="joint_position", rate_hz=30)
    feats = {k: {"dtype": "float32", "shape": (2,), "names": list(names)} for k in ("observation.state", "action")}
    ds = LeRobotDataset.create("t/crash", fps=30, features=feats, root=sys.argv[2], use_videos=False)
    log = RunLogger(ds, s, live=True)
    i = 0
    while True:
        with log.run(task="t", policy=PolicyInfo("p", "c")) as run:
            for k in range(20):
                t = time.monotonic_ns()
                a = np.array([k, k], np.float32)
                mode = dict(expert_action=a, expert_source="x", control_mode="expert") if k % 3 == 0 else \\
                    dict(base_action=a, control_mode="base", inference_id=k, chunk_offset=0)
                run.add_frame({"observation.state": a, "task": "t"}, sent_action=a, obs_ns=t, sent_ns=t + 1, **mode)
            run.end(termination_reason="success")
        i += 1
''')


def test_a_writer_killed_mid_write_leaves_whole_files_and_the_next_writer_continues(tmp_path):
    root = tmp_path / "crash"
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(SRC), str(root)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    try:
        assert wait_for(lambda: len(L.read_events(root)) > 300, 60, 0.01)
    finally:
        child.send_signal(signal.SIGKILL)
        child.wait()
    st = L.read_status(root)
    assert st is not None and st["closed"] is False and st["pid"] == child.pid
    events = L.read_events(root)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    last = events[-1]["seq"]
    # a writer that died in the middle of an append, and in the middle of a status write
    with open(L.live_dir(root) / L.EVENTS, "ab") as f:
        f.write(b'{"seq": ' + str(last + 1).encode() + b', "kind": "run_st')
    (L.live_dir(root) / f".status.json.{child.pid}.tmp").write_bytes(b'{"live_form')
    assert L.read_events(root)[-1]["seq"] == last                # the torn line is never read
    assert L.read_status(root)["seq"] == st["seq"]
    # the next writer (a new process opens the same root; here a fresh dataset holding the same live folder)
    ds = make_ds(tmp_path, "next")
    shutil.copytree(L.live_dir(root), L.live_dir(ds.root))
    with RunLogger(ds, SCHEMA, live=True) as log:
        run = record(log, n=5)
    after = L.read_events(ds.root, last)
    assert after and after[0]["seq"] == last + 1 and after[0]["kind"] == "run_start"
    assert after[0]["run_id"] == run.run_id
    assert [e["seq"] for e in after] == list(range(last + 1, last + 1 + len(after)))
    raw = (L.live_dir(ds.root) / L.EVENTS).read_bytes()
    assert all(json.loads(line) for line in raw.splitlines())
    assert not list(L.live_dir(ds.root).glob(".*.tmp"))
    st2 = L.read_status(ds.root)
    assert st2["seq"] > st["seq"] and st2["closed"] is True and st2["pid"] == os.getpid()


def test_events_rotate_and_a_reader_behind_the_rotation_sees_a_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "ROTATE_RECORDS", 20)
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, live=True) as log:
        for _ in range(6):
            record(log, n=12, takeover=(3, 4, 8))
            time.sleep(0.6)
    d = L.live_dir(ds.root)
    assert (d / L.EVENTS_OLD).exists()
    for p in (d / L.EVENTS, d / L.EVENTS_OLD):
        lines = [json.loads(x) for x in p.read_bytes().splitlines()]
        assert lines[0]["kind"] == "file_start" and lines[1]["seq"] == lines[0]["first_seq"]
        assert len(lines) - 1 <= 20
    total = 6 * (1 + 5 + 1 + 1)                                  # start, 5 mode changes, end, committed per run
    everything = L.read_events(ds.root)
    assert everything[0]["kind"] == "gap" and everything[-1]["seq"] == total
    first = everything[1]["seq"]
    assert everything[0] == {"kind": "gap", "from_seq": 1, "to_seq": first - 1}
    assert [e["seq"] for e in everything[1:]] == list(range(first, total + 1))
    assert L.read_events(ds.root, total - 2) == everything[-2:]
    assert L.read_events(ds.root, total + 50) == [{"kind": "reset", "last_seq": total}]


def test_no_home_path_root_or_host_name_in_live_files(tmp_path, monkeypatch):
    home = os.path.expanduser("~")
    ds = make_ds(tmp_path)
    policy = PolicyInfo("policy", f"{home}/ckpts/{socket.gethostname()}/step10k")

    def failing(f):
        raise OSError(28, f"No space left on device: '{ds.root}/data/chunk-000'")
    with RunLogger(ds, BOUNDED, live=True) as log:
        with pytest.raises(ControlledStop):
            with log.run(task=TASK, policy=policy) as run:
                q = run.queue()
                monkeypatch.setattr(ds, "add_frame", failing)
                for i in range(30):
                    q.put(frame(i), **kw(i))
                    time.sleep(0.005)
                run.end(termination_reason="success")
        time.sleep(0.7)
        st = L.read_status(ds.root)
    stop = st["controlled_stop"]
    assert stop["run_id"] == run.run_id and stop["reason"] == "write_failed"
    assert stop["quarantine"] == f"dataop/quarantine/{run.run_id}" and stop["quarantine_written"]
    assert "<root>" in stop["detail"]
    ev = L.read_events(ds.root)
    assert [e["kind"] for e in ev if e.get("run_id") == run.run_id][-3:] == ["controlled_stop", "run_end",
                                                                             "run_state"]
    assert kinds(ev, "run_state")[-1]["state"] == "quarantined"
    text = "".join(p.read_text() for p in L.live_dir(ds.root).rglob("*") if p.is_file())
    for secret in (home, str(ds.root), str(Path(ds.root).resolve()), socket.gethostname()):
        assert secret not in text


def test_rec05_and_rec04_while_recording(tmp_path):
    ds = make_ds(tmp_path, cams=2)
    out = lambda i: np.full(6, 500.0 if 10 <= i < 13 else i, np.float32)        # noqa: E731

    def cams(i, t):                                              # the wrist camera stalls for frames 20 to 39
        return {"top": t, "wrist": 1_000_000 + 19 * PERIOD_NS if 20 <= i < 40 else t}
    with RunLogger(ds, BOUNDED, live=True) as log:
        run = record(log, n=60, cams=2, camera_ns=cams, action=out)
        time.sleep(0.7)
        st = L.read_status(ds.root)
    rec = [e for e in L.read_events(ds.root) if e["kind"] == "rec_result"]
    r05 = [(e["state"], e["frames"]) for e in rec if e["check_id"] == "REC-05"]
    assert r05 == [("open", [10, None]), ("closed", [10, 13])]
    r04 = [(e["state"], e["camera"], e["frames"]) for e in rec if e["check_id"] == "REC-04"]
    assert r04 == [("open", "wrist", [20, None]), ("closed", "wrist", [20, 40])]
    closed = [e for e in rec if e["check_id"] == "REC-04" and e["state"] == "closed"][0]
    assert closed["review"] is False and 0.6 < closed["seconds"] < 0.7
    assert st["last_run"]["rec"] == {"REC-04": {"gaps": 1, "frames_missed": 20, "open": []},
                                     "REC-05": {"frames": 3, "spans": 1, "open": False}}
    assert run.state == "committed"
    m = json.loads((store.runs_dir(ds.root) / run.run_id / "manifest.json").read_text())
    assert m["events"] == []                                     # live results never enter the sidecar


def test_close_writes_closed_and_stops_the_thread(tmp_path):
    ds = make_ds(tmp_path)
    log = RunLogger(ds, SCHEMA, live=True)
    record(log, n=3)
    thread = log.live_tap._thread
    log.close()
    assert not thread.is_alive()
    st = L.read_status(ds.root)
    assert st["closed"] is True and st["run"] is None
    # a run discarded on close shows as discarded
    log = RunLogger(ds, SCHEMA, live=True)
    with pytest.raises(ZeroDivisionError):
        with log.run(task=TASK, policy=POLICY) as run:
            run.add_frame(frame(0), **kw(0))
            1 / 0
    log.close()
    last = kinds(L.read_events(ds.root), "run_state")[-1]
    assert (last["run_id"], last["state"]) == (run.run_id, "discarded")
