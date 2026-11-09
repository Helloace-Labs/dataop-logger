"""Drive sessions (logger task L9): runs and gaps cover every session frame once, the evidence log, write status.

Scripted sessions on real LeRobot datasets with one small image camera, so the gap dataset is shown to carry every
camera, observation.state and action. Set AFTERSHOCK_CONTRACTS to a checkout with the Drive contracts
(contracts/schema/drive.py) to also load the written records with the cloud models.
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataop_logger import ActionSchema, PolicyInfo, RunError, RunLogger, SessionError, SessionLogger  # noqa: E402
from dataop_logger import session as S  # noqa: E402
from dataop_logger import store  # noqa: E402
from dataop_logger import validate as V  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCHEMA = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
TASK = "pick up the tube"
POLICY = PolicyInfo("pi05", "local/checkpoint")
GO = {"by": "owner@example.com", "at": "2026-11-07T09:10:00Z",
      "scope": {"model_id": "m_1", "planner": None, "task_id": "tube-pick", "task_rev": 3, "station_rev": 7}}


def make_ds(tmp_path, name="ds"):
    features = {"observation.state": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
                "action": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
                "observation.images.top": {"dtype": "image", "shape": (8, 8, 3),
                                           "names": ["height", "width", "channels"]}}
    return lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features=features, root=tmp_path / name,
                                                 use_videos=False)


class Clock:
    def __init__(self):
        self.t = time.monotonic_ns()

    def __call__(self):
        self.t += 33_000_000
        return {"obs_ns": self.t, "sent_ns": self.t + 1000}


def frame(i):
    img = np.full((8, 8, 3), i % 251, np.uint8)
    return {"observation.state": np.full(6, i, np.float32), "observation.images.top": img, "task": TASK}


class Script:
    """Drives a SessionLogger the way the station's session runner would."""

    def __init__(self, sess):
        self.sess, self.clock, self.inf = sess, Clock(), 0

    def hold(self, n, moved=False):
        for _ in range(n):
            i = self.sess.frames
            if moved:
                self.sess.add_frame(frame(i), sent_action=np.full(6, i, np.float32), control_mode="expert",
                                    expert_action=np.full(6, i, np.float32), expert_source="human", **self.clock())
            else:
                self.sess.add_frame(frame(i), sent_action=np.zeros(6, np.float32), control_mode="none",
                                    **self.clock())
            self.sess.evidence("reset_check", {"home": True, "zone": True, "expert_active": False, "torque": True,
                                               "cameras": True, "passed": False})

    def policy(self, n):
        for k in range(n):
            i = self.sess.frames
            a = np.arange(6, dtype=np.float32) + i
            self.sess.add_frame(frame(i), sent_action=a, base_action=a, control_mode="base", inference_id=self.inf,
                                chunk_offset=k % 50, control_gen=1, **self.clock())
            self.sess.evidence("criteria", {"grip": False, "lift": False, "all": False})
        self.inf += 1

    def expert(self, n):
        for _ in range(n):
            i = self.sess.frames
            a = np.arange(6, dtype=np.float32) + i
            self.sess.add_frame(frame(i), sent_action=a + 1, base_action=a, expert_action=a + 1,
                                control_mode="expert", expert_source="human", inference_id=self.inf,
                                chunk_offset=0, control_gen=2, **self.clock())
            self.sess.evidence("criteria", {"grip": True, "lift": True, "all": True})


def open_session(log, sid=None):
    return SessionLogger(log, session_id=sid or str(uuid.uuid4()), station_id="st_1", go=GO, model_id="m_1",
                         task=TASK, task_id="tube-pick", task_rev=3, policy=POLICY)


def run_full_session(tmp_path):
    """Three saved episodes (a success, a rescue, a success) and a short stop that ends the session."""
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with open_session(log) as sess:
            s = Script(sess)
            s.hold(40, moved=True)
            sess.state("reset_wait", "policy", "reset_pass")
            sess.start_episode(moved_by="person", reset_checks={"home": True, "zone": True, "expert_active": False,
                                                                 "torque": True, "cameras": True})
            s.policy(60)
            sess.state("policy", "ending", "success")
            assert sess.seal_episode("success", task_success=True) == "pending"
            assert sess.pending_writes == 1
            s.hold(5)                                          # frames while the run is written are gap frames
            assert sess.write_pending() == "committed"
            sess.state("ending", "reset_wait", "run_written")
            s.hold(30, moved=True)
            sess.start_episode(moved_by="person", reset_checks={"home": True, "zone": True})
            s.policy(30)
            sess.state("policy", "expert", "takeover")
            s.expert(40)
            sess.rescue_success()
            assert sess.seal_episode("failure", rescue_outcome="success") == "pending"
            assert sess.write_pending() == "committed"
            s.hold(40)
            sess.start_episode(moved_by="reset_skill")
            s.policy(50)
            assert sess.seal_episode("success") == "pending"
            assert sess.write_pending() == "committed"
            s.hold(40)
            sess.start_episode(moved_by="person")
            s.policy(20)                                       # 20 frames at 30 fps: a short stop
            sess.state("policy", "stopped", "stop")
            assert sess.seal_episode("operator_abort") == "short_stop"
            s.hold(10)
            rec = sess.finalize("user_stop")
    ds.finalize()
    return ds, sess, rec


def test_full_session_covers_every_frame_and_validates(tmp_path):
    ds, sess, rec = run_full_session(tmp_path)
    assert rec["ended_by"] == "user_stop" and rec["fault"] is None and rec["planner"] is None
    assert rec["session_frames"] == 40 + 60 + 5 + 30 + 70 + 40 + 50 + 40 + 20 + 10
    assert [r["write_status"] for r in rec["runs"]] == ["committed"] * 3
    assert [r["termination"] for r in rec["runs"]] == ["success", "failure", "success"]
    assert [r["rescue_outcome"] for r in rec["runs"]] == [None, "success", None]
    assert [r["range"] for r in rec["runs"]] == [{"start": 40, "end": 100}, {"start": 135, "end": 205},
                                                 {"start": 245, "end": 295}]
    assert [g["range"] for g in rec["gaps"]] == [{"start": 0, "end": 40}, {"start": 100, "end": 135},
                                                 {"start": 205, "end": 245}, {"start": 295, "end": 335},
                                                 {"start": 335, "end": 365}]
    assert rec["gaps"][-1]["short_stop"] == {"start": 335, "end": 355}
    assert [g["gap_episode_index"] for g in rec["gaps"]] == [0, 1, 2, 3, 4]
    assert rec["gaps"][0]["moved_by"] == "person" and rec["gaps"][0]["reset_checks"]["home"] is True
    assert rec["gaps"][1]["reset_checks"] == {"home": True, "zone": True, "expert_active": None, "torque": None,
                                              "cameras": None}
    assert S.coverage_problems([(r["range"]["start"], r["range"]["end"]) for r in rec["runs"]],
                               [(g["range"]["start"], g["range"]["end"]) for g in rec["gaps"]],
                               rec["session_frames"]) == []
    assert S.read_session(ds.root, sess.session_id) == rec
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert res["sessions"] == 1 and res["runs"] == 3
    assert V.validate(sess.dir / "gaps")["ok"]


def test_session_fields_in_manifests_and_events(tmp_path):
    ds, sess, rec = run_full_session(tmp_path)
    ms = {r["run_id"]: store.read_manifest(store.runs_dir(ds.root) / r["run_id"]) for r in rec["runs"]}
    for i, r in enumerate(rec["runs"], 1):
        m = ms[r["run_id"]]
        assert m["session_id"] == sess.session_id and m["episode_in_session"] == i and m["go_ref"] == GO
        assert (m["session_frame_start"], m["session_frame_end"]) == (r["range"]["start"], r["range"]["end"])
        assert m["session_frame_end"] - m["session_frame_start"] == m["length"]
        kinds = [e["kind"] for e in m["events"]]
        assert kinds[0] == "episode_start" and kinds[-1] == "episode_end"
    rescued = ms[rec["runs"][1]["run_id"]]
    assert rescued["rescue_outcome"] == "success" and "rescue_success" in [e["kind"] for e in rescued["events"]]
    assert ms[rec["runs"][0]["run_id"]]["rescue_outcome"] is None
    # the gap episode before the first run ends with the reset handoff
    gap_ms = sorted((m for _, m in store.iter_manifests(sess.dir / "gaps", "committed")),
                    key=lambda m: m["episode_index"])
    assert len(gap_ms) == 5
    assert [e["kind"] for e in gap_ms[0]["events"]] == ["reset_pass"]
    assert gap_ms[-1]["events"] == [] and gap_ms[-1]["termination_reason"] == "unknown"
    assert gap_ms[-1]["session_gap"]["short_stop"] == {"start": 335, "end": 355}


def test_gap_dataset_has_every_feature_and_sidecar_ticks(tmp_path):
    ds, sess, rec = run_full_session(tmp_path)
    info = json.loads((sess.dir / "gaps" / "meta" / "info.json").read_text())
    main = json.loads((ds.root / "meta" / "info.json").read_text())
    assert info["features"].keys() == main["features"].keys()
    assert info["total_episodes"] == 5 and info["total_frames"] == sum(g["range"]["end"] - g["range"]["start"]
                                                                       for g in rec["gaps"])
    last = [m for _, m in store.iter_manifests(sess.dir / "gaps", "committed") if m["episode_index"] == 4][0]
    t = pq.read_table(store.runs_dir(sess.dir / "gaps") / last["run_id"] / "ticks.parquet")
    modes = t.column("control_mode").to_pylist()
    assert modes == ["base"] * 20 + ["none"] * 10            # the short stop's policy frames, then the held pose
    assert t.column("terminated").to_pylist() == [False] * 30 and t.column("truncated").to_pylist() == [False] * 30
    sent = np.asarray(t.column("sent_action").to_pylist(), np.float32)
    assert sent[0].tolist() == (np.arange(6, dtype=np.float32) + 335).tolist()


def test_evidence_log(tmp_path):
    ds, sess, rec = run_full_session(tmp_path)
    rows = S.read_evidence(ds.root, sess.session_id)
    kinds = {r["kind"] for r in rows}
    assert kinds == {"state", "criteria", "reset_check", "write"}
    writes = [(r["payload"]["run_id"], r["payload"]["write_status"]) for r in rows if r["kind"] == "write"]
    ids = [r["run_id"] for r in rec["runs"]]
    assert writes == [(ids[0], "pending"), (ids[0], "committed"), (ids[1], "pending"), (ids[1], "committed"),
                      (ids[2], "pending"), (ids[2], "committed")]
    first_pending = [r for r in rows if r["kind"] == "write"][0]
    assert first_pending["session_frame"] == 99                # the sealing tick, the run's last frame
    assert all(0 <= r["session_frame"] < rec["session_frames"] for r in rows)
    crit = [r["session_frame"] for r in rows if r["kind"] == "criteria"]
    assert len(crit) == 60 + 70 + 50 + 20


def run_failed_session(tmp_path, monkeypatch):
    """Three episodes (a success, a failed rescue, a success whose write fails)."""
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with open_session(log) as sess:
            s = Script(sess)
            s.hold(35)
            sess.start_episode(moved_by="person")
            s.policy(40)
            sess.seal_episode("success")
            assert sess.write_pending() == "committed"
            s.hold(35)
            sess.start_episode(moved_by="person")
            s.policy(20)
            s.expert(20)
            sess.seal_episode("failure", rescue_outcome="failure")
            assert sess.write_pending() == "committed"
            s.hold(35)
            sess.start_episode(moved_by="person")
            s.policy(45)
            sess.seal_episode("success")

            def boom():
                raise OSError("disk full")
            monkeypatch.setattr(ds, "save_episode", boom)
            s.hold(4)
            assert sess.write_pending() == "failed"
            failed_id = sess.runs[-1]["run_id"]
            with pytest.raises(RunError, match="failed save"):
                sess.start_episode()                          # the RunLogger stopped: no further episode
            s.hold(6)                                         # recording continues into the gap
            rec = sess.finalize("user_stop")
    ds.finalize()
    return ds, sess, rec, failed_id


def test_save_failure_quarantines_the_run_and_ends_fault_stop(tmp_path, monkeypatch):
    ds, sess, rec, failed_id = run_failed_session(tmp_path, monkeypatch)
    assert rec["ended_by"] == "fault_stop" and rec["fault"] == "persistence_error"
    assert [r["write_status"] for r in rec["runs"]] == ["committed", "committed", "failed"]
    assert rec["runs"][2]["termination"] == "success"         # the sealed result is kept
    assert (store.quarantine_dir(ds.root) / failed_id / "manifest.json").exists()
    assert not (store.runs_dir(ds.root) / failed_id).exists()
    rows = S.read_evidence(ds.root, sess.session_id)
    pe = [r for r in rows if r["kind"] == "persistence_error"]
    assert len(pe) == 1 and pe[0]["payload"]["run_id"] == failed_id and "disk full" in pe[0]["payload"]["error"]
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]


def test_fault_in_expert_and_stop_over_one_second(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with open_session(log) as sess:
            s = Script(sess)
            s.hold(31)
            sess.start_episode()
            s.policy(10)
            s.expert(10)
            sess.fault("camera_lost", "top camera sent no frame for 3 ticks")
            with pytest.raises(ValueError, match="rescue_outcome"):
                sess.seal_episode("system_error")
            assert sess.seal_episode("system_error", rescue_outcome="unknown") == "pending"
            sess.write_pending()
            rec = sess.finalize("fault_stop")
        with open_session(log) as sess2:
            s = Script(sess2)
            s.hold(31)
            sess2.start_episode()
            s.policy(31)                                      # just over 1 s: saved, tagged stopped
            assert sess2.seal_episode("operator_abort") == "pending"
            sess2.write_pending()
            rec2 = sess2.finalize("user_stop")
    ds.finalize()
    assert rec["fault"] == "camera_lost" and rec["runs"][0]["rescue_outcome"] == "unknown"
    assert rec2["runs"][0]["stopped"] is True and all(g["short_stop"] is None for g in rec2["gaps"])
    m = store.read_manifest(store.runs_dir(ds.root) / rec2["runs"][0]["run_id"])
    assert m["termination_reason"] == "operator_abort" and "stopped" in m["labels"]
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert res["sessions"] == 2


def test_rules_refused(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:                        # planner unknown: refused for Drive
        with pytest.raises(ValueError, match="planner=None"):
            open_session(log)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with pytest.raises(ValueError, match="uuid4"):
            open_session(log, sid="not-a-uuid")
        with open_session(log) as sess:
            s = Script(sess)
            s.hold(3)
            with pytest.raises(SessionError, match="no episode"):
                sess.seal_episode("success")
            sess.start_episode()
            with pytest.raises(SessionError, match="already open"):
                sess.start_episode()
            s.policy(40)
            with pytest.raises(ValueError, match="no takeover"):
                sess.seal_episode("failure", rescue_outcome="failure")
            with pytest.raises(SessionError, match="seal the open episode"):
                sess.finalize("user_stop")
            sess.seal_episode("timeout")
            with pytest.raises(SessionError, match="pending"):
                sess.finalize("user_stop")
            with pytest.raises(SessionError, match="write_pending"):
                sess.start_episode()
            sess.write_pending()
            s.hold(2)
            sess.start_episode()
            s.expert(5)
            with pytest.raises(ValueError, match="takeover is sealed only"):
                sess.seal_episode("success")
            assert sess.seal_episode("failure", rescue_outcome="success") == "pending"
            sess.write_pending()
            with pytest.raises(ValueError, match="user's Stop names no fault"):
                sess.finalize("user_stop", fault="camera_lost")
            sid = sess.session_id
            sess.finalize("user_stop")
            with pytest.raises(SessionError, match="finalized"):
                sess.add_frame(frame(0), sent_action=np.zeros(6), control_mode="none", **Clock()())
        with pytest.raises(SessionError, match="never reused"):
            open_session(log, sid=sid)
    ds.finalize()
    assert V.validate(ds.root)["ok"]


def test_buffers_are_copied_for_the_short_stop(tmp_path):
    """A caller that reuses one array for every frame must not change the frames a short stop moves to the gap."""
    ds = make_ds(tmp_path)
    buf = np.zeros(6, np.float32)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with open_session(log) as sess:
            c = Clock()
            sess.add_frame(frame(0), sent_action=np.zeros(6), control_mode="none", **c())
            sess.start_episode()
            for i in range(5):
                buf[:] = i + 100
                sess.add_frame(frame(i), sent_action=buf, base_action=buf, control_mode="base", inference_id=0,
                               chunk_offset=i, **c())
            buf[:] = -1
            assert sess.seal_episode("operator_abort") == "short_stop"
            rec = sess.finalize("user_stop")
    ds.finalize()
    gm = [m for _, m in store.iter_manifests(sess.dir / "gaps", "committed") if m["episode_index"] == 1][0]
    t = pq.read_table(store.runs_dir(sess.dir / "gaps") / gm["run_id"] / "ticks.parquet")
    assert [row[0] for row in t.column("sent_action").to_pylist()] == [100, 101, 102, 103, 104]
    assert rec["gaps"][1]["short_stop"] == {"start": 1, "end": 6}
    assert V.validate(ds.root)["ok"]


def test_frames_added_around_the_session_are_refused(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with pytest.raises(SessionError, match="directly"):      # close() cannot seal it either
            with open_session(log) as sess:
                c = Clock()
                sess.add_frame(frame(0), sent_action=np.zeros(6), control_mode="none", **c())
                run = sess.start_episode()
                run.add_frame(frame(1), sent_action=np.zeros(6), control_mode="none", **c())
                with pytest.raises(SessionError, match="directly"):
                    sess.seal_episode("success")


def test_validate_finds_broken_session_records(tmp_path):
    ds, sess, rec = run_full_session(tmp_path)
    path = sess.dir / "session.json"
    good = json.loads(path.read_text())

    def check(mut, want):
        bad = json.loads(json.dumps(good))
        mut(bad)
        path.write_text(json.dumps(bad))
        res = V.validate(ds.root)
        assert not res["ok"] and any(want in e for e in res["errors"]), res["errors"]
        path.write_text(json.dumps(good))

    check(lambda r: r["gaps"].pop(1), "in no run or gap")
    check(lambda r: r["runs"][0].update(range={"start": 40, "end": 101}), "session_frame_end")
    check(lambda r: r["runs"][1].update(rescue_outcome="failure"), "rescue_outcome")
    check(lambda r: r["runs"][0].update(write_status="pending"), "pending")
    check(lambda r: r["runs"].pop(2), "does not list it")
    check(lambda r: r["gaps"][0].update(gap_episode_index=9), "no committed episode 9")
    check(lambda r: r["gaps"][3].update(short_stop={"start": 300, "end": 310}), "does not match the session record")
    check(lambda r: r["gaps"][0].update(moved_by="task"), "does not match the session record")
    check(lambda r: r["gaps"][0]["reset_checks"].update(home=False), "does not match the session record")
    check(lambda r: r["gaps"][0]["reset_checks"].pop("home"), "reset_checks must have exactly")
    check(lambda r: r.update(started_at="2026-11-07 09:10"), "started_at must be a UTC time")
    check(lambda r: r["go"].update(scope="pi0.5 + rules"), "go.scope must have exactly")
    check(lambda r: r["go"].update(at="2026-11-07T09:10:00+00:00"), "go.at must be a UTC time")
    check(lambda r: r["runs"][0].update(extra=1), "a run entry must have exactly")
    check(lambda r: r["runs"][0].update(stopped=True), "stopped is true exactly")
    check(lambda r: r.update(fault="camera_lost"), "names no fault")
    check(lambda r: r["gaps"][0].update(gap_episode_index=[]), "gap_episode_index must be an int")
    check(lambda r: r["gaps"][0].update(gap_episode_index={"a": 1}), "gap_episode_index must be an int")
    check(lambda r: r["runs"][0].update(run_id=["x"]), "run_id")
    assert V.validate(ds.root)["ok"]


def _split_review(sid, *edits):
    """A session review with each (kind, payload) as a person's edit, in effect at once."""
    from dataop_logger import review_format as fmt
    r = fmt.empty_review(sid)
    for i, (kind, payload) in enumerate(edits):
        r, _ = fmt.propose(r, kind=kind, payload=payload, source="human", by="p1", expect_rev=r["rev"],
                           now=f"2026-11-07T10:00:{i:02d}Z")
    return r


def test_the_session_review_takes_split_edits_that_fit_the_session(tmp_path):
    """Aftershock issue #447: review.json holds only episode_boundary, episode_merge and episode_split, inside the
    session's frames, and the edits in effect apply to the recorded timeline as the cloud applies them."""
    from dataop_logger import review_format as fmt
    ds, sess, rec = run_full_session(tmp_path)
    path, sid = sess.dir / "review.json", sess.session_id

    def check(review, want=None):
        path.write_text(fmt.dump_review(review))
        res = V.validate(ds.root)
        if want is None:
            assert res["ok"], res["errors"]
        else:
            assert not res["ok"] and any("review.json" in e and want in e for e in res["errors"]), res["errors"]

    # runs [40,100), [135,205), [245,295); gaps [0,40), [100,135), [205,245), [295,335), [335,365) (a short stop)
    good = _split_review(sid, ("episode_split", {"at": 70}), ("episode_boundary", {"at": 100, "to": 104}),
                         ("episode_merge", {"start": 70, "end": 205}), ("episode_boundary", {"at": 295, "to": 295}))
    check(good)
    tl = S.effective_timeline(rec, good)
    eps = [(x["start"], x["end"], x["from_runs"], x["confirmed"]) for x in tl if x["kind"] == "episode"]
    r1, r2, r3 = (r["run_id"] for r in rec["runs"])
    assert eps == [(40, 70, [r1], False), (70, 205, [r1, r2], False), (245, 295, [r3], True)]
    assert [x["id"] for x in tl if x["kind"] == "episode"][2] == r3          # untouched: still run 3
    assert [x["id"] for x in tl if x["kind"] == "gap"] == [g["gap_id"] for i, g in enumerate(rec["gaps"]) if i != 1]
    # a boundary moved away and back gives the run's id again, still waiting for review of its outcome
    back = S.effective_timeline(rec, _split_review(sid, ("episode_boundary", {"at": 100, "to": 90}),
                                                   ("episode_boundary", {"at": 90, "to": 100})))
    assert [(x["id"], x["confirmed"]) for x in back if x["kind"] == "episode"][0] == (r1, False)

    check(_split_review(sid, ("note", {"frame": 3, "text": "x"})), "takes only split edits")
    check(_split_review(sid, ("episode_split", {"at": 366})), "past the session")
    check(_split_review(sid, ("episode_boundary", {"at": 335, "to": 340})), "between two gaps")
    check(_split_review(sid, ("episode_boundary", {"at": 41, "to": 44})), "no boundary at session frame 41")
    check(_split_review(sid, ("episode_boundary", {"at": 40, "to": 100})), "at least one frame")
    check(_split_review(sid, ("episode_merge", {"start": 0, "end": 100})), "no two episodes")
    check(_split_review(sid, ("episode_merge", {"start": 40, "end": 100})), "no two episodes")
    check(_split_review(sid, ("episode_split", {"at": 20})), "no episode holds session frame 20")
    check(_split_review(sid, ("episode_split", {"at": 40})), "no episode holds session frame 40")
    check(fmt.empty_review(str(uuid.uuid4())), "another session")
    # a proposal does not apply until a person accepts it, but its frames must still be inside the session
    r, _ = fmt.propose(fmt.empty_review(sid), kind="episode_merge", payload={"start": 0, "end": 100},
                       source="splitter", by="splitter", job_id="split-1", agent_version="s1", reason="evidence",
                       expect_rev=0, now="2026-11-07T10:00:00Z")
    check(r)
    check(good)


def test_no_split_edit_changes_a_failed_run(tmp_path, monkeypatch):
    ds, sess, rec, failed_id = run_failed_session(tmp_path, monkeypatch)
    path, sid = sess.dir / "review.json", sess.session_id
    from dataop_logger import review_format as fmt
    # runs [35,75), [110,150), [185,230) whose write failed; gaps [0,35), [75,110), [150,185), [230,240)
    for edits in ([("episode_split", {"at": 200})], [("episode_boundary", {"at": 185, "to": 180})],
                  [("episode_merge", {"start": 110, "end": 230})], [("episode_boundary", {"at": 230, "to": 231})]):
        path.write_text(fmt.dump_review(_split_review(sid, *edits)))
        res = V.validate(ds.root)
        assert not res["ok"] and any(f"run {failed_id} failed to write" in e for e in res["errors"]), \
            (edits, res["errors"])
    path.write_text(fmt.dump_review(_split_review(sid, ("episode_merge", {"start": 35, "end": 150}),
                                                  ("episode_boundary", {"at": 185, "to": 185}))))
    assert V.validate(ds.root)["ok"]


def test_an_open_session_review_is_checked_for_its_id_kinds_and_frames(tmp_path):
    from dataop_logger import review_format as fmt
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        sess = open_session(log)
        s = Script(sess)
        s.hold(10)
        sess.start_episode(moved_by="person")
        s.policy(30)
        sess.seal_episode("success")
        assert sess.write_pending() == "committed"
        s.hold(5)
        path = sess.dir / "review.json"
        # the dataset is not finalized while the session records, so the session folder is checked on its own;
        # the record on disk counts the 40 frames up to its last boundary (the write)
        path.write_text(fmt.dump_review(_split_review(sess.session_id, ("episode_split", {"at": 20}))))
        errors, warns, _ = S.session_problems(ds.root)
        assert not [e for e in errors if "review.json" in e], errors
        assert any("is open" in w for w in warns)
        path.write_text(fmt.dump_review(_split_review(sess.session_id, ("label", {
            "start": 1, "end": 2, "text": "x", "driver": None, "segment": None}))))
        assert any("takes only split edits" in e for e in S.session_problems(ds.root)[0])
        path.write_text(fmt.dump_review(_split_review(sess.session_id, ("episode_split", {"at": 41}))))
        assert any("past the session's 40 frames" in e for e in S.session_problems(ds.root)[0])
        path.write_text(fmt.dump_review(_split_review(sess.session_id)))
        sess.finalize("user_stop")


def test_failed_run_needs_its_quarantined_record(tmp_path, monkeypatch):
    ds, sess, rec, failed_id = run_failed_session(tmp_path, monkeypatch)
    q = store.quarantine_dir(ds.root) / failed_id
    (q / "ticks.parquet").unlink()
    res = V.validate(ds.root)
    assert not res["ok"] and any("ticks.parquet unreadable" in e for e in res["errors"]), res["errors"]


def test_failed_run_ticks_must_cover_its_range(tmp_path, monkeypatch):
    ds, sess, rec, failed_id = run_failed_session(tmp_path, monkeypatch)
    q = store.quarantine_dir(ds.root) / failed_id
    t = pq.read_table(q / "ticks.parquet").slice(0, 10)
    data = store.table_bytes(t)
    store.atomic_write_bytes(q / "ticks.parquet", data)
    m = store.read_manifest(q)
    m["integrity"]["ticks_sha256"] = store.sha256(data)
    store.atomic_write_json(q / "manifest.json", m)
    res = V.validate(ds.root)
    assert not res["ok"] and any("are not frames 0 to 44" in e for e in res["errors"]), res["errors"]


def test_close_after_a_terminal_tick_keeps_the_episode(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with pytest.raises(KeyError):
            with open_session(log) as sess:
                c = Clock()
                Script(sess).hold(3)
                sess.start_episode()
                a = np.zeros(6, np.float32)
                sess.add_frame(frame(0), sent_action=a, base_action=a, control_mode="base", inference_id=0,
                               chunk_offset=0, terminated=True, **c())
                raise KeyError("station crashed")
    ds.finalize()
    rec = S.read_session(ds.root, sess.session_id)
    assert [(r["termination"], r["write_status"]) for r in rec["runs"]] == [("system_error", "committed")]
    assert V.validate(ds.root)["ok"]


def test_close_keeps_an_open_episode_and_the_active_gap(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with pytest.raises(ZeroDivisionError):
            with open_session(log) as sess:
                s = Script(sess)
                s.hold(31)
                sess.start_episode()
                s.policy(10)
                s.expert(5)
                1 / 0
        with open_session(log) as sess2:
            s = Script(sess2)
            s.hold(31)
            sess2.start_episode()
            s.policy(31)
            sess2.seal_episode("operator_abort")
            sess2.write_pending()
            s.hold(12)                                        # the active gap when the logger closes
    ds.finalize()
    rec = S.read_session(ds.root, sess.session_id)
    assert rec["ended_at"] is None and [r["termination"] for r in rec["runs"]] == ["system_error"]
    assert rec["runs"][0]["rescue_outcome"] == "unknown" and rec["runs"][0]["write_status"] == "committed"
    assert [g["range"] for g in rec["gaps"]] == [{"start": 0, "end": 31}]
    rows = S.read_evidence(ds.root, sess.session_id)
    assert [r["payload"]["fault"] for r in rows if r["kind"] == "fault"] == ["session_closed"]
    rec2 = S.read_session(ds.root, sess2.session_id)
    assert rec2["gaps"][-1]["range"] == {"start": 62, "end": 74}
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert sum("is open" in w for w in res["warnings"]) == 2


def test_short_stop_moves_ticks_as_recorded(tmp_path):
    """A reward added later, the run's events, and a stopping tick marked truncated move into the gap as recorded;
    terminal flags are cleared so the gap goes on."""
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with open_session(log) as sess:
            c = Clock()
            sess.add_frame(frame(0), sent_action=np.zeros(6), control_mode="none", **c())
            run = sess.start_episode()
            for i in range(4):
                a = np.full(6, i + 50, np.float32)
                sess.add_frame(frame(i), sent_action=a, base_action=a, control_mode="base", inference_id=0,
                               chunk_offset=i, prompt="pick", prompt_rev=0, control_gen=3, **c())
                if i == 1:
                    run.complete_transition(0.5)
                    run.event("drop", source="station", reason="slipped")
            a = np.full(6, 60, np.float32)
            sess.add_frame(frame(4), sent_action=a, base_action=a, control_mode="base", inference_id=0,
                           chunk_offset=4, truncated=True, **c())
            assert sess.seal_episode("operator_abort") == "short_stop"
            sess.add_frame(frame(5), sent_action=np.zeros(6), control_mode="none", **c())
            rec = sess.finalize("user_stop")
    ds.finalize()
    gm = [m for _, m in store.iter_manifests(sess.dir / "gaps", "committed") if m["episode_index"] == 1][0]
    t = pq.read_table(store.runs_dir(sess.dir / "gaps") / gm["run_id"] / "ticks.parquet")
    assert t.column("reward_valid").to_pylist() == [False, True, False, False, False, False]
    assert t.column("reward").to_pylist()[1] == 0.5
    assert t.column("truncated").to_pylist() == [False] * 6 and t.column("terminated").to_pylist() == [False] * 6
    assert t.column("prompt").to_pylist()[:4] == ["pick"] * 4 and t.column("control_gen").to_pylist()[:4] == [3] * 4
    assert [(e["kind"], e["frame_index"]) for e in gm["events"]] == [("drop", 1)]
    assert rec["gaps"][1]["short_stop"] == {"start": 1, "end": 6}
    assert V.validate(ds.root)["ok"]


def test_a_failure_after_the_commit_keeps_the_run_committed(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with open_session(log) as sess:
            s = Script(sess)
            s.hold(31)
            sess.start_episode()
            s.policy(40)
            sess.seal_episode("success")
            real = store.rebuild_index

            def boom(root):
                raise OSError("index disk full")
            monkeypatch.setattr(store, "rebuild_index", boom)
            with pytest.warns(UserWarning, match="was committed"):
                assert sess.write_pending() == "committed"
            monkeypatch.setattr(store, "rebuild_index", real)
            rec = sess.finalize("user_stop")
    ds.finalize()
    assert rec["ended_by"] == "user_stop" and rec["runs"][0]["write_status"] == "committed"
    store.rebuild_index(ds.root)
    assert V.validate(ds.root)["ok"]


def test_a_gap_dataset_that_cannot_finalize_fails_the_session(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with open_session(log) as sess:
            Script(sess).hold(5)
            real = sess.gap_dataset.finalize
            monkeypatch.setattr(sess.gap_dataset, "finalize", lambda: (_ for _ in ()).throw(OSError("no space")))
            with pytest.raises(SessionError, match="could not be finalized"):
                sess.finalize("user_stop")
            rec = S.read_session(ds.root, sess.session_id)
            assert rec["ended_at"] is None
            assert [r["kind"] for r in S.read_evidence(ds.root, sess.session_id)][-1] == "persistence_error"
            with pytest.raises(SessionError, match="call finalize"):
                sess.add_frame(frame(9), sent_action=np.zeros(6), control_mode="none", **Clock()())
            monkeypatch.setattr(sess.gap_dataset, "finalize", real)
            rec = sess.finalize("user_stop")                   # the retry finishes it
    ds.finalize()
    assert rec["ended_by"] == "user_stop" and S.read_session(ds.root, sess.session_id)["ended_at"] is not None
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]


def test_datasets_without_sessions_still_validate(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, policy=POLICY) as run:
            run.add_frame(frame(0), sent_action=np.zeros(6), control_mode="none", **Clock()())
            run.end("success")
    ds.finalize()
    m = store.read_manifest(run.run_path)
    assert "session_id" not in m and "rescue_outcome" not in m    # absent: null for session fields
    res = V.validate(ds.root)
    assert res["ok"] and res["sessions"] == 0


def test_cloud_models_load_the_records(tmp_path):
    """Round trip: the cloud models in contracts/schema/drive.py load session.json, the evidence rows and the run
    manifests (Aftershock issue #371, acceptance 2)."""
    root = os.environ.get("AFTERSHOCK_CONTRACTS")
    if not root or not (Path(root) / "contracts" / "schema" / "drive.py").exists():
        pytest.skip("set AFTERSHOCK_CONTRACTS to a checkout with contracts/schema/drive.py")
    sys.path.insert(0, root)
    from contracts.schema import drive as D
    from contracts.schema.manifest import RunManifest
    ds, sess, rec = run_full_session(tmp_path)
    loaded = D.SessionRecord.model_validate(json.loads((sess.dir / "session.json").read_text()))
    assert loaded.session_frames == rec["session_frames"] and len(loaded.gaps) == 5
    for row in S.read_evidence(ds.root, sess.session_id):
        D.EvidenceRow.model_validate(row)
    for r in rec["runs"]:
        RunManifest.model_validate(store.read_manifest(store.runs_dir(ds.root) / r["run_id"]))
