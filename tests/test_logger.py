"""Integration tests against a real LeRobotDataset (no cameras, so they run in seconds)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataop_logger import ActionSchema, EnvInfo, PolicyInfo, RunError, RunLogger  # noqa: E402
from dataop_logger import store  # noqa: E402
from dataop_logger import validate as V  # noqa: E402
from dataop_logger.reader import DataopDataset  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCHEMA = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
TASK = "pick up the tube"


def features(dim=6):
    names = list(NAMES[:dim]) if dim <= 6 else [f"a{i}" for i in range(dim)]
    return {"observation.state": {"dtype": "float32", "shape": (dim,), "names": names},
            "action": {"dtype": "float32", "shape": (dim,), "names": names}}


def make_ds(tmp_path, name="ds", dim=6):
    return lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features=features(dim),
                                                 root=tmp_path / name, use_videos=False)


def frame(i, dim=6):
    return {"observation.state": np.full(dim, i, np.float32), "task": TASK}


def ts():
    t = time.monotonic_ns()
    return {"obs_ns": t, "sent_ns": t + 1000}


def record(log, n=10, takeover_at=None, reason="success", offset=0.0, **kw):
    with log.run(task=TASK, policy=PolicyInfo("smolvla", "train/smolvla_100"),
                 env=EnvInfo(robot_id="so101", extras={"light": "night"})) as run:
        for i in range(n):
            base = np.arange(6, dtype=np.float32) + i + offset
            if takeover_at is not None and i >= takeover_at:
                exp = base + 1
                run.add_frame(frame(i), sent_action=exp, base_action=base, expert_action=exp,
                              control_mode="expert", expert_source="skill:rescue", inference_id=i, chunk_offset=0, **ts())
            else:
                run.add_frame(frame(i), sent_action=base, base_action=base, control_mode="base",
                              inference_id=i, chunk_offset=0, **ts())
        run.end(termination_reason=reason, **kw)
    return run


def none_frame(run, i, **kw):
    return run.add_frame(frame(i), sent_action=np.zeros(6), control_mode="none", **ts(), **kw)


def test_round_trip_and_validate(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        r0 = record(log, 10)
        r1 = record(log, 8, takeover_at=5, reason="failure", labels=["failure:dropped"])
    ds.finalize()
    assert r0.state == r1.state == "committed"
    m1 = store.read_manifest(r1.run_path)
    assert m1["episode_index"] == 1 and m1["length"] == 8
    assert m1["takeovers"] == [{"start_frame": 5, "end_frame": 8, "control_mode": "expert",
                                "expert_source": "skill:rescue"}]
    assert m1["action_schema"]["bounds"] is None
    t = pq.read_table(r1.run_path / "ticks.parquet")
    assert t.column("frame_index").to_pylist() == list(range(8))
    assert t.column("terminated").to_pylist() == [False] * 7 + [True]
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    idx = [json.loads(line) for line in (ds.root / "dataop/index.jsonl").read_text().splitlines()]
    assert [i["episode_index"] for i in idx] == [0, 1]


@pytest.mark.parametrize("reason,flag", [("success", "terminated"), ("failure", "terminated"),
                                         ("timeout", "truncated"), ("operator_abort", "truncated"),
                                         ("system_error", "truncated")])
def test_end_sets_terminal_flag_from_reason(tmp_path, reason, flag):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        r = record(log, 3, reason=reason)
    ds.finalize()
    t = pq.read_table(r.run_path / "ticks.parquet")
    assert t.column(flag).to_pylist() == [False, False, True]
    assert V.validate(ds.root)["ok"]


def test_contradicting_flag_rejected(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            none_frame(run, 0, truncated=True)
            with pytest.raises(RunError, match="contradicts"):
                run.end("success")
            run.discard()
        with pytest.raises(ValueError, match="both"):
            with log.run(task=TASK) as run:
                none_frame(run, 0, truncated=True, terminated=True)


def test_exception_discards_episode(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ZeroDivisionError):
            with log.run(task=TASK) as run:
                none_frame(run, 0)
                1 / 0
        assert not ds.has_pending_frames() and run.state == "discarded"
        r = record(log, 3)
    ds.finalize()
    assert store.read_manifest(r.run_path)["episode_index"] == 0
    assert V.validate(ds.root)["ok"]


def test_missing_end_is_error_not_success(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(RunError, match="without run.end"):
            with log.run(task=TASK) as run:
                none_frame(run, 0)
        assert not ds.has_pending_frames() and ds.meta.total_episodes == 0


def test_shape_and_nan_rejected_before_lerobot(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            with pytest.raises(ValueError, match="6"):
                run.add_frame(frame(0), sent_action=np.zeros(5), control_mode="none", **ts())
            with pytest.raises(ValueError, match="NaN"):
                run.add_frame(frame(0), sent_action=np.full(6, np.nan), control_mode="none", **ts())
            assert not ds.has_pending_frames()
            run.discard()


def test_task_mismatch_rejected(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            with pytest.raises(ValueError, match="differs from the run task"):
                run.add_frame({**frame(0), "task": "other"}, sent_action=np.zeros(6), control_mode="none", **ts())
            run.discard()


def test_direct_add_frame_detected(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(RunError, match="only through run.add_frame"):
            with log.run(task=TASK) as run:
                none_frame(run, 0)
                ds.add_frame({**frame(1), "action": np.zeros(6, np.float32)})
                none_frame(run, 2)


def test_mutation_after_end_blocks_save(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(RunError, match="changed after run.end"):
            with log.run(task=TASK) as run:
                none_frame(run, 0)
                run.end("success")
                ds.add_frame({**frame(1), "action": np.zeros(6, np.float32)})
        assert run.state == "quarantined" and ds.meta.total_episodes == 0 and not ds.has_pending_frames()
        with pytest.raises(RunError, match="changed after run.end"):
            with log.run(task=TASK) as run:
                none_frame(run, 0)
                run.end("success")
                ds.writer.episode_buffer["action"][0] = np.ones(6, np.float32)
        assert ds.meta.total_episodes == 0


def test_save_failure_keeps_run_for_repair(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    log = RunLogger(ds, SCHEMA)
    real_save = ds.save_episode

    def save_then_crash(*a, **k):                              # the episode reaches disk, then save "fails"
        real_save(*a, **k)
        raise OSError("disk hiccup after write")
    monkeypatch.setattr(ds, "save_episode", save_then_crash)
    with pytest.raises(OSError):
        record(log, 4)
    assert log.broken
    with pytest.raises(RunError, match="failed save"):
        record(log, 4)
    log.close()
    ds.finalize()
    res = V.validate(ds.root)
    assert not res["ok"] and any("save_error" in e for e in res["errors"])
    report = V.repair(ds.root)
    assert any(line.startswith("committed") for line in report)
    assert V.validate(ds.root)["ok"]


def test_rl_fields_and_delayed_reward(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            for i in range(4):
                if i:
                    run.complete_transition(reward=0.0)
                none_frame(run, i)
            run.complete_transition(reward=1.0, terminated=True)
            with pytest.raises(RunError):
                none_frame(run, 9)
            run.end("success", task_success=True)
    ds.finalize()
    t = pq.read_table(run.run_path / "ticks.parquet")
    assert t.column("reward").to_pylist() == [0.0, 0.0, 0.0, 1.0]
    assert store.read_manifest(run.run_path)["rl_ready"] is True
    assert V.validate(ds.root)["ok"]


def test_prediction_rules(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, log_predictions=True) as run:
            t0 = time.monotonic_ns()
            run.predictions(0, np.zeros((5, 6)), obs_frame_index=-1, infer_start_ns=t0, infer_end_ns=t0 + 10,
                            valid_len=2)
            for i in range(2):
                run.add_frame(frame(i), sent_action=np.zeros(6), base_action=np.zeros(6), control_mode="base",
                              inference_id=0, chunk_offset=i, **ts())
            with pytest.raises(ValueError, match="inference_id"):
                run.add_frame(frame(2), sent_action=np.zeros(6), base_action=np.zeros(6), control_mode="base", **ts())
            run.end("success")
        with log.run(task=TASK, log_predictions=True) as run2:                 # offset past valid_len
            t0 = time.monotonic_ns()
            run2.predictions(0, np.zeros((5, 6)), obs_frame_index=-1, infer_start_ns=t0, infer_end_ns=t0, valid_len=1)
            run2.add_frame(frame(0), sent_action=np.zeros(6), base_action=np.zeros(6), control_mode="base",
                           inference_id=0, chunk_offset=3, **ts())
            with pytest.raises(RunError, match="valid_len"):
                run2.end("success")
            run2.discard()
        with log.run(task=TASK, log_predictions=True) as run3:                 # enabled but empty
            none_frame(run3, 0)
            run3.end("success")
    ds.finalize()
    assert (run3.run_path / "predictions.parquet").exists()
    assert store.read_manifest(run3.run_path)["predictions_count"] == 0
    assert V.validate(ds.root)["ok"], V.validate(ds.root)["errors"]


def test_single_writer_lock(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA):
        with pytest.raises(RuntimeError, match="another dataop-logger writer"):
            RunLogger(ds, SCHEMA)


def test_repair_commits_staged_and_quarantines_orphans(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        r = record(log, 5)
    ds.finalize()
    m = store.read_manifest(r.run_path)
    m["status"] = "staged"                                   # simulate a crash after save, before commit
    store.atomic_write_json(r.run_path / "manifest.json", m)
    orphan = store.runs_dir(ds.root) / "orphan"
    import shutil
    shutil.copytree(r.run_path, orphan)
    store.atomic_write_json(orphan / "manifest.json", {**m, "run_id": "orphan", "length": 99})
    broken = store.runs_dir(ds.root) / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json")
    assert not V.validate(ds.root)["ok"]
    report = V.repair(ds.root)
    assert any("committed" in line for line in report)
    assert any("quarantined orphan" in line for line in report)
    assert any("left untouched broken" in line for line in report)
    shutil_rm(broken)
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert res["quarantined"] == ["orphan"]


def shutil_rm(p):
    import shutil
    shutil.rmtree(p)


def _rewrite_ticks(run_path, t, fix_digest):
    data = store.table_bytes(pa.table(t))
    store.atomic_write_bytes(run_path / "ticks.parquet", data)
    if fix_digest:                                         # simulate a writer bug: digest matches bad content
        m = store.read_manifest(run_path)
        m["integrity"]["ticks_sha256"] = store.sha256(data)
        store.atomic_write_json(run_path / "manifest.json", m)


def test_validate_reports_corruption_without_crashing(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        r = record(log, 4, takeover_at=2)
        r2 = record(log, 3)
    ds.finalize()
    orig = pq.read_table(r.run_path / "ticks.parquet").to_pydict()
    t = {k: list(v) for k, v in orig.items()}
    t["expert_action"][3] = [999.0, -999.0, 1.0, 1.0, 1.0, 1.0]   # finite, plausible: only the digest sees it
    _rewrite_ticks(r.run_path, t, fix_digest=False)
    (r2.run_path / "ticks.parquet").write_bytes(b"garbage")
    (store.runs_dir(ds.root) / "half-staged").mkdir()
    errs = "\n".join(V.validate(ds.root)["errors"])
    assert f"{r.run_id}: ticks.parquet does not match its recorded digest" in errs
    assert f"{r2.run_id}: ticks.parquet does not match its recorded digest" in errs
    assert "no manifest" in errs
    t = {k: list(v) for k, v in orig.items()}                       # semantic checks behind a matching digest
    t["control_mode"][1] = "expert"
    t["expert_source"][1] = "skill:rescue"
    t["expert_valid"][1] = True
    t["expert_action"][1] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    _rewrite_ticks(r.run_path, t, fix_digest=True)
    assert any("takeovers summary" in e for e in V.validate(ds.root)["errors"])
    t = {k: list(v) for k, v in orig.items()}
    t["control_mode"][0] = "blend"
    t["expert_source"][0] = "x"
    t["blend_weight"][0] = 0.5
    _rewrite_ticks(r.run_path, t, fix_digest=True)
    assert any("blend without both" in e for e in V.validate(ds.root)["errors"])


def test_events_tamper_detected(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            none_frame(run, 0)
            run.event("takeover", source="human", reason="drop")
            run.end("failure")
    ds.finalize()
    m = store.read_manifest(run.run_path)
    m["events"][0]["reason"] = "stall"
    store.atomic_write_json(run.run_path / "manifest.json", m)
    assert any("events do not match" in e for e in V.validate(ds.root)["errors"])


def test_incomplete_save_is_not_repaired(tmp_path, monkeypatch):
    """Data parquet written, LeRobot metadata never registered: repair must not commit it."""
    ds = make_ds(tmp_path)
    log = RunLogger(ds, SCHEMA)

    def boom(*a, **k):
        raise OSError("metadata write failed")
    monkeypatch.setattr(ds.writer._meta, "save_episode", boom)
    with pytest.raises(OSError):
        record(log, 4)
    log.close()
    ds.finalize()
    report = V.repair(ds.root)
    assert not any(line.startswith("committed") for line in report)
    res = V.validate(ds.root)
    assert not res["ok"]
    assert any("incomplete save" in e for e in res["errors"])


def test_reused_action_buffer_is_safe(tmp_path):
    ds = make_ds(tmp_path)
    buf = np.zeros(6, np.float32)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            for i in range(3):
                buf[:] = i                                             # a control loop reusing one array
                run.add_frame(frame(i), sent_action=buf, control_mode="none", **ts())
            run.end("success")
    ds.finalize()
    assert V.validate(ds.root)["ok"]


def test_blend_and_expert_source_rules(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            with pytest.raises(ValueError, match="blend"):
                run.add_frame(frame(0), sent_action=np.zeros(6), control_mode="blend", expert_source="h",
                              blend_weight=0.5, **ts())
            with pytest.raises(ValueError, match="expert_source"):
                run.add_frame(frame(0), sent_action=np.zeros(6), base_action=np.zeros(6), expert_action=np.ones(6),
                              control_mode="base", inference_id=0, chunk_offset=0, **ts())
            run.discard()


def test_one_dof_round_trip_and_repair(tmp_path):
    schema = ActionSchema(names=("gripper",), units=("pct",), space="joint_position", rate_hz=30)
    ds = make_ds(tmp_path, dim=1)
    with RunLogger(ds, schema) as log:
        with log.run(task=TASK) as run:
            for i in range(3):
                run.add_frame(frame(i, dim=1), sent_action=[float(i)], control_mode="none", **ts())
            run.end("success")
    ds.finalize()
    assert V.validate(ds.root)["ok"], V.validate(ds.root)["errors"]
    m = store.read_manifest(run.run_path)
    store.atomic_write_json(run.run_path / "manifest.json", {**m, "status": "staged"})
    assert any(line.startswith("committed") for line in V.repair(ds.root))


def test_reader_filtered_views_and_missing_policy(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        record(log, 4)
        record(log, 6, takeover_at=3, offset=100)
    ds.add_frame({**frame(0), "action": np.zeros(6, np.float32)})   # a legacy episode with no run
    ds.save_episode()
    with RunLogger(ds, SCHEMA) as log:
        record(log, 2, offset=200)
    ds.finalize()
    full = lerobot_dataset.LeRobotDataset("test/ds", root=ds.root)
    with pytest.raises(ValueError, match="no committed dataop run"):
        DataopDataset(full)
    dd = DataopDataset(full, missing="exclude")
    assert len(dd) == 12 and dd.bound_episodes == [0, 1, 3]
    masked = DataopDataset(full, missing="mask")
    assert len(masked) == 13 and not bool(masked[10]["dataop.run_bound"])
    only1 = DataopDataset(lerobot_dataset.LeRobotDataset("test/ds", root=ds.root, episodes=[1]))
    assert len(only1) == 6 and only1.bound_episodes == [1] and only1.all_bound_episodes == [0, 1, 3]
    item = only1[4]
    assert int(item["episode_index"]) == 1 and int(item["frame_index"]) == 4
    assert bool(item["dataop.dagger_mask"]) and np.allclose(np.asarray(item["dataop.expert_residual"]), 1.0)
    assert item["dataop.expert_source"] == "skill:rescue"
    sel = DataopDataset(lerobot_dataset.LeRobotDataset("test/ds", root=ds.root, episodes=[0, 3]))
    assert len(sel) == 6
    assert [int(sel[k]["episode_index"]) for k in range(6)] == [0, 0, 0, 0, 3, 3]
    assert np.allclose(np.asarray(sel[5]["dataop.sent_action"]), np.asarray(sel[5]["action"]))


def test_schema_validation():
    a = ActionSchema(names=("a",), units=("deg",), space="joint_position", rate_hz=30)
    b = ActionSchema(names=("a",), units=("rad",), space="joint_position", rate_hz=30)
    assert a.id != b.id and len(a.id) == 64
    assert ActionSchema.from_dict({**a.to_dict(), "id": a.id}).id == a.id
    bad = [dict(space="weird"), dict(space="custom:"), dict(rate_hz=0), dict(names=("a", "a"), units=("x", "x")),
           dict(low=(0.0,)), dict(low=(1.0,), high=(0.0,))]
    for kw in bad:
        args = dict(names=("a",), units=("deg",), space="joint_position", rate_hz=30)
        args.update(kw)
        with pytest.raises(ValueError):
            ActionSchema(**args)


def test_malformed_buffer_blocks_save(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(RunError, match="changed after run.end"):
            with log.run(task=TASK) as run:
                none_frame(run, 0)
                none_frame(run, 1)
                run.end("success")
                ds.writer.episode_buffer["task"].pop()
        assert ds.meta.total_episodes == 0 and not ds.has_pending_frames()
        with pytest.raises(RunError, match="changed after run.end"):
            with log.run(task=TASK) as run:
                none_frame(run, 0)
                run.end("success")
                ds.writer.episode_buffer["action"][0] = "not an action"
        assert ds.meta.total_episodes == 0


def test_unexpected_episode_count_stops_logger(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    log = RunLogger(ds, SCHEMA)
    monkeypatch.setattr(ds, "save_episode", lambda *a, **k: ds.clear_episode_buffer())   # saves nothing
    with pytest.raises(RunError, match="unexpected episode index"):
        record(log, 2)
    assert log.broken
    with pytest.raises(RunError, match="failed save"):
        record(log, 2)
    log.close()


def test_reader_follows_lerobot_order(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        for k in range(3):
            record(log, 2, offset=10 * k)
    ds.finalize()
    view = lerobot_dataset.LeRobotDataset("test/ds", root=ds.root, episodes=[2, 0, 2])
    dd = DataopDataset(view)
    assert len(dd) == len(view)
    for k in range(len(view)):
        assert (int(dd[k]["episode_index"]), int(dd[k]["frame_index"])) == \
               (int(view[k]["episode_index"]), int(view[k]["frame_index"]))


def test_prediction_future_frame_must_exist(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, log_predictions=True) as run:
            none_frame(run, 0)
            t0 = time.monotonic_ns()
            run.predictions(0, np.zeros((2, 6)), obs_frame_index=1, infer_start_ns=t0, infer_end_ns=t0)
            with pytest.raises(RunError, match="refers to frame 1"):
                run.end("success")
            run.discard()


def test_repair_survives_malformed_integrity(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        r = record(log, 3)
    ds.finalize()
    m = store.read_manifest(r.run_path)
    store.atomic_write_json(r.run_path / "manifest.json", {**m, "status": "staged", "integrity": "bad"})
    report = V.repair(ds.root)
    assert any("malformed" in line for line in report)


def test_missing_video_is_incomplete(tmp_path):
    feats = {**features(), "observation.images.cam": {"dtype": "video", "shape": (32, 32, 3),
                                                      "names": ["height", "width", "channels"]}}
    ds = lerobot_dataset.LeRobotDataset.create("test/vid", fps=30, features=feats, root=tmp_path / "vid")
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            for i in range(3):
                run.add_frame({**frame(i), "observation.images.cam": np.zeros((32, 32, 3), np.uint8)},
                              sent_action=np.zeros(6), control_mode="none", **ts())
            run.end("success")
    ds.finalize()
    assert V.validate(ds.root)["ok"], V.validate(ds.root)["errors"]
    for mp4 in (ds.root / "videos").rglob("*.mp4"):
        mp4.unlink()
    res = V.validate(ds.root)
    assert not res["ok"] and any("incomplete save" in e for e in res["errors"])


def test_viewer_collect_and_build(tmp_path):
    from dataop_logger import viewer
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        record(log, 10)
        with log.run(task=TASK, policy=PolicyInfo("smolvla", "train/smolvla_100")) as run:
            for i in range(8):
                base = np.arange(6, dtype=np.float32) + i
                if i == 5:
                    run.event("takeover", source="skill:rescue", reason="dropped </script>")
                if i >= 5:
                    run.add_frame(frame(i), sent_action=base + 1, expert_action=base + 1, control_mode="expert",
                                  expert_source="skill:rescue", **ts())
                else:
                    run.add_frame(frame(i), sent_action=base, base_action=base, control_mode="base",
                                  inference_id=i, chunk_offset=0, **ts())
            run.end(termination_reason="failure", labels=["rescued"])
    ds.finalize()
    data = viewer.collect(ds.root)
    assert [r["episode"] for r in data["runs"]] == [0, 1]
    r1 = data["runs"][1]
    assert r1["segments"] == [[0, 5, "base", None], [5, 8, "expert", "skill:rescue"]]
    assert r1["events"][0]["frame"] == 4 and r1["videos"] == []
    html = viewer.build(ds.root).read_text()
    assert "/*__DATA__*/null" not in html and "dropped <\\/script>" in html


def test_more_episodes_than_lerobot_metadata_buffer(tmp_path):
    # LeRobot flushes episode metadata every 10 episodes into a parquet writer it keeps open until finalize();
    # the post-save check must not read that file mid-recording.
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        runs = [record(log, 4) for _ in range(12)]
    ds.finalize()
    assert all(r.state == "committed" for r in runs)
    assert V.validate(ds.root)["ok"]


def test_subgoals_and_failure_mode(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            run.subgoal("grasp", "started")
            for i in range(10):
                if i == 4:
                    run.subgoal("grasp", "completed")
                    run.subgoal("lift", "started")
                run.add_frame(frame(i), sent_action=np.zeros(6), base_action=np.zeros(6), control_mode="base",
                              inference_id=i, chunk_offset=0, **ts())
            run.subgoal("lift", "failed", reason="tube slipped")
            with pytest.raises(ValueError):
                run.subgoal("lift", "done")
            run.end(termination_reason="failure", failure_mode="dropped_during_lift")
        record(log, 5)
    ds.finalize()
    m = store.read_manifest(run.run_path)
    assert m["failure_mode"] == "dropped_during_lift"
    assert m["subgoals"] == [
        {"name": "grasp", "start_frame": 0, "end_frame": 4, "status": "completed", "reason": None},
        {"name": "lift", "start_frame": 4, "end_frame": 10, "status": "failed", "reason": "tube slipped"}]
    assert V.validate(ds.root)["ok"]
    info = V.inspect(ds.root)
    assert info["failure_modes"] == {"dropped_during_lift": 1} and info["failed_subgoals"] == {"lift": 1}
