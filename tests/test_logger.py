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


def _failed_run_with_steps(log, n=40, take=30):
    with log.run(task=TASK) as run:
        run.subgoal("grasp", "started")
        for i in range(n):
            if i == 10:
                run.subgoal("grasp", "completed")
                run.subgoal("lift", "started")
            a = np.full(6, i, np.float32)
            if i < take:
                run.add_frame(frame(i), sent_action=a, base_action=a, control_mode="base",
                              inference_id=i, chunk_offset=0, **ts())
            else:
                if i == take:
                    run.subgoal("lift", "failed", reason="slipped")
                run.add_frame(frame(i), sent_action=a, expert_action=a, control_mode="expert",
                              expert_source="skill:rescue", **ts())
        run.end(termination_reason="failure", failure_mode="no_lift")
    return run


def _review_fixture(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        run = _failed_run_with_steps(log)
    ds.finalize()
    return ds, run, store.read_manifest(run.run_path)


def test_review_edits_effective_and_train_mask(tmp_path):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    modes = ["base"] * 30 + ["expert"] * 10
    p = rv.add_edit(run.run_path, m, "failure_start", {"frame": 12}, source="agent", reason="test")
    assert rv.add_edit(run.run_path, m, "failure_start", {"frame": 12}, source="agent")["id"] == p["id"]
    assert rv.effective(m, rv.load(run.run_path))["failure_start"] is None
    rv.decide(run.run_path, m, p["id"], True, {"frame": 20})
    assert rv.effective(m, rv.load(run.run_path))["failure_start"] == 20
    mask = rv.train_mask(m, rv.load(run.run_path), modes)
    assert mask[:20].all() and not mask[20:30].any() and mask[30:].all()
    rv.add_edit(run.run_path, m, "exclude", {"start": 0, "end": 3})
    rv.add_edit(run.run_path, m, "step_boundary", {"index": 0, "frame": 8})
    r = rv.load(run.run_path)
    eff = rv.effective(m, r)
    assert eff["excludes"] == [{"start": 0, "end": 3}]
    assert [(s["name"], s["start_frame"], s["end_frame"]) for s in eff["subgoals"]] == [("grasp", 0, 8), ("lift", 8, 30)]
    assert not rv.train_mask(m, r, modes)[:3].any()
    for bad in [("exclude", {"start": 5, "end": 5}), ("failure_start", {"frame": 40}), ("failure_start", {"frame": True}),
                ("outcome", {"termination_reason": "success", "failure_mode": "x"}),
                ("step_boundary", {"index": 1, "frame": 3}), ("step_boundary", {"index": 0, "frame": 41}),
                ("step_boundary", {"index": 0, "before": "lift", "frame": 5}), ("bogus", {})]:
        with pytest.raises(rv.ReviewError):
            rv.add_edit(run.run_path, m, *bad)
    rv.add_edit(run.run_path, m, "reject_run", {})
    assert not rv.train_mask(m, rv.load(run.run_path), modes).any()
    assert V.validate(ds.root)["ok"]                    # the recorded run is untouched
    item = DataopDataset(lerobot_dataset.LeRobotDataset("test/ds", root=ds.root))[35]
    assert not bool(item["dataop.train_mask"]) and not bool(item["dataop.dagger_mask"])


@pytest.mark.parametrize("modes,fs,masked", [
    (["base", "none", "base", "expert"], 0, [0, 1, 2]),          # a pause does not end the bad stretch
    (["base", "blend", "expert"], 0, [0]),                       # blend is a takeover; keep its labels
    (["base", "base", "base"], 1, [1, 2]),                       # no takeover: to the end
    (["expert", "base", "expert"], 0, []),                       # starting on a takeover frame masks nothing
    (["none", "base", "expert"], 0, [0, 1]),
])
def test_train_mask_failure_start_across_modes(modes, fs, masked):
    from dataop_logger import review as rv
    m = {"length": len(modes), "termination_reason": "failure", "failure_mode": None, "subgoals": []}
    r = rv.empty()
    r["seq"] = 1
    r["edits"].append({"id": "a" * 12, "kind": "failure_start", "payload": {"frame": fs}, "source": "human",
                       "status": "accepted", "accepted_seq": 1})
    mask = rv.train_mask(m, r, modes)
    assert [i for i in range(len(modes)) if not mask[i]] == masked
    m["termination_reason"] = "success"                          # an effective success ignores failure_start
    assert rv.train_mask(m, r, modes).all()


def test_review_undo_chain_and_reviewed_rules(tmp_path):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    a = rv.add_edit(run.run_path, m, "outcome", {"termination_reason": "failure", "failure_mode": "first"})
    b = rv.add_edit(run.run_path, m, "outcome", {"termination_reason": "failure", "failure_mode": "second"})
    assert rv.effective(m, rv.load(run.run_path))["failure_mode"] == "second"
    rv.decide(run.run_path, m, b["id"], False)                    # undo B brings A back
    assert rv.effective(m, rv.load(run.run_path))["failure_mode"] == "first"
    rv.decide(run.run_path, m, a["id"], False)
    assert rv.effective(m, rv.load(run.run_path))["failure_mode"] == "no_lift"
    # an outcome corrected to success retires the failure start
    rv.add_edit(run.run_path, m, "failure_start", {"frame": 5})
    rv.add_edit(run.run_path, m, "outcome", {"termination_reason": "success", "failure_mode": None})
    assert rv.effective(m, rv.load(run.run_path))["failure_start"] is None
    # only proposals can be accepted; accepted edits can only be undone
    p = rv.add_edit(run.run_path, m, "exclude", {"start": 0, "end": 2}, source="agent")
    rv.decide(run.run_path, m, p["id"], True)
    with pytest.raises(rv.ReviewError):
        rv.decide(run.run_path, m, p["id"], True, {"start": 5, "end": 9})
    with pytest.raises(rv.ReviewError):
        rv.decide(run.run_path, m, p["id"], "false")
    with pytest.raises(rv.ReviewConflict):
        rv.add_edit(run.run_path, m, "note", {"frame": 1, "text": "x"}, expect_rev=0)


def _add_notes(run_path, m, k):
    from dataop_logger import review as rv
    for i in range(k):
        rv.add_edit(run_path, m, "note", {"frame": 1, "text": f"{run_path.name}-{i}"})


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_review_concurrent_writers_lose_nothing(tmp_path):
    import multiprocessing as mp
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=_add_notes, args=(run.run_path, m, 15)) for _ in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    r = rv.load(run.run_path)
    assert len(r["edits"]) == 60 and r["rev"] == 60


def test_find_idle_and_suggest_cut(tmp_path):
    from dataop_logger import review as rv
    from dataop_logger.cli import main
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:                 # moves, sits still for 2 s, moves again
            for i in range(120):
                q = np.full(6, 50.0 if 30 <= i < 90 else float(i), np.float32)
                run.add_frame({"observation.state": q, "task": TASK}, sent_action=q, base_action=q,
                              control_mode="base", inference_id=i, chunk_offset=0, **ts())
            run.end(termination_reason="success")
    ds.finalize()
    m = store.read_manifest(run.run_path)
    found = rv.find_idle(ds.root, run.run_path, m)
    sugg = [f["edit"] for f in found]
    assert [(e["payload"]["start"], e["payload"]["end"]) for e in sugg] == [(31, 89)] and found[0]["status"] == "created"
    assert sugg[0]["reason"] == "no movement for 1.9 s mid-run, policy driving"
    assert sugg[0]["by"] == "dataop idle finder (measured state)"
    again = rv.find_idle(ds.root, run.run_path, m)
    assert again[0]["edit"]["id"] == sugg[0]["id"] and again[0]["status"] == "already_suggested"
    assert rv.train_mask(m, rv.load(run.run_path), ["base"] * 120).all()             # a suggestion cuts nothing
    rv.decide(run.run_path, m, sugg[0]["id"], True)
    assert not rv.train_mask(m, rv.load(run.run_path), ["base"] * 120)[31:89].any()
    assert rv.find_idle(ds.root, run.run_path, m)[0]["status"] == "already_cut"
    assert main(["suggest-cut", str(ds.root), run.run_id, "0", "10", "--reason", "warm-up", "--by", "my-agent"]) == 0
    mine = [e for e in rv.load(run.run_path)["edits"] if e.get("by") == "my-agent"]
    assert len(mine) == 1 and mine[0]["status"] == "proposed"
    rv.decide(run.run_path, m, mine[0]["id"], False)                                  # the person keeps it
    assert rv.suggest_cut(run.run_path, m, 0, 10, "again", by="other-agent")["status"] == "previously_kept"
    assert main(["suggest-cut", str(ds.root), run.run_id, "5", "5", "--reason", "x"]) == 2
    rv.add_edit(run.run_path, m, "exclude", {"start": 80, "end": 100})               # overlaps the accepted 31-89
    assert V.inspect(ds.root)["after_review"]["cut_frames"] == 69
    rv.add_edit(run.run_path, m, "exclude", {"start": 0, "end": 10})                 # a person cuts what they kept
    assert rv.suggest_cut(run.run_path, m, 0, 10, "again", by="x")["status"] == "already_cut"
    rev = rv.load(run.run_path)["rev"]
    with pytest.raises(rv.ReviewConflict):                                            # stale batch writes nothing
        rv.find_idle(ds.root, run.run_path, m, expect_rev=rev - 1)
    assert rv.load(run.run_path)["rev"] == rev


@pytest.mark.parametrize("series,rate,want", [
    ([0] * 15 + [10] * 15, 10, [(0, 14), (16, 30)]),          # two plateaus: the movement between stays uncut
    ([0] * 12 + [10] + [0] * 12, 10, [(0, 11), (14, 25)]),    # a one-frame spike splits the stretch
    ([0] * 9, 10, []),                                        # shorter than one window
    ([0] * 10, 10, [(0, 10)]),                                # exactly one window
    (list(range(30)), 10, []),                                # always moving
    ([0] * 14 + [10] * 8, 10, [(0, 13)]),                     # second plateau shorter than a second
])
def test_idle_spans_boundaries(series, rate, want):
    from dataop_logger import review as rv
    motion = np.asarray(series, np.float32).reshape(-1, 1)
    spans = rv.idle_spans(motion, rate)
    assert spans == want
    cut = np.zeros(len(series), bool)                          # cutting every suggestion keeps every movement
    for a, b in spans:
        cut[a:b] = True
    moves = [i for i in range(1, len(series)) if series[i] != series[i - 1]]
    assert not any(cut[i] or cut[i - 1] for i in moves)


def test_review_validation_covers_history_and_sequences(tmp_path):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    lb = rv.add_edit(run.run_path, m, "label", {"start": 1, "end": 3, "text": "x"})
    rv.interpret(run.run_path, lb["id"], "keep", "fine")
    good = json.loads((run.run_path / "review.json").read_text())
    for mutate in (lambda r: r["edits"][0].__setitem__("interpretation_history", "oops"),
                   lambda r: r["edits"][0].__setitem__("interpretation", {}),
                   lambda r: r["edits"][0]["interpretation"].__setitem__("by", ""),
                   lambda r: r["edits"][0].__setitem__("interpretation_history", [{"train": "keep"}]),
                   lambda r: r.__setitem__("seq", 0),
                   lambda r: r["edits"][0].__setitem__("by", 5)):
        bad = json.loads(json.dumps(good))
        mutate(bad)
        (run.run_path / "review.json").write_text(json.dumps(bad))
        with pytest.raises(rv.ReviewError):
            rv.effective(m, rv.load(run.run_path))
    # a pending proposal of a kind the viewer no longer creates is reported on its own
    (run.run_path / "review.json").write_text(json.dumps(good))
    rv.add_edit(run.run_path, m, "failure_start", {"frame": 5}, source="agent", reason="old")
    after = V.inspect(ds.root)["after_review"]
    assert after["other_open_proposals"] == 1 and after["open_cut_suggestions"] == 0


def test_interpretation_is_not_silently_replaced(tmp_path):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    lb = rv.add_edit(run.run_path, m, "label", {"start": 1, "end": 3, "text": "blocked camera"})
    rv.interpret(run.run_path, lb["id"], "exclude", "camera blocked", by="agent-a")
    with pytest.raises(rv.ReviewError):
        rv.interpret(run.run_path, lb["id"], "keep", "looks fine", by="agent-b")
    e = rv.interpret(run.run_path, lb["id"], "keep", "looks fine", by="agent-b", replace=True)
    assert e["interpretation"]["train"] == "keep" and e["interpretation_history"][0]["by"] == "agent-a"
    with pytest.raises(rv.ReviewError):                          # an interpreted label cannot be re-aimed in place
        rv.decide(run.run_path, m, lb["id"], True, {"start": 6, "end": 9, "text": "other"})


def test_malformed_stored_review_fails_closed(tmp_path):
    from dataop_logger import review as rv, viewer
    ds, run, m = _review_fixture(tmp_path)
    lb = rv.add_edit(run.run_path, m, "label", {"start": 1, "end": 3, "text": "x"})
    rv.interpret(run.run_path, lb["id"], "exclude", "drop")
    r = json.loads((run.run_path / "review.json").read_text())
    r["edits"][0]["interpretation"]["train"] = "exlcude"
    (run.run_path / "review.json").write_text(json.dumps(r))
    with pytest.raises(rv.ReviewError):
        rv.effective(m, rv.load(run.run_path))
    assert not V.validate(ds.root)["ok"]
    with pytest.raises(ValueError):
        DataopDataset(lerobot_dataset.LeRobotDataset("test/ds", root=ds.root), check=False)
    state = viewer.collect(ds.root)["runs"][0]["review"]
    assert state["error"] and state["rev"] == -1
    r["edits"][0]["interpretation"]["train"] = "exclude"
    r["edits"][0]["payload"]["start"] = -4
    (run.run_path / "review.json").write_text(json.dumps(r))
    with pytest.raises(rv.ReviewError):
        rv.effective(m, rv.load(run.run_path))


def test_logger_rejects_bad_step_order_and_success_failure_mode(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:
            run.subgoal("a", "started")
            with pytest.raises(ValueError):
                run.subgoal("b", "started")
            with pytest.raises(ValueError):
                run.subgoal("b", "completed")
            run.subgoal("a", "completed")
            run.subgoal("b", "completed")                         # implicit start after a
            none_frame(run, 0)
            with pytest.raises(ValueError):
                run.end(termination_reason="success", failure_mode="dropped")
            run.end(termination_reason="success")
    ds.finalize()
    assert V.validate(ds.root)["ok"]


def test_save_check_fails_closed_without_saved_record(tmp_path, monkeypatch):
    from dataop_logger import adapter
    ds = make_ds(tmp_path)
    monkeypatch.setattr(adapter, "saved_episode", lambda *_: None)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(RunError):
            record(log, 3)
        assert log.broken
    run_dirs = list(store.runs_dir(ds.root).iterdir())
    assert store.read_manifest(run_dirs[0])["status"] == "staged"


def test_range_parsing():
    from dataop_logger.viewer import _parse_range
    assert _parse_range("bytes=0-99", 1000) == (0, 99)
    assert _parse_range("bytes=900-", 1000) == (900, 999)
    assert _parse_range("bytes=-500", 1000) == (500, 999)
    assert _parse_range("bytes=-5000", 1000) == (0, 999)
    for bad in ("bytes=5-1", "bytes=1000-", "bytes=a-b", "bytes=0-1,5-9", "bytes=-", "bytes=-0", "items=0-1"):
        assert _parse_range(bad, 1000) is None


def test_review_api(tmp_path):
    import functools
    import http.client
    import http.server
    import threading
    from dataop_logger import review as rv, viewer
    ds, run, m = _review_fixture(tmp_path)

    def start(editable):
        h = type("H", (viewer._RangeHandler,), {"dataset_root": Path(ds.root), "editable": editable})
        if not editable:
            h.do_POST = lambda self: self._json(403, {"error": "read-only"})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(h, directory=str(ds.root)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    srv, ro = start(True), start(False)
    port = srv.server_address[1]

    def post(body, port=port, **hdr):
        c = http.client.HTTPConnection("127.0.0.1", port)
        h = {"Content-Type": "application/json", "X-Dataop": "1", **hdr}
        c.request("POST", viewer.API, json.dumps(body), h)
        r = c.getresponse()
        return r.status, json.loads(r.read())

    rid = run.run_id
    lbl = {"op": "add", "kind": "label", "payload": {"start": 1, "end": 4, "text": "x"}}
    try:
        base = {"run_id": rid, "rev": 0}
        assert post({**base, **lbl}, **{"X-Dataop": "0"})[0] == 403
        for origin in ("http://evil.example", "null"):
            assert post({**base, **lbl}, Origin=origin)[0] == 403
        assert post({**base, **lbl}, Host=f"evil.example:{port}")[0] == 403
        assert post({**base, **lbl}, port=ro.server_address[1])[0] == 403
        assert post({"run_id": "nope", "rev": 0, **lbl})[0] == 400
        assert post({"run_id": rid, **lbl})[0] == 400                                      # rev missing
        assert post({**base, "op": "add", "kind": "reject_run", "payload": {}})[0] == 400  # viewer adds labels and cuts only
        assert post({**base, "op": "add", "kind": "label", "payload": {"start": 1, "end": 4, "driver": "policy"}})[0] == 400
        code, state = post({**base, **lbl})
        assert code == 200 and state["rev"] == 1 and state["effective"]["labels"][0]["text"] == "x"
        assert post({**base, **lbl})[0] == 409                                             # stale rev
        lid = state["effective"]["labels"][0]["id"]
        code, state = post({"run_id": rid, "rev": 1, "op": "replace", "edit_id": lid,
                            "payload": {"start": 2, "end": 5, "text": "y", "segment": "regrasp"}})
        assert code == 200 and state["effective"]["labels"][0]["segment"] == "regrasp"
        code, state = post({"run_id": rid, "rev": state["rev"], "op": "find_idle"})
        assert code == 200
        code, state = post({"run_id": rid, "rev": state["rev"], "op": "add", "kind": "exclude", "payload": {"start": 0, "end": 3}})
        cut = next(e for e in state["edits"] if e["kind"] == "exclude" and e["status"] == "accepted")
        for bad in ("false", 0, None):
            assert post({"run_id": rid, "rev": state["rev"], "op": "decide", "edit_id": cut["id"], "accept": bad})[0] == 400
        code, state = post({"run_id": rid, "rev": state["rev"], "op": "decide", "edit_id": cut["id"], "accept": False})
        assert code == 200 and state["effective"]["excludes"] == []
        step = state["step"]["entry"]
        code, state = post({"run_id": rid, "rev": state["rev"], "op": "undo", "entry_id": step})
        assert code == 200 and state["undone"]["entry"] == step and len(state["effective"]["excludes"]) == 1
        assert post({"run_id": rid, "rev": state["rev"], "op": "undo", "entry_id": step})[0] == 400
        c = http.client.HTTPConnection("127.0.0.1", port)
        c.request("GET", "/dataop/view/")
        resp = c.getresponse()
        page = resp.read().decode()
        assert resp.getheader("X-Frame-Options") == "DENY" and "frame-ancestors 'none'" in resp.getheader("Content-Security-Policy")
        assert '"editable": true' in page and "/*__DATA__*/null" not in page and "__NONCE__" not in page
    finally:
        srv.shutdown()
        ro.shutdown()


def test_viewer_skips_runs_with_unexpected_ids(tmp_path):
    from dataop_logger import viewer
    ds, run, m = _review_fixture(tmp_path)
    bad = dict(m, run_id='x" onmouseover="alert(1)')
    store.atomic_write_json(run.run_path / "manifest.json", bad)
    assert viewer.collect(ds.root)["runs"] == []
    assert not V.validate(ds.root)["ok"]


def test_labels_replace_interpret_and_mask(tmp_path):
    from dataop_logger import review as rv
    from dataop_logger.cli import main
    ds, run, m = _review_fixture(tmp_path)
    modes = ["base"] * 30 + ["expert"] * 10
    lb = rv.add_edit(run.run_path, m, "label", {"start": 2, "end": 6, "text": "my hand blocks the camera",
                                                "driver": "policy", "segment": "grasp"})
    for bad in [{"start": 6, "end": 6, "text": "x"}, {"start": 0, "end": 3}, {"start": 0, "end": 3, "text": "x", "driver": "d" * 51},
                {"start": 0, "end": 3, "segment": " "}, {"start": 0, "end": 41, "text": "x"}]:
        with pytest.raises(rv.ReviewError):
            rv.add_edit(run.run_path, m, "label", bad)
    labels = rv.list_labels(ds.root, only_uninterpreted=True)
    assert len(labels) == 1 and labels[0]["recorded"]["control_modes"] == {"base": 4}
    assert labels[0]["recorded"]["steps"] == ["grasp"]
    assert rv.train_mask(m, rv.load(run.run_path), modes).all()           # text alone changes nothing
    assert main(["interpret", str(ds.root), run.run_id, lb["id"], "--train", "exclude",
                 "--meaning", "camera view blocked"]) == 0
    assert rv.list_labels(ds.root, only_uninterpreted=True) == []
    mask = rv.train_mask(m, rv.load(run.run_path), modes)
    assert not mask[2:6].any() and mask[:2].all() and mask[6:].all()
    # rewriting a label drops the old interpretation; deleting it brings the frames back
    new = rv.replace_edit(run.run_path, m, lb["id"], {"start": 2, "end": 8, "text": "hand in view"})
    eff = rv.effective(m, rv.load(run.run_path))
    assert [(x["id"], x["end"], x["interpretation"]) for x in eff["labels"]] == [(new["id"], 8, None)]
    assert rv.train_mask(m, rv.load(run.run_path), modes).all()
    rv.decide(run.run_path, m, new["id"], False)
    assert rv.effective(m, rv.load(run.run_path))["labels"] == []
    with pytest.raises(rv.ReviewError):
        rv.interpret(run.run_path, new["id"], "exclude", "gone")
    assert V.validate(ds.root)["ok"]


def test_undo_steps(tmp_path):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    with rv.recording("cut") as rec:
        rv.add_edit(run.run_path, m, "exclude", {"start": 0, "end": 5})
    cut_step = rec["entry"]
    lb = rv.add_edit(run.run_path, m, "label", {"start": 1, "end": 3, "text": "first"})
    rv.interpret(run.run_path, lb["id"], "exclude", "drop it")
    with rv.recording("change label") as rec:
        rv.replace_edit(run.run_path, m, lb["id"], {"start": 1, "end": 4, "text": "second"})
    rv.undo(run.run_path, m, rec["entry"])                 # the old label and its reading come back
    eff = rv.effective(m, rv.load(run.run_path))
    assert [(x["text"], x["interpretation"]["train"]) for x in eff["labels"]] == [("first", "exclude")]
    with pytest.raises(rv.ReviewError):
        rv.undo(run.run_path, m, rec["entry"])             # once only
    rv.undo(run.run_path, m, cut_step)
    assert rv.effective(m, rv.load(run.run_path))["excludes"] == []
    # one step for a batch; a later change to the same edit blocks undo of the earlier step
    a = rv.add_edit(run.run_path, m, "exclude", {"start": 10, "end": 12}, source="agent", reason="idle")
    b = rv.add_edit(run.run_path, m, "exclude", {"start": 20, "end": 22}, source="agent", reason="idle")
    with rv.recording("cut all") as rec:
        rv.decide_many(run.run_path, m, [a["id"], b["id"]], True)
    assert len(rv.effective(m, rv.load(run.run_path))["excludes"]) == 2
    with rv.recording("undo one") as rec2:
        rv.decide(run.run_path, m, a["id"], False)
    with pytest.raises(rv.ReviewError):
        rv.undo(run.run_path, m, rec["entry"])
    rv.undo(run.run_path, m, rec2["entry"])
    rv.undo(run.run_path, m, rec["entry"])
    r = rv.load(run.run_path)
    assert {e["id"]: e["status"] for e in r["edits"] if e["id"] in (a["id"], b["id"])} == {a["id"]: "proposed", b["id"]: "proposed"}
    assert V.validate(ds.root)["ok"]
    with rv.recording("nothing") as rec:                # a step that changes nothing is not recorded
        pass
    assert rec["entry"] is None


def test_undo_history_is_validated_and_one_step_per_recording(tmp_path):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    with pytest.raises(rv.ReviewError):
        with rv.recording("two cuts"):
            rv.add_edit(run.run_path, m, "exclude", {"start": 0, "end": 2})
            rv.add_edit(run.run_path, m, "exclude", {"start": 4, "end": 6})
    with rv.recording("cut") as rec:
        rv.add_edit(run.run_path, m, "exclude", {"start": 8, "end": 9})
    with rv.recording("undo cut") as rec:
        rv.decide(run.run_path, m, next(e["id"] for e in rv.load(run.run_path)["edits"] if e["payload"]["start"] == 8), False)
    r = json.loads((run.run_path / "review.json").read_text())
    assert [u["entry"] for u in rv.undoable(r)][-1] == rec["entry"]
    good = json.dumps(r)
    bad = json.loads(good)
    bad["history"][-1]["changes"][0]["after"]["payload"]["start"] = -50          # corrupt snapshot
    bad["history"][-1]["changes"][0]["before"] = {"id": "nope"}
    (run.run_path / "review.json").write_text(json.dumps(bad))
    with pytest.raises(rv.ReviewError):
        rv.effective(m, rv.load(run.run_path))
    bad = json.loads(good)                                                        # a restore that would be invalid
    bad["history"][-1]["changes"][0]["before"]["payload"]["start"] = -50
    (run.run_path / "review.json").write_text(json.dumps(bad))
    with pytest.raises(rv.ReviewError):
        rv.undo(run.run_path, m, rec["entry"])
    assert json.loads((run.run_path / "review.json").read_text()) == bad          # nothing was saved


def test_undoable_hides_steps_changed_later(tmp_path):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    with rv.recording("cut") as first:
        rv.add_edit(run.run_path, m, "exclude", {"start": 0, "end": 2})
    with rv.recording("label") as second:
        lb = rv.add_edit(run.run_path, m, "label", {"start": 4, "end": 6, "text": "x"})
    rv.interpret(run.run_path, lb["id"], "keep", "fine")          # not a viewer step: the label step can't be undone
    assert [u["entry"] for u in rv.undoable(rv.load(run.run_path))] == [first["entry"]]
    rv.undo(run.run_path, m, first["entry"])                      # the older independent step is still reachable
    assert rv.undoable(rv.load(run.run_path)) == []
    r = json.loads((run.run_path / "review.json").read_text())   # a snapshot missing fields fails validation
    del r["history"][0]["changes"][0]["after"]["payload"]
    (run.run_path / "review.json").write_text(json.dumps(r))
    with pytest.raises(rv.ReviewError):
        rv.effective(m, rv.load(run.run_path))


def test_history_byte_cap(tmp_path, monkeypatch):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    monkeypatch.setattr(rv, "HISTORY_BYTES", 2000)
    with rv.recording("big") as rec:
        rv.add_edit(run.run_path, m, "label", {"start": 1, "end": 3, "text": "y" * 1990})
    assert rec["entry"] is None and rv.load(run.run_path).get("history", []) == []
    for i in range(10):
        with rv.recording("cut"):
            rv.add_edit(run.run_path, m, "exclude", {"start": i, "end": i + 1})
    hist = rv.load(run.run_path)["history"]
    assert 0 < len(hist) < 10 and len(json.dumps(hist)) <= 2000


def test_label_custom_drive_mode(tmp_path):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)
    rv.add_edit(run.run_path, m, "label", {"start": 1, "end": 3, "text": "teleop nudge", "driver": " teleop "})
    assert rv.effective(m, rv.load(run.run_path))["labels"][0]["driver"] == "teleop"


def test_major_events_three_kinds_and_notes(tmp_path):
    from dataop_logger import events, review as rv
    from dataop_logger.cli import main
    ds, run, m = _review_fixture(tmp_path)               # grasp 0-10, lift 10-30 failed, rescue from 30
    modes = ["base"] * 30 + ["expert"] * 10
    srcs = [None] * 30 + ["skill:rescue"] * 10
    evs = events.major_events(m, rv.load(run.run_path), modes, srcs)
    got = [(e["kind"], e["frame"], e["title"]) for e in evs]
    assert got == [("drive", 0, "Policy drives"), ("segment", 0, "grasp"), ("segment", 10, "lift"),
                   ("drive", 30, "Rescue drives"), ("failure", 30, "Policy fails"), ("end", 40, "Run ends: failure")]
    fail = next(e for e in evs if e["kind"] == "failure")
    assert fail["estimated"] and fail["window"] == [0, 30] and "lift failed" in fail["note"]
    assert next(e for e in evs if e["key"] == "segment:lift@10")["note"] == "failed: slipped"
    # a marked failure replaces the estimate; labels correct drive mode and segments
    rv.add_edit(run.run_path, m, "failure_start", {"frame": 22})
    rv.add_edit(run.run_path, m, "label", {"start": 25, "end": 30, "text": "operator nudged", "driver": "teleop"})
    evs = events.major_events(m, rv.load(run.run_path), modes, srcs)
    got = [(e["kind"], e["frame"], e["title"]) for e in evs]
    assert ("failure", 22, "Policy fails") in got and ("drive", 25, "Teleop drives") in got
    assert not next(e for e in evs if e["kind"] == "failure")["estimated"]
    # notes: an agent's note replaces the recorded one; undo-able like any edit
    assert main(["note", str(ds.root), run.run_id, "failure", "--text", "gripper slips at the cap", "--by", "vlm-agent"]) == 0
    evs = events.major_events(m, rv.load(run.run_path), modes, srcs)
    fail = next(e for e in evs if e["kind"] == "failure")
    assert (fail["note"], fail["note_by"]) == ("gripper slips at the cap", "vlm-agent")
    with pytest.raises(rv.ReviewError):
        rv.add_event_note(run.run_path, m, "failure", " ", by="x")
    assert V.validate(ds.root)["ok"]


def test_segment_outer_edges(tmp_path):
    from dataop_logger import events, review as rv
    ds, run, m = _review_fixture(tmp_path)                 # grasp 0-10, lift 10-30
    rv.add_edit(run.run_path, m, "segment_edge", {"index": 0, "edge": "start", "frame": 4})
    rv.add_edit(run.run_path, m, "segment_edge", {"index": 1, "edge": "end", "frame": 36})
    steps = rv.effective(m, rv.load(run.run_path))["subgoals"]
    assert [(s["start_frame"], s["end_frame"]) for s in steps] == [(4, 10), (10, 36)]
    rv.add_edit(run.run_path, m, "segment_edge", {"index": 0, "edge": "start", "frame": 25})   # clamped: one frame left
    assert rv.effective(m, rv.load(run.run_path))["subgoals"][0]["start_frame"] == 9
    with pytest.raises(rv.ReviewError):
        rv.add_edit(run.run_path, m, "segment_edge", {"index": 5, "edge": "start", "frame": 1})
    segs = events.major_events(m, rv.load(run.run_path), ["base"] * 40, [None] * 40, with_segments=True)[1]
    assert [(s["name"], s["start"], s["end"], s["index"]) for s in segs] == [("grasp", 9, 10, 0), ("lift", 10, 36, 1)]


def test_step_boundary_follows_corrected_neighbours(tmp_path):
    from dataop_logger import review as rv
    ds, run, m = _review_fixture(tmp_path)                 # grasp 0-10, lift 10-30
    rv.add_edit(run.run_path, m, "segment_edge", {"index": 1, "edge": "end", "frame": 38})
    rv.add_edit(run.run_path, m, "step_boundary", {"index": 0, "frame": 33})   # past lift's recorded end: allowed now
    steps = rv.effective(m, rv.load(run.run_path))["subgoals"]
    assert [(s["start_frame"], s["end_frame"]) for s in steps] == [(0, 33), (33, 38)]
    rv.add_edit(run.run_path, m, "step_boundary", {"index": 0, "frame": 40})   # clamped to leave lift one frame
    steps = rv.effective(m, rv.load(run.run_path))["subgoals"]
    assert [(s["start_frame"], s["end_frame"]) for s in steps] == [(0, 37), (37, 38)]


def test_drive_boundary_moves_the_switch(tmp_path):
    from dataop_logger import events, review as rv
    ds, run, m = _review_fixture(tmp_path)
    modes, srcs = ["base"] * 30 + ["expert"] * 10, [None] * 30 + ["skill:rescue"] * 10
    rv.add_edit(run.run_path, m, "drive_boundary", {"index": 0, "frame": 26})
    evs, segs, drive = events.major_events(m, rv.load(run.run_path), modes, srcs, with_segments=True)
    assert [(d["driver"], d["start"], d["end"], d["rec_start"], d["rec_end"]) for d in drive] == \
        [("policy", 0, 26, 0, 30), ("rescue", 26, 40, 30, 40)]
    assert ("drive", 26, "Rescue drives") in [(e["kind"], e["frame"], e["title"]) for e in evs]
    assert next(e for e in evs if e["kind"] == "failure")["frame"] == 26      # the estimate follows the switch
    rv.add_edit(run.run_path, m, "drive_boundary", {"index": 0, "frame": 40})  # clamped: rescue keeps a frame
    drive = events.major_events(m, rv.load(run.run_path), modes, srcs, with_segments=True)[2]
    assert [(d["start"], d["end"]) for d in drive] == [(0, 39), (39, 40)]


def test_lists_schema_seed_add_rename_delete(tmp_path):
    from dataop_logger import lists, review as rv
    ds, run, m = _review_fixture(tmp_path)              # failure_mode no_lift, steps grasp and lift
    d = lists.load(ds.root)
    assert [e["name"] for e in d["failure_types"]] == ["no_lift"]
    assert [e["name"] for e in d["task_segments"]] == ["grasp", "lift"]
    lists.add(ds.root, "failure_types", "collision", "hit something")
    with pytest.raises(lists.ListError):
        lists.add(ds.root, "failure_types", "collision")
    d = lists.rename(ds.root, "task_segments", "lift", "raise")
    assert lists.canonical(d, "task_segments", "lift") == "raise"          # recorded names still map
    with pytest.raises(lists.ListError):
        lists.delete(ds.root, "task_segments", "raise")                    # in use by the run
    lists.delete(ds.root, "failure_types", "collision")
    with pytest.raises(lists.ListError):
        lists.add(ds.root, "failure_types", "x", expect_rev=0)             # stale revision
    assert [e["name"] for e in lists.load(ds.root)["failure_types"]] == ["no_lift"]


def test_segment_plan_replaces_segments(tmp_path):
    from dataop_logger import events, review as rv
    ds, run, m = _review_fixture(tmp_path)
    plan = [{"name": "reach", "start": 0, "end": 5}, {"name": "grasp", "start": 5, "end": 10},
            {"name": "lift", "start": 12, "end": 30}]
    rv.add_edit(run.run_path, m, "segment_plan", {"segments": plan})
    steps = rv.effective(m, rv.load(run.run_path))["subgoals"]
    assert [(s["name"], s["start_frame"], s["end_frame"], s["status"]) for s in steps] == \
        [("reach", 0, 5, "completed"), ("grasp", 5, 10, "completed"), ("lift", 12, 30, "failed")]
    for bad in ([{"name": "a", "start": 0, "end": 5}, {"name": "b", "start": 4, "end": 8}],
                [{"name": "a", "start": 3, "end": 3}], [{"name": "", "start": 0, "end": 2}]):
        with pytest.raises(rv.ReviewError):
            rv.add_edit(run.run_path, m, "segment_plan", {"segments": bad})
    segs = events.major_events(m, rv.load(run.run_path), ["base"] * 40, [None] * 40, with_segments=True)[1]
    assert [(s["name"], s["start"], s["end"]) for s in segs] == [("reach", 0, 5), ("grasp", 5, 10), ("lift", 12, 30)]


def test_agent_annotates_through_the_schema(tmp_path):
    from dataop_logger import review as rv
    from dataop_logger.annotate import runs_json
    from dataop_logger.cli import main
    ds, run, m = _review_fixture(tmp_path)                 # 30 fps, 40 frames, grasp 0-10, lift 10-30 failed
    segs = '[{"name": "grasp", "start_s": 0, "end_s": 0.3}, {"name": "lift", "start_s": 0.3, "end_s": 1.0}]'
    assert main(["annotate", str(ds.root), run.run_id, "--segments", segs, "--failure-at", "0.8",
                 "--failure-type", "no_lift", "--by", "vlm-agent"]) == 0
    assert main(["annotate", str(ds.root), run.run_id, "--failure-type", "teleported", "--by", "x"]) == 2   # not on the list
    out = runs_json(ds.root)
    r = out["runs"][0]
    assert [(s["name"], s["start_frame"], s["end_frame"]) for s in r["segments"]] == [("grasp", 0, 9), ("lift", 9, 30)]
    assert r["failure"]["frame"] == 24 and r["failure"]["type"] == "no_lift" and not r["failure"]["estimated"]
    assert r["failure"]["set_by"]["failure_start"] == "vlm-agent" and r["segments_set_by"] == "vlm-agent"
    assert {e["name"] for e in out["lists"]["task_segments"]} == {"grasp", "lift"}
    # each agent change is one undoable step for a person
    undo = rv.undoable(rv.load(run.run_path))
    assert [u["what"] for u in undo] == ["vlm-agent: task segments", "vlm-agent: failure type no_lift",
                                          "vlm-agent: policy fails at 0.8 s"]
    rv.undo(run.run_path, m, undo[-1]["entry"])
    assert runs_json(ds.root)["runs"][0]["failure"]["estimated"]


# ── round 16 fixes, batches and import ─────────────────────────────────────────

def test_idle_spans_do_not_depend_on_units():
    from dataop_logger import review as rv
    moving = np.linspace(0.0, 1.0, 60, dtype=np.float32)[:, None]          # moves its whole range in 2 s
    assert rv.idle_spans(moving, 30) == []
    plateaus = np.asarray([0] * 15 + [10] * 15, np.float32)[:, None]
    assert rv.idle_spans(plateaus, 10) == rv.idle_spans(plateaus * 0.001, 10) == rv.idle_spans(plateaus * 1000, 10)


def test_annotate_writes_nothing_when_one_field_is_bad(tmp_path):
    from dataop_logger import annotate, lists, review as rv
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        run = record(log, 30)                                    # the policy drove all of it: no takeover to move
    ds.finalize()
    lists.add(ds.root, "task_segments", "reach")
    with pytest.raises(rv.ReviewError):
        annotate.annotate(ds.root, run.run_id, "agent", segments=[{"name": "reach", "start_s": 0, "end_s": 0.5}],
                          takeover_at=0.2)
    assert rv.load(run.run_path)["edits"] == []


def test_noop_review_change_keeps_revision(tmp_path):
    from dataop_logger import review as rv
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        run = record(log, 10)
    ds.finalize()
    rv.set_reviewed(run.run_path, True)
    rev = rv.load(run.run_path)["rev"]
    rv.set_reviewed(run.run_path, True)
    assert rv.load(run.run_path)["rev"] == rev


def test_lists_cli_describe_and_delete(tmp_path):
    from dataop_logger import lists
    from dataop_logger.cli import main
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        record(log, 5)
    ds.finalize()
    assert main(["lists", str(ds.root), "add", "failure_types", "slip"]) == 0
    assert main(["lists", str(ds.root), "describe", "failure_types", "slip", "--description", "object slides out"]) == 0
    assert lists.load(ds.root)["failure_types"][-1]["description"] == "object slides out"
    assert main(["lists", str(ds.root), "delete", "failure_types", "slip"]) == 0
    assert all(e["name"] != "slip" for e in lists.load(ds.root)["failure_types"])


def test_batches_runs_join_lock_and_unlock(tmp_path):
    from dataop_logger import batches as B, review as rv
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, batch="round 1") as log:
        a = record(log, 6)
        b = record(log, 6, takeover_at=3, reason="failure")
    ds.finalize()
    s = B.summary(ds.root)
    assert [(x["name"], x["status"], x["n_runs"], x["rescued"]) for x in s] == [("round 1", "collecting", 2, 1)]
    with pytest.raises(B.BatchError):                       # every run must be reviewed before the lock
        B.set_status(ds.root, "round 1", "locked")
    for r in (a, b):
        rv.set_reviewed(r.run_path, True)
    B.set_status(ds.root, "round 1", "locked", by="tester")
    m = store.read_manifest(a.run_path)
    with pytest.raises(rv.ReviewError, match="locked"):
        rv.add_edit(a.run_path, m, "note", {"frame": 1, "text": "x"})
    with pytest.raises(B.BatchError):
        B.set_status(ds.root, "round 1", "reviewing")       # unlocking needs a reason
    B.set_status(ds.root, "round 1", "reviewing", reason="fix a label")
    rv.add_edit(a.run_path, m, "note", {"frame": 1, "text": "x"})
    with RunLogger(ds, SCHEMA, batch="round 1") as log:     # a closed batch takes no new runs
        with pytest.raises(Exception, match="reviewing"):
            record(log, 4)


def test_batches_place_existing_runs_by_policy(tmp_path):
    from dataop_logger import batches as B
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        record(log, 5)
        record(log, 5)
    ds.finalize()
    (ds.root / "dataop" / "batches.json").unlink()          # as if recorded before batches existed
    s = B.summary(ds.root)
    assert len(s) == 1 and s[0]["n_runs"] == 2 and s[0]["policy"]["source"] == "train/smolvla_100"


def _plain_dataset(tmp_path, episodes):
    """A LeRobot dataset recorded without dataop, with an intervention flag and LeRobot's next.success."""
    feats = {**features(), "intervention": {"dtype": "bool", "shape": (1,), "names": None},
             "next.success": {"dtype": "bool", "shape": (1,), "names": None}}
    ds = lerobot_dataset.LeRobotDataset.create("test/plain", fps=30, features=feats, root=tmp_path / "plain",
                                               use_videos=False)
    for n, take, ok in episodes:
        for i in range(n):
            a = np.arange(6, dtype=np.float32) + i
            ds.add_frame({"observation.state": a, "action": a, "task": TASK,
                          "intervention": np.array([take is not None and i >= take]),
                          "next.success": np.array([ok and i == n - 1])})
        ds.save_episode()
    ds.finalize()
    return ds


def test_import_plain_and_hil_mapping(tmp_path):
    from dataop_logger import batches as B, importer
    ds = _plain_dataset(tmp_path, [(12, None, True), (15, 6, False)])
    dry = importer.import_dataset(str(ds.root), preset="plain", dry_run=True)
    assert dry["to_bind"] == 2 and not (ds.root / "dataop" / "runs").exists()
    mapping = {"control": {"column": "intervention"}, "expert_source": "human", "base_action": "sent_on_policy_frames",
               "outcome": {"column": "next.success"}}
    rep = importer.import_dataset(str(ds.root), mapping=mapping, batch="hil")
    assert [r["outcome"] for r in rep["runs"]] == ["success", "failure"]
    assert [r["policy_frames"] for r in rep["runs"]] == [12, 6]
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert [(b["name"], b["n_runs"], b["rescued"]) for b in B.summary(ds.root)] == [("hil", 2, 1)]
    again = importer.import_dataset(str(ds.root), mapping=mapping)
    assert again["runs"] == [] and len(again["skipped"]) == 2                     # idempotent


def test_import_unknown_fields_stay_unknown(tmp_path):
    import pyarrow.parquet as pq
    from dataop_logger import importer
    ds = _plain_dataset(tmp_path, [(10, None, True)])
    rep = importer.import_dataset(str(ds.root))
    assert rep["runs"][0]["outcome"] == "unknown"
    run = store.runs_dir(ds.root) / rep["runs"][0]["run_id"]
    t = pq.read_table(run / "ticks.parquet")
    assert set(t.column("control_mode").to_pylist()) == {"unknown"} and not any(t.column("base_valid").to_pylist())
    assert V.validate(ds.root)["ok"]


def test_import_refuses_lerobot_v21(tmp_path):
    import json
    from dataop_logger import importer
    (tmp_path / "old" / "meta").mkdir(parents=True)
    (tmp_path / "old" / "meta" / "info.json").write_text(json.dumps({"codebase_version": "v2.1"}))
    with pytest.raises(importer.ImportError_, match="convert_dataset_v21_to_v30"):
        importer.import_dataset(str(tmp_path / "old"))
