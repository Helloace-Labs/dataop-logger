"""Logger task L2: optional task, planner and prompt fields, and unknown values.

Manifest: task_id, task_rev, planner ({name, id}, null for no planner, "unknown" when not recorded).
Per tick: prompt, prompt_rev, subtask_cmd, control_gen. Imports record what they cannot know as unknown.
Datasets written before these fields existed keep loading, and their missing fields read as unknown.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aftershock_logger import UNKNOWN, ActionSchema, PlannerInfo, PolicyInfo, RunLogger  # noqa: E402
from aftershock_logger import batches as B  # noqa: E402
from aftershock_logger import adapter, importer, store  # noqa: E402
from aftershock_logger import validate as V  # noqa: E402
from aftershock_logger.reader import planned_columns, tick_records  # noqa: E402
from aftershock_logger.schema import manifest_fields  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCHEMA = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
TASK = "pick up the tube"
OLD_DATASET = Path(__file__).resolve().parent / "data" / "dataop_v0_1_0"   # written by dataop-logger 0.1.0
RULES = PlannerInfo(name="Rules", id="rules-v3")


def make_ds(tmp_path, name="ds", extra=None):
    feats = {k: {"dtype": "float32", "shape": (6,), "names": list(NAMES)} for k in ("observation.state", "action")}
    return lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features={**feats, **(extra or {})},
                                                 root=tmp_path / name, use_videos=False)


def ts():
    t = time.monotonic_ns()
    return {"obs_ns": t, "sent_ns": t + 1000}


def add(run, i, **kw):
    a = np.arange(6, dtype=np.float32) + i
    return run.add_frame({"observation.state": a, "task": TASK}, sent_action=a, base_action=a, control_mode="base",
                         inference_id=i, chunk_offset=0, **ts(), **kw)


def record(log, ticks, run_kw=None, reason="success"):
    """ticks: one dict of add_frame keyword arguments per frame."""
    with log.run(task=TASK, policy=PolicyInfo("smolvla", "train/smolvla_100"), **(run_kw or {})) as run:
        for i, kw in enumerate(ticks):
            add(run, i, **kw)
        run.end(termination_reason=reason)
    return run


PLANNED = [
    {"prompt": "reach the tube", "prompt_rev": 0, "subtask_cmd": "st_01", "control_gen": 0},
    {"prompt": "reach the tube", "prompt_rev": 0, "subtask_cmd": "st_01", "control_gen": 0},
    {"prompt": "grasp the cap", "prompt_rev": 1, "subtask_cmd": "st_02", "control_gen": 0},
    {"prompt": "grasp the cap", "prompt_rev": 1, "subtask_cmd": "st_02", "control_gen": 1},
]


def test_every_field_is_written_and_read_back(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=RULES) as log:
        r = record(log, PLANNED, {"task_id": "tube-pick", "task_rev": 3})
    ds.finalize()
    m = store.read_manifest(r.run_path)
    assert (m["task_id"], m["task_rev"], m["planner"]) == ("tube-pick", 3, {"name": "Rules", "id": "rules-v3"})
    t = pq.read_table(r.run_path / "ticks.parquet")
    assert t.schema.field("prompt").type == pa.string() and t.schema.field("subtask_cmd").type == pa.string()
    assert t.schema.field("prompt_rev").type == pa.int64() and t.schema.field("control_gen").type == pa.int64()
    for k in ("prompt", "prompt_rev", "subtask_cmd", "control_gen"):
        assert t.column(k).to_pylist() == [p[k] for p in PLANNED]
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert manifest_fields(m) == {"task_id": "tube-pick", "task_rev": 3, "planner": {"name": "Rules", "id": "rules-v3"},
                                  "policy": m["policy"]}


def test_reader_exposes_the_tick_fields(tmp_path):
    pytest.importorskip("torch")
    from aftershock_logger.reader import DataopDataset
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=RULES) as log:
        record(log, PLANNED, {"task_id": "tube-pick", "task_rev": 3})
        record(log, [{}, {}])                                  # nothing given: every field unknown
    ds.finalize()
    dd = DataopDataset(lerobot_dataset.LeRobotDataset("test/ds", root=ds.root))
    item = dd[2]
    assert item["dataop.prompt"] == "grasp the cap" and item["dataop.subtask_cmd"] == "st_02"
    assert int(item["dataop.prompt_rev"]) == 1 and int(dd[3]["dataop.control_gen"]) == 1
    blank = dd[4]
    assert blank["dataop.prompt"] == UNKNOWN and blank["dataop.subtask_cmd"] == UNKNOWN
    assert int(blank["dataop.prompt_rev"]) == -1 and int(blank["dataop.control_gen"]) == -1
    assert dd.runs[0]["fields"]["task_id"] == "tube-pick" and dd.runs[1]["fields"]["task_id"] == UNKNOWN


def test_fields_not_given_are_unknown(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        r = record(log, [{}, {}, {}])
    ds.finalize()
    m = store.read_manifest(r.run_path)
    assert (m["task_id"], m["task_rev"], m["planner"]) == (UNKNOWN, UNKNOWN, UNKNOWN)
    t = pq.read_table(r.run_path / "ticks.parquet").to_pydict()
    assert t["prompt"] == [UNKNOWN] * 3 and t["subtask_cmd"] == [UNKNOWN] * 3
    assert t["prompt_rev"] == [-1] * 3 and t["control_gen"] == [-1] * 3
    assert m["policy"]["family"] == "smolvla"
    assert V.validate(ds.root)["ok"]


def test_no_planner_is_null_and_kept_apart_from_unknown(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        r0 = record(log, [{}, {}])
        with log.run(task=TASK, policy=PolicyInfo("smolvla", "x")) as run:
            r1 = run
            add(run, 0)
            with pytest.raises(ValueError, match="no planner"):
                add(run, 1, subtask_cmd="st_01")
            assert len(run._rows) == 1 and adapter.buffer_size(ds) == 1   # refused before LeRobot saw it
            add(run, 1, subtask_cmd=None)
            run.end(termination_reason="success")
        r2 = record(log, [{"subtask_cmd": "st_01"}], {"planner": RULES})    # a run can name its own planner
        r3 = record(log, [{}], {"planner": UNKNOWN})
    ds.finalize()
    m0, m2, m3 = (store.read_manifest(r.run_path) for r in (r0, r2, r3))
    assert m0["planner"] is None and "planner" in m0
    # an omitted subtask_cmd stays unknown even without a planner; None only when the caller says so
    assert pq.read_table(r0.run_path / "ticks.parquet").column("subtask_cmd").to_pylist() == [UNKNOWN, UNKNOWN]
    assert pq.read_table(r1.run_path / "ticks.parquet").column("subtask_cmd").to_pylist() == [UNKNOWN, None]
    assert m2["planner"] == {"name": "Rules", "id": "rules-v3"} and m3["planner"] == UNKNOWN
    assert pq.read_table(r3.run_path / "ticks.parquet").column("subtask_cmd").to_pylist() == [UNKNOWN]
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]


@pytest.mark.parametrize("kw", [{"task_id": ""}, {"task_id": None}, {"task_rev": -1}, {"task_rev": True},
                                {"task_rev": "3"}, {"task_rev": 2.0}, {"planner": "rules-v3"},
                                {"planner": {"name": "Rules"}}, {"planner": {"name": "Rules", "id": ""}},
                                {"planner": {"name": "Rules", "id": "r", "extra": 1}}])
def test_bad_run_fields_are_refused(tmp_path, kw):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with pytest.raises(ValueError):
            with log.run(task=TASK, **kw):
                pass
        r = record(log, [{}])                                  # the writer is still usable
    assert r.state == "committed"


def test_bad_planner_on_the_writer_is_refused(tmp_path):
    ds = make_ds(tmp_path)
    for bad in ("Rules", {"id": "x"}, 3):
        with pytest.raises(ValueError):
            RunLogger(ds, SCHEMA, planner=bad)
    with pytest.raises(ValueError):
        PlannerInfo(name="", id="x")


@pytest.mark.parametrize("kw", [{"prompt": None}, {"prompt": 3}, {"prompt_rev": -1},
                                {"prompt_rev": 1.5}, {"prompt_rev": True}, {"control_gen": -1},
                                {"control_gen": "2"}, {"subtask_cmd": ""}, {"subtask_cmd": 2}])
def test_bad_tick_fields_are_refused_before_lerobot(tmp_path, kw):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, policy=PolicyInfo("smolvla", "x")) as run:
            add(run, 0)
            with pytest.raises(ValueError):
                add(run, 1, **kw)
            assert len(run._rows) == 1 and adapter.buffer_size(ds) == 1
            run.end(termination_reason="success")
    ds.finalize()
    assert V.validate(ds.root)["ok"]


@pytest.mark.parametrize("ticks,match", [
    ([{"prompt_rev": 2}, {"prompt_rev": 1}], "prompt_rev went down"),
    ([{"prompt_rev": 2}, {}, {"prompt_rev": 1}], "prompt_rev went down"),       # unknown ticks in between
    ([{"control_gen": 3}, {"control_gen": 2}], "control_gen went down"),
    ([{"prompt": "a", "prompt_rev": 4}, {"prompt": "b", "prompt_rev": 4}], "without a new prompt_rev"),
    ([{"prompt": "a", "prompt_rev": 4}, {"prompt_rev": 4}, {"prompt": "b", "prompt_rev": 4}], "without a new"),
])
def test_counters_and_prompt_follow_on(tmp_path, ticks, match):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK, policy=PolicyInfo("smolvla", "x")) as run:
            for i, kw in enumerate(ticks[:-1]):
                add(run, i, **kw)
            with pytest.raises(ValueError, match=match):
                add(run, len(ticks) - 1, **ticks[-1])
            run.end(termination_reason="success")
    ds.finalize()
    assert V.validate(ds.root)["ok"]


def test_counters_may_stay_skip_and_restart_across_runs(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        record(log, [{"prompt": "a", "prompt_rev": 5, "control_gen": 7}, {"prompt_rev": 5},
                     {"prompt": "a", "prompt_rev": 5}, {"prompt": "b", "prompt_rev": 9, "control_gen": 9}])
        record(log, [{"prompt": "b", "prompt_rev": 0, "control_gen": 0}])   # counters belong to one run's ticks
    ds.finalize()
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]


def _tamper_ticks(run_path, change):
    t = pq.read_table(run_path / "ticks.parquet")
    d = {k: t.column(k).to_pylist() for k in t.column_names}
    types = {f.name: f.type for f in t.schema}
    change(d, types)
    data = store.table_bytes(pa.table({k: pa.array(v, types.get(k)) for k, v in d.items()}))
    store.atomic_write_bytes(run_path / "ticks.parquet", data)
    m = store.read_manifest(run_path)
    m["integrity"]["ticks_sha256"] = store.sha256(data)        # a matching digest: only the content rules catch it
    store.atomic_write_json(run_path / "manifest.json", m)


def _set(col, i, v):
    def f(d, types):
        d[col][i] = v
    return f


def _drop(col):
    def f(d, types):
        d.pop(col)
    return f


def _as_string(col):
    def f(d, types):
        d[col] = [str(x) for x in d[col]]
        types[col] = pa.string()
    return f


@pytest.mark.parametrize("change,match", [
    (_set("prompt_rev", 3, 0), "prompt_rev went down"),
    (_set("control_gen", 1, 5), "control_gen went down"),
    (_set("prompt", 1, "other"), "without a new prompt_rev"),
    (_set("prompt", 0, None), "prompt is null"),
    (_set("subtask_cmd", 0, ""), "subtask_cmd is an empty string"),
    (_set("prompt_rev", 0, -2), "prompt_rev must be >= 0"),
    (_set("control_gen", 0, None), "control_gen must be >= 0"),
    (_drop("control_gen"), "only some of the columns"),
    (_as_string("prompt_rev"), "prompt_rev is not an integer column"),
])
def test_validate_checks_the_tick_fields(tmp_path, change, match):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=RULES) as log:
        r = record(log, PLANNED)
    ds.finalize()
    _tamper_ticks(r.run_path, change)
    errs = "\n".join(V.validate(ds.root)["errors"])
    assert match in errs, errs


def test_validate_checks_subtask_cmd_against_no_planner(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        r = record(log, [{}, {}])
    ds.finalize()
    _tamper_ticks(r.run_path, _set("subtask_cmd", 1, "st_02"))
    assert any("no planner" in e for e in V.validate(ds.root)["errors"])


@pytest.mark.parametrize("key,value,match", [
    ("task_id", "", "task_id must be"), ("task_id", None, "task_id must be"), ("task_rev", -3, "task_rev must be"),
    ("task_rev", True, "task_rev must be"), ("task_rev", None, "task_rev must be"),
    ("planner", "rules", "planner must be"), ("planner", {"name": "x"}, "planner must be"),
    ("policy", "smolvla", "policy must be"),
])
def test_validate_checks_the_manifest_fields(tmp_path, key, value, match):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        r = record(log, [{}])
    ds.finalize()
    m = store.read_manifest(r.run_path)
    m[key] = value
    store.atomic_write_json(r.run_path / "manifest.json", m)
    errs = "\n".join(V.validate(ds.root)["errors"])
    assert match in errs, errs


# ── datasets written before L2 ─────────────────────────────────────────────────────────────────────────────


def test_old_dataset_keeps_loading_and_reads_as_unknown(tmp_path):
    root = tmp_path / "old"
    shutil.copytree(OLD_DATASET, root)
    res = V.validate(root)
    assert res["ok"], res["errors"]
    for run_path, m in store.iter_manifests(root, "committed"):
        assert not any(k in m for k in ("task_id", "task_rev", "planner"))
        f = manifest_fields(m)
        assert (f["task_id"], f["task_rev"], f["planner"]) == (UNKNOWN, UNKNOWN, UNKNOWN)
        assert f["policy"]["family"] == "smolvla"               # a recorded policy is kept
        t = pq.read_table(run_path / "ticks.parquet")
        assert "prompt" not in t.column_names
        p = planned_columns(t)
        assert p["prompt"] == [UNKNOWN] * t.num_rows and p["subtask_cmd"] == [UNKNOWN] * t.num_rows
        assert p["prompt_rev"].tolist() == [-1] * t.num_rows and p["control_gen"].tolist() == [-1] * t.num_rows


def test_old_dataset_reader_gives_unknown_tick_fields(tmp_path):
    pytest.importorskip("torch")
    from aftershock_logger.reader import DataopDataset
    root = tmp_path / "old"
    shutil.copytree(OLD_DATASET, root)
    dd = DataopDataset(lerobot_dataset.LeRobotDataset("test/old_v0_1_0", root=root))
    item = dd[3]
    assert item["dataop.prompt"] == UNKNOWN and item["dataop.subtask_cmd"] == UNKNOWN
    assert int(item["dataop.prompt_rev"]) == -1 and int(item["dataop.control_gen"]) == -1


def test_old_and_new_runs_in_one_dataset(tmp_path):
    root = tmp_path / "old"
    shutil.copytree(OLD_DATASET, root)
    ds = lerobot_dataset.LeRobotDataset.resume("test/old_v0_1_0", root=root)
    with RunLogger(ds, SCHEMA, planner=RULES) as log:
        r = record(log, PLANNED, {"task_id": "tube-pick", "task_rev": 1})
    ds.finalize()
    res = V.validate(root)
    assert res["ok"], res["errors"]
    assert store.read_manifest(r.run_path)["episode_index"] == 2
    assert len(B.summary(root)) >= 1


def test_legacy_null_policy_reads_as_unknown_only_without_task_rev():
    old = {"policy": None}
    assert manifest_fields(old)["policy"] == UNKNOWN                    # before L2, null meant not recorded
    assert manifest_fields({"policy": None, "task_rev": UNKNOWN})["policy"] is None   # after L2, null is no policy
    assert manifest_fields({"policy": None, "task_rev": 2})["policy"] is None
    assert manifest_fields({"policy": UNKNOWN, "task_rev": 2})["policy"] == UNKNOWN
    assert manifest_fields({})["planner"] == UNKNOWN


# ── imports ────────────────────────────────────────────────────────────────────────────────────────────────


def _plain(tmp_path, n=6):
    ds = make_ds(tmp_path, "plain", {"intervention": {"dtype": "bool", "shape": (1,), "names": None}})
    for i in range(n):
        a = np.arange(6, dtype=np.float32) + i
        ds.add_frame({"observation.state": a, "action": a, "task": TASK, "intervention": np.array([i >= 4])})
    ds.save_episode()
    ds.finalize()
    return ds


def _imported(root):
    (run_path, m), = list(store.iter_manifests(root, "committed"))
    return m, pq.read_table(run_path / "ticks.parquet").to_pydict()


def test_import_records_what_it_cannot_know_as_unknown(tmp_path):
    ds = _plain(tmp_path)
    importer.import_dataset(str(ds.root))
    m, t = _imported(ds.root)
    assert (m["task_id"], m["task_rev"], m["planner"], m["policy"]) == (UNKNOWN,) * 4
    assert m["termination_reason"] == UNKNOWN and set(t["control_mode"]) == {UNKNOWN}
    assert set(t["prompt"]) == {UNKNOWN} and set(t["subtask_cmd"]) == {UNKNOWN}
    assert set(t["prompt_rev"]) == {-1} and set(t["control_gen"]) == {-1}
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert manifest_fields(m)["policy"] == UNKNOWN
    assert B.summary(ds.root)[0]["policy"] is None              # batches group an unknown policy as before


def test_import_watcher_style_without_policy_mapping_keeps_policy_unknown(tmp_path):
    ds = _plain(tmp_path)
    importer.import_dataset(str(ds.root), mapping={"control": {"column": "intervention"},
                                                   "base_action": "sent_on_policy_frames"})
    m, t = _imported(ds.root)
    assert m["policy"] == UNKNOWN and m["planner"] == UNKNOWN
    assert t["control_mode"] == ["base"] * 4 + ["expert"] * 2
    assert V.validate(ds.root)["ok"]


def test_import_demos_know_no_policy_or_planner_ran(tmp_path):
    ds = _plain(tmp_path)
    importer.import_dataset(str(ds.root), preset="demos")
    m, t = _imported(ds.root)
    assert m["policy"] is None and m["planner"] is None and "task_rev" in m
    assert manifest_fields(m)["policy"] is None                 # not legacy: null stays "no policy"
    assert set(t["subtask_cmd"]) == {UNKNOWN} and set(t["prompt"]) == {UNKNOWN}
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]


def test_import_with_a_mapped_policy_records_it(tmp_path):
    ds = _plain(tmp_path)
    importer.import_dataset(str(ds.root), mapping={"control": {"column": "intervention"},
                                                   "base_action": "sent_on_policy_frames",
                                                   "policy": {"family": "pi05", "source": "org/pi05"}})
    m, _ = _imported(ds.root)
    assert m["policy"]["family"] == "pi05" and m["planner"] == UNKNOWN
    assert B.summary(ds.root)[0]["policy"]["source"] == "org/pi05"


def test_manifest_json_round_trip_keeps_unknown_null_and_values_apart(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        r = record(log, [{}], {"task_id": "tube-pick"})
    ds.finalize()
    raw = json.loads((r.run_path / "manifest.json").read_text())
    assert raw["planner"] is None and raw["task_rev"] == UNKNOWN and raw["task_id"] == "tube-pick"


# ── round 1 review ─────────────────────────────────────────────────────────────────────────────────────────


def test_empty_prompt_is_kept(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        r = record(log, [{"prompt": "", "prompt_rev": 0}, {"prompt": "", "prompt_rev": 0}])
    ds.finalize()
    assert pq.read_table(r.run_path / "ticks.parquet").column("prompt").to_pylist() == ["", ""]
    assert V.validate(ds.root)["ok"]


def test_policy_not_given_is_unknown_and_none_means_no_policy(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        with log.run(task=TASK) as run:                          # Watch mode that does not name its policy
            add(run, 0)
            run.end(termination_reason="success")
        unnamed = run
        with log.run(task=TASK, policy=None) as run:             # a teleop demo: no policy ran
            with pytest.raises(ValueError, match="policy=None"):
                add(run, 0)
            a = np.zeros(6, np.float32)
            with pytest.raises(ValueError, match="policy=None"):
                run.add_frame({"observation.state": a, "task": TASK}, sent_action=a, expert_action=a,
                              expert_source="human", base_action=a, inference_id=0, chunk_offset=0,
                              control_mode="expert", **ts())
            assert adapter.buffer_size(ds) == 0
            run.add_frame({"observation.state": a, "task": TASK}, sent_action=a, expert_action=a,
                          expert_source="human", control_mode="expert", **ts())
            run.end(termination_reason="success")
        demo = run
        with pytest.raises(ValueError, match="policy must be"):
            with log.run(task=TASK, policy="smolvla"):
                pass
    ds.finalize()
    assert store.read_manifest(unnamed.run_path)["policy"] == UNKNOWN
    m = store.read_manifest(demo.run_path)
    assert m["policy"] is None and manifest_fields(m)["policy"] is None
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert sum(b["n_runs"] for b in B.summary(ds.root)) == 2


def test_no_planner_means_the_prompt_never_changes(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        with log.run(task=TASK, policy=PolicyInfo("smolvla", "x")) as run:
            add(run, 0, prompt="reach", prompt_rev=0)
            add(run, 1, prompt=UNKNOWN)
            with pytest.raises(ValueError, match="planner is present"):
                add(run, 2, prompt="grasp", prompt_rev=1)
            add(run, 2, prompt="reach", prompt_rev=0)
            run.end(termination_reason="success")
        r = run
    ds.finalize()
    assert V.validate(ds.root)["ok"]
    _tamper_ticks(r.run_path, lambda d, types: (d["prompt"].__setitem__(2, "grasp"),
                                                d["prompt_rev"].__setitem__(2, 1)))
    assert any("planner is present" in e for e in V.validate(ds.root)["errors"])


def test_tick_records_use_the_unknown_literal(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=RULES) as log:
        r = record(log, [{"prompt": "a", "prompt_rev": 2, "control_gen": 0, "subtask_cmd": "st_01"},
                         {"subtask_cmd": None}])
    ds.finalize()
    rows = tick_records(r.run_path)
    assert [(x["prompt"], x["prompt_rev"], x["subtask_cmd"], x["control_gen"]) for x in rows] == \
        [("a", 2, "st_01", 0), (UNKNOWN, UNKNOWN, None, UNKNOWN)]
    assert rows[0]["control_mode"] == "base" and len(rows[0]["sent_action"]) == 6
    root = tmp_path / "old"
    shutil.copytree(OLD_DATASET, root)
    for run_path, _ in store.iter_manifests(root, "committed"):
        for x in tick_records(run_path):
            assert (x["prompt"], x["prompt_rev"], x["subtask_cmd"], x["control_gen"]) == (UNKNOWN,) * 4


def test_import_all_expert_mapping_without_declarations_stays_unknown(tmp_path):
    ds = _plain(tmp_path)                    # a scripted skill drove every frame; a planner may have run
    importer.import_dataset(str(ds.root), mapping={"control": {"constant": "expert"}, "expert_source": "skill:pick"})
    m, t = _imported(ds.root)
    assert m["policy"] == UNKNOWN and m["planner"] == UNKNOWN and set(t["subtask_cmd"]) == {UNKNOWN}
    assert m["origin"]["fields"]["policy"] == "unknown" and m["origin"]["fields"]["planner"] == "unknown"
    assert V.validate(ds.root)["ok"]


def test_import_declarations_for_policy_and_planner(tmp_path):
    ds = _plain(tmp_path)
    rep = importer.import_dataset(str(ds.root), mapping={"control": {"column": "intervention"},
                                                         "base_action": "sent_on_policy_frames", "policy": None})
    assert rep["runs"] == [] and "no policy ran" in rep["skipped"][0]["why"]     # the policy drove frames 0 to 3
    importer.import_dataset(str(ds.root), mapping={"control": {"column": "intervention"},
                                                   "base_action": "sent_on_policy_frames",
                                                   "planner": {"name": "Rules", "id": "rules-v3"}})
    m, _ = _imported(ds.root)
    assert m["planner"] == {"name": "Rules", "id": "rules-v3"} and m["origin"]["fields"]["planner"] == "mapping"
    assert V.validate(ds.root)["ok"]
    for bad in ({"planner": "Rules"}, {"planner": {"name": "Rules"}}, {"policy": "smolvla"}):
        with pytest.raises(importer.ImportError_):
            importer.import_dataset(str(ds.root), mapping=bad, dry_run=True)


def test_viewer_lists_imports_with_unknown_and_null_policy(tmp_path):
    from aftershock_logger import viewer
    ds = _plain(tmp_path)
    importer.import_dataset(str(ds.root))
    runs = viewer.collect(ds.root)["runs"]
    assert len(runs) == 1 and runs[0]["policy"] is None
    ds2 = _plain(tmp_path / "b")
    importer.import_dataset(str(ds2.root), preset="demos")
    assert viewer.collect(ds2.root)["runs"][0]["policy"] is None
