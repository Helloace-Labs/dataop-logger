"""Logger task L3: one file per batch, pack identity and mixed_policy, and frozen membership of locked batches.

Batches live in dataop/batches/<batch_id>.json. Datasets written before L3 keep one dataop/batches.json; it reads the
same, and the first change moves it to per-batch files (the old file is renamed, never deleted).
"""
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aftershock_logger import UNKNOWN, ActionSchema, PlannerInfo, PolicyInfo, RunLogger  # noqa: E402
from aftershock_logger import batches as B  # noqa: E402
from aftershock_logger import review as rv  # noqa: E402
from aftershock_logger import store  # noqa: E402
from aftershock_logger import validate as V  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCHEMA = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
TASK = "pick up the tube"
POL_A = PolicyInfo("smolvla", "train/smolvla_100")
POL_B = PolicyInfo("pi05", "train/pi05_lora", revision="abc123")
# a pre-L3 dataset to test on, read from a temporary copy; set to the dataset root (the folder holding dataop/)
LEGACY_FIXTURE = os.environ.get("AFTERSHOCK_LOGGER_LEGACY_BATCHES_FIXTURE")


def make_ds(tmp_path, name="ds"):
    feats = {k: {"dtype": "float32", "shape": (6,), "names": list(NAMES)} for k in ("observation.state", "action")}
    return lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features=feats, root=tmp_path / name,
                                                 use_videos=False)


def record(log, n=6, policy=POL_A, **kw):
    with log.run(task=TASK, policy=policy, **kw) as run:
        for i in range(n):
            a = np.arange(6, dtype=np.float32) + i
            t = time.monotonic_ns()
            if policy is None:
                run.add_frame({"observation.state": np.full(6, i, np.float32), "task": TASK}, sent_action=a,
                              expert_action=a, control_mode="expert", expert_source="teleop:leader", obs_ns=t,
                              sent_ns=t + 1000)
            else:
                run.add_frame({"observation.state": np.full(6, i, np.float32), "task": TASK}, sent_action=a,
                              base_action=a, control_mode="base", inference_id=i, chunk_offset=0, obs_ns=t,
                              sent_ns=t + 1000)
        run.end(termination_reason="success")
    return run


def files(root):
    return sorted(p.name for p in (Path(root) / "dataop" / "batches").glob("[!.]*.json"))


def on_disk(root, bid):
    return json.loads((Path(root) / "dataop" / "batches" / f"{bid}.json").read_text())


def snapshot(root):
    d = Path(root) / "dataop"
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*"))
            if p.is_file() and not p.name.endswith(".lock")}


def to_legacy(root):
    """Rewrite the registry as the single batches.json a logger before task L3 wrote."""
    d = B._read(root)
    old = {"version": 1, "rev": d["rev"], "members": d["members"],
           "batches": [{k: v for k, v in b.items() if k not in ("seq", "identity", "mixed_policy")}
                       for b in d["batches"]]}
    shutil.rmtree(Path(root) / "dataop" / "batches")
    (Path(root) / "dataop" / "batches.json").write_text(json.dumps(old))
    return old


def review_all(ds):
    for p, _ in store.iter_manifests(ds.root, "committed"):
        rv.set_reviewed(p, True)


def two_batches(tmp_path):
    """Batch "a" (2 runs, locked) and batch "b" (1 run, reviewing)."""
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, batch="a") as log:
        a1, a2 = record(log), record(log)
    with RunLogger(ds, SCHEMA, batch="b") as log:
        b1 = record(log)
    ds.finalize()
    review_all(ds)
    B.set_status(ds.root, "a", "locked")
    return ds, a1, a2, b1


def test_each_batch_is_its_own_file(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, batch="round 1") as log:
        r1, r2 = record(log), record(log)
    ds.finalize()
    assert not (ds.root / "dataop" / "batches.json").exists()
    d = B._read(ds.root)
    (b,) = d["batches"]
    assert files(ds.root) == [f"{b['batch_id']}.json"]
    f = on_disk(ds.root, b["batch_id"])
    assert f["version"] == 1 and f["runs"] == sorted([r1.run_id, r2.run_id]) and f["registry_rev"] == d["rev"]
    assert f["identity"] == {"station": UNKNOWN, "task": UNKNOWN, "planner": UNKNOWN,
                             "policy": {"family": "smolvla", "source": "train/smolvla_100", "revision": None}}
    assert f["mixed_policy"] is False
    assert not (ds.root / "dataop" / "batches" / ".journal.json").exists()
    assert V.validate(ds.root)["ok"]


def test_identity_keeps_null_and_unknown_apart_and_uses_task_and_planner_ids(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, planner=PlannerInfo("rules", "rules-v3")) as log:
        r = record(log, task_id="tube-pick", task_rev=2)
    with RunLogger(ds, SCHEMA, planner=None) as log:
        t = record(log, policy=None, task_id="tube-pick", task_rev=2)
    ds.finalize()
    i1 = B.run_identity(store.read_manifest(r.run_path))
    i2 = B.run_identity(store.read_manifest(t.run_path))
    assert i1["task"] == "tube-pick" and i1["planner"] == "rules-v3" and i1["station"] == UNKNOWN
    assert i2["policy"] is None and i2["planner"] is None
    assert B.run_identity({"policy": None})["policy"] == UNKNOWN      # before L2, a null policy was not recorded
    assert B.run_identity({"policy": None, "task_rev": 1})["policy"] is None


def test_a_run_with_another_identity_opens_a_new_batch(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        a1, a2 = record(log), record(log)
        b1 = record(log, policy=POL_B)
        a3 = record(log)
    ds.finalize()
    s = B.summary(ds.root)                                  # newest first
    assert [x["n_runs"] for x in s] == [1, 1, 2] and [x["status"] for x in s] == ["collecting", "reviewing", "reviewing"]
    assert s[1]["runs"] == [b1.run_id] and s[0]["runs"] == [a3.run_id]
    assert s[0]["name"] != s[2]["name"]                     # the logger never repeats a name it chose
    assert not any(x["mixed_policy"] for x in s)
    assert "another station, task, policy or planner" in s[2]["history"][-1]["what"]
    assert V.validate(ds.root)["ok"]


def test_a_named_batch_keeps_a_different_run_and_flags_mixed_policy(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, batch="round 3") as log:
        record(log)
        record(log, policy=POL_B)
    ds.finalize()
    (s,) = B.summary(ds.root)
    assert s["n_runs"] == 2 and s["mixed_policy"] is True
    assert on_disk(ds.root, s["batch_id"])["mixed_policy"] is True
    assert s["identity"]["policy"]["source"] == "train/smolvla_100"


def test_moving_a_run_out_of_a_locked_batch_fails_and_changes_nothing(tmp_path):
    ds, a1, a2, b1 = two_batches(tmp_path)
    before = snapshot(ds.root)
    with pytest.raises(B.BatchError, match=r"batch 'a' is locked, so its runs are frozen: run .* cannot move out"):
        B.move(ds.root, [a1.run_id], "b")
    with pytest.raises(B.BatchError, match="locked, so its runs are frozen: no run can move into it"):
        B.move(ds.root, [b1.run_id], "a")
    with pytest.raises(B.BatchError, match="frozen"):
        B.add_runs(ds.root, [b1.run_id], "a")
    with pytest.raises(B.BatchError, match="frozen"):       # an import cannot pull a run out of a locked batch
        B.add_runs(ds.root, [a1.run_id], "b")
    assert snapshot(ds.root) == before
    B.set_status(ds.root, "a", "reviewing", reason="regroup")
    B.move(ds.root, [a1.run_id], "b")                       # allowed once unlocked
    assert B._read(ds.root)["members"][a1.run_id] == B.find(B._read(ds.root), "b")["batch_id"]


def test_a_move_updates_both_files_and_marks_source_training_sets_out_of_date(tmp_path):
    ds, a1, a2, b1 = two_batches(tmp_path)
    B.set_status(ds.root, "a", "reviewing", reason="regroup")
    d = B._read(ds.root)
    a, b = B.find(d, "a"), B.find(d, "b")
    B.add_export([(ds.root, a["batch_id"], None)], {"export_id": "x1", "recipe": "dagger", "path": "/out"})
    rev = B._read(ds.root)["rev"]
    with pytest.raises(B.BatchError, match="changed"):
        B.move(ds.root, [a1.run_id], "b", expect_rev=rev - 1)
    B.move(ds.root, [a1.run_id], "b", by="tester", expect_rev=rev)
    fa, fb = on_disk(ds.root, a["batch_id"]), on_disk(ds.root, b["batch_id"])
    assert fa["runs"] == [a2.run_id] and fb["runs"] == sorted([a1.run_id, b1.run_id])
    assert fa["registry_rev"] == fb["registry_rev"] == rev + 1 == B._read(ds.root)["rev"]
    assert all(e["stale"] for e in fa["exports"])
    assert "moved out to 'b'" in fa["history"][-1]["what"] and "moved in" in fb["history"][-1]["what"]
    assert V.validate(ds.root)["ok"]


def test_placement_never_puts_a_run_into_a_locked_batch(tmp_path):
    ds, a1, a2, c1 = two_batches(tmp_path)
    d = B._read(ds.root)
    a, c = B.find(d, "a"), B.find(d, "b")
    m = store.read_manifest(c1.run_path)                   # a run that names the locked batch but never joined it
    m["batch"] = {"batch_id": a["batch_id"], "name": "a"}
    store.atomic_write_json(c1.run_path / "manifest.json", m)
    f = on_disk(ds.root, c["batch_id"])
    f["runs"] = []
    store.atomic_write_json(ds.root / "dataop" / "batches" / f"{c['batch_id']}.json", f)
    d = B.load(ds.root)
    assert d["members"][c1.run_id] == c["batch_id"]        # placed with its own kind, not into the locked batch
    assert on_disk(ds.root, a["batch_id"])["runs"] == sorted([a1.run_id, a2.run_id])
    assert V.validate(ds.root)["ok"]


def test_old_batches_json_reads_the_same_then_moves_to_per_batch_files(tmp_path):
    ds, a1, a2, b1 = two_batches(tmp_path)
    B.add_export([(ds.root, "b", None)], {"export_id": "x1", "recipe": "dagger", "path": "/out"})
    expected = B.summary(ds.root)
    old = to_legacy(ds.root)
    legacy = ds.root / "dataop" / "batches.json"
    raw = legacy.read_bytes()

    os.chmod(ds.root / "dataop", 0o555)                     # read only: read in place, nothing written
    try:
        assert B.summary(ds.root) == expected
        assert B.load(ds.root)["members"] == old["members"]
        with pytest.raises(rv.ReviewError, match="locked"):  # the old file's lock still holds
            rv.set_reviewed(a1.run_path, False)
    finally:
        os.chmod(ds.root / "dataop", 0o755)
    assert legacy.read_bytes() == raw and not (ds.root / "dataop" / "batches").exists()
    assert V.validate(ds.root)["ok"]

    d = B.load(ds.root)                                     # writable: the first read moves it
    assert not legacy.exists() and (ds.root / "dataop" / "batches.json.migrated").read_bytes() == raw
    assert files(ds.root) == sorted(f"{b['batch_id']}.json" for b in old["batches"])
    assert d["members"] == old["members"] and d["rev"] == old["rev"] + 1
    for b in old["batches"]:
        f = on_disk(ds.root, b["batch_id"])
        assert {k: f[k] for k in b} == b                      # every field kept exactly
        assert f["runs"] == sorted(r for r, x in old["members"].items() if x == b["batch_id"])
    assert B.summary(ds.root) == expected
    assert V.validate(ds.root)["ok"]
    with pytest.raises(B.BatchError, match="frozen"):
        B.move(ds.root, [a1.run_id], "b")


def test_old_batches_json_moves_on_the_first_change_too(tmp_path):
    ds, a1, a2, b1 = two_batches(tmp_path)
    to_legacy(ds.root)
    B.edit(ds.root, "b", purpose="second round")
    assert not (ds.root / "dataop" / "batches.json").exists()
    assert len(files(ds.root)) == 2 and on_disk(ds.root, B.find(B._read(ds.root), "b")["batch_id"])["purpose"] \
        == "second round"


def test_an_interrupted_change_is_seen_whole_and_finished_by_the_next_writer(tmp_path, monkeypatch):
    ds, a1, a2, b1 = two_batches(tmp_path)
    B.set_status(ds.root, "a", "reviewing", reason="regroup")
    real, calls = store.atomic_write_json, []

    def crash_after_journal_and_one_file(path, obj):
        calls.append(Path(path).name)
        if len(calls) == 3:
            raise OSError("power cut")
        return real(path, obj)
    monkeypatch.setattr(store, "atomic_write_json", crash_after_journal_and_one_file)
    with pytest.raises(OSError):
        B.move(ds.root, [a1.run_id], "b")
    monkeypatch.setattr(store, "atomic_write_json", real)
    assert calls[0] == ".journal.json"
    assert (ds.root / "dataop" / "batches" / ".journal.json").exists()
    d = B._read(ds.root)                                    # the journal is overlaid: the move is seen whole
    b = B.find(d, "b")
    assert d["members"][a1.run_id] == b["batch_id"]
    assert sorted(r for r, x in d["members"].items() if x == b["batch_id"]) == sorted([a1.run_id, b1.run_id])
    B.edit(ds.root, "a", purpose="x")                       # the next writer finishes it first
    assert not (ds.root / "dataop" / "batches" / ".journal.json").exists()
    assert on_disk(ds.root, b["batch_id"])["runs"] == sorted([a1.run_id, b1.run_id])
    assert V.validate(ds.root)["ok"]


def test_a_run_in_two_batch_files_is_an_error_and_a_locked_copy_wins(tmp_path):
    ds, a1, a2, b1 = two_batches(tmp_path)
    bid = B.find(B._read(ds.root), "b")["batch_id"]
    f = on_disk(ds.root, bid)
    f["runs"] = sorted(f["runs"] + [a1.run_id])
    (ds.root / "dataop" / "batches" / f"{bid}.json").write_text(json.dumps(f))
    res = V.validate(ds.root)
    assert not res["ok"] and any("several batches" in e for e in res["errors"])
    with pytest.raises(rv.ReviewError, match="locked"):
        rv.set_reviewed(a1.run_path, False)
    with pytest.raises(B.BatchError, match="several batches"):
        B.move(ds.root, [b1.run_id], "b")


def test_a_file_named_for_another_batch_is_an_error(tmp_path):
    ds, a1, a2, b1 = two_batches(tmp_path)
    bid = B.find(B._read(ds.root), "b")["batch_id"]
    src = ds.root / "dataop" / "batches" / f"{bid}.json"
    os.replace(src, src.with_name("0123456789ab.json"))
    res = V.validate(ds.root)
    assert not res["ok"] and any("file name must be its batch id" in e for e in res["errors"])


def test_remap_copies_the_batch_files(tmp_path):
    ds, a1, a2, b1 = two_batches(tmp_path)
    new = tmp_path / "copy"
    shutil.copytree(ds.root, new, ignore=shutil.ignore_patterns("dataop"))
    rep = V.remap(ds.root, new)
    assert any(line.startswith("copied 2 batch") for line in rep)
    assert files(new) == files(ds.root)
    assert B._read(new)["members"] == B._read(ds.root)["members"]
    assert B.copy_to(ds.root, new) == "batches already match"
    B.edit(new, "b", purpose="changed here")
    assert "kept the destination's batches" in B.copy_to(ds.root, new)


def test_a_hint_with_an_unusable_batch_id_gets_a_new_one(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, batch="r") as log:
        r = record(log)
    ds.finalize()
    shutil.rmtree(ds.root / "dataop" / "batches")
    m = store.read_manifest(r.run_path)
    m["batch"] = {"batch_id": "../Escape", "name": "r"}
    store.atomic_write_json(r.run_path / "manifest.json", m)
    d = B.load(ds.root)
    (b,) = d["batches"]
    assert b["name"] == "r" and B.BATCH_ID.match(b["batch_id"]) and files(ds.root) == [f"{b['batch_id']}.json"]


@pytest.mark.skipif(not LEGACY_FIXTURE, reason="set AFTERSHOCK_LOGGER_LEGACY_BATCHES_FIXTURE to a pre-L3 dataset")
def test_a_real_pre_l3_dataset_reads_the_same_on_a_copy(tmp_path):
    src = Path(LEGACY_FIXTURE).expanduser()
    old = json.loads((src / "dataop" / "batches.json").read_text())
    root = tmp_path / "copy"
    shutil.copytree(src / "dataop", root / "dataop", ignore=shutil.ignore_patterns("*.lock"))
    before = B._read(root)
    assert before["members"] == old["members"] and before["rev"] == old["rev"]
    expected = B.summary(root)                              # migrates on this copy
    assert not (root / "dataop" / "batches.json").exists()
    d = B._read(root)
    assert d["members"] == old["members"] and d["rev"] == old["rev"] + 1
    for b in old["batches"]:
        f = on_disk(root, b["batch_id"])
        assert {k: f[k] for k in b} == b
    assert B.summary(root) == expected and B.check(root) == []
    assert len(expected) == len(old["batches"]) and sum(x["n_runs"] for x in expected) == len(old["members"])


def test_pack_fields_kind_station_task_planner(tmp_path):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, batch="policy runs", planner=PlannerInfo("rules", "rules-v3")) as log:
        record(log, task_id="tube-pick", task_rev=1)
    with RunLogger(ds, SCHEMA, batch="demos", planner=None) as log:
        record(log, policy=None, task_id="tube-pick", task_rev=1)
    ds.finalize()
    d = B._read(ds.root)
    p, t = (on_disk(ds.root, B.find(d, n)["batch_id"]) for n in ("policy runs", "demos"))
    assert (p["kind"], p["station"], p["task"], p["planner"]) == ("model_run", UNKNOWN, "tube-pick", "rules-v3")
    assert (t["kind"], t["planner"]) == ("teleop", None)
    to_legacy(ds.root)                                      # an old file gets them on its move
    B.load(ds.root)
    assert on_disk(ds.root, p["batch_id"])["kind"] == "model_run" and on_disk(ds.root, t["batch_id"])["kind"] == "teleop"


def test_a_review_edit_waits_for_a_batch_change(tmp_path):
    import fcntl
    import threading
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA, batch="r") as log:
        r = record(log)
    ds.finalize()
    done = threading.Event()
    with open(ds.root / "dataop" / "batches.lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)                      # as a lock or move in progress holds it
        t = threading.Thread(target=lambda: (rv.set_reviewed(r.run_path, True), done.set()))
        t.start()
        time.sleep(0.3)
        assert not done.is_set()
        fcntl.flock(lk, fcntl.LOCK_UN)
    t.join(5)
    assert done.is_set()


def test_a_reader_racing_the_move_off_the_old_file_still_sees_the_lock(tmp_path, monkeypatch):
    ds, a1, a2, b1 = two_batches(tmp_path)
    to_legacy(ds.root)
    real = B._read_json

    def migrate_right_after_reading_the_old_file(p):
        out = real(p)
        if p.name == "batches.json":
            monkeypatch.setattr(B, "_read_json", real)
            B.load(ds.root)                                 # another process moves it to per-batch files now
        return out
    monkeypatch.setattr(B, "_read_json", migrate_right_after_reading_the_old_file)
    assert B.batch_of(ds.root, a1.run_id)["status"] == "locked"
    assert not (ds.root / "dataop" / "batches.json").exists()


def test_a_run_placed_before_its_join_goes_where_join_would_put_it(tmp_path, monkeypatch):
    ds = make_ds(tmp_path)
    with RunLogger(ds, SCHEMA) as log:
        record(log)
    B.new(ds.root, "later")                                 # the first batch is now closed, same identity
    real = B.join
    monkeypatch.setattr(B, "join", lambda *a, **k: None)   # the run commits, but a reader comes before join
    with RunLogger(ds, SCHEMA) as log:
        r = record(log)
    ds.finalize()
    d = B.load(ds.root)
    later = B.find(d, "later")
    assert d["members"][r.run_id] == later["batch_id"] and later["status"] == "collecting"
    assert real(ds.root, r.run_id, None, None)["batch_id"] == later["batch_id"]


def test_an_import_that_moves_a_run_marks_the_source_training_sets_out_of_date(tmp_path):
    ds, a1, a2, b1 = two_batches(tmp_path)
    B.add_export([(ds.root, "b", None)], {"export_id": "x1", "recipe": "dagger", "path": "/out"})
    B.add_runs(ds.root, [b1.run_id], "imported")
    d = B._read(ds.root)
    assert all(e["stale"] for e in B.find(d, "b")["exports"])
    assert d["members"][b1.run_id] == B.find(d, "imported")["batch_id"]
    assert "moved out" in B.find(d, "b")["history"][-1]["what"]


def test_a_move_in_one_dataset_puts_a_combined_training_set_out_of_date_in_the_other(tmp_path):
    from aftershock_logger import export as X
    roots = []
    for name in ("a", "b"):
        ds = make_ds(tmp_path, name)
        with RunLogger(ds, SCHEMA, batch="r") as log:
            record(log), record(log)
        with RunLogger(ds, SCHEMA, batch="s") as log:
            record(log)
        ds.finalize()
        review_all(ds)
        B.set_status(ds.root, "r", "locked")
        roots.append(ds.root)
    X.export([(roots[0], "r"), (roots[1], "r")], "filtered", str(tmp_path / "out"))
    assert not any(x["out_of_date"] for x in X.training_sets(roots[0]))
    B.set_status(roots[1], "r", "reviewing", reason="regroup")
    moved = B._read(roots[1])["members"]
    rid = next(r for r, x in moved.items() if x == B.find(B._read(roots[1]), "r")["batch_id"])
    B.move(roots[1], [rid], "s")
    (x,) = X.training_sets(roots[0])
    assert x["moved_runs"] == 1 and x["out_of_date"]
