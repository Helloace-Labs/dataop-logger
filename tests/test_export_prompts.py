"""Export with subtask prompts (prompts="task+subtasks"): every kept span is written with the task prompt and then once
more per subtask stretch with that subtask's prompt, from the subtask segments in effect: a format 2 `segments` edit,
else the recorded steps with boundary edits applied, mapped to subtask ids by the review's legacy_subtask_map. Frames
with no segment, or an unresolved one, get only the task prompt. The default export is unchanged.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aftershock_logger import ActionSchema, RunLogger, store  # noqa: E402
from aftershock_logger import batches as B  # noqa: E402
from aftershock_logger import export as X  # noqa: E402
from aftershock_logger import review as rv  # noqa: E402
from aftershock_logger import review_format as fmt  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCHEMA = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
TASK = "put the cube in the bin"
TEXTS = [{"subtask_id": "reach", "text": "reach the cube"}, {"subtask_id": "grasp", "text": "grasp the cube"},
         {"subtask_id": "lift", "text": "lift the cube"}]
TEXT = {x["subtask_id"]: x["text"] for x in TEXTS}


def _ds(root: Path):
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
             "action": {"dtype": "float32", "shape": (6,), "names": list(NAMES)}}
    return lerobot_dataset.LeRobotDataset.create(f"test/{root.name}", fps=30, features=feats, root=root,
                                                 use_videos=False)


def _teleop_run(log, n, steps=()):
    """A person drives every frame; steps are (name, start frame), each ending where the next starts."""
    with log.run(task=TASK) as run:
        starts = dict((at, name) for name, at in steps)
        cur = None
        for i in range(n):
            if i in starts:
                if cur:
                    run.subgoal(cur, "completed")
                cur = starts[i]
                run.subgoal(cur, "started")
            a = np.full(6, i, np.float32)
            t = time.monotonic_ns()
            run.add_frame({"observation.state": a, "task": TASK}, sent_action=a, expert_action=a,
                          control_mode="expert", expert_source="human", obs_ns=t, sent_ns=t + 1000)
        if cur:
            run.subgoal(cur, "completed")
        run.end(termination_reason="success")
    return run


def _lock(root):
    for p, _ in store.iter_manifests(root, "committed"):
        rv.set_reviewed(p, True)
    B.set_status(root, "b1", "locked")


def _with_map(run, entries):
    """Give the run a format 2 review with a legacy_subtask_map, as a review migrated against a task has."""
    m = store.read_manifest(run.run_path)
    r = fmt.empty_review(m["run_id"])
    r["legacy_subtask_map"] = {"run_id": m["run_id"], "task_id": "cube-bin", "entries": entries}
    rv._path(run.run_path).write_text(fmt.dump_review(r))
    return m


@pytest.fixture
def labeled(tmp_path):
    """Run `seg`: three subtask segments by a segments edit (frames 0-1 and 26-29 in none) and a cut at [12, 14).
    Run `plain`: no segments. Run `unres`: one segment that names no subtask id."""
    ds = _ds(tmp_path / "ds")
    with RunLogger(ds, SCHEMA, batch="b1") as log:
        seg = _teleop_run(log, 30)
        plain = _teleop_run(log, 10)
        unres = _teleop_run(log, 8)
    ds.finalize()
    m = store.read_manifest(seg.run_path)
    rv.add_edit(seg.run_path, m, "segments", {"segments": [
        {"subtask_id": "reach", "start": 2, "end": 8, "legacy_name": None},
        {"subtask_id": "grasp", "start": 8, "end": 18, "legacy_name": None},
        {"subtask_id": "lift", "start": 18, "end": 26, "legacy_name": None}]})
    rv.add_edit(seg.run_path, m, "exclude", {"start": 12, "end": 14})
    mu = store.read_manifest(unres.run_path)
    rv.add_edit(unres.run_path, mu, "segments", {"segments": [{"name": "wiggle", "start": 0, "end": 8}]})
    _lock(ds.root)
    return ds, seg, plain, unres


def episodes(out: Path) -> list[tuple[str, list[int]]]:
    """(prompt, source frame indices) per written episode, in episode order; the state holds the source frame."""
    got = lerobot_dataset.LeRobotDataset(f"test/{out.name}", root=out)
    tasks = {int(i): t for t, i in zip(got.meta.tasks.index, got.meta.tasks["task_index"])}
    rows = [pq.read_table(f).to_pydict() for f in sorted((out / "data").rglob("*.parquet"))]
    eps: dict[int, list] = {}
    for r in rows:
        for e, f, ti, st in zip(r["episode_index"], r["frame_index"], r["task_index"], r["observation.state"]):
            eps.setdefault(e, []).append((f, tasks[ti], int(st[0])))
    out_ = []
    for e in sorted(eps):
        fr = sorted(eps[e])
        assert len({p for _, p, _ in fr}) == 1, "an episode has one prompt"
        out_.append((fr[0][1], [s for _, _, s in fr]))
    return out_


def test_a_labeled_run_is_written_once_per_prompt_and_unlabeled_frames_only_with_the_task(labeled, tmp_path):
    ds, seg, plain, unres = labeled
    out = tmp_path / "set"
    rep = X.export([(ds.root, "b1")], "teleop", str(out), repo_id="test/set", prompts="task+subtasks",
                   subtask_prompts=TEXTS)
    by = {r["run_id"]: r for r in rep["runs"]}
    r = by[seg.run_id]
    assert r["spans"] == [(0, 12), (14, 30)]
    assert r["subtask_spans"] == [["reach", 2, 8], ["grasp", 8, 12], ["grasp", 14, 18], ["lift", 18, 26]]
    assert r["prompts"] == [{"prompt": TASK, "subtask_id": None, "frames": 28, "episodes": 2},
                            {"prompt": "reach the cube", "subtask_id": "reach", "frames": 6, "episodes": 1},
                            {"prompt": "grasp the cube", "subtask_id": "grasp", "frames": 8, "episodes": 2},
                            {"prompt": "lift the cube", "subtask_id": "lift", "frames": 8, "episodes": 1}]
    assert r["kept"] == 28 and r["written"] == 50
    for other in (plain, unres):                                   # no subtask id: the task prompt only
        o = by[other.run_id]
        assert o["subtask_spans"] == [] and [p["subtask_id"] for p in o["prompts"]] == [None]
        assert o["written"] == o["kept"]
    assert rep["frames"] == 50 + 10 + 8 and rep["episodes"] == 6 + 1 + 1
    assert rep["options"]["prompts"] == "task+subtasks" and rep["options"]["prompt_texts"] == TEXTS

    eps = episodes(out)
    assert len(eps) == rep["episodes"] and sum(len(f) for _, f in eps) == rep["frames"]
    # members are written in run id order; find the labeled run's six episodes by their frames
    want = [(TASK, list(range(0, 12))), ("reach the cube", list(range(2, 8))), ("grasp the cube", list(range(8, 12))),
            (TASK, list(range(14, 30))), ("grasp the cube", list(range(14, 18))), ("lift the cube", list(range(18, 26)))]
    i = next(k for k, e in enumerate(eps) if e == want[0])
    assert eps[i:i + 6] == want
    for prompt, frames in eps:                                     # a frame's subtask prompt matches its segment
        if prompt in TEXT.values():
            sid = next(k for k, v in TEXT.items() if v == prompt)
            assert all({"reach": range(2, 8), "grasp": range(8, 18), "lift": range(18, 26)}[sid].__contains__(f)
                       for f in frames)
    rec = json.loads((out / "dataop_export.json").read_text())
    assert rec["options"]["prompt_texts"] == TEXTS and rec["runs"] == json.loads(json.dumps(rep["runs"]))


def test_mapped_recorded_steps_with_a_moved_boundary_match_the_segments_edit(tmp_path):
    """Recorded steps grasp [0, 10) and lift [10, 20), mapped by legacy_subtask_map, with the boundary moved to 14,
    give the same subtask segments, report and written frames as a segments edit grasp [0, 14), lift [14, 20)."""
    outs, reps = [], []
    for kind in ("steps", "edit"):
        root = tmp_path / kind
        ds = _ds(root)
        with RunLogger(ds, SCHEMA, batch="b1") as log:
            run = _teleop_run(log, 20, steps=(("grasp", 0), ("lift", 10)))
        ds.finalize()
        m = _with_map(run, [{"step_index": 0, "recorded_name": "grasp", "subtask_id": "grasp"},
                            {"step_index": None, "recorded_name": "lift", "subtask_id": "lift"}])
        if kind == "steps":
            rv.add_edit(run.run_path, m, "step_boundary", {"index": 0, "before": "grasp", "after": "lift", "frame": 14})
        else:
            rv.add_edit(run.run_path, m, "segments", {"segments": [
                {"subtask_id": "grasp", "start": 0, "end": 14, "legacy_name": None},
                {"subtask_id": "lift", "start": 14, "end": 20, "legacy_name": None}]})
        assert rv.subtask_segments(m, rv.load(run.run_path)) == [("grasp", 0, 14), ("lift", 14, 20)]
        _lock(root)
        out = tmp_path / f"set_{kind}"
        reps.append(X.export([(root, "b1")], "teleop", str(out), repo_id="test/set", prompts="task+subtasks",
                             subtask_prompts={x["subtask_id"]: x["text"] for x in TEXTS}))
        outs.append(out)
    a, b = ([{k: v for k, v in r.items() if k not in ("run_id", "review_digest", "review_rev", "ticks_sha256",
                                                       "manifest_sha256", "root", "batch_id")} for r in x["runs"]] for x in reps)
    assert a == b and a[0]["subtask_spans"] == [["grasp", 0, 14], ["lift", 14, 20]]
    assert episodes(outs[0]) == episodes(outs[1]) == [(TASK, list(range(20))), ("grasp the cube", list(range(14))),
                                                      ("lift the cube", list(range(14, 20)))]
    assert reps[0]["written"]["train"]["tree_sha256"] == reps[1]["written"]["train"]["tree_sha256"]


def test_a_step_the_map_leaves_unresolved_gets_only_the_task_prompt(tmp_path):
    ds = _ds(tmp_path / "ds")
    with RunLogger(ds, SCHEMA, batch="b1") as log:
        run = _teleop_run(log, 12, steps=(("grasp", 0), ("lift", 6)))
    ds.finalize()
    m = _with_map(run, [{"step_index": 0, "recorded_name": "grasp", "subtask_id": "unresolved"},
                        {"step_index": 1, "recorded_name": "lift", "subtask_id": "lift"}])
    assert rv.subtask_segments(m, rv.load(run.run_path)) == [(None, 0, 6), ("lift", 6, 12)]
    assert rv.subtask_segments(m, rv.empty(m["run_id"])) == [(None, 0, 6), (None, 6, 12)]   # no map: no ids
    _lock(ds.root)
    rep = X.export([(ds.root, "b1")], "teleop", str(tmp_path / "o"), dry_run=True, prompts="task+subtasks",
                   subtask_prompts=TEXTS)
    assert rep["runs"][0]["subtask_spans"] == [["lift", 6, 12]] and rep["frames"] == 12 + 6


def test_the_default_export_is_unchanged(labeled, tmp_path):
    """No new keys in the report or the record, and the same bytes as an explicit prompts="task". The comparison with
    the logger from before this change is test_the_default_export_matches_the_baseline_from_before_subtask_prompts."""
    ds, seg, plain, unres = labeled
    a = X.export([(ds.root, "b1")], "teleop", str(tmp_path / "a"), repo_id="test/x")
    b = X.export([(ds.root, "b1")], "teleop", str(tmp_path / "b"), repo_id="test/x", prompts="task")
    assert a["written"]["train"]["tree_sha256"] == b["written"]["train"]["tree_sha256"]
    assert set(a["options"]) == {"with_success", "zero_on_success", "segments", "outcomes", "min_span", "base", "val",
                                 "hook", "drop_features", "runs", "runs_by_root"}
    for r in a["runs"]:
        assert set(r) == {"root", "batch", "batch_id", "run_id", "episode", "as", "outcome", "split", "frames", "kept",
                          "spans", "dropped", "review_rev", "review_digest", "ticks_sha256", "manifest_sha256"}
    assert a["frames"] == 28 + 10 + 8 and a["episodes"] == 2 + 1 + 1
    assert [p for p, _ in episodes(tmp_path / "a")] == [TASK] * 4
    s = X.export([(ds.root, "b1")], "teleop", str(tmp_path / "s"), repo_id="test/x", prompts="task+subtasks",
                 subtask_prompts=TEXTS)
    task_eps = [e for e in episodes(tmp_path / "s") if e[0] == TASK]
    assert task_eps == episodes(tmp_path / "a")                   # the task prompt episodes are the default set
    assert s["frames"] - a["frames"] == 22


def test_prompt_options_are_checked(labeled, tmp_path):
    ds = labeled[0]
    go = lambda **kw: X.export([(ds.root, "b1")], "teleop", str(tmp_path / "o"), dry_run=True, **kw)
    with pytest.raises(X.ExportError, match="prompts must be one of"):
        go(prompts="subtasks")
    with pytest.raises(X.ExportError, match="needs the subtask prompt texts"):
        go(prompts="task+subtasks")
    with pytest.raises(X.ExportError, match="only with"):
        go(subtask_prompts=TEXTS)
    with pytest.raises(X.ExportError, match=r"subtasks \['lift'\]"):
        go(prompts="task+subtasks", subtask_prompts=TEXTS[:2])      # a kept subtask without a text fails, never drops
    for bad in ([], [{"subtask_id": "reach"}], [["Reach", "x"]], [["unresolved", "x"]], [["reach", " x"]],
                [["reach", "a\nb"]], [["reach", ""]], [["reach", "x"], ["reach", "y"]]):
        with pytest.raises(X.ExportError):
            go(prompts="task+subtasks", subtask_prompts=bad)
    ok = go(prompts="task+subtasks", subtask_prompts=[["lift", "lift"], ["grasp", "grasp"], ["reach", "reach"],
                                                      ["place", "place"]])
    seg = next(r for r in ok["runs"] if r["subtask_spans"])
    assert [p["subtask_id"] for p in seg["prompts"]] == [None, "lift", "grasp", "reach"]   # the order given


def test_cli_passes_the_prompt_texts(labeled, tmp_path, capsys):
    from aftershock_logger import cli
    ds = labeled[0]
    f = tmp_path / "texts.json"
    f.write_text(json.dumps(TEXTS))
    assert cli.main(["export", "--batch", f"{ds.root}:b1", "--recipe", "teleop", "--into", str(tmp_path / "c"),
                     "--dry-run", "--prompts", "task+subtasks", "--subtask-prompts", str(f)]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 68
    assert cli.main(["export", "--batch", f"{ds.root}:b1", "--recipe", "teleop", "--into", str(tmp_path / "c"),
                     "--dry-run", "--prompts", "task+subtasks"]) == 2


def test_the_default_export_matches_the_baseline_from_before_subtask_prompts(tmp_path):
    """Every recipe's default export on a fixed fixture: the report, the record and every written episode (prompt,
    frames, actions) equal what the logger wrote before subtask prompts existed (tests/export_baseline.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import export_baseline
    want = json.loads((Path(__file__).resolve().parent / "data" / "export_default_baseline.json").read_text())
    got = json.loads(json.dumps(export_baseline.baseline(tmp_path)))
    assert sorted(got) == sorted(want)
    for recipe in want:
        assert got[recipe] == want[recipe], recipe


def test_a_subtask_mapping_change_puts_a_subtask_prompt_set_out_of_date(tmp_path):
    """The review digest does not cover the subtask ids a legacy_subtask_map gives recorded steps (a format 1 review
    migrated against a task changes only those), so a task+subtasks set records a subtask digest per run and is out
    of date when it changes. A default set records none and is judged as before."""
    ds = _ds(tmp_path / "ds")
    with RunLogger(ds, SCHEMA, batch="b1") as log:
        run = _teleop_run(log, 12, steps=(("grasp", 0), ("lift", 6)))
    ds.finalize()
    entries = [{"step_index": 0, "recorded_name": "grasp", "subtask_id": "unresolved"},
               {"step_index": 1, "recorded_name": "lift", "subtask_id": "lift"}]
    m = _with_map(run, entries)
    sub = X.export([(ds.root, "b1")], "teleop", str(tmp_path / "sub"), allow_unlocked=True, prompts="task+subtasks",
                   subtask_prompts=TEXTS)
    plain = X.export([(ds.root, "b1")], "teleop", str(tmp_path / "plain"), allow_unlocked=True)
    assert "subtask_digest" in sub["runs"][0] and "subtask_digest" not in plain["runs"][0]
    sets = {s["export_id"]: s for s in X.training_sets(ds.root)}
    assert not sets[sub["export_id"]]["out_of_date"] and not sets[plain["export_id"]]["out_of_date"]
    before = B.review_digest(m, rv.load(run.run_path))
    r = rv.load(run.run_path)
    r["legacy_subtask_map"]["entries"][0]["subtask_id"] = "grasp"             # the step now names a subtask
    rv._path(run.run_path).write_text(fmt.dump_review(r))
    assert B.review_digest(m, rv.load(run.run_path)) == before
    sets = {s["export_id"]: s for s in X.training_sets(ds.root)}
    assert sets[sub["export_id"]]["out_of_date"] and sets[sub["export_id"]]["changed_runs"] == 1
    assert not sets[plain["export_id"]]["out_of_date"]
