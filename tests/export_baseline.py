"""The default export's output on a fixed fixture, kept as a baseline so a change to the default export shows up.

  python tests/export_baseline.py <logger src folder> <baseline.json>

builds the fixture in a temporary folder, exports it with every recipe through the logger found in <logger src folder>
and writes what came out. tests/data/export_default_baseline.json was written this way from the logger before subtask
prompts existed, and tests/test_export_prompts.py compares the current logger's default export against it.

The summary holds what the export decides and writes, not what differs between two builds of the same fixture: per
recipe the report (without digests of files that carry times, the export id or paths), the record's keys and the
same fields, with the sources' values, and every written episode's prompt, source frames (the state holds the source frame index), actions and
extra features, in episode order. Run ids and batch ids come from a seeded generator, so runs keep their order.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

RECIPES = ("teleop", "lora", "filtered", "dagger", "residual")
TASK = "put the cube in the bin"
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# values that differ between two builds of the fixture (paths, times, digests of files with times, LeRobot's
# stats.json key order); the key stays, with its value masked, so a key that goes missing still shows
VARY = ("root", "review_digest", "ticks_sha256", "manifest_sha256", "at", "lerobot")


def _seeded_uuids(seed: int = 7):
    rng = random.Random(seed)
    return lambda: uuid.UUID(int=rng.getrandbits(128), version=4)


def build_fixture(root: Path) -> None:
    """Three runs in batch b1, locked: a teleop run with recorded steps, a segments edit and a cut; a short teleop
    run; a policy run with a rescue, a cut inside the rescue and a failure."""
    from aftershock_logger import ActionSchema, RunLogger, store
    from aftershock_logger import batches as B
    from aftershock_logger import review as rv
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    schema = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": list(NAMES)},
             "action": {"dtype": "float32", "shape": (6,), "names": list(NAMES)}}
    real = uuid.uuid4
    uuid.uuid4 = _seeded_uuids()
    try:
        ds = LeRobotDataset.create("test/baseline", fps=30, features=feats, root=root, use_videos=False)
        with RunLogger(ds, schema, batch="b1") as log:
            for n, steps, take in ((30, (("grasp", 0), ("lift", 12)), None), (10, (), None), (16, (), 6)):
                with log.run(task=TASK) as run:
                    starts, cur = dict((at, name) for name, at in steps), None
                    for i in range(n):
                        if i in starts:
                            if cur:
                                run.subgoal(cur, "completed")
                            cur = starts[i]
                            run.subgoal(cur, "started")
                        a = np.full(6, i, np.float32)
                        t = time.monotonic_ns()
                        frame = {"observation.state": a, "task": TASK}
                        if take is None:
                            run.add_frame(frame, sent_action=a, expert_action=a, control_mode="expert",
                                          expert_source="human", obs_ns=t, sent_ns=t + 1000)
                        elif i < take:
                            run.add_frame(frame, sent_action=a, base_action=a, control_mode="base", inference_id=i,
                                          chunk_offset=0, obs_ns=t, sent_ns=t + 1000)
                        else:
                            e = a + 1
                            run.add_frame(frame, sent_action=e, base_action=a, expert_action=e, control_mode="expert",
                                          expert_source="skill:rescue", inference_id=i, chunk_offset=0, obs_ns=t,
                                          sent_ns=t + 1000)
                    if cur:
                        run.subgoal(cur, "completed")
                    run.end(termination_reason="success" if take is None else "failure")
                if n == 30:
                    m = store.read_manifest(run.run_path)
                    rv.add_edit(run.run_path, m, "segments", {"segments": [
                        {"subtask_id": "grasp", "start": 2, "end": 18, "legacy_name": None}]})
                    rv.add_edit(run.run_path, m, "exclude", {"start": 12, "end": 14})
                if take is not None:
                    rv.add_edit(run.run_path, store.read_manifest(run.run_path), "exclude", {"start": 9, "end": 11})
        ds.finalize()
        for p, _ in store.iter_manifests(root, "committed"):
            rv.set_reviewed(p, True)
        B.set_status(root, "b1", "locked")
    finally:
        uuid.uuid4 = real


def _mask(v):
    """v as JSON, with the value of every VARY key masked (None stays None, so a value that disappears shows)."""
    if isinstance(v, dict):
        return {k: (x if x is None else "<varies>") if k in VARY else _mask(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_mask(x) for x in v]
    return v


def _episodes(out: Path) -> list[dict]:
    import pyarrow.parquet as pq
    info = json.loads((out / "meta" / "info.json").read_text())
    tasks = pq.read_table(out / "meta" / "tasks.parquet").to_pydict()
    name = next(k for k in tasks if k != "task_index")
    by_index = dict(zip(tasks["task_index"], tasks[name]))
    rows: dict[int, list] = {}
    for f in sorted((out / "data").rglob("*.parquet")):
        t = pq.read_table(f).to_pydict()
        cols = [c for c in t if c not in ("index", "episode_index", "frame_index", "timestamp", "task_index")]
        for k in range(len(t["episode_index"])):
            rows.setdefault(t["episode_index"][k], []).append(
                (t["frame_index"][k], by_index[t["task_index"][k]], {c: t[c][k] for c in cols}))
    eps = []
    for e in sorted(rows):
        fr = sorted(rows[e], key=lambda x: x[0])
        eps.append({"prompts": sorted({p for _, p, _ in fr}),
                    "frames": [int(c["observation.state"][0]) for _, _, c in fr],
                    "values": {c: [v[c] for _, _, v in fr] for c in sorted(fr[0][2]) if c != "observation.state"}})
    return [{"total_episodes": info["total_episodes"], "total_frames": info["total_frames"],
             "features": sorted(info["features"])}] + eps


def summary(rep: dict, out: Path) -> dict:
    rec = json.loads((out / "dataop_export.json").read_text()) if (out / "dataop_export.json").exists() else None
    keep = ("recipe", "options", "dry_run", "episodes", "frames", "hook", "split", "this_split", "registered", "stale")
    rep = json.loads(json.dumps(rep))
    return {"report": {"keys": sorted(rep), "runs": _mask(rep["runs"]), "sources": _mask(rep["sources"]),
                       **{k: _mask(rep.get(k)) for k in keep if k in rep}},
            "record": rec and {"keys": sorted(rec), "runs": _mask(rec["runs"]), "sources": _mask(rec["sources"]),
                               "versions": sorted(rec["versions"]),
                               **{k: _mask(rec.get(k)) for k in keep if k in rec}},
            "written": rep.get("written") and {k: v["episodes"] for k, v in rep["written"].items()},
            "dataset": _episodes(out) if (out / "meta").exists() else None}


def baseline(work: Path) -> dict:
    from aftershock_logger import export as X
    build_fixture(work / "ds")
    got = {}
    for recipe in RECIPES:
        out = work / f"set_{recipe}"
        rep = X.export([(work / "ds", "b1")], recipe, str(out), repo_id="test/set")
        got[recipe] = summary(rep, out)
    return got


if __name__ == "__main__":
    sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
    with tempfile.TemporaryDirectory() as d:
        data = baseline(Path(d))
    Path(sys.argv[2]).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
