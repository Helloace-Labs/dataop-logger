"""Import presets for LeRobot's own human-in-the-loop recordings, each tested on a small synthetic dataset in the shape
that LeRobot release writes (importer.py docstring):

  lerobot-rollout     `lerobot-rollout --strategy.type=dagger` (0.6 and later): a bool `intervention` column, with or
                      without the policy's own frames (record_autonomous)
  lerobot-hil         HIL-SERL replay buffers saved with `complementary_info.is_intervention` (0.3.x, and 0.6 and
                      later; the buffer stores the flag as a float32 0 or 1)
  lerobot-hil-record  HIL-SERL recordings with no intervention flag (gym_manipulator record mode, 0.4 and 0.5 buffers)

No LeRobot release writes `next.success`, so both HIL presets read success from `next.reward`.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aftershock_logger import importer, store  # noqa: E402
from aftershock_logger import validate as V  # noqa: E402
from aftershock_logger.export import tree_digest  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def _create(tmp_path, name, extra):
    feats = {"observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES},
             "action": {"dtype": "float32", "shape": (6,), "names": NAMES}, **extra}
    return lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features=feats, root=tmp_path / name,
                                                 use_videos=False)


def _frame(i, task, **cols):
    a = np.full(6, i, np.float32)
    return {"observation.state": a, "action": a, "task": task, **cols}


def dagger_rollout(tmp_path, name="rollout", episodes=((6, 4), (0, 5))):
    """lerobot-rollout --strategy.type=dagger: per episode (policy frames, then intervention frames). (0, n) is the
    default corrections-only recording, where every frame is a correction."""
    ds = _create(tmp_path, name, {"intervention": {"dtype": "bool", "shape": (1,), "names": None}})
    for pol, cor in episodes:
        for i in range(pol + cor):
            ds.add_frame(_frame(i, "put the cube in the bin", intervention=np.array([i >= pol], bool)))
        ds.save_episode()
    ds.finalize()
    return ds


def hil(tmp_path, name, flag: bool, penalty: bool = True, episodes=((8, 5, True), (6, 3, False))):
    """HIL-SERL: per episode (frames, first intervention frame, success). Success is a reward of 1 on the last step,
    which also ends the episode (next.done)."""
    extra = {"next.reward": {"dtype": "float32", "shape": (1,), "names": None},
             "next.done": {"dtype": "bool", "shape": (1,), "names": None}}
    if penalty:
        extra["complementary_info.discrete_penalty"] = {"dtype": "float32", "shape": (1,), "names": ["discrete_penalty"]}
    if flag:          # the replay buffer stores the bool in a float tensor, so the column is float32 0 or 1
        extra["complementary_info.is_intervention"] = {"dtype": "float32", "shape": (1,), "names": None}
    ds = _create(tmp_path, name, extra)
    for n, take, ok in episodes:
        for i in range(n):
            last = i == n - 1
            cols = {"next.reward": np.array([1.0 if (ok and last) else 0.0], np.float32),
                    "next.done": np.array([last], bool)}
            if penalty:
                cols["complementary_info.discrete_penalty"] = np.array([0.0], np.float32)
            if flag:
                cols["complementary_info.is_intervention"] = np.array([float(i >= take)], np.float32)
            ds.add_frame(_frame(i, "from_replay_buffer", **cols))
        ds.save_episode()
    ds.finalize()
    return ds


def runs(root):
    out = {}
    for p, m in store.iter_manifests(root, "committed"):
        out[m["episode_index"]] = (m, pq.read_table(p / "ticks.parquet").to_pydict())
    return out


def lerobot_files(root):
    return {k: tree_digest(Path(root) / k) for k in ("meta", "data")}


def test_dagger_rollout_is_detected_and_maps_intervention_to_who_drove(tmp_path):
    ds = dagger_rollout(tmp_path)
    before = lerobot_files(ds.root)
    assert importer.detect(ds.root) == "lerobot-rollout"
    rep = importer.import_dataset(str(ds.root), preset="auto")
    assert rep["preset"] == "lerobot-rollout" and rep["fields"]["control"] == "column intervention"
    assert [r["policy_frames"] for r in rep["runs"]] == [6, 0]
    got = runs(ds.root)
    m, t = got[0]
    assert t["control_mode"] == ["base"] * 6 + ["expert"] * 4
    assert t["base_valid"] == [True] * 6 + [False] * 4 and t["expert_valid"] == [False] * 6 + [True] * 4
    assert t["expert_source"] == [None] * 6 + ["human"] * 4
    assert np.allclose(np.asarray(t["base_action"][:6]), np.asarray(t["sent_action"][:6]))
    assert m["termination_reason"] == "unknown" and m["policy"] == "unknown"     # a rollout records no outcome
    assert [(x["start_frame"], x["end_frame"]) for x in m["takeovers"]] == [(6, 10)]
    m1, t1 = got[1]                                               # corrections only: a person drove every frame
    assert t1["control_mode"] == ["expert"] * 5 and not any(t1["base_valid"])
    assert V.validate(ds.root)["ok"]
    assert lerobot_files(ds.root) == before                      # LeRobot files are never rewritten


def test_imported_rollouts_stay_mergeable_with_lerobot(tmp_path):
    """Import only adds the dataop/ sidecar, so LeRobot's own merge still combines two imported rollouts."""
    agg = pytest.importorskip("lerobot.datasets.aggregate")
    a = dagger_rollout(tmp_path, "a")
    b = dagger_rollout(tmp_path, "b", episodes=((3, 2),))
    for d in (a, b):
        importer.import_dataset(str(d.root), preset="auto")
    agg.aggregate_datasets(["test/a", "test/b"], "test/ab", roots=[a.root, b.root], aggr_root=tmp_path / "ab")
    ab = lerobot_dataset.LeRobotDataset("test/ab", root=tmp_path / "ab")
    assert ab.num_episodes == 3 and ab.num_frames == 10 + 5 + 5
    assert "intervention" in ab.features
    rep = importer.import_dataset(str(tmp_path / "ab"), preset="auto")
    assert rep["preset"] == "lerobot-rollout" and [r["policy_frames"] for r in rep["runs"]] == [6, 0, 3]


def test_hil_buffer_with_the_intervention_flag(tmp_path):
    ds = hil(tmp_path, "buffer", flag=True)
    assert importer.detect(ds.root) == "lerobot-hil"
    rep = importer.import_dataset(str(ds.root), preset="auto")
    assert rep["preset"] == "lerobot-hil" and rep["missing_columns"] == []
    assert rep["fields"]["outcome"] == "column next.reward above 0"
    got = runs(ds.root)
    assert got[0][0]["termination_reason"] == "success" and got[1][0]["termination_reason"] == "failure"
    t = got[0][1]
    assert t["control_mode"] == ["base"] * 5 + ["expert"] * 3
    assert t["base_valid"] == [True] * 5 + [False] * 3 and t["expert_source"][5:] == ["human"] * 3
    assert t["reward"][-1] == 1.0 and all(t["reward_valid"])
    assert got[0][0]["rl_ready"] is True
    assert V.validate(ds.root)["ok"]


@pytest.mark.parametrize("penalty", [True, False])
def test_hil_recording_without_the_flag_keeps_who_drove_unknown(tmp_path, penalty):
    """gym_manipulator record mode, and the 0.4 and 0.5 buffers: no column says who drove a frame."""
    ds = hil(tmp_path, "record", flag=False, penalty=penalty)
    assert importer.detect(ds.root) == "lerobot-hil-record"
    rep = importer.import_dataset(str(ds.root), preset="auto")
    assert rep["preset"] == "lerobot-hil-record" and rep["fields"]["control"] == "unknown"
    got = runs(ds.root)
    assert got[0][0]["termination_reason"] == "success" and got[1][0]["termination_reason"] == "failure"
    assert set(got[0][1]["control_mode"]) == {"unknown"} and not any(got[0][1]["base_valid"])
    assert got[0][0]["policy"] == "unknown"
    assert V.validate(ds.root)["ok"]


def test_the_older_hil_preset_on_a_dataset_without_the_flag_reports_the_column_missing(tmp_path):
    ds = hil(tmp_path, "record", flag=False)
    rep = importer.import_dataset(str(ds.root), preset="lerobot-hil", dry_run=True)
    assert rep["missing_columns"] == ["complementary_info.is_intervention"]
    assert rep["fields"]["control"].endswith("(missing)")


def test_a_success_column_still_maps_and_the_threshold_is_checked(tmp_path):
    ds = _create(tmp_path, "own", {"next.success": {"dtype": "bool", "shape": (1,), "names": None},
                                  "next.reward": {"dtype": "float32", "shape": (1,), "names": None}})
    for i in range(4):
        ds.add_frame(_frame(i, "pick", **{"next.success": np.array([i == 3], bool),
                                          "next.reward": np.array([-1.0], np.float32)}))
    ds.save_episode()
    ds.finalize()
    assert importer.detect(ds.root) == "demos"                    # no next.done: not a HIL-SERL recording
    rep = importer.import_dataset(str(ds.root), preset="plain", mapping={"outcome": {"column": "next.success"}},
                                  dry_run=True)
    assert rep["runs"][0]["outcome"] == "success"
    neg = importer.import_dataset(str(ds.root), preset="plain", dry_run=True,
                                  mapping={"outcome": {"column": "next.reward", "above": 0}})
    assert neg["runs"][0]["outcome"] == "failure"                 # a negative reward is not a success
    for bad in ({"column": "next.reward", "above": True}, {"column": "next.reward", "above": "0"},
                {"column": "next.reward", "above": float("nan")}, {"constant": "success", "above": 0}):
        with pytest.raises(importer.ImportError_, match="threshold"):
            importer.import_dataset(str(ds.root), preset="plain", mapping={"outcome": bad}, dry_run=True)


def test_every_preset_has_a_label_and_the_cli_lists_it():
    assert set(importer.PRESET_LABELS) == set(importer.PRESETS)
    from aftershock_logger import cli
    import inspect
    src = inspect.getsource(cli)
    for name in importer.PRESETS:
        assert name in src
    json.dumps(importer.PRESETS)                                  # presets stay plain JSON (the mapping digest)
