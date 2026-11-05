"""Import: action units per dimension (one unit, a list in action order, a map by action name, or a named unit set).

An import records the units it is given in each run's action schema, and the schema id covers them. A single string
still applies to every dimension, and with nothing given every dimension stays unknown. Units are never guessed from
the dataset: a unit set is used only when the mapping names it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aftershock_logger import UNKNOWN, ActionSchema, importer, store  # noqa: E402
from aftershock_logger import validate as V  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

SO_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos",
            "gripper.pos"]
SO_UNITS = ["deg"] * 5 + ["percent"]


def dataset(tmp_path, names=SO_NAMES, name="so"):
    dim = len(names)
    feats = {"observation.state": {"dtype": "float32", "shape": (dim,), "names": names},
             "action": {"dtype": "float32", "shape": (dim,), "names": names}}
    ds = lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features=feats, root=tmp_path / name,
                                               use_videos=False)
    for i in range(8):
        a = np.arange(dim, dtype=np.float32) + i
        ds.add_frame({"observation.state": a, "action": a, "task": "pick"})
    ds.save_episode()
    ds.finalize()
    return ds


def imported_schema(ds, **kw) -> dict:
    rep = importer.import_dataset(str(ds.root), **kw)
    m = json.loads((store.runs_dir(ds.root) / rep["runs"][0]["run_id"] / "manifest.json").read_text())
    assert V.validate(ds.root)["ok"]
    return m["action_schema"]


@pytest.mark.parametrize("units", [SO_UNITS, "lerobot:so_follower:degrees",
                                   dict(zip(SO_NAMES, SO_UNITS))])
def test_per_dimension_units_are_recorded_and_the_schema_id_covers_them(tmp_path, units):
    ds = dataset(tmp_path)
    s = imported_schema(ds, mapping={"action_units": units, "action_space": "joint_position"})
    assert s["units"] == SO_UNITS and s["names"] == SO_NAMES
    same = ActionSchema(names=tuple(SO_NAMES), units=tuple(SO_UNITS), space="joint_position", rate_hz=30.0)
    assert s["id"] == same.id
    other = ActionSchema(names=tuple(SO_NAMES), units=(UNKNOWN,) * 6, space="joint_position", rate_hz=30.0)
    assert s["id"] != other.id


def test_a_single_unit_still_applies_to_every_dimension_and_nothing_given_stays_unknown(tmp_path):
    assert imported_schema(dataset(tmp_path, name="a"), mapping={"action_units": "deg"})["units"] == ["deg"] * 6
    assert imported_schema(dataset(tmp_path, name="b"))["units"] == [UNKNOWN] * 6


def test_a_map_leaves_the_actions_it_does_not_name_unknown(tmp_path):
    s = imported_schema(dataset(tmp_path), mapping={"action_units": {"gripper.pos": "percent"}})
    assert s["units"] == [UNKNOWN] * 5 + ["percent"]


@pytest.mark.parametrize("units, why", [
    (["deg"] * 5, "lists 5 units, the action has 6"),
    ({"elbow.pos": "deg"}, "does not have"),
    ("lerobot:so_follower:radians", "unknown unit set"),
    (["deg", ""] + ["deg"] * 4, "non-empty string"),
    (7, "non-empty string"),
])
def test_bad_units_are_refused_before_anything_is_written(tmp_path, units, why):
    ds = dataset(tmp_path)
    with pytest.raises(importer.ImportError_, match=why):
        importer.import_dataset(str(ds.root), mapping={"action_units": units})
    assert not (ds.root / "dataop" / "runs").exists()


def test_a_unit_set_refuses_a_dataset_with_other_action_names(tmp_path):
    ds = dataset(tmp_path, names=["a0", "a1"], name="other")
    with pytest.raises(importer.ImportError_, match=r"has no unit for actions \['a0', 'a1'\]"):
        importer.import_dataset(str(ds.root), mapping={"action_units": "lerobot:so_follower:degrees"})


def test_a_dry_run_shows_the_units(tmp_path):
    ds = dataset(tmp_path)
    p = importer.plan(ds.root, "plain", {"action_units": "lerobot:so_follower:degrees"})
    assert list(p["schema"].units) == SO_UNITS
