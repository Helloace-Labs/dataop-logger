"""The rename to aftershock-logger keeps the old names working.

Covers: the old package import (`dataop_logger` and its submodules are the same objects as `aftershock_logger`),
both commands (`aftershock-logger` and `dataop-logger`) as a built wheel declares them, `python -m` with either
name, the `dataop/` sidecar folder (still written, and a dataset written by dataop-logger 0.1.0 before the rename
still validates, inspects and reads, unchanged on disk).
"""
import hashlib
import importlib
import json
import os
import pkgutil
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
OLD_DATASET = Path(__file__).resolve().parent / "data" / "dataop_v0_1_0"   # written by dataop-logger 0.1.0
sys.path.insert(0, str(SRC))

import aftershock_logger  # noqa: E402
import dataop_logger  # noqa: E402

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TASK = "pick up the tube"
SUBMODULES = sorted(m.name for m in pkgutil.iter_modules(aftershock_logger.__path__))


def tree_digest(root: Path) -> dict:
    """Content hash of every file except lock files (their content is a pid)."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not p.name.endswith(".lock")}


def copy_old(tmp_path) -> Path:
    dst = tmp_path / "old"
    shutil.copytree(OLD_DATASET, dst)
    return dst


def run_py(args, *, pythonpath, cwd, extra_env=None):
    env = {**os.environ, "PYTHONPATH": str(pythonpath), "HF_HUB_OFFLINE": "1", **(extra_env or {})}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


def record_two_runs(pkg, root: Path):
    """Write two committed runs through the given package object (old or new name)."""
    schema = pkg.ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
    feats = {k: {"dtype": "float32", "shape": (6,), "names": list(NAMES)} for k in ("observation.state", "action")}
    ds = lerobot_dataset.LeRobotDataset.create("test/rename", fps=30, features=feats, root=root, use_videos=False)
    with pkg.RunLogger(ds, schema) as log:
        for n, reason in ((6, "success"), (4, "failure")):
            with log.run(task=TASK, policy=pkg.PolicyInfo("smolvla", "train/smolvla_100")) as run:
                for i in range(n):
                    t = time.monotonic_ns()
                    a = np.arange(6, dtype=np.float32) + i
                    run.add_frame({"observation.state": np.full(6, i, np.float32), "task": TASK}, sent_action=a,
                                  base_action=a, control_mode="base", inference_id=i, chunk_offset=0,
                                  obs_ns=t, sent_ns=t + 1000)
                run.end(termination_reason=reason)
    ds.finalize()
    return ds


# ── old package import ──────────────────────────────────────────────────────────────────────────────────────

def test_old_package_is_the_new_package():
    assert dataop_logger is aftershock_logger
    assert aftershock_logger.__name__ == "aftershock_logger"
    for name in aftershock_logger.__all__:
        assert getattr(dataop_logger, name) is getattr(aftershock_logger, name)


@pytest.mark.parametrize("sub", SUBMODULES)
def test_every_old_submodule_is_the_new_module(sub):
    old = importlib.import_module(f"dataop_logger.{sub}")
    new = importlib.import_module(f"aftershock_logger.{sub}")
    assert old is new
    assert new.__name__ == new.__spec__.name == f"aftershock_logger.{sub}"   # the alias leaves the real spec
    assert new.__package__ == "aftershock_logger"


def test_old_from_imports_give_the_same_objects():
    from dataop_logger import store as old_store
    from dataop_logger.reader import DataopDataset as OldReader
    from dataop_logger.review import ReviewError as OldReviewError
    from aftershock_logger import store
    from aftershock_logger.reader import DataopDataset
    from aftershock_logger.review import ReviewError
    assert old_store is store and OldReader is DataopDataset and OldReviewError is ReviewError
    assert issubclass(OldReviewError, Exception) and OldReviewError.__module__ == "aftershock_logger.review"


def test_unknown_old_submodule_still_fails():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dataop_logger.no_such_module")


def test_old_name_first_in_a_fresh_process_loads_each_module_once(tmp_path):
    """Importing only old names first (as existing user code does) must not load a second copy of any module."""
    code = f"""
import importlib, sys
subs = {SUBMODULES!r}
olds = [importlib.import_module("dataop_logger." + s) for s in subs]   # old names first, no new import yet
import aftershock_logger
news = [importlib.import_module("aftershock_logger." + s) for s in subs]
assert all(o is n for o, n in zip(olds, news))
assert sys.modules["dataop_logger"] is aftershock_logger
mods = [m for k, m in sys.modules.items() if k.split(".")[0] in ("dataop_logger", "aftershock_logger")]
real = {{k for k in sys.modules if k.split(".")[0] == "aftershock_logger"}}
assert len({{id(m) for m in mods}}) == len(real), "a module was loaded twice"
assert all(m.__spec__.name.startswith("aftershock_logger") for m in mods)
print("ok")
"""
    r = run_py(["-W", "error::DeprecationWarning", "-W", "error::ImportWarning", "-c", code],
               pythonpath=SRC, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


# ── old and new commands ────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory):
    """Build the wheel from a copy of the repo (so no build files land in the checkout) and unpack it."""
    pytest.importorskip("setuptools.build_meta")
    tmp = tmp_path_factory.mktemp("wheel")
    proj = tmp / "proj"
    proj.mkdir()
    for f in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(REPO / f, proj / f)
    shutil.copytree(SRC, proj / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    out = tmp / "dist"
    out.mkdir()
    r = subprocess.run([sys.executable, "-c", "import sys; from setuptools import build_meta as b; "
                        "print(b.build_wheel(sys.argv[1]))", str(out)],
                       cwd=proj, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    wheel = out / r.stdout.strip().splitlines()[-1]
    site = tmp / "site"
    with zipfile.ZipFile(wheel) as z:
        z.extractall(site)
        names = z.namelist()
    return {"wheel": wheel, "site": site, "names": names}


def console_scripts(site: Path) -> dict:
    ep = next(site.glob("*.dist-info/entry_points.txt")).read_text().splitlines()
    i = ep.index("[console_scripts]")
    out = {}
    for line in ep[i + 1:]:
        if line.startswith("["):
            break
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            out[k] = v
    return out


def run_command(site: Path, name: str, args, cwd):
    """Run a console script the way pip's generated wrapper does: argv[0] is the command, then target()."""
    target = console_scripts(site)[name]
    mod, func = target.split(":")
    code = (f"import sys; sys.argv[0] = {name!r}; from {mod} import {func} as f; sys.exit(f(sys.argv[1:]))")
    return run_py(["-W", "error::DeprecationWarning", "-c", code, *args], pythonpath=site, cwd=cwd)


def test_wheel_is_aftershock_logger_and_declares_both_commands(built_wheel):
    assert built_wheel["wheel"].name.startswith("aftershock_logger-")
    meta = next(built_wheel["site"].glob("*.dist-info/METADATA")).read_text()
    assert "\nName: aftershock-logger\n" in meta
    assert console_scripts(built_wheel["site"]) == {"aftershock-logger": "aftershock_logger.cli:main",
                                                    "dataop-logger": "dataop_logger.cli:main"}
    names = set(built_wheel["names"])
    for f in ("aftershock_logger/__init__.py", "aftershock_logger/cli.py", "aftershock_logger/viewer.html",
              "aftershock_logger/viewer.js", "dataop_logger/__init__.py"):
        assert f in names, f
    assert not any(n.startswith("dataop_logger/") and n != "dataop_logger/__init__.py" for n in names)


@pytest.mark.parametrize("command", ["aftershock-logger", "dataop-logger"])
def test_both_commands_validate_and_inspect_the_old_dataset(built_wheel, tmp_path, command):
    root = copy_old(tmp_path)
    before = tree_digest(root)
    r = run_command(built_wheel["site"], command, ["validate", str(root)], cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.startswith("OK: 2 runs, 2 LeRobot episodes"), r.stdout + r.stderr
    r = run_command(built_wheel["site"], command, ["validate", str(root), "--json"], cwd=tmp_path)
    rep = json.loads(r.stdout)
    assert r.returncode == 0 and rep["ok"] and rep["errors"] == []
    r = run_command(built_wheel["site"], command, ["inspect", str(root)], cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.strip(), r.stderr
    r = run_command(built_wheel["site"], command, ["--help"], cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.startswith(f"usage: {command} "), r.stdout
    assert tree_digest(root) == before                       # reading an old dataset changes nothing on disk


@pytest.mark.parametrize("module", ["aftershock_logger.cli", "dataop_logger.cli"])
def test_python_dash_m_works_with_both_names(tmp_path, module):
    root = copy_old(tmp_path)
    r = run_py(["-W", "error::DeprecationWarning", "-W", "error::RuntimeWarning", "-m", module, "validate",
                str(root)], pythonpath=SRC, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.startswith("OK: 2 runs"), r.stdout + r.stderr
    r = run_py(["-m", module, "--help"], pythonpath=SRC, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.startswith("usage: aftershock-logger "), r.stdout


# ── the dataop/ sidecar folder ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pkg_name", ["aftershock_logger", "dataop_logger"])
def test_writes_still_go_to_the_dataop_folder(tmp_path, pkg_name):
    pkg = importlib.import_module(pkg_name)
    ds = record_two_runs(pkg, tmp_path / "ds")
    side = ds.root / "dataop"
    assert (side / "index.jsonl").exists() and len(list((side / "runs").iterdir())) == 2
    assert not (ds.root / "aftershock").exists()
    m = json.loads(next((side / "runs").glob("*/manifest.json")).read_text())
    assert set(m["env"]["software"]) == {"lerobot", "dataop_logger"}    # manifest key keeps its old name
    assert m["env"]["software"]["dataop_logger"]
    from aftershock_logger import validate as V
    assert V.validate(ds.root)["ok"]


def test_old_dataset_validates_and_runs_list_with_the_new_package(tmp_path):
    from aftershock_logger import store
    from aftershock_logger import validate as V
    from aftershock_logger.annotate import runs_json
    root = copy_old(tmp_path)
    before = tree_digest(root)
    res = V.validate(root)
    assert res["ok"], res["errors"]
    ms = sorted((m for _, m in store.iter_manifests(root, "committed")), key=lambda m: m["episode_index"])
    assert [(m["episode_index"], m["length"], m["termination_reason"]) for m in ms] == [(0, 10, "success"),
                                                                                         (1, 8, "failure")]
    assert ms[1]["takeovers"][0]["start_frame"] == 5
    assert ms[0]["env"]["software"]["dataop_logger"] == "0.1.0"
    runs = runs_json(root)["runs"]
    assert len(runs) == 2
    assert tree_digest(root) == before


def test_old_dataset_reads_with_the_new_reader(tmp_path):
    pytest.importorskip("torch")
    from aftershock_logger.reader import DataopDataset
    root = copy_old(tmp_path)
    ds = DataopDataset(lerobot_dataset.LeRobotDataset("test/old_v0_1_0", root=root))
    assert len(ds) == 18
    base = ds[12]                                           # episode 1, frame 2: the policy drove
    assert base["dataop.control_mode"] == "base"
    assert base["dataop.base_action"].tolist() == (np.arange(6) + 2).tolist()
    rescue = ds[15]                                         # episode 1, frame 5: the rescue took over
    assert rescue["dataop.control_mode"] == "expert" and rescue["dataop.expert_source"] == "skill:rescue"
    assert rescue["dataop.expert_action"].tolist() == (np.arange(6) + 6).tolist()
