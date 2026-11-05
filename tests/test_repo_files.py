"""The repo carries no build output, and a wheel built from the committed files holds only the sources in src/.

A stale `build/lib/` in the checkout is picked up by setuptools: its files are copied into the wheel next to the
real modules. These tests build from exactly the files git tracks, so they fail if build output is committed again.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60)


def tracked_files():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    r = git("ls-files", "-z")
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    return [p for p in r.stdout.split("\0") if p]


def test_no_build_output_is_tracked():
    files = tracked_files()
    assert [p for p in files if p == "build" or p.startswith("build/")] == []
    assert [p for p in files if p == "dist" or p.startswith("dist/")] == []


@pytest.mark.parametrize("path", ["build/lib/dataop_logger/adapter.py", "build/anything"])
def test_build_folder_is_ignored(path):
    tracked_files()
    r = git("check-ignore", "-q", "--no-index", path)
    assert r.returncode == 0, f"{path} is not ignored by .gitignore"


def test_wheel_from_the_tracked_files_holds_only_the_sources(tmp_path):
    pytest.importorskip("setuptools.build_meta")
    proj = tmp_path / "proj"
    for rel in tracked_files():
        src = REPO / rel
        if src.is_file():
            dst = proj / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    out = tmp_path / "dist"
    out.mkdir()
    r = subprocess.run([sys.executable, "-c", "import sys; from setuptools import build_meta as b; "
                        "print(b.build_wheel(sys.argv[1]))", str(out)],
                       cwd=proj, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    wheel = out / r.stdout.strip().splitlines()[-1]
    with zipfile.ZipFile(wheel) as z:
        in_wheel = sorted(n for n in z.namelist() if n.endswith(".py"))
    in_src = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
                    if (proj / "src" / p.relative_to(SRC)).exists())
    assert in_wheel == in_src
