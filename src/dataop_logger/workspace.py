"""The datasets this machine's viewer knows about, so one viewer can switch between them and import new ones.

  ~/.dataop/workspace.json   {"version": 1, "datasets": ["/abs/path", ...]}   (DATAOP_HOME moves it, for tests)

Nothing here is part of a dataset; deleting the file only empties the viewer's dataset list.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import store


def _path() -> Path:
    return Path(os.environ.get("DATAOP_HOME") or Path.home() / ".dataop") / "workspace.json"


def _read() -> list[str]:
    try:
        d = json.loads(_path().read_text())
        return [str(x) for x in d.get("datasets", []) if isinstance(x, str)]
    except (OSError, ValueError, AttributeError):
        return []


def remember(root: Path) -> None:
    root = str(Path(root).expanduser().resolve())
    known = _read()
    if root not in known:
        _path().parent.mkdir(parents=True, exist_ok=True)
        store.atomic_write_json(_path(), {"version": 1, "datasets": [*known, root]})


def _info(root: Path) -> dict | None:
    try:
        return json.loads((root / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return None


def describe(root: Path) -> dict | None:
    """Name, episode count and how many episodes have no dataop run yet; None if it is not a LeRobot dataset."""
    info = _info(root)
    if info is None:
        return None
    runs = sum(1 for _ in store.iter_manifests(root, "committed")) if store.dataop_dir(root).exists() else 0
    eps = int(info.get("total_episodes") or 0)
    complete = any((root / "data").rglob("*.parquet")) if (root / "data").is_dir() else False
    return {"root": str(root), "name": root.name, "episodes": eps, "runs": runs, "not_imported": max(0, eps - runs),
            "complete": complete,
            "version": str(info.get("codebase_version", "")), "fps": info.get("fps"),
            "robot_type": info.get("robot_type")}


def datasets() -> list[dict]:
    """Known datasets that still exist, in the order they were added."""
    return [x for x in (describe(Path(r)) for r in _read()) if x]


def candidates(extra: list[Path] = ()) -> list[dict]:
    """LeRobot datasets on this machine that could be imported: folders next to known datasets and LeRobot's own
    download cache. Only one level deep, so the scan stays fast."""
    known = set(_read())
    dirs = {Path(r).parent for r in known} | {Path(x) for x in extra}
    cache = Path(os.environ.get("HF_LEROBOT_HOME") or Path.home() / ".cache" / "huggingface" / "lerobot")
    if cache.is_dir():
        dirs |= {d for d in cache.iterdir() if d.is_dir()}      # <owner>/<name>
    out, seen = [], set()
    for d in sorted(dirs):
        try:
            kids = sorted(d.iterdir())
        except OSError:
            continue
        for k in kids:
            r = str(k.resolve())
            if r in seen or not k.is_dir():
                continue
            seen.add(r)
            x = describe(k)
            if x:
                out.append({**x, "known": r in known})
    return out
