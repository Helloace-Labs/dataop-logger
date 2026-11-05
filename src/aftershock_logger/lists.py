"""The dataset's agreed lists (its schema): failure types and task segments.

Stored in `dataop/lists.json`:
  {"version": 1, "rev": int, "failure_types": [entry], "task_segments": [entry]}
  entry = {"name": str, "description": str, "aliases": [str]}

People and agents pick from these lists, so counts stay comparable. A rename keeps the old name as an alias, so names
already in recordings (which never change) still map to the entry. A list is seeded from the names the dataset already
uses the first time it is read.
"""
from __future__ import annotations

import fcntl
import json
import re
from contextlib import contextmanager
from pathlib import Path

from . import store

KINDS = ("failure_types", "task_segments")
NAME = re.compile(r"^[^\n\r]{1,60}$")


class ListError(ValueError):
    pass


def _path(root: Path) -> Path:
    return store.dataop_dir(Path(root)) / "lists.json"


def _seed(root: Path) -> dict:
    from . import review as rv
    fails, segs = [], []
    for run_path, m in store.iter_manifests(Path(root), "committed"):
        try:
            eff = rv.effective(m, rv.load(run_path))
        except rv.ReviewError:
            eff = {"failure_mode": m.get("failure_mode"), "subgoals": m.get("subgoals", [])}
        for name in [m.get("failure_mode"), eff.get("failure_mode")]:
            if name and name not in fails:
                fails.append(name)
        for s in list(m.get("subgoals", [])) + list(eff.get("subgoals", [])):
            if s["name"] not in segs:
                segs.append(s["name"])
    entry = lambda n: {"name": n, "description": "", "aliases": []}
    return {"version": 1, "rev": 0, "failure_types": [entry(n) for n in fails], "task_segments": [entry(n) for n in segs]}


def load(root: Path) -> dict:
    p = _path(root)
    if not p.exists():
        return _seed(root)
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ListError(f"{p} is unreadable ({e})") from e
    if not isinstance(d, dict) or d.get("version") != 1 or not all(isinstance(d.get(k), list) for k in KINDS):
        raise ListError(f"{p} is not a version 1 lists file")
    for k in KINDS:
        for e in d[k]:
            if not isinstance(e, dict) or not isinstance(e.get("name"), str) or not NAME.match(e["name"]) \
                    or not isinstance(e.get("aliases", []), list):
                raise ListError(f"{p} has a malformed {k} entry")
    return d


@contextmanager
def _transaction(root: Path, expect_rev: int | None = None):
    store.dataop_dir(Path(root)).mkdir(parents=True, exist_ok=True)
    with open(store.dataop_dir(Path(root)) / "lists.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            d = load(root)
            if expect_rev is not None and d["rev"] != expect_rev:
                raise ListError(f"the lists changed (now at rev {d['rev']}, you had {expect_rev}); reload")
            yield d
            d["rev"] += 1
            store.atomic_write_json(_path(root), d)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _check(kind: str, name) -> str:
    if kind not in KINDS:
        raise ListError(f"kind must be one of {KINDS}")
    if not isinstance(name, str) or not NAME.match(name.strip()):
        raise ListError("a name must be 1 to 60 characters on one line")
    return name.strip()


def _find(d: dict, kind: str, name: str):
    return next((e for e in d[kind] if e["name"] == name or name in e.get("aliases", [])), None)


def canonical(d: dict, kind: str, name):
    """The list's name for a name found in data (follows renames); unknown names are returned unchanged."""
    if not name:
        return name
    e = _find(d, kind, name)
    return e["name"] if e else name


def add(root: Path, kind: str, name: str, description: str = "", expect_rev: int | None = None) -> dict:
    name = _check(kind, name)
    if not isinstance(description, str) or len(description) > 300:
        raise ListError("a description must be at most 300 characters")
    with _transaction(root, expect_rev) as d:
        if _find(d, kind, name):
            raise ListError(f"{name!r} is already on the list")
        d[kind].append({"name": name, "description": description.strip(), "aliases": []})
    return d


def rename(root: Path, kind: str, old: str, new: str, expect_rev: int | None = None) -> dict:
    new = _check(kind, new)
    with _transaction(root, expect_rev) as d:
        e = _find(d, kind, old)
        if e is None:
            raise ListError(f"{old!r} is not on the list")
        if new != e["name"] and _find(d, kind, new):
            raise ListError(f"{new!r} is already on the list")
        if new != e["name"]:
            e.setdefault("aliases", [])
            if e["name"] not in e["aliases"]:
                e["aliases"].append(e["name"])
            e["aliases"] = [a for a in e["aliases"] if a != new]
            e["name"] = new
    return d


def describe(root: Path, kind: str, name: str, description: str, expect_rev: int | None = None) -> dict:
    if not isinstance(description, str) or len(description) > 300:
        raise ListError("a description must be at most 300 characters")
    with _transaction(root, expect_rev) as d:
        e = _find(d, kind, name)
        if e is None:
            raise ListError(f"{name!r} is not on the list")
        e["description"] = description.strip()
    return d


def in_use(root: Path, kind: str, entry: dict) -> int:
    """How many runs use this entry (by its name or an alias)."""
    from . import review as rv
    names = {entry["name"], *entry.get("aliases", [])}
    n = 0
    for run_path, m in store.iter_manifests(Path(root), "committed"):
        try:
            eff = rv.effective(m, rv.load(run_path))
        except rv.ReviewError:
            continue
        if kind == "failure_types" and eff.get("failure_mode") in names:
            n += 1
        elif kind == "task_segments" and any(s["name"] in names for s in eff.get("subgoals", [])):
            n += 1
    return n


def delete(root: Path, kind: str, name: str, expect_rev: int | None = None) -> dict:
    with _transaction(root, expect_rev) as d:
        e = _find(d, kind, name)
        if e is None:
            raise ListError(f"{name!r} is not on the list")
        used = in_use(root, kind, e)
        if used:
            raise ListError(f"{e['name']!r} is used by {used} run{'s' if used > 1 else ''}; change those first")
        d[kind] = [x for x in d[kind] if x is not e]
    return d
