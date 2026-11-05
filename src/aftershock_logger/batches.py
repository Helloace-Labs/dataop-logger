"""Batches (packs in Aftershock): the runs collected with one policy version and reviewed together, so one training
set can be made from them. In DAgger, a batch is one round.

Stored as one file per batch, `dataop/batches/<batch_id>.json`:
  {"version": 1, "registry_rev": int, "seq": int, "batch_id", "name", "purpose", "status", "created_at",
   "policy": {family, source, revision} | None, "identity": identity | None, "mixed_policy": bool,
   "parent": {"batch_id", "export_id"} | None, "model": str | None, "runs": [run_id], "locked": lock | None,
   "exports": [export record], "history": [{"at", "by", "what"}]}
  identity = {"station", "task", "policy", "planner"}   (see run_identity)
  lock     = {"at", "by", "runs": {run_id: {"review_rev", "review_digest"}}, "forced"}

Every change runs under `dataop/batches.lock`. A change that touches several batch files (a move, or opening a new
collecting batch) first writes the new content of every file it changes to `dataop/batches/.journal.json`, then
the files, then removes the journal, so a crash in between is finished by the next writer and readers overlay the
journal meanwhile. The registry revision is the highest `registry_rev` of any file; every change raises it by one.

Datasets written before logger task L3 keep every batch in one `dataop/batches.json`
({"version": 1, "rev", "batches": [...], "members": {run_id: batch_id}}). Readers accept it as it is; the first change
writes the per-batch files and renames the old file to `batches.json.migrated` (kept, never deleted).

Lifecycle: collecting -> reviewing -> locked. New runs join only a collecting batch; review edits on a locked batch's
runs are refused (review.transaction checks), and export needs a locked batch. A locked batch's membership is frozen:
no run moves into or out of it, and no run is placed or imported into it, until it is unlocked with a reason.
Unlocking marks the batch's earlier exports stale. Run manifests never change: membership lives here. A committed run
that is not a member yet (recorded before batches existed, or a crash between commit and joining) is placed on first
read: into the batch named in its manifest's `batch` field if there is one and it is not locked, otherwise into an
unlocked batch with the same identity, otherwise into a new one.
"""
from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from . import store
from .schema import UNKNOWN, manifest_fields

STATUSES = ("collecting", "reviewing", "locked")
NAME = re.compile(r"^[^\n\r]{1,80}$")
BATCH_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")    # lower case only: the id is a file name on any file system
HISTORY_LIMIT = 200
FILE_VERSION = 1
JOURNAL = ".journal.json"
LEGACY = "batches.json"
_FILE_ONLY = ("version", "registry_rev", "runs")


class BatchError(ValueError):
    pass


def _dir(root: Path) -> Path:
    return store.dataop_dir(Path(root)) / "batches"


def _file(root: Path, batch_id: str) -> Path:
    return _dir(root) / f"{batch_id}.json"


def _legacy(root: Path) -> Path:
    return store.dataop_dir(Path(root)) / LEGACY


def _lock_path(root: Path) -> Path:
    return store.dataop_dir(Path(root)) / "batches.lock"


def _now() -> str:
    from .review import _now as now
    return now()


def _policy_key(policy: dict | str | None) -> dict | None:
    if not isinstance(policy, dict) or not policy:      # null (no policy) and "unknown" (not recorded)
        return None
    return {"family": policy.get("family"), "source": policy.get("source"), "revision": policy.get("revision")}


def _policy_name(policy: dict | str | None) -> str:
    if not isinstance(policy, dict) or not policy:
        return "unknown policy"
    rev = policy.get("revision")
    return f"{policy.get('source') or policy.get('family') or 'policy'}" + (f" @ {rev[:12]}" if rev else "")


def run_identity(manifest: dict) -> dict:
    """What a run's batch should share: station, task, policy and planner. Null (known to be none) and "unknown"
    (not recorded) stay apart. The logger records no station yet, so it is "unknown" unless the manifest names one.
    The task is the Aftershock task id; the planner is its id, since a changed planner gets a new id."""
    f = manifest_fields(manifest)
    st = manifest.get("station")
    pol = f["policy"]
    plan = f["planner"]
    return {"station": st if isinstance(st, str) and st else UNKNOWN,
            "task": f["task_id"] if isinstance(f["task_id"], str) else UNKNOWN,
            "policy": _policy_key(pol) if isinstance(pol, dict) else (None if pol is None else UNKNOWN),
            "planner": plan.get("id") if isinstance(plan, dict) else (None if plan is None else UNKNOWN)}


def _new(name: str, status: str, policy: dict | None = None, purpose: str = "", by: str = "dataop",
         identity: dict | None = None) -> dict:
    return {"batch_id": uuid.uuid4().hex[:12], "name": name, "purpose": purpose, "status": status,
            "created_at": _now(), "policy": _policy_key(policy), "identity": identity, "mixed_policy": False,
            "parent": None, "model": None, "locked": None, "exports": [],
            "history": [{"at": _now(), "by": by, "what": f"created ({status})"}]}


def _add(d: dict, b: dict) -> dict:
    b["seq"] = max((x.get("seq", -1) for x in d["batches"]), default=-1) + 1
    d["batches"].append(b)
    return b


def _unique_name(d: dict, base: str) -> str:
    """A name no batch has yet, for batches the logger names itself."""
    names = {b["name"] for b in d["batches"]}
    if base not in names:
        return base
    base, i = base[:72], 2
    while f"{base} ({i})" in names:
        i += 1
    return f"{base} ({i})"


def _empty() -> dict:
    return {"version": 1, "rev": 0, "batches": [], "members": {}}


def _check_batch(p, b) -> None:
    """Every field a batch needs, with its type, so nothing downstream trips over a hand-edited file."""
    bad = lambda what: BatchError(f"{p} has a malformed batch ({what})")
    if not isinstance(b, dict):
        raise bad("not an object")
    for k, t in (("batch_id", str), ("name", str), ("status", str), ("exports", list), ("history", list)):
        if not isinstance(b.get(k), t):
            raise bad(f"{k} missing or not a {t.__name__}")
    if b["status"] not in STATUSES:
        raise bad(f"status {b['status']!r}")
    for k in ("policy", "parent", "locked", "identity"):
        if b.get(k) is not None and not isinstance(b[k], dict):
            raise bad(f"{k} is not an object")
    if "mixed_policy" in b and not isinstance(b["mixed_policy"], bool):
        raise bad("mixed_policy is not true or false")
    lk = b.get("locked")
    if (b["status"] == "locked") != (lk is not None):
        raise bad("status and lock disagree")
    if lk is not None:
        runs = lk.get("runs")
        if not isinstance(runs, dict) or not all(
                isinstance(v, dict) and isinstance(v.get("review_rev"), int) and isinstance(v.get("review_digest"), str)
                for v in runs.values()):
            raise bad("lock runs")
    if not all(isinstance(x, dict) for x in b["exports"] + b["history"]):
        raise bad("exports or history entries")


def _parse_file(p: Path, f) -> tuple[dict, list, int]:
    """One per-batch file -> (batch, runs, registry_rev)."""
    if not isinstance(f, dict) or f.get("version") != FILE_VERSION:
        raise BatchError(f"{p} is not a version {FILE_VERSION} batch file")
    _check_batch(p, f)
    if f["batch_id"] != p.stem or not BATCH_ID.match(f["batch_id"]):
        raise BatchError(f"{p} holds batch {f['batch_id']!r}; the file name must be its batch id")
    runs, rev = f.get("runs"), f.get("registry_rev")
    if not isinstance(runs, list) or not all(isinstance(r, str) and r for r in runs) or len(set(runs)) != len(runs):
        raise BatchError(f"{p} has a malformed batch (runs must be distinct run ids)")
    if not isinstance(rev, int) or isinstance(rev, bool) or rev < 0 or not isinstance(f.get("seq", 0), int):
        raise BatchError(f"{p} has a malformed batch (registry_rev or seq)")
    return {k: v for k, v in f.items() if k not in _FILE_ONLY}, runs, rev


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise BatchError(f"{p} is unreadable ({e})") from e


def _journal(root: Path) -> dict | None:
    p = _dir(root) / JOURNAL
    try:
        j = _read_json(p)
    except FileNotFoundError:
        return None
    if not isinstance(j, dict) or j.get("version") != 1 or not isinstance(j.get("files"), dict):
        raise BatchError(f"{p} is not a version 1 batches journal")
    return j


def _read_legacy(p: Path) -> dict:
    d = _read_json(p)
    if not isinstance(d, dict) or d.get("version") != 1 or not isinstance(d.get("batches"), list) \
            or not isinstance(d.get("members"), dict) or not isinstance(d.get("rev", 0), int):
        raise BatchError(f"{p} is not a version 1 batches file")
    for b in d["batches"]:
        _check_batch(p, b)
    ids = [b["batch_id"] for b in d["batches"]]
    if len(set(ids)) != len(ids):
        raise BatchError(f"{p} has repeated batch ids")
    if any(v not in ids for v in d["members"].values()):
        raise BatchError(f"{p} lists a run in a batch that does not exist")
    for i, b in enumerate(d["batches"]):
        b.setdefault("seq", i)
        b.setdefault("identity", None)
        b.setdefault("mixed_policy", False)
    return {"version": 1, "rev": d.get("rev", 0), "batches": d["batches"], "members": d["members"]}


def _load(root: Path, strict: bool = True, derive: bool = True) -> tuple[dict, bool, dict]:
    """The registry as it is on disk, without locking: (registry, from the old single file, runs in several batches).
    A journal left by an interrupted change is overlaid, so readers see the change it was making."""
    root = Path(root)
    files = {}
    dr = _dir(root)
    if dr.is_dir():
        for p in dr.iterdir():
            if p.suffix == ".json" and not p.name.startswith("."):
                try:
                    files[p.stem] = (p, _read_json(p))
                except FileNotFoundError:                 # replaced between listing and reading
                    continue
    j = _journal(root) if dr.is_dir() else None
    for bid, f in (j or {}).get("files", {}).items():
        files[bid] = (_file(root, bid), f)
    legacy = _legacy(root)
    if not files and legacy.exists() and not (j and j.get("retire_legacy")):
        d = _read_legacy(legacy)
        if derive:
            _derive(root, d)
        return d, True, {}
    d = _empty()
    seen: dict[str, list[str]] = {}
    for bid, (p, f) in sorted(files.items()):
        b, runs, rev = _parse_file(p, f)
        d["batches"].append(b)
        d["rev"] = max(d["rev"], rev)
        for r in runs:
            seen.setdefault(r, []).append(bid)
    d["batches"].sort(key=lambda b: (b.get("seq", 0), b["created_at"] if isinstance(b.get("created_at"), str) else "",
                                     b["batch_id"]))
    dups = {r: bs for r, bs in seen.items() if len(bs) > 1}
    if dups and strict:
        r, bs = sorted(dups.items())[0]
        raise BatchError(f"run {r} is listed in several batches ({', '.join(bs)}); a run belongs to exactly one batch")
    locked = {b["batch_id"] for b in d["batches"] if b["status"] == "locked"}
    for r, bs in seen.items():                            # in a broken registry, a locked batch wins the run
        d["members"][r] = next((x for x in bs if x in locked), bs[0])
    return d, False, dups


def _read(root: Path) -> dict:
    """The registry, read under a shared lock so no change is half seen. A read-only dataset is read without it."""
    return _read_full(root)[0]


def _read_full(root: Path) -> tuple[dict, bool]:
    try:
        lock = open(_lock_path(root), "a+")
    except OSError:
        d, legacy, _ = _load(root)
        return d, legacy
    with lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            d, legacy, _ = _load(root)
            return d, legacy
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def has_batches(root: Path) -> bool:
    """Whether the dataset has a batch registry in either layout."""
    dr = _dir(root)
    return _legacy(root).exists() or (dr.is_dir() and any(p.suffix == ".json" for p in dr.iterdir()))


def _members_of(d: dict, bid: str) -> list[str]:
    return sorted(r for r, x in d["members"].items() if x == bid)


def _record(d: dict, b: dict) -> dict:
    return {**b, "runs": _members_of(d, b["batch_id"])}


def _derive(root: Path, d: dict, only: set | None = None) -> None:
    """Fill identity (from the runs, when the batch has none yet) and mixed_policy (some run's identity differs)."""
    for b in d["batches"]:
        if only is not None and b["batch_id"] not in only:
            continue
        ids = []
        for rid in _members_of(d, b["batch_id"]):
            try:
                ids.append(run_identity(store.read_manifest(store.runs_dir(Path(root)) / rid)))
            except Exception:                             # a missing run is reported by check, not here
                continue
        if not ids:
            b["mixed_policy"] = False
            continue
        if b.get("identity") is None:
            counts = Counter(json.dumps(i, sort_keys=True) for i in ids)
            best = max(counts.values())
            b["identity"] = next(i for i in ids if counts[json.dumps(i, sort_keys=True)] == best)
        b["mixed_policy"] = any(i != b["identity"] for i in ids)


def _place(root: Path, d: dict) -> bool:
    """Put committed runs that are not members yet into a batch, never a locked one. Returns True if anything
    changed."""
    changed = False
    for _, m in store.iter_manifests(Path(root), "committed"):
        rid = m["run_id"]
        if rid in d["members"]:
            continue
        ident = run_identity(m)
        hint = m.get("batch")
        b = None
        if isinstance(hint, dict):
            b = next((x for x in d["batches"] if x["batch_id"] == hint.get("batch_id")), None)
            if b is None and hint.get("name"):
                b = next((x for x in d["batches"] if x["name"] == hint["name"]), None)
            if b is not None and b["status"] == "locked":   # frozen: the run goes to a batch like it instead
                b = None
            elif b is None and isinstance(hint.get("name"), str) and NAME.match(hint["name"].strip() or "-"):
                b = _add(d, _new(hint["name"], "collecting" if not _collecting(d) else "reviewing", m.get("policy"),
                                 identity=ident))
                if isinstance(hint.get("batch_id"), str) and BATCH_ID.match(hint["batch_id"]):
                    b["batch_id"] = hint["batch_id"]
        if b is None:
            b = next((x for x in d["batches"] if x.get("identity") == ident and x["status"] != "locked"), None)
            if b is None:
                b = _add(d, _new(_unique_name(d, _policy_name(m.get("policy"))), "reviewing", m.get("policy"),
                                 by="dataop (existing runs)", identity=ident))
        d["members"][rid] = b["batch_id"]
        changed = True
    return changed


def _collecting(d: dict) -> dict | None:
    return next((b for b in d["batches"] if b["status"] == "collecting"), None)


@contextmanager
def _exclusive(root: Path):
    store.dataop_dir(Path(root)).mkdir(parents=True, exist_ok=True)
    with open(_lock_path(root), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _recover(root)
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _apply(root: Path, j: dict) -> None:
    """Write what a journal says, then drop the journal. Safe to repeat."""
    for bid, f in sorted(j["files"].items()):
        if not isinstance(bid, str) or not BATCH_ID.match(bid):
            raise BatchError(f"the batches journal names a bad batch id {bid!r}")
        store.atomic_write_json(_file(root, bid), f)
    legacy = _legacy(root)
    if j.get("retire_legacy") and legacy.exists():
        dst = legacy.with_name(LEGACY + ".migrated")
        if dst.exists():                                  # never overwrite an earlier copy
            dst = legacy.with_name(f"{LEGACY}.migrated-{uuid.uuid4().hex[:8]}")
        os.replace(legacy, dst)
    (_dir(root) / JOURNAL).unlink(missing_ok=True)
    store._fsync_dir(_dir(root))
    store._fsync_dir(store.dataop_dir(Path(root)))


def _recover(root: Path) -> None:
    """Under the exclusive lock: finish a change a crash interrupted."""
    j = _journal(root) if _dir(root).is_dir() else None
    if j is not None:
        _apply(root, j)


def _commit(root: Path, d: dict, before: dict, migrate: bool) -> bool:
    """Write every batch that changed (all of them when moving off the old single file). Returns True if written."""
    changed = [b for b in d["batches"] if migrate or _record(d, b) != before.get(b["batch_id"])]
    if not changed and not migrate:
        return False
    rev = d["rev"] + 1
    files = {}
    for b in changed:
        if not BATCH_ID.match(b["batch_id"]):
            raise BatchError(f"batch id {b['batch_id']!r} cannot be a file name; rename it in {LEGACY} by hand "
                             "(lower case letters, digits, - and _)")
        files[b["batch_id"]] = {**_record(d, b), "version": FILE_VERSION, "registry_rev": rev}
    j = {"version": 1, "files": files, "retire_legacy": migrate}
    if len(files) > 1 or migrate:
        store.atomic_write_json(_dir(root) / JOURNAL, j)
    _apply(root, j)
    d["rev"] = rev
    return True


@contextmanager
def _transaction(root: Path, expect_rev: int | None = None, place: bool = True):
    with _exclusive(root):
        d, legacy, _ = _load(root)
        before = {b["batch_id"]: copy.deepcopy(_record(d, b)) for b in d["batches"]}
        if expect_rev is not None and d["rev"] != expect_rev:
            raise BatchError(f"the batches changed (now at rev {d['rev']}, you had {expect_rev}); reload")
        if place:                               # callers adding runs themselves place them first
            _place(root, d)
        yield d
        moved = {b["batch_id"] for b in d["batches"]
                 if b["batch_id"] not in before or _record(d, b)["runs"] != before[b["batch_id"]]["runs"]}
        _derive(root, d, moved)
        _commit(root, d, before, legacy)


def load(root: Path) -> dict:
    """The batches with every committed run placed. Saves the placement (and moves an old single batches.json to
    per-batch files) the first time it is needed."""
    d, legacy = _read_full(root)
    probe = copy.deepcopy(d)
    if _place(root, probe) or legacy:
        _derive(root, probe)
        try:
            with _transaction(root):
                pass
        except OSError:                                  # read-only dataset: placement stays in memory
            return probe
        return _read(root)
    return d


def find(d: dict, ref: str) -> dict:
    """A batch by id or exact name."""
    b = next((x for x in d["batches"] if x["batch_id"] == ref), None) or \
        next((x for x in d["batches"] if x["name"] == ref), None)
    if b is None:
        raise BatchError(f"no batch {ref!r}")
    return b


def batch_of(root: Path, run_id: str) -> dict | None:
    """The batch a run belongs to, read without locking (the review lock check runs inside a review lock, and the
    batch lock is taken before review locks). Every lock and unlock rewrites one file atomically, and runs only move
    between unlocked batches, so the locked answer is never torn. An unreadable registry raises, so a broken file
    never unlocks a locked batch."""
    d, _, _ = _load(root, strict=False, derive=False)
    bid = d["members"].get(run_id)
    return next((b for b in d["batches"] if b["batch_id"] == bid), None) if bid else None


def check_editable(root: Path, run_id: str) -> None:
    b = batch_of(root, run_id)
    if b is not None and b["status"] == "locked":
        raise BatchError(f"this run is in batch {b['name']!r}, which is locked for export; unlock the batch to edit")


def _log(b: dict, by: str, what: str) -> None:
    b["history"].append({"at": _now(), "by": by, "what": what})
    del b["history"][:-HISTORY_LIMIT]


def _check_name(name) -> str:
    if not isinstance(name, str) or not NAME.match(name.strip()):
        raise BatchError("a batch name must be 1 to 80 characters on one line")
    return name.strip()


def _frozen(b: dict, what: str) -> BatchError:
    return BatchError(f"batch {b['name']!r} is locked, so its runs are frozen: {what}. Unlock it first "
                      "(unlocking needs a reason and marks its training sets out of date)")


def new(root: Path, name: str, purpose: str = "", by: str = "human", collecting: bool = True,
        expect_rev: int | None = None) -> dict:
    """Open a new batch. With collecting=True it becomes the batch new runs join; the previous collecting batch is
    closed for review."""
    name = _check_name(name)
    with _transaction(root, expect_rev) as d:
        if any(b["name"] == name for b in d["batches"]):
            raise BatchError(f"a batch named {name!r} already exists")
        if collecting:
            old = _collecting(d)
            if old:
                old["status"] = "reviewing"
                _log(old, by, f"closed: new batch {name!r} opened")
        b = _add(d, _new(name, "collecting" if collecting else "reviewing", purpose=purpose, by=by))
    return b


def join(root: Path, run_id: str, name: str | None, policy: dict | None, identity: dict | None = None) -> dict:
    """Called by the logger at commit: add the run to the batch named `name` (opened if needed), or to the collecting
    batch when the run's identity (station, task, policy, planner) matches it, or to a new batch named after the
    policy. A run added to a named batch with another identity is kept there and flags the batch mixed_policy."""
    if identity is None:
        try:
            identity = run_identity(store.read_manifest(store.runs_dir(Path(root)) / run_id))
        except Exception:
            identity = None
    with _transaction(root, place=False) as d:
        b = d["members"].get(run_id)
        if b:
            return next(x for x in d["batches"] if x["batch_id"] == b)
        target = None
        if name:
            target = next((x for x in d["batches"] if x["name"] == name), None)
            if target is not None and target["status"] != "collecting":
                raise BatchError(f"batch {name!r} is {target['status']}; open a new batch for new runs")
        if target is None:
            cur = _collecting(d)
            other = cur is not None and None not in (cur.get("identity"), identity) and cur["identity"] != identity
            if name or cur is None or other:
                if cur is not None:
                    cur["status"] = "reviewing"
                    _log(cur, "dataop", "closed: a new batch was opened by the logger" + (
                        " for a run with another station, task, policy or planner" if other and not name else ""))
                target = _add(d, _new(name or _unique_name(d, _policy_name(policy)), "collecting", policy,
                                      identity=identity))
            else:
                target = cur
        if target["policy"] is None and policy:
            target["policy"] = _policy_key(policy)
        d["members"][run_id] = target["batch_id"]
        return target


def add_runs(root: Path, run_ids: list[str], name: str, policy: dict | None = None, by: str = "import") -> dict:
    """Put new runs (an import) into the batch called `name`, opened for review if it does not exist."""
    name = _check_name(name)
    with _transaction(root, place=False) as d:
        b = next((x for x in d["batches"] if x["name"] == name), None)
        if b is None:
            b = _add(d, _new(name, "reviewing", policy, by=by))
        elif b["status"] == "locked":
            raise _frozen(b, f"{len(run_ids)} imported run(s) cannot join it")
        for rid in run_ids:
            cur = d["members"].get(rid)
            if cur is not None and cur != b["batch_id"]:
                src = find(d, cur)
                if src["status"] == "locked":
                    raise _frozen(src, f"run {rid} cannot move to {name!r}")
                _log(src, by, f"run {rid} moved out to {name!r} by an import")
        for rid in run_ids:
            d["members"][rid] = b["batch_id"]
        _log(b, by, f"{len(run_ids)} run(s) imported")
    return b


def target_for(root: Path, name: str | None) -> dict | None:
    """Which batch a new run would join, checked before recording so a run never records into a closed batch."""
    d = _read(root)
    if name:
        b = next((x for x in d["batches"] if x["name"] == name), None)
        if b is not None and b["status"] != "collecting":
            raise BatchError(f"batch {name!r} is {b['status']}; open a new batch for new runs")
        return b
    return _collecting(d)


def move(root: Path, run_ids: list[str], ref: str, by: str = "human", expect_rev: int | None = None) -> dict:
    """Move runs into another batch. The lock checks and the move happen under one lock, so a batch cannot be locked
    or unlocked halfway through. Training sets made from a batch a run leaves are marked out of date."""
    with _transaction(root, expect_rev) as d:
        dest = find(d, ref)
        if dest["status"] == "locked":
            raise _frozen(dest, "no run can move into it")
        sources = {}
        for rid in run_ids:
            if rid not in d["members"]:
                raise BatchError(f"no committed run {rid!r}")
            src = find(d, d["members"][rid])
            if src["status"] == "locked":
                raise _frozen(src, f"run {rid} cannot move out of it")
            if src is not dest:
                sources.setdefault(src["batch_id"], (src, []))[1].append(rid)
        for src, rids in sources.values():
            for rid in rids:
                d["members"][rid] = dest["batch_id"]
            for e in src["exports"]:
                e["stale"] = True
            _log(src, by, f"{len(rids)} run(s) moved out to {dest['name']!r}: {', '.join(sorted(rids))}")
        n = sum(len(r) for _, r in sources.values())
        if n:
            _log(dest, by, f"{n} run(s) moved in")
    return dest


def set_status(root: Path, ref: str, status: str, by: str = "human", reason: str | None = None,
               expect_rev: int | None = None, force: bool = False) -> dict:
    """close (-> reviewing), reopen (-> collecting), lock and unlock. Locking needs every run reviewed and no open
    suggestions, unless force=True (recorded in the history)."""
    from . import review as rv
    if status not in STATUSES:
        raise BatchError(f"status must be one of {STATUSES}")
    with _transaction(root, expect_rev) as d:
        b = find(d, ref)
        if b["status"] == status:
            return b
        if b["status"] == "locked":
            if not reason or not reason.strip():
                raise BatchError("unlocking a batch needs a reason")
            for e in b["exports"]:
                e["stale"] = True
            b["locked"] = None
            _log(b, by, f"unlocked: {reason.strip()}")
        if status == "collecting":
            cur = _collecting(d)
            if cur is not None and cur is not b:
                cur["status"] = "reviewing"
                _log(cur, by, f"closed: {b['name']!r} reopened")
        if status == "locked":
            runs, unreviewed, open_ = {}, [], []
            paths = {p.name: (p, m) for p, m in store.iter_manifests(Path(root), "committed")}
            for rid, bid in sorted(d["members"].items()):
                if bid != b["batch_id"]:
                    continue
                if rid not in paths:
                    raise BatchError(f"run {rid} is not a committed run; run validate")
                p, m = paths[rid]
                with open(p / "review.lock", "a+") as lk:      # wait for any edit in flight on this run
                    fcntl.flock(lk, fcntl.LOCK_EX)
                    try:
                        r = rv.load(p)
                        rv.check_review(m, r)
                        runs[rid] = {"review_rev": r["rev"], "review_digest": review_digest(m, r)}
                        if not r.get("reviewed"):
                            unreviewed.append(rid)
                        if any(e["status"] == "proposed" for e in r["edits"]):
                            open_.append(rid)
                    finally:
                        fcntl.flock(lk, fcntl.LOCK_UN)
            if not runs:
                raise BatchError("an empty batch cannot be locked")
            if not force and (unreviewed or open_):
                parts = [f"{len(unreviewed)} run{'s are' if len(unreviewed) != 1 else ' is'} not marked reviewed"
                         if unreviewed else "", f"{len(open_)} run{'s have' if len(open_) != 1 else ' has'} open "
                         "suggestions" if open_ else ""]
                raise BatchError(" and ".join(x for x in parts if x) + "; finish the review first or lock with force")
            b["locked"] = {"at": _now(), "by": by, "runs": runs, "forced": bool(force and (unreviewed or open_))}
            _log(b, by, f"locked ({len(runs)} runs)" + (f", forced with {len(unreviewed)} unreviewed and "
                                                         f"{len(open_)} with open suggestions" if b["locked"]["forced"] else ""))
        elif b["status"] != "locked":
            _log(b, by, {"collecting": "reopened for new runs", "reviewing": "closed to new runs"}[status])
        b["status"] = status
    return b


def review_digest(manifest: dict, review: dict) -> str:
    from . import review as rv
    return store.json_sha256(rv.effective(manifest, review))


def edit(root: Path, ref: str, by: str = "human", name: str | None = None, purpose: str | None = None,
         model: str | None = None, expect_rev: int | None = None) -> dict:
    """Rename, describe, or record the model trained from this batch's newest export (lineage)."""
    with _transaction(root, expect_rev) as d:
        b = find(d, ref)
        if name is not None:
            name = _check_name(name)
            if any(x["name"] == name and x is not b for x in d["batches"]):
                raise BatchError(f"a batch named {name!r} already exists")
            _log(b, by, f"renamed from {b['name']!r}")
            b["name"] = name
        if purpose is not None:
            if not isinstance(purpose, str) or len(purpose) > 300:
                raise BatchError("a purpose must be at most 300 characters")
            b["purpose"] = purpose.strip()
        if model is not None:
            if not isinstance(model, str) or not model.strip() or len(model) > 300:
                raise BatchError("a model must be a source or source@revision, at most 300 characters")
            b["model"] = model.strip()
            _log(b, by, f"model trained from it: {b['model']}")
            src, _, rev = b["model"].partition("@")
            for c in d["batches"]:
                p = c.get("policy") if isinstance(c.get("policy"), dict) else {}
                if c is not b and c["parent"] is None and p.get("source") == src and (not rev or p.get("revision") == rev):
                    newest = b["exports"][-1]["export_id"] if b["exports"] else None
                    c["parent"] = {"batch_id": b["batch_id"], "export_id": newest}
    return b


def add_export(sources: list[tuple[Path, str, dict | None]], record: dict) -> bool:
    """Register a finished export with every source batch at once: (root, batch_id, lock it was planned from).
    All dataset roots are locked together (in path order), so the export is current everywhere only if every batch
    still holds exactly its planned lock; otherwise it is registered as stale everywhere. Returns stale."""
    from contextlib import ExitStack
    roots = sorted({str(Path(r).resolve()) for r, _, _ in sources})
    with ExitStack() as stack:
        for r in roots:
            stack.enter_context(_exclusive(Path(r)))
        regs = {}
        for r in roots:
            d, legacy, _ = _load(Path(r))
            regs[r] = (d, {b["batch_id"]: copy.deepcopy(_record(d, b)) for b in d["batches"]}, legacy)
        targets, stale = [], False
        for root, bid, planned in sources:
            d = regs[str(Path(root).resolve())][0]
            b = find(d, bid)
            # a set made from a locked batch is current only while that lock holds; one made from an unlocked batch
            # is judged later, run by run, against the review digests it recorded (export.training_sets)
            stale |= planned is not None and (b["status"] != "locked" or b["locked"] != planned)
            targets.append((d, b, planned))
        for d, b, planned in targets:
            b["exports"].append({**record, "stale": stale})
            _log(b, record.get("by", "dataop"), f"exported {record.get('recipe')} to {record.get('path')}"
                 + (" (already out of date: a source batch changed while it was written)" if stale and planned else ""))
        for r in roots:
            d, before, legacy = regs[r]
            _commit(Path(r), d, before, legacy)
    return stale


def check(root: Path) -> list[str]:
    """Problems with the batch registry, for validate: schema, unique ids and names, one batch per run, membership
    of committed runs, and that locked runs still match their lock."""
    from . import review as rv
    try:
        d, legacy, dups = _load(root, strict=False, derive=False)
    except (BatchError, AttributeError, TypeError, KeyError) as e:
        return [f"batches: {e}"]
    errs = [f"batches: run {r} is listed in several batches ({', '.join(bs)})" for r, bs in sorted(dups.items())]
    ids = [b["batch_id"] for b in d["batches"]]
    names = [b["name"] for b in d["batches"]]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        errs.append("batches: repeated batch ids or names")
    if sum(b["status"] == "collecting" for b in d["batches"]) > 1:
        errs.append("batches: more than one collecting batch")
    runs = {p.name: (p, m) for p, m in store.iter_manifests(Path(root), "committed")}
    for rid in d["members"]:
        if rid not in runs:
            errs.append(f"batches: run {rid} is listed but is not a committed run")
    for b in d["batches"]:
        if b["status"] == "locked":
            lk = b.get("locked") or {}
            mine = _members_of(d, b["batch_id"]) + sorted(r for r, bs in dups.items()
                                                          if b["batch_id"] in bs and d["members"][r] != b["batch_id"])
            if sorted(lk.get("runs", {})) != sorted(mine):
                errs.append(f"batch {b['name']!r} is locked but its runs differ from the lock")
            for rid in mine:
                if rid in runs and rid in lk.get("runs", {}):
                    p, m = runs[rid]
                    try:
                        r = rv.load(p)
                        if review_digest(m, r) != lk["runs"][rid]["review_digest"]:
                            errs.append(f"run {rid} changed after batch {b['name']!r} was locked")
                    except rv.ReviewError as e:
                        errs.append(f"run {rid}: {e}")
    return errs


def copy_to(src: Path, dst: Path) -> str | None:
    """Copy a dataset's batches to another root (remap). Keeps the destination's own batches if it has different
    ones. Returns a line for the report, or None when the source has no batches."""
    if not has_batches(src):
        return None
    d, _ = _read_full(src)
    with _exclusive(dst):
        cur, _, _ = _load(dst)
        if cur["batches"] or has_batches(dst):
            strip = lambda x: [_record(x, b) for b in x["batches"]]
            if strip(cur) == strip(d):
                return "batches already match"
            return "kept the destination's batches; they differ from the source's (merge by hand)"
        _commit(Path(dst), d, {}, False)
    return f"copied {len(d['batches'])} batch(es)"


def summary(root: Path) -> list[dict]:
    """Every batch with its runs and counts, newest first, for the viewer and `batches list`."""
    from . import review as rv
    d = load(root)
    stats = {b["batch_id"]: {"runs": 0, "reviewed": 0, "success": 0, "rescued": 0, "ids": set()} for b in d["batches"]}
    runs = {b["batch_id"]: [] for b in d["batches"]}
    for p, m in store.iter_manifests(Path(root), "committed"):
        bid = d["members"].get(m["run_id"])
        if bid is None:
            continue
        s = stats[bid]
        s["runs"] += 1
        runs[bid].append(m["run_id"])
        s["ids"].add(json.dumps(run_identity(m), sort_keys=True))
        try:
            r = rv.load(p)
            eff = rv.effective(m, r)
        except rv.ReviewError:
            continue
        s["reviewed"] += bool(r.get("reviewed"))
        s["success"] += eff["termination_reason"] == "success" and not eff["rejected"]
        s["rescued"] += any(t.get("control_mode") in ("expert", "blend") for t in m.get("takeovers", []))
    out = []
    for b in reversed(d["batches"]):
        s = stats[b["batch_id"]]
        ident = b.get("identity")
        mixed = len(s["ids"]) > 1 or (ident is not None and bool(s["ids"])
                                      and s["ids"] != {json.dumps(ident, sort_keys=True)})
        out.append({**{k: b[k] for k in ("batch_id", "name", "purpose", "status", "created_at", "policy", "parent",
                                          "model")},
                    "identity": ident, "runs": sorted(runs[b["batch_id"]]), "n_runs": s["runs"],
                    "reviewed": s["reviewed"], "success": s["success"], "rescued": s["rescued"], "mixed_policy": mixed,
                    "locked": b["locked"] and {"at": b["locked"]["at"], "by": b["locked"]["by"]},
                    "exports": b["exports"], "history": b["history"][-20:]})
    return out
