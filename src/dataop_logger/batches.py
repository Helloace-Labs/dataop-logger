"""Batches: the runs collected with one policy version and reviewed together, so one training set can be made from
them. In DAgger, a batch is one round.

Stored in `dataop/batches.json`:
  {"version": 1, "rev": int, "batches": [batch], "members": {run_id: batch_id}}
  batch = {"batch_id", "name", "purpose", "status", "created_at", "policy": {family, source, revision} | None,
           "parent": {"batch_id", "export_id"} | None, "model": str | None, "locked": lock | None,
           "exports": [export record], "history": [{"at", "by", "what"}]}
  lock  = {"at", "by", "runs": {run_id: {"review_rev", "review_digest"}}}

Lifecycle: collecting -> reviewing -> locked. New runs join only a collecting batch; review edits on a locked batch's
runs are refused (review.transaction checks), and export needs a locked batch. Unlocking needs a reason and marks the
batch's earlier exports stale. Run manifests never change: membership lives here. A committed run that is not a
member yet (recorded before batches existed, or a crash between commit and joining) is placed on first read: into the
batch named in its manifest's `batch` field if there is one, otherwise into a batch for its policy.
"""
from __future__ import annotations

import copy
import fcntl
import json
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import store

STATUSES = ("collecting", "reviewing", "locked")
NAME = re.compile(r"^[^\n\r]{1,80}$")
HISTORY_LIMIT = 200


class BatchError(ValueError):
    pass


def _path(root: Path) -> Path:
    return store.dataop_dir(Path(root)) / "batches.json"


def _now() -> str:
    from .review import _now as now
    return now()


def _policy_key(policy: dict | None) -> dict | None:
    if not policy:
        return None
    return {"family": policy.get("family"), "source": policy.get("source"), "revision": policy.get("revision")}


def _policy_name(policy: dict | None) -> str:
    if not policy:
        return "unknown policy"
    rev = policy.get("revision")
    return f"{policy.get('source') or policy.get('family') or 'policy'}" + (f" @ {rev[:12]}" if rev else "")


def _new(name: str, status: str, policy: dict | None = None, purpose: str = "", by: str = "dataop") -> dict:
    return {"batch_id": uuid.uuid4().hex[:12], "name": name, "purpose": purpose, "status": status,
            "created_at": _now(), "policy": _policy_key(policy), "parent": None, "model": None, "locked": None,
            "exports": [], "history": [{"at": _now(), "by": by, "what": f"created ({status})"}]}


def _empty() -> dict:
    return {"version": 1, "rev": 0, "batches": [], "members": {}}


def _read(root: Path) -> dict:
    p = _path(root)
    if not p.exists():
        return _empty()
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise BatchError(f"{p} is unreadable ({e})") from e
    if not isinstance(d, dict) or d.get("version") != 1 or not isinstance(d.get("batches"), list) \
            or not isinstance(d.get("members"), dict):
        raise BatchError(f"{p} is not a version 1 batches file")
    for b in d["batches"]:
        _check_batch(p, b)
    ids = {b["batch_id"] for b in d["batches"]}
    if any(v not in ids for v in d["members"].values()):
        raise BatchError(f"{p} lists a run in a batch that does not exist")
    return d


def _check_batch(p: Path, b) -> None:
    """Every field a batch needs, with its type, so nothing downstream trips over a hand-edited file."""
    bad = lambda what: BatchError(f"{p} has a malformed batch ({what})")
    if not isinstance(b, dict):
        raise bad("not an object")
    for k, t in (("batch_id", str), ("name", str), ("status", str), ("exports", list), ("history", list)):
        if not isinstance(b.get(k), t):
            raise bad(f"{k} missing or not a {t.__name__}")
    if b["status"] not in STATUSES:
        raise bad(f"status {b['status']!r}")
    for k in ("policy", "parent", "locked"):
        if b.get(k) is not None and not isinstance(b[k], dict):
            raise bad(f"{k} is not an object")
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


def _place(root: Path, d: dict) -> bool:
    """Put committed runs that are not members yet into a batch. Returns True if anything changed."""
    changed = False
    for _, m in store.iter_manifests(Path(root), "committed"):
        rid = m["run_id"]
        if rid in d["members"]:
            continue
        hint = m.get("batch")
        b = None
        if isinstance(hint, dict):
            b = next((x for x in d["batches"] if x["batch_id"] == hint.get("batch_id")), None)
            if b is None and hint.get("name"):
                b = next((x for x in d["batches"] if x["name"] == hint["name"]), None)
            if b is None and hint.get("name"):
                b = _new(hint["name"], "collecting" if not _collecting(d) else "reviewing", m.get("policy"))
                if hint.get("batch_id"):
                    b["batch_id"] = hint["batch_id"]
                d["batches"].append(b)
        if b is None:
            key = _policy_key(m.get("policy"))
            b = next((x for x in d["batches"] if x["policy"] == key and x["status"] != "locked"), None)
            if b is None:
                b = _new(_policy_name(m.get("policy")), "reviewing", m.get("policy"), by="dataop (existing runs)")
                d["batches"].append(b)
        d["members"][rid] = b["batch_id"]
        changed = True
    return changed


def _collecting(d: dict) -> dict | None:
    return next((b for b in d["batches"] if b["status"] == "collecting"), None)


@contextmanager
def _transaction(root: Path, expect_rev: int | None = None, place: bool = True):
    store.dataop_dir(Path(root)).mkdir(parents=True, exist_ok=True)
    with open(store.dataop_dir(Path(root)) / "batches.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            d = _read(root)
            if expect_rev is not None and d["rev"] != expect_rev:
                raise BatchError(f"the batches changed (now at rev {d['rev']}, you had {expect_rev}); reload")
            if place:                               # callers adding runs themselves place them first
                _place(root, d)
            yield d
            d["rev"] += 1
            store.atomic_write_json(_path(root), d)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load(root: Path) -> dict:
    """The batches with every committed run placed. Saves the placement the first time it is needed."""
    d = _read(root)
    probe = copy.deepcopy(d)
    if _place(root, probe):
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
    """The batch a run belongs to, read without writing (used by the review lock check). An unreadable batches file
    raises, so a broken registry never unlocks a locked batch."""
    d = _read(root)
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
        b = _new(name, "collecting" if collecting else "reviewing", purpose=purpose, by=by)
        d["batches"].append(b)
    return b


def join(root: Path, run_id: str, name: str | None, policy: dict | None) -> dict:
    """Called by the logger at commit: add the run to the batch named `name` (opened if needed), or to the collecting
    batch, or to a new batch named after the policy."""
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
            if name or cur is None:
                if cur is not None:
                    cur["status"] = "reviewing"
                    _log(cur, "dataop", "closed: a new batch was opened by the logger")
                target = _new(name or _policy_name(policy), "collecting", policy)
                d["batches"].append(target)
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
            b = _new(name, "reviewing", policy, by=by)
            d["batches"].append(b)
        elif b["status"] == "locked":
            raise BatchError(f"batch {name!r} is locked")
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
    with _transaction(root, expect_rev) as d:
        dest = find(d, ref)
        if dest["status"] == "locked":
            raise BatchError(f"batch {dest['name']!r} is locked")
        for rid in run_ids:
            if rid not in d["members"]:
                raise BatchError(f"no committed run {rid!r}")
            src = next(x for x in d["batches"] if x["batch_id"] == d["members"][rid])
            if src["status"] == "locked":
                raise BatchError(f"run {rid} is in locked batch {src['name']!r}")
        for rid in run_ids:
            d["members"][rid] = dest["batch_id"]
        _log(dest, by, f"{len(run_ids)} run(s) moved in")
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
                p = c.get("policy") or {}
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
            store.dataop_dir(Path(r)).mkdir(parents=True, exist_ok=True)
            f = stack.enter_context(open(store.dataop_dir(Path(r)) / "batches.lock", "a+"))
            fcntl.flock(f, fcntl.LOCK_EX)
            stack.callback(fcntl.flock, f, fcntl.LOCK_UN)
        regs = {r: _read(Path(r)) for r in roots}
        targets, stale = [], False
        for root, bid, planned in sources:
            d = regs[str(Path(root).resolve())]
            b = find(d, bid)
            stale |= planned is None or b["status"] != "locked" or b["locked"] != planned
            targets.append((d, b, planned))
        for d, b, planned in targets:
            b["exports"].append({**record, "stale": stale})
            _log(b, record.get("by", "dataop"), f"exported {record.get('recipe')} to {record.get('path')}"
                 + (" (already out of date: a source batch changed while it was written)" if stale and planned else ""))
        for r in roots:
            regs[r]["rev"] += 1
            store.atomic_write_json(_path(Path(r)), regs[r])
    return stale


def check(root: Path) -> list[str]:
    """Problems with dataop/batches.json, for validate: schema, unique ids and names, membership of committed runs,
    and that locked runs still match their lock."""
    from . import review as rv
    try:
        d = _read(root)
    except (BatchError, AttributeError, TypeError, KeyError) as e:
        return [f"batches.json: {e}"]
    errs = []
    ids = [b["batch_id"] for b in d["batches"]]
    names = [b["name"] for b in d["batches"]]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        errs.append("batches.json has repeated batch ids or names")
    if sum(b["status"] == "collecting" for b in d["batches"]) > 1:
        errs.append("batches.json has more than one collecting batch")
    runs = {p.name: (p, m) for p, m in store.iter_manifests(Path(root), "committed")}
    for rid in d["members"]:
        if rid not in runs:
            errs.append(f"batches.json lists run {rid}, which is not a committed run")
    for b in d["batches"]:
        if b["status"] == "locked":
            lk = b.get("locked") or {}
            mine = sorted(r for r, x in d["members"].items() if x == b["batch_id"])
            if sorted(lk.get("runs", {})) != mine:
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


def summary(root: Path) -> list[dict]:
    """Every batch with its runs and counts, newest first, for the viewer and `batches list`."""
    from . import review as rv
    d = load(root)
    stats = {b["batch_id"]: {"runs": 0, "reviewed": 0, "success": 0, "rescued": 0, "policies": set()} for b in d["batches"]}
    runs = {b["batch_id"]: [] for b in d["batches"]}
    for p, m in store.iter_manifests(Path(root), "committed"):
        bid = d["members"].get(m["run_id"])
        if bid is None:
            continue
        s = stats[bid]
        s["runs"] += 1
        runs[bid].append(m["run_id"])
        s["policies"].add(json.dumps(_policy_key(m.get("policy")), sort_keys=True))
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
        out.append({**{k: b[k] for k in ("batch_id", "name", "purpose", "status", "created_at", "policy", "parent",
                                          "model")},
                    "runs": sorted(runs[b["batch_id"]]), "n_runs": s["runs"], "reviewed": s["reviewed"],
                    "success": s["success"], "rescued": s["rescued"], "mixed_policy": len(s["policies"]) > 1,
                    "locked": b["locked"] and {"at": b["locked"]["at"], "by": b["locked"]["by"]},
                    "exports": b["exports"], "history": b["history"][-20:]})
    return out
