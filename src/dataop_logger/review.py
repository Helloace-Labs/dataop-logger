"""Human review of committed runs: corrections stored next to the run, never applied to the recorded data.

Each run may have `dataop/runs/<run_id>/review.json`. It holds edits (agent proposals and human edits) and whether
the run has been reviewed. The recorded frames, the manifest and its checksums never change; `effective()` folds the
accepted edits into what training should use, and the reader turns that into `dataop.train_mask`.

Edit kinds and their payloads:
  outcome        {"termination_reason": str, "failure_mode": str | None}
  failure_start  {"frame": int}       on a failed run, frames from here to the next takeover are not good behaviour
  exclude        {"start": int, "end": int}   frames [start, end) are left out of training
  step_boundary  {"index": int, "before": str, "after": str, "frame": int}
                 moves the shared boundary between recorded steps `index` and `index + 1`
  reject_run     {}                           the whole run is left out of training
  note           {"frame": int, "text": str}
  label          {"start": int, "end": int, "text": str, "driver": drive mode, e.g. "policy" or "rescue" | None, "segment": str | None}
                 a person's description of frames [start, end) in plain language. An agent reads it later and
                 records what it means with `interpret()`; an interpretation of train="exclude" leaves the span
                 out of training.

Single-valued kinds (outcome, failure_start, reject_run, and step_boundary per index): the most recently accepted
edit wins, and undoing it brings back the one accepted before it. Every change to review.json happens under a
cross-process lock and bumps `rev`.
"""
from __future__ import annotations

import contextvars
import copy
import fcntl
import json
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import store
from .schema import TERMINATION_REASONS

KINDS = ("outcome", "failure_start", "exclude", "step_boundary", "reject_run", "note", "label", "event_note")
TRAIN_USE = ("keep", "exclude")
SOURCES = ("agent", "human")
STATUSES = ("proposed", "accepted", "rejected")
MULTI = ("exclude", "note", "label")        # kinds where every accepted edit applies
TAKEOVER_MODES = ("expert", "blend")
EDIT_ID = re.compile(r"^[0-9a-f]{12}$")


class ReviewError(ValueError):
    pass


class ReviewConflict(ReviewError):
    """The review changed since the caller read it."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _path(run_path: Path) -> Path:
    return Path(run_path) / "review.json"


def empty() -> dict:
    return {"version": 1, "rev": 0, "seq": 0, "reviewed": False, "reviewed_at": None, "edits": []}


def load(run_path: Path) -> dict:
    p = _path(run_path)
    if not p.exists():
        return empty()
    try:
        r = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ReviewError(f"{p} is unreadable ({e})") from e
    if not isinstance(r, dict) or r.get("version") != 1 or not isinstance(r.get("edits"), list):
        raise ReviewError(f"{p} is not a version 1 review file")
    for e in r["edits"]:
        if not isinstance(e, dict) or not EDIT_ID.match(str(e.get("id"))) or e.get("kind") not in KINDS \
                or e.get("status") not in STATUSES or e.get("source") not in SOURCES:
            raise ReviewError(f"{p} has a malformed edit")
    r.setdefault("rev", 0)
    r.setdefault("seq", 0)
    return r


_RECORDING = contextvars.ContextVar("dataop_review_recording", default=None)
HISTORY_LIMIT = 200                  # steps kept for undo
HISTORY_BYTES = 2_000_000            # and at most this much JSON; the oldest steps are dropped first


@contextmanager
def recording(what: str):
    """Record the edits made inside this block as one undoable step in review["history"].
    Yields a dict whose "entry" is the history id once the step is saved (None if nothing changed)."""
    box = {"what": what, "entry": None}
    token = _RECORDING.set(box)
    try:
        yield box
    finally:
        _RECORDING.reset(token)


@contextmanager
def transaction(run_path: Path, expect_rev: int | None = None):
    """Load, change and save one run's review under an exclusive cross-process lock."""
    with open(Path(run_path) / "review.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            review = load(run_path)
            if expect_rev is not None and review["rev"] != expect_rev:
                raise ReviewConflict(f"the review changed (now at rev {review['rev']}, you had {expect_rev})")
            box = _RECORDING.get()
            if box is not None and box.get("used"):
                raise ReviewError("a recorded step may contain one transaction only")
            before = {e["id"]: copy.deepcopy(e) for e in review["edits"]} if box is not None else None
            yield review
            if box is not None:
                after = {e["id"]: e for e in review["edits"]}
                changes = [{"id": i, "before": before.get(i), "after": copy.deepcopy(after.get(i))}
                           for i in list(before) + [i for i in after if i not in before]
                           if before.get(i) != after.get(i)]
                box["used"] = True
                if changes:
                    entry = {"id": uuid.uuid4().hex[:12], "at": _now(), "t": time.time_ns(), "what": box["what"],
                             "changes": changes, "undone": False}
                    hist = review.setdefault("history", [])
                    if len(json.dumps(entry)) <= HISTORY_BYTES:   # a single huge step is not kept for undo
                        hist.append(entry)
                    del hist[:-HISTORY_LIMIT]
                    while hist and len(json.dumps(hist)) > HISTORY_BYTES:
                        del hist[0]
                    box["entry"] = entry["id"] if hist and hist[-1] is entry else None
            review["rev"] += 1
            store.atomic_write_json(_path(run_path), review)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def undoable(review: dict) -> list[dict]:
    """Steps that can be undone right now, oldest first: [{entry, what, t}]. A step is offered only while every
    edit it touched is still exactly as the step left it, so a later change (from the page, an agent or a script)
    hides it until that later change is undone."""
    current = {e["id"]: e for e in review["edits"]}
    return [{"entry": h["id"], "what": h["what"], "t": h["t"]} for h in review.get("history", [])
            if not h["undone"] and all(current.get(c["id"]) == c["after"] for c in h["changes"])]


def undo(run_path: Path, manifest: dict, entry_id: str, expect_rev: int | None = None) -> dict:
    """Revert one recorded step: every edit it touched goes back to how it was, if nothing changed it since.
    The result is validated like any stored review before it is saved."""
    with transaction(run_path, expect_rev) as review:
        entry = next((h for h in review.get("history", []) if h.get("id") == entry_id), None)
        if entry is None:
            raise ReviewError("nothing to undo with that id")
        if entry.get("undone"):
            raise ReviewError("that step was already undone")
        current = {e["id"]: e for e in review["edits"]}
        for c in entry["changes"]:
            if current.get(c["id"]) != c["after"]:
                raise ReviewError("can't undo: those frames were changed again since; undo the later change first")
        for c in entry["changes"]:
            if c["before"] is None:
                review["edits"] = [e for e in review["edits"] if e["id"] != c["id"]]
            elif c["id"] in current:
                review["edits"][review["edits"].index(current[c["id"]])] = copy.deepcopy(c["before"])
            else:
                review["edits"].append(copy.deepcopy(c["before"]))
        entry["undone"], entry["undone_at"] = True, _now()
        check_review(manifest, review)
        return entry


def check_payload(kind: str, payload: dict, manifest: dict) -> dict:
    """Validate and normalise one edit's payload against the recorded run."""
    length = manifest["length"]
    if kind not in KINDS:
        raise ReviewError(f"kind must be one of {KINDS}")
    if not isinstance(payload, dict):
        raise ReviewError("payload must be an object")

    def frame(v, name, lo, hi):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ReviewError(f"{name} must be an integer frame in [{lo}, {hi}]")
        return int(v)

    if kind == "outcome":
        tr = payload.get("termination_reason")
        if tr not in TERMINATION_REASONS:
            raise ReviewError(f"termination_reason must be one of {TERMINATION_REASONS}")
        fm = payload.get("failure_mode")
        if fm is not None and (not isinstance(fm, str) or not fm.strip() or len(fm) > 200):
            raise ReviewError("failure_mode must be a short non-empty string or null")
        if tr == "success" and fm is not None:
            raise ReviewError("a successful run has no failure_mode")
        return {"termination_reason": tr, "failure_mode": fm}
    if kind == "failure_start":
        return {"frame": frame(payload.get("frame"), "frame", 0, length - 1)}
    if kind == "exclude":
        a, b = frame(payload.get("start"), "start", 0, length), frame(payload.get("end"), "end", 0, length)
        if a >= b:
            raise ReviewError("exclude needs start < end")
        return {"start": a, "end": b}
    if kind == "step_boundary":
        steps = manifest.get("subgoals", [])
        i = payload.get("index")
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(steps) - 1:
            raise ReviewError("step_boundary needs the index of a recorded step that has a next step")
        a, b = steps[i], steps[i + 1]
        if a["end_frame"] != b["start_frame"]:
            raise ReviewError(f"steps {a['name']!r} and {b['name']!r} do not share a boundary")
        if payload.get("before", a["name"]) != a["name"] or payload.get("after", b["name"]) != b["name"]:
            raise ReviewError("step names do not match the steps at that index")
        f = frame(payload.get("frame"), "frame", a["start_frame"] + 1, b["end_frame"] - 1)   # each step keeps a frame
        return {"index": i, "before": a["name"], "after": b["name"], "frame": f}
    if kind == "reject_run":
        return {}
    if kind == "event_note":
        key, text = payload.get("key"), payload.get("text")
        if not isinstance(key, str) or not 0 < len(key) <= 120 or any(c in key for c in "\n\r"):
            raise ReviewError("event_note needs the event's key")
        if not isinstance(text, str) or not text.strip() or len(text) > 500:
            raise ReviewError("a note must be 1 to 500 characters")
        return {"key": key, "text": text.strip()}
    if kind == "label":
        a, b = frame(payload.get("start"), "start", 0, length), frame(payload.get("end"), "end", 0, length)
        if a >= b:
            raise ReviewError("a label needs start < end")
        text = payload.get("text") or ""
        driver, segment = payload.get("driver"), payload.get("segment")
        if not isinstance(text, str) or len(text) > 2000:
            raise ReviewError("label text must be at most 2000 characters")
        if driver is not None and (not isinstance(driver, str) or not driver.strip() or len(driver) > 50):
            raise ReviewError("drive mode must be a short name (policy, rescue or your own) or null")
        if segment is not None and (not isinstance(segment, str) or not segment.strip() or len(segment) > 100):
            raise ReviewError("segment must be a short name or null")
        if not text.strip():
            raise ReviewError("a label needs a description of what happens")
        return {"start": a, "end": b, "text": text.strip(), "driver": driver.strip() if driver else None,
                "segment": segment.strip() if segment else None}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ReviewError("note text must be 1 to 2000 characters")
    return {"frame": frame(payload.get("frame"), "frame", 0, length - 1), "text": text}


def _slot(e: dict):
    """Edits that compete: same kind, and for step boundaries the same index."""
    if e["kind"] == "step_boundary":
        return (e["kind"], e["payload"].get("index"))
    if e["kind"] == "event_note":
        return (e["kind"], e["payload"].get("key"))
    return e["kind"]


def _accept(review: dict, edit: dict) -> None:
    review["seq"] += 1
    edit["status"], edit["decided_at"], edit["accepted_seq"] = "accepted", _now(), review["seq"]


def add_edit(run_path: Path, manifest: dict, kind: str, payload: dict, source: str = "human",
             reason: str | None = None, expect_rev: int | None = None, by: str | None = None) -> dict:
    """Add an edit. Human edits are accepted at once; agent edits wait as proposals and reopen the run's review."""
    if source not in SOURCES:
        raise ReviewError(f"source must be one of {SOURCES}")
    payload = check_payload(kind, payload, manifest)
    with transaction(run_path, expect_rev) as review:
        if source == "agent":
            prior, _ = _prior_suggestion(review, kind, payload)
            if prior is not None:
                return prior
        edit = {"id": uuid.uuid4().hex[:12], "kind": kind, "payload": payload, "source": source, "reason": reason,
                "status": "proposed", "created_at": _now(), "decided_at": None}
        if by is not None:
            edit["by"] = by
        review["edits"].append(edit)
        if source == "human":
            _accept(review, edit)
    return edit


def _prior_suggestion(review: dict, kind: str, payload: dict) -> tuple[dict | None, str]:
    """An agent suggestion for exactly this change that already exists, and what became of it.
    A person's decision stands: a span someone kept is not suggested again, whichever agent asks."""
    if kind == "exclude":                       # these exact frames are cut right now, by anyone
        for e in active_edits(review):
            if e["kind"] == "exclude" and e["payload"] == payload:
                return e, "already_cut"
    for e in review["edits"]:
        if e["source"] == "agent" and e["kind"] == kind and e["payload"] == payload and e["status"] != "accepted":
            return e, {"proposed": "already_suggested",
                       "rejected": "previously_kept" if kind == "exclude" else "previously_dismissed"}[e["status"]]
    return None, "created"


def add_suggestions(run_path: Path, manifest: dict, items: list[dict], by: str,
                    expect_rev: int | None = None) -> list[dict]:
    """Add a batch of agent cut suggestions in one locked, revision-checked transaction.
    items: [{"start", "end", "reason"}]. Returns one {"status", "edit"} per item, status one of created,
    already_suggested, already_cut or previously_kept."""
    if not isinstance(by, str) or not by.strip() or len(by) > 100:
        raise ReviewError("by must name the agent in 1 to 100 characters")
    checked = []
    for it in items:
        reason = it.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 300:
            raise ReviewError("a suggestion needs a short description (1 to 300 characters)")
        checked.append((check_payload("exclude", {"start": it.get("start"), "end": it.get("end")}, manifest),
                        reason.strip()))
    out = []
    with transaction(run_path, expect_rev) as review:
        for payload, reason in checked:
            prior, status = _prior_suggestion(review, "exclude", payload)
            if prior is None:
                prior = {"id": uuid.uuid4().hex[:12], "kind": "exclude", "payload": payload, "source": "agent",
                         "reason": reason, "by": by.strip(), "status": "proposed", "created_at": _now(),
                         "decided_at": None}
                review["edits"].append(prior)
            out.append({"status": status, "edit": prior})
    return out


def decide(run_path: Path, manifest: dict, edit_id: str, accept: bool, payload: dict | None = None,
           expect_rev: int | None = None) -> dict:
    """Accept (optionally adjusted) or reject an edit. Rejecting an accepted edit undoes it."""
    if not isinstance(accept, bool):
        raise ReviewError("accept must be true or false")
    with transaction(run_path, expect_rev) as review:
        edit = next((e for e in review["edits"] if e["id"] == edit_id), None)
        if edit is None:
            raise ReviewError(f"no edit {edit_id!r} on this run")
        if accept and edit["status"] != "proposed":
            raise ReviewError("only a proposal can be accepted; change an accepted edit by replacing it")
        if not accept and edit["status"] == "rejected":
            raise ReviewError("this edit is already rejected")
        if accept:
            if payload is not None:
                edit["payload"] = check_payload(edit["kind"], payload, manifest)
                edit["adjusted"] = True
            _accept(review, edit)
        else:
            edit["status"], edit["decided_at"] = "rejected", _now()
    return edit


def add_event_note(run_path: Path, manifest: dict, key: str, text: str, by: str, source: str = "agent",
                   expect_rev: int | None = None) -> dict:
    """A note on a major event (its details). Notes are descriptions, not training changes, so an agent's note
    applies at once like a person's; the newest note for an event wins and undo restores the previous one."""
    if source not in SOURCES:
        raise ReviewError(f"source must be one of {SOURCES}")
    if not isinstance(by, str) or not by.strip() or len(by) > 100:
        raise ReviewError("by must name who wrote the note in 1 to 100 characters")
    payload = check_payload("event_note", {"key": key, "text": text}, manifest)
    with transaction(run_path, expect_rev) as review:
        edit = {"id": uuid.uuid4().hex[:12], "kind": "event_note", "payload": payload, "source": source,
                "reason": None, "by": by.strip(), "status": "proposed", "created_at": _now(), "decided_at": None}
        review["edits"].append(edit)
        _accept(review, edit)
    return edit


def decide_many(run_path: Path, manifest: dict, edit_ids: list, accept: bool, expect_rev: int | None = None) -> int:
    """Accept or dismiss several proposals in one step (for "Cut all" and "Keep all")."""
    if not isinstance(accept, bool):
        raise ReviewError("accept must be true or false")
    if not isinstance(edit_ids, list) or not edit_ids or len(edit_ids) > 1000:
        raise ReviewError("edit_ids must be a non-empty list")
    with transaction(run_path, expect_rev) as review:
        by_id = {e["id"]: e for e in review["edits"]}
        todo = [by_id.get(i) for i in edit_ids]
        if any(e is None or e["status"] != "proposed" for e in todo):
            raise ReviewError("every id must be an open proposal on this run")
        for e in todo:
            if accept:
                _accept(review, e)
            else:
                e["status"], e["decided_at"] = "rejected", _now()
    return len(todo)


def replace_edit(run_path: Path, manifest: dict, edit_id: str, payload: dict, expect_rev: int | None = None) -> dict:
    """A person rewrites an accepted edit (for example a label): the old one is retired, the new one is in effect.
    Any agent interpretation of the old text does not carry over."""
    with transaction(run_path, expect_rev) as review:
        old = next((e for e in review["edits"] if e["id"] == edit_id), None)
        if old is None or old["status"] != "accepted":
            raise ReviewError(f"no accepted edit {edit_id!r} on this run")
        new_payload = check_payload(old["kind"], payload, manifest)
        edit = {"id": uuid.uuid4().hex[:12], "kind": old["kind"], "payload": new_payload, "source": "human",
                "reason": None, "status": "proposed", "created_at": _now(), "decided_at": None, "replaces": old["id"]}
        old["status"], old["decided_at"], old["replaced_by"] = "rejected", _now(), edit["id"]
        review["edits"].append(edit)
        _accept(review, edit)
    return edit


def interpret(run_path: Path, edit_id: str, train: str, meaning: str, by: str = "agent",
              expect_rev: int | None = None, replace: bool = False) -> dict:
    """Record what a label means for training, as read by an agent: keep or exclude the span, and why.
    A label that already has an interpretation is refused unless replace=True; the old one goes to history."""
    if train not in TRAIN_USE:
        raise ReviewError(f"train must be one of {TRAIN_USE}")
    if not isinstance(meaning, str) or not meaning.strip() or len(meaning) > 500:
        raise ReviewError("meaning must be 1 to 500 characters")
    with transaction(run_path, expect_rev) as review:
        edit = next((e for e in review["edits"] if e["id"] == edit_id), None)
        if edit is None or edit["kind"] != "label" or edit["status"] != "accepted":
            raise ReviewError(f"no accepted label {edit_id!r} on this run")
        if edit.get("interpretation"):
            if not replace:
                raise ReviewError(f"label {edit_id} was already interpreted as {edit['interpretation']['train']!r} "
                                  f"by {edit['interpretation']['by']}; pass replace to change it")
            edit.setdefault("interpretation_history", []).append(edit["interpretation"])
        if not isinstance(by, str) or not by.strip() or len(by) > 100:
            raise ReviewError("by must name the agent in 1 to 100 characters")
        edit["interpretation"] = {"train": train, "meaning": meaning.strip(), "by": by.strip(), "at": _now()}
    return edit


def active_edits(review: dict) -> list[dict]:
    """Accepted edits that are in effect: all of the multi-valued kinds, the latest accepted per single slot."""
    acc = sorted((e for e in review["edits"] if e["status"] == "accepted"), key=lambda e: e.get("accepted_seq", 0))
    latest = {}
    for e in acc:
        if e["kind"] not in MULTI:
            latest[_slot(e)] = e
    return [e for e in acc if e["kind"] in MULTI or latest.get(_slot(e)) is e]


def _valid_reading(it) -> bool:
    """The same limits interpret() enforces when it writes a reading."""
    def text(v, hi):
        return isinstance(v, str) and 0 < len(v.strip()) and len(v) <= hi
    return (isinstance(it, dict) and it.get("train") in TRAIN_USE and text(it.get("meaning"), 500)
            and text(it.get("by"), 100) and text(it.get("at"), 40))


def _valid_edit(e: dict, manifest: dict) -> bool:
    """One stored edit as this code writes it (used for history snapshots)."""
    if not EDIT_ID.match(str(e.get("id"))) or e.get("kind") not in KINDS or e.get("status") not in STATUSES \
            or e.get("source") not in SOURCES:
        return False
    try:
        if check_payload(e["kind"], e.get("payload"), manifest) != e.get("payload"):
            return False
    except ReviewError:
        return False
    if e["status"] == "accepted" and (isinstance(e.get("accepted_seq"), bool) or not isinstance(e.get("accepted_seq"), int)):
        return False
    readings = ([e["interpretation"]] if e.get("interpretation") is not None else []) + e.get("interpretation_history", [])
    return isinstance(e.get("interpretation_history", []), list) and all(_valid_reading(x) for x in readings)


def check_review(manifest: dict, review: dict) -> None:
    """Reject a stored review that this code would never have written: fail closed before it shapes training."""
    for k in ("rev", "seq"):
        if isinstance(review.get(k), bool) or not isinstance(review.get(k), int) or review[k] < 0:
            raise ReviewError(f"review {k} must be a non-negative integer")
    ids = set()
    for e in review["edits"]:
        if e["id"] in ids:
            raise ReviewError(f"edit id {e['id']} appears twice")
        ids.add(e["id"])
        try:
            if check_payload(e["kind"], e["payload"], manifest) != e["payload"]:
                raise ReviewError("payload is not in canonical form")
        except ReviewError as err:
            raise ReviewError(f"edit {e['id']} ({e['kind']}): {err}") from None
        if e["status"] == "accepted" and (isinstance(e.get("accepted_seq"), bool)
                                          or not isinstance(e.get("accepted_seq"), int)):
            raise ReviewError(f"edit {e['id']} is accepted without an accepted_seq")
        for k in ("reason", "by"):
            if e.get(k) is not None and not isinstance(e[k], str):
                raise ReviewError(f"edit {e['id']} has a non-text {k}")
        hist = e.get("interpretation_history", [])
        if not isinstance(hist, list):
            raise ReviewError(f"edit {e['id']} has a malformed interpretation history")
        current = [e["interpretation"]] if "interpretation" in e and e["interpretation"] is not None else []
        for it in current + hist:
            if e["kind"] != "label" or not _valid_reading(it):
                raise ReviewError(f"edit {e['id']} has a malformed interpretation")
    hist = review.get("history", [])
    if not isinstance(hist, list) or len(hist) > HISTORY_LIMIT or len(json.dumps(hist)) > HISTORY_BYTES:
        raise ReviewError("review history must be a list of at most 200 steps and 2 MB")
    step_ids = set()
    for h in hist:
        if not isinstance(h, dict) or not EDIT_ID.match(str(h.get("id"))) or h["id"] in step_ids \
                or not isinstance(h.get("what"), str) or not isinstance(h.get("at"), str) \
                or isinstance(h.get("t"), bool) or not isinstance(h.get("t"), int) or h["t"] <= 0 \
                or not isinstance(h.get("undone"), bool) or not isinstance(h.get("changes"), list) or not h["changes"]:
            raise ReviewError("review history has a malformed step")
        step_ids.add(h["id"])
        touched = set()
        for c in h["changes"]:
            if not isinstance(c, dict) or not EDIT_ID.match(str(c.get("id"))) or c["id"] in touched:
                raise ReviewError("review history has a malformed change")
            touched.add(c["id"])
            for snap in (c.get("before"), c.get("after")):
                if snap is not None and (not isinstance(snap, dict) or snap.get("id") != c["id"]
                                         or not _valid_edit(snap, manifest)):
                    raise ReviewError("review history has a malformed snapshot")
    seqs = [e["accepted_seq"] for e in review["edits"] if "accepted_seq" in e]
    if any(isinstance(q, bool) or not isinstance(q, int) or not 1 <= q <= review["seq"] for q in seqs) \
            or len(seqs) != len(set(seqs)):
        raise ReviewError("accepted_seq values must be unique integers between 1 and the review seq")


def effective(manifest: dict, review: dict) -> dict:
    """The run as training should see it after the edits in effect."""
    check_review(manifest, review)
    act = active_edits(review)
    single = {e["kind"]: e["payload"] for e in act if e["kind"] not in MULTI and e["kind"] not in ("step_boundary", "event_note")}
    out = {
        "termination_reason": manifest["termination_reason"], "failure_mode": manifest.get("failure_mode"),
        "failure_start": None, "excludes": [e["payload"] for e in act if e["kind"] == "exclude"],
        "rejected": "reject_run" in single, "notes": [e["payload"] for e in act if e["kind"] == "note"],
        "subgoals": [dict(s) for s in manifest.get("subgoals", [])], "reviewed": review["reviewed"],
        "labels": [{"id": e["id"], **e["payload"], "interpretation": e.get("interpretation")}
                   for e in act if e["kind"] == "label"],
        "event_notes": {e["payload"]["key"]: {"text": e["payload"]["text"], "id": e["id"],
                                              "by": e.get("by") or ("person" if e["source"] == "human" else "agent")}
                        for e in act if e["kind"] == "event_note"},
    }
    if "outcome" in single:
        out["termination_reason"] = single["outcome"]["termination_reason"]
        out["failure_mode"] = single["outcome"]["failure_mode"]
    if "failure_start" in single and out["termination_reason"] != "success":
        out["failure_start"] = single["failure_start"]["frame"]
    steps = out["subgoals"]
    for e in act:
        if e["kind"] == "step_boundary":
            p = e["payload"]
            i = p["index"]
            if i + 1 < len(steps) and steps[i]["name"] == p["before"] and steps[i + 1]["name"] == p["after"]:
                f = min(max(p["frame"], steps[i]["start_frame"] + 1), steps[i + 1]["end_frame"] - 1)
                steps[i]["end_frame"] = steps[i + 1]["start_frame"] = f
    return out


def train_mask(manifest: dict, review: dict, control_modes: list[str]) -> np.ndarray:
    """Per-frame bool: False for a rejected run, excluded spans, labels an agent read as "exclude", and, on a failed
    run, every frame from the accepted
    failure start up to the next takeover (the first expert or blend frame), or to the end if none follows."""
    n = manifest["length"]
    eff = effective(manifest, review)
    mask = np.ones(n, dtype=bool)
    if eff["rejected"]:
        mask[:] = False
        return mask
    for x in eff["excludes"]:
        mask[x["start"]:x["end"]] = False
    for lb in eff["labels"]:
        if (lb.get("interpretation") or {}).get("train") == "exclude":
            mask[lb["start"]:lb["end"]] = False
    fs = eff["failure_start"]
    if fs is not None and control_modes[fs] not in TAKEOVER_MODES:
        end = next((i for i in range(fs, n) if control_modes[i] in TAKEOVER_MODES), n)
        mask[fs:end] = False
    return mask


# ── AI feature: find idle frames ───────────────────────────────────────────────

def _episode_state(root: Path, episode_index: int, n: int) -> np.ndarray | None:
    """The measured joint positions (observation.state) for one episode, or None if the dataset has none."""
    import pyarrow.dataset as pads
    d = Path(root) / "data"
    if not d.exists():
        return None
    ds = pads.dataset(str(d), format="parquet")
    if "observation.state" not in ds.schema.names:
        return None
    t = ds.to_table(columns=["frame_index", "observation.state"],
                    filter=pads.field("episode_index") == episode_index).sort_by("frame_index")
    if t.num_rows != n:
        return None
    return np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32).reshape(n, -1)


def idle_spans(motion: np.ndarray, rate: float, min_s: float = 1.0, still: float = 1.5) -> list[tuple[int, int]]:
    """Stretches covered by consecutive one-second windows in which no joint moves more than `still`
    (max - min). Returns [start, end) spans of at least min_s; spans on either side of a movement never merge."""
    n = len(motion)
    w = max(2, int(round(rate)))
    if n < w:
        return []
    quiet = [float((motion[i:i + w].max(axis=0) - motion[i:i + w].min(axis=0)).max()) < still
             for i in range(n - w + 1)]
    # Each run of consecutive quiet windows is one stretch: windows that share frames belong together, and a
    # movement between two still plateaus breaks the run, so the two plateaus stay separate stretches.
    spans, i, k = [], 0, max(1, int(round(min_s * rate)))
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            a, b = i, j - 1 + w
            a += a > 0                               # leave the frames where the arm starts or stops moving
            b -= b < n                               # outside every stretch, so cutting all keeps them
            if spans and a < spans[-1][1]:
                a = spans[-1][1]
            if b - a >= k:
                spans.append((a, b))
            i = j
        else:
            i += 1
    return spans


def find_idle(root: Path, run_path: Path, manifest: dict, by: str = "dataop idle finder",
              min_s: float = 1.0, still: float = 1.5, expect_rev: int | None = None) -> list[dict]:
    """Suggest cutting idle stretches of one run (our built-in model for the "find idle frames" AI feature).
    Uses measured joint positions when the dataset has them, otherwise the sent commands. Suggestions are agent
    proposals of kind "exclude" with a short description; a person cuts or keeps each one."""
    n, rate = manifest["length"], manifest["action_schema"]["rate_hz"]
    motion = _episode_state(root, manifest["episode_index"], n)
    source = "measured joints"
    if motion is None:
        t = pq.read_table(run_path / "ticks.parquet", columns=["sent_action"])
        motion = np.asarray(t.column("sent_action").to_pylist(), dtype=np.float32).reshape(n, -1)
        source = "commands"
    modes = pq.read_table(run_path / "ticks.parquet", columns=["control_mode"]).column("control_mode").to_pylist()
    items = []
    by = f"{by} ({source})"
    for a, b in idle_spans(motion, rate, min_s, still):
        where = "at the start" if a == 0 else "at the end" if b == n else "mid-run"
        span = modes[a:b]
        who = max(set(span), key=span.count)
        who = {"base": "policy driving", "expert": "rescue driving", "blend": "shared control", "none": "nobody driving"}[who]
        items.append({"start": a, "end": b, "reason": f"arm still for {(b - a) / rate:.1f} s {where}, {who}"})
    return add_suggestions(run_path, manifest, items, by, expect_rev=expect_rev)


def suggest_cut(run_path: Path, manifest: dict, start: int, end: int, reason: str, by: str = "agent") -> dict:
    """Any agent's suggestion to cut frames [start, end): shown in the viewer for a person to cut or keep.
    Returns {"status", "edit"} like add_suggestions."""
    return add_suggestions(run_path, manifest, [{"start": start, "end": end, "reason": reason}], by)[0]


def list_labels(root: Path, only_uninterpreted: bool = False) -> list[dict]:
    """Every label in effect, with what was recorded in its span, for an agent to read."""
    out = []
    for run_path, m in store.iter_manifests(Path(root), "committed"):
        r = load(run_path)
        eff = effective(m, r)
        if not eff["labels"]:
            continue
        rate = m["action_schema"]["rate_hz"]
        modes = pq.read_table(run_path / "ticks.parquet", columns=["control_mode"]).column("control_mode").to_pylist()
        for lb in eff["labels"]:
            if only_uninterpreted and lb["interpretation"]:
                continue
            span = modes[lb["start"]:lb["end"]]
            out.append({
                "run_id": m["run_id"], "episode_index": m["episode_index"], "label_id": lb["id"],
                "start_frame": lb["start"], "end_frame": lb["end"],
                "start_s": round(lb["start"] / rate, 2), "end_s": round(lb["end"] / rate, 2),
                "text": lb["text"], "driver": lb["driver"], "segment": lb["segment"],
                "interpretation": lb["interpretation"],
                "recorded": {
                    "control_modes": {k: span.count(k) for k in sorted(set(span))},
                    "steps": [s["name"] for s in eff["subgoals"]
                              if s["start_frame"] < lb["end"] and s["end_frame"] > lb["start"]],
                    "run_outcome": eff["termination_reason"], "failure_mode": eff["failure_mode"],
                    "task": m["task"],
                },
            })
    return out
