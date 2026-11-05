"""Review format 2: the review file shared by the logger and the Aftershock cloud, as plain dicts.

A review holds every edit people, agents, checks and the episode splitter made to one run; the recorded run never
changes. This module is the format itself: the strict reader, the canonical writer, the operations that change a
review, and the lossless reading of the logger's earlier format 1 files. `review.py` builds the logger's review
features (the viewer, the CLI, the training mask) on top of it.

The cloud implements the same rules and both sides test against the same round trip fixtures, so a review written
here and read there, or the other way round, keeps its exact bytes:

  canonical text   json.dumps(review, indent=1, sort_keys=True) + "\\n", every key written, nulls included
  digest           sha256 of the canonical text

Reading is fail closed: a key, kind, source or value this format would not write is refused, never dropped.

Format 1 (`{"version": 1, ...}`, edit `id`, sources agent and human, recorded steps by index) is read through
`migrate_format1`. Names map to subtask ids only by recorded name, against a task the caller gives (never by
position); without a task every name stays "unresolved" with its frames kept. `to_format1` gives the exact format 1
file back for any migrated review, which is the proof that nothing is lost.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from typing import Any

REVIEW_FORMAT = 2
HISTORY_LIMIT = 200          # undo steps kept
HISTORY_BYTES = 2_000_000    # and at most this much JSON (plain json.dumps length); the oldest steps go first
UNRESOLVED = "unresolved"
UNKNOWN = "unknown"

SOURCES = ("human", "agent", "check", "splitter")
STATUSES = ("proposed", "accepted", "rejected")
KINDS = ("outcome", "failure_start", "failure_type", "exclude", "segments", "step_boundary", "segment_edge",
         "drive_boundary", "reject_run", "note", "event_note", "label", "reading", "name")
MULTI = ("exclude", "note", "label")   # every accepted edit of these kinds applies; the other kinds have one slot
TERMINATION_REASONS = ("success", "failure", "timeout", "operator_abort", "system_error", "unknown")
TRAIN_USE = ("keep", "exclude")

EDIT_ID = re.compile(r"[0-9a-f]{12}")
SLUG = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
UUID4 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
ONE_LINE = re.compile(r"[^\r\n]+")

REVIEW_KEYS = ("review_format", "run_id", "rev", "seq", "reviewed", "reviewed_at", "reviewed_by", "edits",
               "history", "locked", "lock_log", "legacy_subtask_map")
EDIT_KEYS = ("edit_id", "kind", "payload", "source", "by", "job_id", "agent_version", "reason", "status",
             "created_at", "decided_at", "decided_by", "accepted_seq", "corrects", "replaces", "replaced_by",
             "adjusted", "legacy_reading", "legacy_reading_history")


class FormatError(ValueError):
    """A review this format would not write."""


class MigrationError(FormatError):
    """A format 1 review that does not migrate cleanly."""


class ReviewError(ValueError):
    """An operation that cannot be done."""


class ReviewConflict(ReviewError):
    """The review moved on since the caller read it."""


class ReviewLocked(ReviewError):
    """The run's pack is locked; its review takes no edits until an owner unlocks it."""


# ------------------------------------------------------------------ value checks


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _int(v, what: str, lo: int = 0) -> int:
    if not _is_int(v) or v < lo:
        raise FormatError(f"{what} must be an integer >= {lo}")
    return v


_SURROGATE = re.compile("[\ud800-\udfff]")


def _str(v, what: str, lo: int = 0, hi: int | None = None, blank_ok: bool = True) -> str:
    """Text. A field with any limit (a length or not blank) is also refused when it holds a lone surrogate code
    point, which is not valid unicode; a field with no limit (by, reason, a step's what) takes any string. The
    reference reader draws the line in the same place."""
    if not isinstance(v, str) or len(v) < lo or (hi is not None and len(v) > hi):
        raise FormatError(f"{what} must be text of {lo} to {hi if hi is not None else 'any'} characters")
    if (lo > 0 or hi is not None or not blank_ok) and _SURROGATE.search(v):
        raise FormatError(f"{what} is not valid unicode text")
    if not blank_ok and not v.strip():
        raise FormatError(f"{what} must not be blank")
    return v


def _opt(check, v, *a, **k):
    return None if v is None else check(v, *a, **k)


def _match(rx: re.Pattern, v, what: str) -> str:
    if not isinstance(v, str) or not rx.fullmatch(v):
        raise FormatError(f"{what} is not valid: {v!r}")
    return v


def _bool(v, what: str) -> bool:
    if not isinstance(v, bool):
        raise FormatError(f"{what} must be true or false")
    return v


def _keys(obj, keys, what: str) -> dict:
    if not isinstance(obj, dict):
        raise FormatError(f"{what} must be an object")
    if set(obj) != set(keys):
        missing, extra = sorted(set(keys) - set(obj)), sorted(set(obj) - set(keys))
        raise FormatError(f"{what} has the wrong keys (missing {missing}, unexpected {extra})")
    return obj


def _range(p: dict, what: str) -> None:
    if not p["start"] < p["end"]:
        raise FormatError(f"{what} needs start < end, got [{p['start']}, {p['end']})")


def outcome_of(termination_reason: str) -> str:
    """success stays success, unknown stays unknown, and every other reason means the task did not succeed."""
    return termination_reason if termination_reason in ("success", "unknown") else "failure"


# ------------------------------------------------------------------ payloads, one check per kind


def _segment(s) -> None:
    _keys(s, ("subtask_id", "start", "end", "legacy_name"), "a segment")
    _match(SLUG, s["subtask_id"], "subtask_id")
    _int(s["start"], "start"), _int(s["end"], "end")
    _range(s, "a segment")
    if s["legacy_name"] is not None:
        _str(s["legacy_name"], "legacy_name", 1, 100, blank_ok=False)
    elif s["subtask_id"] == UNRESOLVED:
        raise FormatError("an unresolved segment keeps its legacy_name")


def check_payload(kind: str, p) -> dict:
    """Check one edit's payload for its kind and return it. Every key of the kind is required, nulls included."""
    if kind not in KINDS:
        raise FormatError(f"kind must be one of {KINDS}")
    if kind == "outcome":
        _keys(p, ("outcome", "termination_reason", "failure_mode"), "an outcome payload")
        if p["outcome"] not in ("success", "failure", "unknown"):
            raise FormatError("outcome must be success, failure or unknown")
        tr = p["termination_reason"]
        if tr is not None:
            if tr not in TERMINATION_REASONS or not isinstance(tr, str):
                raise FormatError(f"termination_reason must be one of {TERMINATION_REASONS} or null")
            if p["outcome"] != outcome_of(tr):
                raise FormatError(f"termination_reason {tr!r} means outcome {outcome_of(tr)!r}, not {p['outcome']!r}")
        if p["failure_mode"] is not None:
            _str(p["failure_mode"], "failure_mode", 1, 200, blank_ok=False)
            if p["outcome"] == "success":
                raise FormatError("a successful run has no failure_mode")
    elif kind in ("failure_start",):
        _keys(p, ("frame",), "a failure_start payload")
        _int(p["frame"], "frame")
    elif kind == "failure_type":
        _keys(p, ("failure_type_id", "vocab_rev"), "a failure_type payload")
        _match(SLUG, p["failure_type_id"], "failure_type_id")
        _int(p["vocab_rev"], "vocab_rev")
    elif kind == "exclude":
        _keys(p, ("start", "end"), "an exclude payload")
        _int(p["start"], "start"), _int(p["end"], "end")
        _range(p, "exclude")
    elif kind == "segments":
        _keys(p, ("segments",), "a segments payload")
        if not isinstance(p["segments"], list) or len(p["segments"]) > 500:
            raise FormatError("segments must be a list of at most 500")
        last = 0
        for s in p["segments"]:
            _segment(s)
            if s["start"] < last:
                raise FormatError("segments must be in order and must not overlap")
            last = s["end"]
    elif kind == "step_boundary":
        _keys(p, ("index", "before", "after", "frame"), "a step_boundary payload")
        _int(p["index"], "index"), _int(p["frame"], "frame")
        _str(p["before"], "before", 1), _str(p["after"], "after", 1)
    elif kind == "segment_edge":
        _keys(p, ("index", "edge", "frame"), "a segment_edge payload")
        _int(p["index"], "index"), _int(p["frame"], "frame")
        if p["edge"] not in ("start", "end"):
            raise FormatError("edge must be start or end")
    elif kind == "drive_boundary":
        _keys(p, ("index", "frame"), "a drive_boundary payload")
        _int(p["index"], "index"), _int(p["frame"], "frame")
    elif kind == "reject_run":
        _keys(p, (), "a reject_run payload")
    elif kind == "note":
        _keys(p, ("frame", "text"), "a note payload")
        _int(p["frame"], "frame")
        _str(p["text"], "a note", 1, 2000)
    elif kind == "event_note":
        _keys(p, ("key", "text"), "an event_note payload")
        _str(p["key"], "an event key", 1, 120)
        _match(ONE_LINE, p["key"], "an event key")
        _str(p["text"], "an event note", 1, 500, blank_ok=False)
    elif kind == "label":
        _keys(p, ("start", "end", "text", "driver", "segment"), "a label payload")
        _int(p["start"], "start"), _int(p["end"], "end")
        _str(p["text"], "a label", 1, 2000)
        _opt(_str, p["driver"], "driver", 1, 50)
        _opt(_str, p["segment"], "segment", 1, 100)
        _range(p, "a label")
        _str(p["text"], "a label", 1, 2000, blank_ok=False)
    elif kind == "reading":
        _keys(p, ("label_id", "train", "meaning"), "a reading payload")
        _match(EDIT_ID, p["label_id"], "label_id")
        if p["train"] not in TRAIN_USE:
            raise FormatError("train must be keep or exclude")
        _str(p["meaning"], "a reading's meaning", 1, 500, blank_ok=False)
    elif kind == "name":
        _keys(p, ("text",), "a name payload")
        _str(p["text"], "a name", 1, 200)
    return p


# ------------------------------------------------------------------ edits, history, the review


def _legacy_reading(r) -> None:
    _keys(r, ("train", "meaning", "by", "at"), "a legacy reading")
    if r["train"] not in TRAIN_USE:
        raise FormatError("a legacy reading's train must be keep or exclude")
    _str(r["meaning"], "a legacy reading's meaning", 1, 500)
    _str(r["by"], "a legacy reading's by", 1, 100)
    _str(r["at"], "a legacy reading's at", 1, 40)


def check_edit(e) -> dict:
    """One edit (or a history snapshot of one), on its own."""
    _keys(e, EDIT_KEYS, "an edit")
    _match(EDIT_ID, e["edit_id"], "edit_id")
    if e["kind"] not in KINDS:
        raise FormatError(f"edit {e['edit_id']}: kind must be one of {KINDS}")
    try:
        check_payload(e["kind"], e["payload"])
    except FormatError as err:
        raise FormatError(f"edit {e['edit_id']} ({e['kind']}): {err}") from None
    if e["source"] not in SOURCES:
        raise FormatError(f"edit {e['edit_id']}: source must be one of {SOURCES}")
    _opt(_str, e["by"], "by")
    for k in ("job_id", "agent_version"):
        _opt(_str, e[k], k, 1)
    _opt(_str, e["reason"], "reason")
    if e["status"] not in STATUSES:
        raise FormatError(f"edit {e['edit_id']}: status must be one of {STATUSES}")
    _str(e["created_at"], "created_at", 1)
    _opt(_str, e["decided_at"], "decided_at", 1)
    _opt(_str, e["decided_by"], "decided_by", 1, 100)
    _opt(_int, e["accepted_seq"], "accepted_seq", 1)
    for k in ("corrects", "replaces", "replaced_by"):
        if e[k] is not None:
            _match(EDIT_ID, e[k], k)
    _bool(e["adjusted"], "adjusted")
    if e["legacy_reading"] is not None:
        _legacy_reading(e["legacy_reading"])
    if not isinstance(e["legacy_reading_history"], list):
        raise FormatError("legacy_reading_history must be a list")
    for r in e["legacy_reading_history"]:
        _legacy_reading(r)
    where = f"edit {e['edit_id']}"
    if e["source"] == "human":
        if e["job_id"] is not None or e["agent_version"] is not None:
            raise FormatError(f"{where}: a person's edit has no job_id or agent_version")
    elif e["job_id"] is None or e["agent_version"] is None:
        raise FormatError(f"{where}: a {e['source']} edit names its job_id and agent_version")
    if e["status"] == "accepted" and e["accepted_seq"] is None:
        raise FormatError(f"{where}: an accepted edit has an accepted_seq")
    if e["status"] != "proposed" and e["decided_at"] is None:
        raise FormatError(f"{where}: a decided edit has decided_at")
    if e["corrects"] is not None and e["source"] != "human":
        raise FormatError(f"{where}: only a person's edit corrects another edit")
    if e["kind"] != "label" and (e["legacy_reading"] is not None or e["legacy_reading_history"]):
        raise FormatError(f"{where}: only a label has a legacy reading")
    return e


def _check_step(h) -> None:
    _keys(h, ("step_id", "at", "t", "what", "changes", "undone", "undone_at"), "a history step")
    _match(EDIT_ID, h["step_id"], "step_id")
    _str(h["at"], "a step's at", 1)
    _opt(_int, h["t"], "a step's t", 1)
    _str(h["what"], "a step's what")
    _bool(h["undone"], "undone")
    _opt(_str, h["undone_at"], "undone_at", 1)
    if not isinstance(h["changes"], list) or not h["changes"]:
        raise FormatError("a history step has at least one change")
    ids = []
    for c in h["changes"]:
        _keys(c, ("edit_id", "before", "after"), "a history change")
        _match(EDIT_ID, c["edit_id"], "a change's edit_id")
        for snap in (c["before"], c["after"]):
            if snap is not None:
                check_edit(snap)
                if snap["edit_id"] != c["edit_id"]:
                    raise FormatError("a change's snapshots are of its own edit")
        if c["before"] is None and c["after"] is None:
            raise FormatError("a change has a before or an after")
        ids.append(c["edit_id"])
    if len(set(ids)) != len(ids):
        raise FormatError("a step touches each edit once")
    if h["undone_at"] is not None and not h["undone"]:
        raise FormatError("undone_at is set only on an undone step")


def _check_lock(x, keys) -> None:
    _keys(x, keys, "a lock record")
    if "action" in x and x["action"] not in ("lock", "unlock"):
        raise FormatError("a lock_log action is lock or unlock")
    _str(x["by"], "a lock's by", 1, 100)
    _str(x["at"], "a lock's at", 1)
    _opt(_str, x["reason"], "a lock's reason")


def all_edit_versions(review: dict) -> list[dict]:
    """Every edit and every snapshot of an edit in history."""
    snaps = [s for h in review["history"] for c in h["changes"] for s in (c["before"], c["after"]) if s is not None]
    return list(review["edits"]) + snaps


def _lookup(lmap: dict, step_index, name: str):
    return next((x["subtask_id"] for x in lmap["entries"]
                 if x["step_index"] == step_index and x["recorded_name"] == name), None)


def check_review(review) -> dict:
    """The whole review as this format writes it, or FormatError (fail closed). Returns the review unchanged."""
    _keys(review, REVIEW_KEYS, "the review")
    if not _is_int(review["review_format"]) or review["review_format"] != REVIEW_FORMAT:
        raise FormatError("not a review_format 2 file")
    _match(UUID4, review["run_id"], "run_id")
    _int(review["rev"], "rev"), _int(review["seq"], "seq")
    _bool(review["reviewed"], "reviewed")
    _opt(_str, review["reviewed_at"], "reviewed_at", 1)
    _opt(_str, review["reviewed_by"], "reviewed_by", 1, 100)
    for k in ("edits", "history", "lock_log"):
        if not isinstance(review[k], list):
            raise FormatError(f"{k} must be a list")
    for e in review["edits"]:
        check_edit(e)
    for h in review["history"]:
        _check_step(h)
    if review["locked"] is not None:
        _check_lock(review["locked"], ("by", "at", "reason"))
    for x in review["lock_log"]:
        _check_lock(x, ("action", "by", "at", "reason"))
    lmap = review["legacy_subtask_map"]
    if lmap is not None:
        _keys(lmap, ("run_id", "task_id", "entries"), "legacy_subtask_map")
        _match(UUID4, lmap["run_id"], "legacy_subtask_map run_id")
        if lmap["task_id"] is not None:
            _match(SLUG, lmap["task_id"], "legacy_subtask_map task_id")
        if not isinstance(lmap["entries"], list):
            raise FormatError("legacy_subtask_map entries must be a list")
        for x in lmap["entries"]:
            _keys(x, ("step_index", "recorded_name", "subtask_id"), "a legacy_subtask_map entry")
            _opt(_int, x["step_index"], "step_index")
            _str(x["recorded_name"], "recorded_name", 1)
            _match(SLUG, x["subtask_id"], "a legacy_subtask_map subtask_id")
        keys = [(x["step_index"], x["recorded_name"]) for x in lmap["entries"]]
        if len(set(keys)) != len(keys):
            raise FormatError("legacy_subtask_map has one entry per step index and name")
    # across the whole review
    by_id: dict[str, dict] = {}
    for e in review["edits"]:
        if e["edit_id"] in by_id:
            raise FormatError(f"edit id {e['edit_id']} appears twice")
        by_id[e["edit_id"]] = e
    seqs = [e["accepted_seq"] for e in review["edits"] if e["accepted_seq"] is not None]
    if len(set(seqs)) != len(seqs) or any(q > review["seq"] for q in seqs):
        raise FormatError("accepted_seq values must be unique and at most the review seq")
    for e in review["edits"]:
        for ref in (e["corrects"], e["replaces"], e["replaced_by"]):
            if ref is not None and ref not in by_id:
                raise FormatError(f"edit {e['edit_id']} refers to {ref}, which is not in this review")
        if e["corrects"] is not None and by_id[e["corrects"]]["source"] == "human":
            raise FormatError(f"edit {e['edit_id']} corrects a person's edit; a person replaces it instead")
        if e["kind"] == "reading":
            label = by_id.get(e["payload"]["label_id"])
            if label is None or label["kind"] != "label":
                raise FormatError(f"reading {e['edit_id']} is about {e['payload']['label_id']}, which is not a label here")
    # The 2 MB cap is applied by the operations when they add a step, not here, so a format 1 history that fit the
    # logger's measure migrates whole.
    if len(review["history"]) > HISTORY_LIMIT:
        raise FormatError("review history must be at most 200 steps")
    step_ids = [h["step_id"] for h in review["history"]]
    if len(set(step_ids)) != len(step_ids):
        raise FormatError("history step ids must be unique")
    if review["reviewed_at"] is not None and not review["reviewed"]:
        raise FormatError("reviewed_at is set only on a reviewed run")
    last = review["lock_log"][-1]["action"] if review["lock_log"] else "unlock"
    if (review["locked"] is not None) != (last == "lock"):
        raise FormatError("locked must agree with the last lock_log entry")
    if lmap is not None:
        if lmap["run_id"] != review["run_id"]:
            raise FormatError("legacy_subtask_map is for another run")
        for e in all_edit_versions(review):
            if e["kind"] == "segments":
                for s in e["payload"]["segments"]:
                    if s["legacy_name"] is not None and _lookup(lmap, None, s["legacy_name"]) != s["subtask_id"]:
                        raise FormatError(f"segment {s['legacy_name']!r} disagrees with legacy_subtask_map")
    return review


def check_against_manifest(review: dict, manifest: dict) -> None:
    """Every edit and history snapshot must fit the recorded run: frames inside it, recorded step indices that exist,
    step names that match and share a boundary. FormatError otherwise (fail closed)."""
    if review["run_id"] != manifest["run_id"]:
        raise FormatError("the review is for another run")
    n = manifest["length"]
    steps = manifest.get("subgoals") or []
    for e in all_edit_versions(review):
        p, kind = e["payload"], e["kind"]

        def within(v: int, hi: int, what: str) -> None:
            if v > hi:
                raise FormatError(f"edit {e['edit_id']} ({kind}): {what} {v} is past {hi}")

        if kind in ("failure_start", "note"):
            within(p["frame"], n - 1, "frame")
        elif kind in ("exclude", "label"):
            within(p["end"], n, "end")
        elif kind == "segments":
            for s in p["segments"]:
                within(s["end"], n, "segment end")
        elif kind == "drive_boundary":
            within(p["index"], n - 1, "index")
            within(p["frame"], n, "frame")
        elif kind == "segment_edge":
            within(p["index"], len(steps) - 1, "recorded step index")
            within(p["frame"], n, "frame")
        elif kind == "step_boundary":
            within(p["index"], len(steps) - 2, "recorded step index")
            within(p["frame"], n, "frame")
            a, b = steps[p["index"]], steps[p["index"] + 1]
            if (a.get("name"), b.get("name")) != (p["before"], p["after"]):
                raise FormatError(f"edit {e['edit_id']}: step names do not match the recorded steps at {p['index']}")
            if a.get("end_frame") != b.get("start_frame"):
                raise FormatError(f"edit {e['edit_id']}: recorded steps {p['index']} and {p['index'] + 1} share no "
                                  "boundary")


def check_against_task(review: dict, task: dict) -> None:
    """Every subtask id and failure type id must be the task's, current or retired. `task` is
    {task_id, subtasks: [{subtask_id, ...}], failure_types: [{failure_type_id, ...}], retired_subtask_ids,
    retired_failure_type_ids}; missing lists count as empty."""
    subtasks = {s["subtask_id"] for s in task.get("subtasks", [])} | set(task.get("retired_subtask_ids", [])) \
        | {UNRESOLVED}
    failures = {f["failure_type_id"] for f in task.get("failure_types", [])} \
        | set(task.get("retired_failure_type_ids", []))
    for e in all_edit_versions(review):
        if e["kind"] == "segments":
            bad = [s["subtask_id"] for s in e["payload"]["segments"] if s["subtask_id"] not in subtasks]
            if bad:
                raise FormatError(f"edit {e['edit_id']}: subtask ids {bad} are not in task {task.get('task_id')}")
        elif e["kind"] == "failure_type" and e["payload"]["failure_type_id"] not in failures:
            raise FormatError(f"edit {e['edit_id']}: failure type {e['payload']['failure_type_id']} is not in task "
                              f"{task.get('task_id')}")
    if review["legacy_subtask_map"] is not None:
        bad = [x["subtask_id"] for x in review["legacy_subtask_map"]["entries"] if x["subtask_id"] not in subtasks]
        if bad:
            raise FormatError(f"legacy_subtask_map ids {bad} are not in task {task.get('task_id')}")


# ------------------------------------------------------------------ the file


def dumps(obj: Any) -> str:
    """The canonical writer, shared with format 1: sorted keys, one space indent, ASCII, a final newline."""
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def dump_review(review: dict) -> str:
    return dumps(review)


def load_review(text: str | bytes) -> dict:
    """Parse and check a format 2 file. Anything this format would not write is refused."""
    data = json.loads(text)
    if not isinstance(data, dict) or data.get("review_format") != REVIEW_FORMAT:
        raise FormatError("not a review_format 2 file")
    return check_review(data)


def review_digest(review: dict) -> str:
    """sha256 of the canonical file text."""
    return hashlib.sha256(dump_review(review).encode()).hexdigest()


def empty_review(run_id: str) -> dict:
    return check_review({"review_format": 2, "run_id": run_id, "rev": 0, "seq": 0, "reviewed": False,
                         "reviewed_at": None, "reviewed_by": None, "edits": [], "history": [], "locked": None,
                         "lock_log": [], "legacy_subtask_map": None})


# ------------------------------------------------------------------ reading a review


def slot(e: dict) -> tuple:
    """Edits that compete for one value. Multi-valued kinds never compete."""
    p, kind = e["payload"], e["kind"]
    if kind in MULTI:
        return (kind, e["edit_id"])
    if kind in ("step_boundary", "drive_boundary"):
        return (kind, p["index"])
    if kind == "segment_edge":
        return (kind, p["index"], p["edge"])
    if kind == "event_note":
        return (kind, p["key"])
    if kind == "reading":
        return (kind, p["label_id"])
    return (kind,)


def active_edits(review: dict) -> list[dict]:
    """Accepted edits in effect: every multi-valued one, and the latest accepted per slot, by accepted_seq."""
    acc = sorted((e for e in review["edits"] if e["status"] == "accepted"), key=lambda e: e["accepted_seq"])
    latest = {slot(e): e for e in acc}
    return [e for e in acc if latest[slot(e)] is e]


def reading_in_effect(review: dict, label_id: str) -> dict | None:
    """The reading in effect for a label in effect: {train, meaning, by, at} from its latest accepted reading edit
    (at is when it was accepted), else its format 1 reading exactly as written. None for a label not in effect."""
    active = active_edits(review)
    if not any(e["edit_id"] == label_id and e["kind"] == "label" for e in active):
        return None
    r = next((e for e in reversed(active) if e["kind"] == "reading" and e["payload"]["label_id"] == label_id), None)
    if r is not None:
        return {"train": r["payload"]["train"], "meaning": r["payload"]["meaning"], "by": r["by"],
                "at": r["decided_at"]}
    legacy = next(e for e in review["edits"] if e["edit_id"] == label_id)["legacy_reading"]
    return None if legacy is None else dict(legacy)


# ------------------------------------------------------------------ operations


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _size(obj: Any) -> int:
    return len(json.dumps(obj))


def _who(by) -> str:
    if not isinstance(by, str) or not by.strip() or len(by) > 100:
        raise ReviewError("by names who acts in 1 to 100 characters")
    return by


def _validated(d: dict) -> dict:
    try:
        return check_review(d)
    except FormatError as e:
        raise ReviewError(f"the result would not be a valid review: {e}") from None


@contextmanager
def _write(review: dict, expect_rev: int, *, what: str | None = None, allow_locked: bool = False):
    """Yield a mutable copy in box["d"]; on exit record one history step for the edits that changed (when `what` is
    given), clear a sign off if edits changed, bump rev if anything changed, and check the result, which the caller
    reads from box["out"]. The input review is never changed."""
    if not _is_int(expect_rev):
        raise ReviewError("every write carries the rev it was based on")
    if review["rev"] != expect_rev:
        raise ReviewConflict(f"the review changed (now at rev {review['rev']}, you had {expect_rev})")
    if review["locked"] is not None and not allow_locked:
        raise ReviewLocked(f"the review is locked by {review['locked']['by']}; unlock the pack first")
    before = review
    d = copy.deepcopy(before)
    box: dict[str, Any] = {"d": d, "now": None, "t": None, "step": None, "what": None}
    yield box
    if what is not None:
        old = {e["edit_id"]: e for e in before["edits"]}
        new = {e["edit_id"]: e for e in d["edits"]}
        changes = [{"edit_id": i, "before": copy.deepcopy(old.get(i)), "after": copy.deepcopy(new.get(i))}
                   for i in list(old) + [i for i in new if i not in old] if old.get(i) != new.get(i)]
        if changes:
            step = {"step_id": new_id(), "at": box["now"], "t": box["t"], "what": box.get("what") or what,
                    "changes": changes,
                    "undone": False, "undone_at": None}
            hist = d["history"]
            if _size(step) <= HISTORY_BYTES:                # one huge step is not kept for undo
                hist.append(step)
            del hist[:-HISTORY_LIMIT]
            while hist and _size(hist) > HISTORY_BYTES:
                del hist[0]
            box["step"] = step["step_id"] if hist and hist[-1] is step else None
    if d["reviewed"] and d["edits"] != before["edits"]:
        d["reviewed"], d["reviewed_at"], d["reviewed_by"] = False, None, None   # changed after sign off
    if d != before:
        d["rev"] += 1
    box["out"] = _validated(d)


def _find(d: dict, edit_id: str) -> dict:
    e = next((x for x in d["edits"] if x["edit_id"] == edit_id), None)
    if e is None:
        raise ReviewError(f"no edit {edit_id!r} on this run")
    return e


def _accept(d: dict, e: dict, by: str, now: str) -> None:
    d["seq"] += 1
    e.update(status="accepted", accepted_seq=d["seq"], decided_at=now, decided_by=by)


def _new_edit(edit_id: str, kind: str, payload: dict, source: str, by: str, reason, job_id, agent_version,
              now: str) -> dict:
    return {"edit_id": edit_id, "kind": kind, "payload": payload, "source": source, "by": by, "job_id": job_id,
            "agent_version": agent_version, "reason": reason, "status": "proposed", "created_at": now,
            "decided_at": None, "decided_by": None, "accepted_seq": None, "corrects": None, "replaces": None,
            "replaced_by": None, "adjusted": False, "legacy_reading": None, "legacy_reading_history": []}


def _payload(kind: str, payload) -> dict:
    try:
        return check_payload(kind, copy.deepcopy(payload))
    except FormatError as e:
        raise ReviewError(f"bad {kind} payload: {e}") from None


def _edit_of(review: dict, edit_id: str) -> dict:
    return next(e for e in review["edits"] if e["edit_id"] == edit_id)


def _same_open(edits: list[dict], kind, payload, source, by, job_id, agent_version) -> dict | None:
    return next((e for e in edits if e["status"] == "proposed" and e["source"] == source and e["by"] == by
                 and e["job_id"] == job_id and e["agent_version"] == agent_version and e["kind"] == kind
                 and e["payload"] == payload), None)


def _check_proposal(kind, payload, source, by, reason, job_id, agent_version, corrects) -> tuple[dict, str]:
    payload = _payload(kind, payload)
    by = _who(by)
    if source == "human":
        if job_id is not None or agent_version is not None:
            raise ReviewError("a person's edit has no job_id or agent_version")
    elif source in ("agent", "check", "splitter"):
        if not job_id or not agent_version:
            raise ReviewError(f"a {source} proposal names its job_id and agent_version")
        if corrects is not None:
            raise ReviewError("only a person's edit corrects another edit")
        if not isinstance(reason, str) or not reason.strip():
            raise ReviewError(f"a {source} proposal says why, in a short reason")
    else:
        raise ReviewError("source must be human, agent, check or splitter")
    return payload, by


def _propose_into(d: dict, kind, payload, source, by, reason, job_id, agent_version, corrects, edit_id,
                  now) -> dict:
    """Add one checked edit to the mutable review d (see propose)."""
    e = _new_edit(edit_id or new_id(), kind, payload, source, by, reason, job_id, agent_version, now)
    if kind == "reading":
        label = next((x for x in d["edits"] if x["edit_id"] == payload["label_id"]), None)
        if label is None or label["kind"] != "label" or label["status"] != "accepted":
            raise ReviewError(f"no accepted label {payload['label_id']!r} on this run")
    if corrects is not None:
        target = _find(d, corrects)
        if target["source"] == "human":
            raise ReviewError("a person's edit is changed with replace, not corrects")
        if target["kind"] != kind:
            raise ReviewError("a correction has the kind of the edit it corrects")
        if target["status"] != "proposed":
            raise ReviewError("only an open proposal is corrected; change an accepted edit with replace")
        target.update(status="rejected", decided_at=now, decided_by=by)
        e["corrects"] = corrects
    d["edits"].append(e)
    if source == "human":
        _accept(d, e, by, now)
    return e


def propose(review: dict, *, kind: str, payload: dict, source: str, by: str, expect_rev: int, now: str,
            reason: str | None = None, job_id: str | None = None, agent_version: str | None = None,
            corrects: str | None = None, edit_id: str | None = None, t: int | None = None,
            what: str | None = None) -> tuple[dict, dict]:
    """Add an edit. A person's edit is accepted at once. An agent, check or splitter edit waits as a proposal and
    never changes another edit, so two proposals for one slot stay side by side until a person accepts one.

    corrects (a person's edit only) names an open agent, check or splitter proposal of the same kind that this edit
    changes; the corrected proposal is rejected in the same step. An accepted edit is changed with replace instead.
    The same open proposal from the same source, by, job_id and agent_version is not added twice: the existing one
    is returned and rev stays. A reading names an accepted label.
    """
    payload, by = _check_proposal(kind, payload, source, by, reason, job_id, agent_version, corrects)
    if source != "human":
        same = _same_open(review["edits"], kind, payload, source, by, job_id, agent_version)
        if same is not None:
            if review["rev"] != expect_rev:
                raise ReviewConflict(f"the review changed (now at rev {review['rev']}, you had {expect_rev})")
            return review, same
    with _write(review, expect_rev, what=what or f"add {kind}") as box:
        box["now"], box["t"] = now, t
        e = _propose_into(box["d"], kind, payload, source, by, reason, job_id, agent_version, corrects, edit_id, now)
    return box["out"], _edit_of(box["out"], e["edit_id"])


def accept(review: dict, edit_id: str, *, by: str, expect_rev: int, now: str, t: int | None = None,
           what: str | None = None) -> tuple[dict, dict]:
    """A person accepts a proposal unchanged. Only a proposal can be accepted."""
    by = _who(by)
    with _write(review, expect_rev, what=what or "accept") as box:
        box["now"], box["t"] = now, t
        e = _find(box["d"], edit_id)
        if e["status"] != "proposed":
            raise ReviewError("only a proposal can be accepted; change an accepted edit with replace")
        _accept(box["d"], e, by, now)
    return box["out"], _edit_of(box["out"], edit_id)


def reject(review: dict, edit_id: str, *, by: str, expect_rev: int, now: str, t: int | None = None,
           what: str | None = None) -> tuple[dict, dict]:
    """A person rejects a proposal. An accepted edit is changed with replace or taken back with undo."""
    by = _who(by)
    with _write(review, expect_rev, what=what or "reject") as box:
        box["now"], box["t"] = now, t
        e = _find(box["d"], edit_id)
        if e["status"] != "proposed":
            raise ReviewError("only a proposal can be rejected; undo or replace an accepted edit")
        e.update(status="rejected", decided_at=now, decided_by=by)
    return box["out"], _edit_of(box["out"], edit_id)


def replace(review: dict, edit_id: str, payload: dict, *, by: str, expect_rev: int, now: str,
            new_edit_id: str | None = None, t: int | None = None, what: str | None = None) -> tuple[dict, dict]:
    """A person rewrites an accepted edit. The old one is retired with replaced_by; the new one is in effect and
    names it in replaces (and in corrects when the old one was not a person's). Readings of a replaced label stay
    with the old label, so they no longer apply."""
    by = _who(by)
    with _write(review, expect_rev, what=what or "replace") as box:
        d = box["d"]
        box["now"], box["t"] = now, t
        old = _find(d, edit_id)
        if old["status"] != "accepted":
            raise ReviewError(f"no accepted edit {edit_id!r} on this run")
        e = _new_edit(new_edit_id or new_id(), old["kind"], _payload(old["kind"], payload), "human", by, None,
                      None, None, now)
        e["replaces"] = edit_id
        if old["source"] != "human":
            e["corrects"] = edit_id
        old.update(status="rejected", decided_at=now, decided_by=by, replaced_by=e["edit_id"])
        d["edits"].append(e)
        _accept(d, e, by, now)
    return box["out"], _edit_of(box["out"], e["edit_id"])


def _undone_edits(edits: list[dict], step: dict) -> list[dict]:
    """The edits after putting back every edit the step touched, or ReviewError when that cannot be done cleanly:
    an edit changed again since, or another edit still refers to one the undo would take away."""
    cur = {e["edit_id"]: e for e in edits}
    if any(cur.get(c["edit_id"]) != c["after"] for c in step["changes"]):
        raise ReviewError("those edits changed again since; undo the later change first")
    out = list(edits)
    for c in step["changes"]:
        if c["before"] is None:
            out = [e for e in out if e["edit_id"] != c["edit_id"]]
        elif c["after"] is None:
            out.append(copy.deepcopy(c["before"]))
        else:
            out[out.index(cur[c["edit_id"]])] = copy.deepcopy(c["before"])
    ids = {e["edit_id"] for e in out}
    for e in out:
        refs = [e["corrects"], e["replaces"], e["replaced_by"]]
        if e["kind"] == "reading":
            refs.append(e["payload"]["label_id"])
        gone = [r for r in refs if r is not None and r not in ids]
        if gone:
            raise ReviewError(f"edit {e['edit_id']} still refers to {gone[0]}; undo that edit's step first")
    return out


def undoable(review: dict) -> list[str]:
    """Step ids that can be undone now, oldest first: every edit the step touched is still as the step left it, and
    no other edit refers to an edit the undo would take away."""
    out = []
    for h in review["history"]:
        if h["undone"]:
            continue
        try:
            _undone_edits(review["edits"], h)
        except ReviewError:
            continue
        out.append(h["step_id"])
    return out


def undo(review: dict, step_id: str, *, expect_rev: int, now: str) -> tuple[dict, str]:
    """Put back every edit one step touched, if none changed since and nothing still refers to an edit it takes
    away. Undo is not itself an undoable step."""
    with _write(review, expect_rev) as box:
        d = box["d"]
        step = next((h for h in d["history"] if h["step_id"] == step_id), None)
        if step is None:
            raise ReviewError("nothing to undo with that id")
        if step["undone"]:
            raise ReviewError("that step was already undone")
        d["edits"] = _undone_edits(d["edits"], step)
        step.update(undone=True, undone_at=now)
    return box["out"], step_id


def set_reviewed(review: dict, reviewed: bool, *, by: str, expect_rev: int, now: str) -> dict:
    """A person signs the run off, or takes that back. Any later change to the edits clears it."""
    if not isinstance(reviewed, bool):
        raise ReviewError("reviewed is true or false")
    by = _who(by)
    with _write(review, expect_rev) as box:
        box["d"].update(reviewed=reviewed, reviewed_at=now if reviewed else None, reviewed_by=by if reviewed else None)
    return box["out"]


def lock(review: dict, *, by: str, expect_rev: int, now: str, reason: str | None = None) -> dict:
    """Freeze the review while its pack is locked."""
    by = _who(by)
    if review["locked"] is not None:
        raise ReviewLocked("the review is already locked")
    with _write(review, expect_rev) as box:
        box["d"]["locked"] = {"by": by, "at": now, "reason": reason}
        box["d"]["lock_log"].append({"action": "lock", "by": by, "at": now, "reason": reason})
    return box["out"]


def unlock(review: dict, *, by: str, reason: str, expect_rev: int, now: str) -> dict:
    """An owner unlocks the review, with a reason, before any change to a locked pack's run."""
    by = _who(by)
    if review["locked"] is None:
        raise ReviewError("the review is not locked")
    if not isinstance(reason, str) or not reason.strip():
        raise ReviewError("unlocking needs a reason")
    with _write(review, expect_rev, allow_locked=True) as box:
        box["d"]["locked"] = None
        box["d"]["lock_log"].append({"action": "unlock", "by": by, "at": now, "reason": reason})
    return box["out"]


# Logger-side operations. Each one is a series of the steps above done as one undoable step, or (retract) the
# one change a person makes in the logger's viewer that the shared operations above do not name. Their results are
# ordinary format 2 reviews and pass check_review like any other.


def propose_many(review: dict, items: list[dict], *, expect_rev: int, now: str, t: int | None = None,
                 what: str) -> tuple[dict, list[dict]]:
    """Several proposals in one step. items are propose's keyword arguments (kind, payload, source, by, reason,
    job_id, agent_version). An item equal to an open proposal (as propose decides it) returns that proposal."""
    checked = [(it, *_check_proposal(it["kind"], it["payload"], it["source"], it["by"], it.get("reason"),
                                     it.get("job_id"), it.get("agent_version"), None)) for it in items]
    with _write(review, expect_rev, what=what) as box:
        box["now"], box["t"] = now, t
        ids = []
        for it, payload, by in checked:
            same = None if it["source"] == "human" else _same_open(box["d"]["edits"], it["kind"], payload,
                                                                   it["source"], by, it.get("job_id"),
                                                                   it.get("agent_version"))
            e = same or _propose_into(box["d"], it["kind"], payload, it["source"], by, it.get("reason"),
                                      it.get("job_id"), it.get("agent_version"), None, None, now)
            ids.append(e["edit_id"])
    return box["out"], [_edit_of(box["out"], i) for i in ids]


def decide_many(review: dict, edit_ids: list[str], accept_them: bool, *, by: str, expect_rev: int, now: str,
                t: int | None = None, what: str) -> dict:
    """A person accepts or rejects several proposals in one step. Every id must be an open proposal."""
    by = _who(by)
    with _write(review, expect_rev, what=what) as box:
        box["now"], box["t"] = now, t
        todo = [next((x for x in box["d"]["edits"] if x["edit_id"] == i), None) for i in edit_ids]
        if any(e is None or e["status"] != "proposed" for e in todo) or len(set(edit_ids)) != len(edit_ids):
            raise ReviewError("every id must be an open proposal on this run, once")
        for e in todo:
            if accept_them:
                _accept(box["d"], e, by, now)
            else:
                e.update(status="rejected", decided_at=now, decided_by=by)
    return box["out"]


def retract(review: dict, edit_id: str, *, by: str, expect_rev: int, now: str, t: int | None = None,
            what: str | None = None) -> tuple[dict, dict]:
    """A person takes back an accepted edit (deletes a label or a cut in the viewer). The edit is kept as rejected
    with its accepted_seq, so the edit accepted before it in the same slot is in effect again. Its acceptance stays
    as it was (decided_at and decided_by still say who accepted it and when); the take back is recorded as its own
    history step, whose `what` names who took it back and whose `at` says when. Format 1 recorded the same state
    (rejected with an accepted_seq). Readings of a label taken back stay with it and no longer apply, as with
    replace."""
    by = _who(by)
    with _write(review, expect_rev, what=what or "take back") as box:
        box["now"], box["t"] = now, t
        e = _find(box["d"], edit_id)
        if e["status"] != "accepted":
            raise ReviewError("only an accepted edit can be taken back")
        box["what"] = f"{what or 'take back'} (taken back by {by})"
        e["status"] = "rejected"
    return box["out"], _edit_of(box["out"], edit_id)


# ------------------------------------------------------------------ format 1


FORMAT1_KINDS = ("outcome", "failure_start", "exclude", "step_boundary", "segment_edge", "drive_boundary",
                 "segment_plan", "reject_run", "note", "label", "event_note")
F1_TOP_KEYS = {"version", "rev", "seq", "reviewed", "reviewed_at", "reviewed_by", "edits", "history"}
F1_EDIT_KEYS = {"id", "kind", "payload", "source", "reason", "status", "created_at", "decided_at", "by",
                "accepted_seq", "adjusted", "replaces", "replaced_by", "interpretation", "interpretation_history"}
F1_STEP_KEYS = {"id", "at", "t", "what", "changes", "undone", "undone_at"}
F1_READING_KEYS = {"train", "meaning", "by", "at"}


def is_format1(raw) -> bool:
    return isinstance(raw, dict) and "review_format" not in raw and raw.get("version") == 1


def dump_format1(review: dict) -> str:
    """The logger's format 1 writer (the same canonical text)."""
    return dumps(review)


def normalize_format1(raw: dict) -> dict:
    """A format 1 review as the logger read it: a missing rev or seq is 0, and a missing history or reviewed_by is
    the same as none (format 1 wrote them only when set)."""
    out = dict(raw)
    out.setdefault("rev", 0)
    out.setdefault("seq", 0)
    if out.get("reviewed_by", None) is None:
        out.pop("reviewed_by", None)
    if not out.get("history"):
        out.pop("history", None)
    return out


def _f1_keys(obj, allowed: set, what: str) -> dict:
    if not isinstance(obj, dict):
        raise MigrationError(f"{what} is not an object")
    extra = set(obj) - allowed
    if extra:
        raise MigrationError(f"{what} has keys format 1 never writes: {sorted(extra)}")
    return obj


def _subgoal_names(manifest: dict) -> list[str]:
    names = []
    for s in manifest.get("subgoals") or []:
        # any non-empty name, spaces included, kept exactly as recorded
        if not isinstance(s, dict) or not isinstance(s.get("name"), str) or not s["name"]:
            raise MigrationError("a recorded subgoal has no name")
        names.append(s["name"])
    return names


def _resolve(name: str, task: dict | None) -> str:
    """The task's one subtask with this name, ignoring case and surrounding spaces; unresolved otherwise."""
    if task is None:
        return UNRESOLVED
    key = name.strip().casefold()
    hit = [s["subtask_id"] for s in task.get("subtasks", []) if s["name"].casefold() == key]
    return hit[0] if len(hit) == 1 else UNRESOLVED


def _f1_edits_and_snapshots(raw: dict) -> list:
    snaps = [s for h in raw.get("history") or [] for c in (h.get("changes") or []) if isinstance(c, dict)
             for s in (c.get("before"), c.get("after")) if isinstance(s, dict)]
    return list(raw["edits"]) + snaps


def _f1_reading(r) -> dict:
    _f1_keys(r, F1_READING_KEYS, "a reading")
    return dict(r)


def _f1_edit(e, segment_ids: dict[str, str]) -> dict:
    _f1_keys(e, F1_EDIT_KEYS, "an edit")
    for k in ("id", "kind", "payload", "source", "status", "created_at"):
        if k not in e:
            raise MigrationError(f"an edit has no {k}")
    if "reason" not in e or "decided_at" not in e:
        raise MigrationError(f"edit {e['id']} lacks reason or decided_at, which format 1 always writes")
    if e["kind"] not in FORMAT1_KINDS:
        raise MigrationError(f"edit {e['id']} has kind {e['kind']!r}, which format 1 never writes")
    if e["source"] not in ("human", "agent"):
        raise MigrationError(f"edit {e['id']} has source {e['source']!r}, which format 1 never writes")
    if e.get("adjusted", True) is not True:
        raise MigrationError(f"edit {e['id']} has adjusted false, which format 1 never writes")
    kind, p = e["kind"], e["payload"]
    if kind == "outcome":
        if not isinstance(p, dict) or set(p) != {"termination_reason", "failure_mode"}:
            raise MigrationError(f"edit {e['id']}: an outcome payload has termination_reason and failure_mode")
        tr = p["termination_reason"]
        if not isinstance(tr, str):
            raise MigrationError(f"edit {e['id']}: termination_reason is not text")
        p = {"outcome": outcome_of(tr), **p}
    elif kind == "segment_plan":
        kind = "segments"
        if not isinstance(p, dict) or set(p) != {"segments"} or not isinstance(p["segments"], list):
            raise MigrationError(f"edit {e['id']}: a segment_plan payload is {{segments: [...]}}")
        segs = []
        for s in p["segments"]:
            _f1_keys(s, {"name", "start", "end"}, "a segment_plan segment")
            if not isinstance(s.get("name"), str):
                raise MigrationError(f"edit {e['id']}: a segment has no name")
            segs.append({"subtask_id": segment_ids[s["name"]], "start": s.get("start"), "end": s.get("end"),
                         "legacy_name": s["name"]})
        p = {"segments": segs}
    hist = e.get("interpretation_history", [])
    if not isinstance(hist, list):
        raise MigrationError(f"edit {e['id']} has a malformed interpretation_history")
    is_human = e["source"] == "human"
    return {
        "edit_id": e["id"], "kind": kind, "payload": copy.deepcopy(p), "source": e["source"], "by": e.get("by"),
        "job_id": None if is_human else UNKNOWN, "agent_version": None if is_human else UNKNOWN,
        "reason": e["reason"], "status": e["status"], "created_at": e["created_at"], "decided_at": e["decided_at"],
        "decided_by": None, "accepted_seq": e.get("accepted_seq"), "corrects": None, "replaces": e.get("replaces"),
        "replaced_by": e.get("replaced_by"), "adjusted": "adjusted" in e,
        "legacy_reading": _f1_reading(e["interpretation"]) if e.get("interpretation") is not None else None,
        "legacy_reading_history": [_f1_reading(r) for r in hist],
    }


def _f1_step(h, segment_ids: dict[str, str]) -> dict:
    _f1_keys(h, F1_STEP_KEYS, "a history step")
    changes = []
    for c in h.get("changes") or []:
        _f1_keys(c, {"id", "before", "after"}, "a history change")
        changes.append({"edit_id": c.get("id"),
                        "before": None if c.get("before") is None else _f1_edit(c["before"], segment_ids),
                        "after": None if c.get("after") is None else _f1_edit(c["after"], segment_ids)})
    return {"step_id": h.get("id"), "at": h.get("at"), "t": h.get("t"), "what": h.get("what"), "changes": changes,
            "undone": h.get("undone"), "undone_at": h.get("undone_at")}


def migrate_format1(raw: dict, manifest: dict, task: dict | None = None) -> dict:
    """A format 1 review of `manifest`'s run as format 2, with its legacy_subtask_map.

    task ({task_id, subtasks: [{subtask_id, name}]}) is the task the run belongs to, or None, which leaves every
    name unresolved. Only names are matched, never positions, so the result does not depend on the order of the
    task's subtasks, and it records no task revision. Anything format 1 never wrote stops the migration.
    """
    _f1_keys(raw, F1_TOP_KEYS, "the review")
    if raw.get("version") != 1:
        raise MigrationError("not a format 1 review (version 1)")
    if not isinstance(raw.get("edits"), list) or not isinstance(raw.get("history", []), list):
        raise MigrationError("edits and history are lists")
    if task is not None:
        if not isinstance(task.get("task_id"), str) or not SLUG.fullmatch(task["task_id"]):
            raise MigrationError("the task has no valid task_id")
        if any(s.get("subtask_id") == UNRESOLVED for s in task.get("subtasks", [])):
            raise MigrationError(f"task {task['task_id']} uses the reserved subtask id {UNRESOLVED!r}")
    entries = [{"step_index": i, "recorded_name": n, "subtask_id": _resolve(n, task)}
               for i, n in enumerate(_subgoal_names(manifest))]
    plan_names = sorted({s["name"] for e in _f1_edits_and_snapshots(raw)
                         if isinstance(e, dict) and e.get("kind") == "segment_plan"
                         for s in (e.get("payload") or {}).get("segments") or []
                         if isinstance(s, dict) and isinstance(s.get("name"), str)})
    segment_ids = {n: _resolve(n, task) for n in plan_names}
    entries += [{"step_index": None, "recorded_name": n, "subtask_id": segment_ids[n]} for n in plan_names]
    d = {
        "review_format": 2, "run_id": manifest["run_id"], "rev": raw.get("rev", 0), "seq": raw.get("seq", 0),
        "reviewed": raw.get("reviewed"), "reviewed_at": raw.get("reviewed_at"), "reviewed_by": raw.get("reviewed_by"),
        "edits": [_f1_edit(e, segment_ids) for e in raw["edits"]],
        "history": [_f1_step(h, segment_ids) for h in raw.get("history", [])],
        "locked": None, "lock_log": [],
        "legacy_subtask_map": {"run_id": manifest["run_id"], "task_id": None if task is None else task["task_id"],
                               "entries": entries},
    }
    try:
        check_review(d)
        check_against_manifest(d, manifest)
    except FormatError as e:
        raise MigrationError(f"the format 1 review does not migrate cleanly: {e}") from None
    return d


def _edit_to_format1(e: dict) -> dict:
    where = f"edit {e['edit_id']}"
    if e["source"] not in ("human", "agent"):
        raise MigrationError(f"{where}: format 1 has no {e['source']} source")
    if e["source"] == "agent" and (e["job_id"], e["agent_version"]) != (UNKNOWN, UNKNOWN):
        raise MigrationError(f"{where}: format 1 cannot keep job_id and agent_version")
    if e["decided_by"] is not None or e["corrects"] is not None:
        raise MigrationError(f"{where}: format 1 cannot keep decided_by or corrects")
    kind, p = e["kind"], e["payload"]
    if kind in ("failure_type", "name", "reading"):
        raise MigrationError(f"{where}: format 1 has no {kind} kind")
    if kind == "outcome":
        if p["termination_reason"] is None:
            raise MigrationError(f"{where}: format 1 needs the outcome's termination_reason")
        p = {"termination_reason": p["termination_reason"], "failure_mode": p["failure_mode"]}
    elif kind == "segments":
        kind = "segment_plan"
        if any(s["legacy_name"] is None for s in p["segments"]):
            raise MigrationError(f"{where}: a segment drawn in format 2 has no format 1 name")
        p = {"segments": [{"name": s["legacy_name"], "start": s["start"], "end": s["end"]} for s in p["segments"]]}
    out = {"id": e["edit_id"], "kind": kind, "payload": copy.deepcopy(p), "source": e["source"],
           "reason": e["reason"], "status": e["status"], "created_at": e["created_at"], "decided_at": e["decided_at"]}
    for k in ("by", "accepted_seq", "replaces", "replaced_by"):
        if e[k] is not None:
            out[k] = e[k]
    if e["adjusted"]:
        out["adjusted"] = True
    if e["legacy_reading"] is not None:
        out["interpretation"] = dict(e["legacy_reading"])
    if e["legacy_reading_history"]:
        out["interpretation_history"] = [dict(r) for r in e["legacy_reading_history"]]
    return out


def to_format1(review: dict) -> dict:
    """The format 1 file a migrated review came from. Refuses anything format 1 cannot say."""
    d = check_review(review)
    if d["locked"] is not None or d["lock_log"]:
        raise MigrationError("format 1 has no lock")
    out: dict[str, Any] = {"version": 1, "rev": d["rev"], "seq": d["seq"], "reviewed": d["reviewed"],
                           "reviewed_at": d["reviewed_at"], "edits": [_edit_to_format1(e) for e in d["edits"]]}
    if d["reviewed_by"] is not None:
        out["reviewed_by"] = d["reviewed_by"]
    hist = []
    for h in d["history"]:
        step = {"id": h["step_id"], "at": h["at"], "what": h["what"], "undone": h["undone"],
                "changes": [{"id": c["edit_id"],
                             "before": None if c["before"] is None else _edit_to_format1(c["before"]),
                             "after": None if c["after"] is None else _edit_to_format1(c["after"])}
                            for c in h["changes"]]}
        if h["t"] is not None:
            step["t"] = h["t"]
        if h["undone_at"] is not None:
            step["undone_at"] = h["undone_at"]
        hist.append(step)
    if hist:
        out["history"] = hist
    return out


def read(text: str | bytes, manifest: dict, task: dict | None = None) -> dict:
    """A review file's text as format 2: format 2 is checked as is, format 1 is migrated. Then checked against the
    recorded run. FormatError (or MigrationError) when it is neither or does not fit."""
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise FormatError(f"unreadable ({e})") from None
    if is_format1(raw):
        return migrate_format1(raw, manifest, task)
    if not isinstance(raw, dict) or raw.get("review_format") != REVIEW_FORMAT:
        raise FormatError("neither a review_format 2 file nor a format 1 (version 1) file")
    check_review(raw)
    check_against_manifest(raw, manifest)
    return raw
