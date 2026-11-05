"""Human review of committed runs: corrections stored next to the run, never applied to the recorded data.

Each run may have `dataop/runs/<run_id>/review.json` in review format 2 (`review_format.py`): every edit people,
agents and checks made, with its source, status and who decided it, an undo history, and whether the run has been
reviewed. The recorded frames, the manifest and its checksums never change; `effective()` folds the accepted edits
into what training should use, and the reader turns that into `dataop.train_mask`.

A file in the logger's earlier format 1 (`{"version": 1, ...}`) is read through the lossless migration and stays as
it is on disk until the next change to that run's review, which writes it as format 2. A format 2 file this logger
wrote at 4f4395e, before edits had retracted_at and retracted_by, is read the same way: upgraded in memory
(review_format._before_take_back_fields) and written with both keys at its next change.

Edit kinds and their payloads (frames are frame indices of the run; ranges are [start, end)):
  outcome        {"outcome": success | failure | unknown, "termination_reason": str | None, "failure_mode": str | None}
                 (callers may pass only termination_reason and failure_mode; outcome follows from the reason)
  failure_start  {"frame": int}       on a failed run, frames from here to the next takeover are not good behaviour
  failure_type   {"failure_type_id": id, "vocab_rev": int}
  exclude        {"start": int, "end": int}   frames [start, end) are left out of training
  segments       {"segments": [{"subtask_id", "start", "end", "legacy_name"}]}: the run's full list of task segments
                 (callers may pass [{"name", "start", "end"}]; see review_format._named_segments)
  step_boundary  {"index": int, "before": str, "after": str, "frame": int}
                 moves the shared boundary between recorded steps `index` and `index + 1`
  segment_edge   {"index": int, "edge": start | end, "frame": int}
  drive_boundary {"index": int, "frame": int}
  reject_run     {}                           the whole run is left out of training
  note           {"frame": int, "text": str}
  event_note     {"key": str, "text": str}
  label          {"start": int, "end": int, "text": str, "driver": str | None, "segment": str | None}
                 a person's description of frames [start, end) in plain language
  reading        {"label_id": id, "train": keep | exclude, "meaning": str}: what a label means for training; the
                 reading in effect of "exclude" leaves the label's span out of training
  name           {"text": str}

A person's edit is accepted at once. An agent's edit is a proposal that a person accepts, corrects or rejects. In the
one-value kinds the most recently accepted edit wins, and taking it back brings back the one accepted before it.
Every change happens under a cross-process lock, is checked against the recorded run, and bumps `rev`.
"""
from __future__ import annotations

import contextvars
import fcntl
import json
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import review_format as fmt
from . import store

KINDS = fmt.KINDS + ("segment_plan",)       # segment_plan: the format 1 name for segments, still accepted as input
TRAIN_USE = fmt.TRAIN_USE
SOURCES = fmt.SOURCES
STATUSES = fmt.STATUSES
MULTI = fmt.MULTI
TAKEOVER_MODES = ("expert", "blend")
EDIT_ID = re.compile(r"^[0-9a-f]{12}$")
HISTORY_LIMIT = fmt.HISTORY_LIMIT
HISTORY_BYTES = fmt.HISTORY_BYTES


class ReviewError(ValueError):
    pass


class ReviewConflict(ReviewError):
    """The review changed since the caller read it."""


@contextmanager
def _errors():
    """Format and operation errors as this module's errors."""
    try:
        yield
    except fmt.ReviewConflict as e:
        raise ReviewConflict(str(e)) from None
    except (fmt.ReviewError, fmt.FormatError) as e:
        raise ReviewError(str(e)) from None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _path(run_path: Path) -> Path:
    return Path(run_path) / "review.json"


def _agent_version() -> str:
    """The version written on an agent proposal when the caller names none: this logger's own, never "unknown"
    (which marks edits migrated from format 1)."""
    try:
        from importlib.metadata import version
        return f"aftershock-logger {version('aftershock-logger')}"
    except Exception:                                   # run from a source checkout
        return "aftershock-logger (source)"


def _job_id() -> str:
    """A job id for one logger command that proposes edits (one find_idle call, one suggest-cut)."""
    return f"local-{uuid.uuid4().hex[:12]}"


def empty(run_id: str) -> dict:
    with _errors():
        return fmt.empty_review(run_id)


def _has_review_file(p: Path) -> bool:
    """False only when there is no review entry at all. A broken link, or an entry that cannot be looked up, is a
    review file that cannot be read (ReviewError), never a missing one."""
    try:
        p.lstat()
    except FileNotFoundError:
        return False
    except OSError as e:
        raise ReviewError(f"{p.name} cannot be looked up ({type(e).__name__})") from None
    return True


def load(run_path: Path, manifest: dict | None = None, task: dict | None = None) -> dict:
    """The run's review as format 2, checked against the recorded run. A missing file is an empty review; a format 1
    file is migrated in memory (by recorded name, against `task` when given) and left on disk as it is."""
    run_path = Path(run_path)
    if manifest is None:
        manifest = store.read_manifest(run_path)
    p = _path(run_path)
    if not _has_review_file(p):
        return empty(manifest["run_id"])
    try:
        return fmt.read(p.read_bytes(), manifest, task)
    except OSError as e:
        raise ReviewError(f"{p}: cannot be read ({type(e).__name__})") from None
    except (fmt.FormatError, KeyError, TypeError, AttributeError) as e:
        raise ReviewError(f"{p}: {e}") from None


def file_format(run_path: Path) -> int | None:
    """1 or 2 for the review file on disk, None only when the run has no review entry at all. An entry that cannot be
    looked up or read (a broken link, for example), is not valid JSON or is in neither format raises ReviewError, so
    callers never mistake a damaged review for a missing one."""
    p = _path(run_path)
    if not _has_review_file(p):
        return None
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        what = "is not valid JSON" if isinstance(e, ValueError) else f"cannot be read ({type(e).__name__})"
        raise ReviewError(f"{p.name} {what}") from None
    if fmt.is_format1(raw):
        return 1
    if isinstance(raw, dict) and raw.get("review_format") == 2:
        return 2
    found = raw.get("review_format", raw.get("version")) if isinstance(raw, dict) else type(raw).__name__
    raise ReviewError(f"{p.name} is in no known review format (found {found!r})")


_RECORDING = contextvars.ContextVar("dataop_review_recording", default=None)


@contextmanager
def recording(what: str):
    """Name the undoable step the review change inside this block makes (one change per block). Yields a dict whose
    "entry" is the step id once it is saved (None if nothing changed or the step was too big to keep)."""
    box = {"what": what, "entry": None}
    token = _RECORDING.set(box)
    try:
        yield box
    finally:
        _RECORDING.reset(token)


@contextmanager
def transaction(run_path: Path, manifest: dict | None = None, expect_rev: int | None = None):
    """Load one run's review under an exclusive cross-process lock, let the caller replace box["review"] with the
    result of format 2 operations, and save it if it changed. The batches lock is held shared from the lock check to
    the write (taken first, as batches.set_status does), so a batch cannot be locked or its files moved between the
    check and the edit. Yields {"review", "what", "now", "t"}; "what" names the step (from recording())."""
    from . import batches
    run_path = Path(run_path)
    if manifest is None:
        manifest = store.read_manifest(run_path)
    root = run_path.parent.parent.parent
    with batches.shared(root), open(run_path / "review.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            try:
                batches.check_editable(root, run_path.name)
            except batches.BatchError as e:
                raise ReviewError(str(e)) from None
            review = load(run_path, manifest)
            if expect_rev is not None and review["rev"] != expect_rev:
                raise ReviewConflict(f"the review changed (now at rev {review['rev']}, you had {expect_rev})")
            rec = _RECORDING.get()
            if rec is not None and rec.get("used"):
                raise ReviewError("a recorded step may contain one transaction only")
            box = {"review": review, "what": rec["what"] if rec else None, "now": _now(), "t": time.time_ns()}
            with _errors():
                yield box
            out = box["review"]
            if rec is not None:
                rec["used"] = True
                rec["entry"] = _new_step(review, out)
            if box.get("force") or (out is not review and out != review):
                with _errors():
                    fmt.check_review(out)
                    fmt.check_against_manifest(out, manifest)
                store.atomic_write_bytes(_path(run_path), fmt.dump_review(out).encode())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _new_step(before: dict, after: dict) -> str | None:
    """The id of the history step an operation added, if it kept one."""
    if not after["history"]:
        return None
    last = after["history"][-1]["step_id"]
    return None if any(h["step_id"] == last for h in before["history"]) else last


def _ints(v):
    """numpy integers as plain ints, so callers can pass frame numbers straight from arrays."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, dict):
        return {k: _ints(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_ints(x) for x in v]
    return v


def _input(kind: str, payload, manifest: dict) -> tuple[str, dict]:
    """A caller's payload in the stored form: the format 1 shapes the viewer and scripts send (an outcome without
    `outcome`, named segments, a step boundary without its step names), texts trimmed, numpy integers as ints.
    The result is checked by the operation itself."""
    if kind not in KINDS:
        raise ReviewError(f"kind must be one of {fmt.KINDS}")
    if not isinstance(payload, dict):
        raise ReviewError("payload must be an object")
    p = _ints(dict(payload))
    if kind == "segment_plan":
        kind = "segments"
    if kind == "outcome":
        tr = p.get("termination_reason")
        out = p.get("outcome", fmt.outcome_of(tr) if isinstance(tr, str) and tr in fmt.TERMINATION_REASONS else None)
        p = {"outcome": out, "termination_reason": tr, "failure_mode": p.get("failure_mode")}
    elif kind == "segments":
        segs = p.get("segments")
        if isinstance(segs, list):
            out = []
            for x in segs:
                if isinstance(x, dict) and "subtask_id" not in x and "name" in x:
                    name = x["name"]
                    if not isinstance(name, str) or not 0 < len(name.strip()) <= 100:
                        raise ReviewError("each segment needs a name")
                    x = {"subtask_id": fmt.UNRESOLVED, "start": x.get("start"), "end": x.get("end"),
                         "legacy_name": name.strip()}
                out.append(x)
            p = {"segments": out}
    elif kind == "step_boundary":
        steps, i = manifest.get("subgoals") or [], p.get("index")
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(steps) - 1:
            p.setdefault("before", steps[i]["name"])
            p.setdefault("after", steps[i + 1]["name"])
    elif kind == "label":
        text, driver, segment = p.get("text") or "", p.get("driver"), p.get("segment")
        if not isinstance(text, str) or not text.strip():
            raise ReviewError("a label needs a description of what happens")
        if driver is not None and (not isinstance(driver, str) or not driver.strip()):
            raise ReviewError("drive mode must be a short name (policy, rescue or your own) or null")
        if segment is not None and (not isinstance(segment, str) or not segment.strip()):
            raise ReviewError("segment must be a short name or null")
        p = {"start": p.get("start"), "end": p.get("end"), "text": text.strip(),
             "driver": driver.strip() if driver else None, "segment": segment.strip() if segment else None}
    elif kind == "event_note":
        if isinstance(p.get("text"), str):
            p["text"] = p["text"].strip()
    return kind, p


def _propose(box: dict, **kw) -> dict:
    """fmt.propose on the transaction's review. A segment given by name gets no invented subtask id: it is stored as
    unresolved with its name, and in a migrated review it agrees with legacy_subtask_map (fmt.propose applies the
    rule, as the reference does). A proposal that changes nothing (the same open one exists) leaves the review
    exactly as it was."""
    box["review"], e = fmt.propose(box["review"], expect_rev=box["review"]["rev"], now=box["now"], t=box["t"],
                                   what=box["what"], **kw)
    return e


def check_payload(kind: str, payload: dict, manifest: dict) -> dict:
    """Check and normalise one edit's payload against the recorded run, as an edit would store it."""
    kind, p = _input(kind, payload, manifest)
    with _errors():
        fmt.check_payload(kind, p)
        probe = fmt.empty_review(manifest["run_id"])
        e = fmt._new_edit("0" * 12, kind, p, "human", "check", None, None, None, "now")
        probe["edits"].append(e)
        fmt.check_against_manifest(probe, manifest)
    return p


def set_reviewed(run_path: Path, reviewed: bool, by: str = "person", expect_rev: int | None = None) -> None:
    """Mark the run as reviewed (a person has checked it), or not. Batches count reviewed runs."""
    if not isinstance(reviewed, bool):
        raise ReviewError("reviewed must be true or false")
    with transaction(run_path, expect_rev=expect_rev) as box:
        box["review"] = fmt.set_reviewed(box["review"], reviewed, by=(by or "person")[:80],
                                         expect_rev=box["review"]["rev"], now=box["now"])


def undoable(review: dict) -> list[dict]:
    """Steps that can be undone right now, oldest first: [{entry, what, t}]. A step is offered only while every edit
    it touched is still exactly as the step left it and nothing else refers to an edit it would take away, so a
    later change (from the page, an agent or a script) hides it until that later change is undone."""
    ok = set(fmt.undoable(review))
    return [{"entry": h["step_id"], "what": h["what"], "t": h["t"] or 0} for h in review["history"]
            if h["step_id"] in ok]


def undo(run_path: Path, manifest: dict, entry_id: str, expect_rev: int | None = None) -> dict:
    """Revert one step: every edit it touched goes back to how it was, if nothing changed it since. Returns the
    step."""
    with transaction(run_path, manifest, expect_rev) as box:
        box["review"], sid = fmt.undo(box["review"], entry_id, expect_rev=box["review"]["rev"], now=box["now"])
    return next(h for h in box["review"]["history"] if h["step_id"] == sid)


def add_edit(run_path: Path, manifest: dict, kind: str, payload: dict, source: str = "human",
             reason: str | None = None, expect_rev: int | None = None, by: str | None = None,
             job_id: str | None = None, agent_version: str | None = None) -> dict:
    """Add an edit. A person's edit is accepted at once; an agent's edit waits as a proposal. An agent proposal that
    repeats one a person already decided, or one that is still open, is not added again: the existing edit is
    returned (see add_suggestions for what became of it)."""
    if source not in SOURCES:
        raise ReviewError(f"source must be one of {SOURCES}")
    kind, payload = _input(kind, payload, manifest)
    if by is None:
        by = "person" if source == "human" else source
    if source != "human":
        job_id, agent_version = job_id or _job_id(), agent_version or _agent_version()
    with transaction(run_path, manifest, expect_rev) as box:
        if source != "human":
            prior, _ = _prior_suggestion(box["review"], kind, payload)
            if prior is not None:
                return prior
        e = _propose(box, kind=kind, payload=payload, source=source, by=by, reason=reason, job_id=job_id,
                     agent_version=agent_version)
    return e


def _prior_suggestion(review: dict, kind: str, payload: dict) -> tuple[dict | None, str]:
    """An agent suggestion for exactly this change that already exists, and what became of it.
    A person's decision stands: a span someone kept is not suggested again, whichever agent asks."""
    if kind == "exclude":                       # these exact frames are cut right now, by anyone
        for e in fmt.active_edits(review):
            if e["kind"] == "exclude" and e["payload"] == payload:
                return e, "already_cut"
    for e in review["edits"]:
        if e["source"] != "human" and e["kind"] == kind and e["payload"] == payload and e["status"] != "accepted":
            return e, {"proposed": "already_suggested",
                       "rejected": "previously_kept" if kind == "exclude" else "previously_dismissed"}[e["status"]]
    return None, "created"


def add_suggestions(run_path: Path, manifest: dict, items: list[dict], by: str, expect_rev: int | None = None,
                    job_id: str | None = None, agent_version: str | None = None) -> list[dict]:
    """Add a batch of agent cut suggestions in one locked, revision-checked step.
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
    job_id, agent_version = job_id or _job_id(), agent_version or _agent_version()
    with transaction(run_path, manifest, expect_rev) as box:
        out, new = [], []
        for payload, reason in checked:
            prior, status = _prior_suggestion(box["review"], "exclude", payload)
            if prior is None and payload in [n["payload"] for n in new]:
                prior, status = None, "already_suggested"
            if prior is None and status == "created":
                new.append({"kind": "exclude", "payload": payload, "source": "agent", "by": by.strip(),
                            "reason": reason, "job_id": job_id, "agent_version": agent_version})
            out.append({"status": status, "edit": prior, "payload": payload})
        if new:
            box["review"], made = fmt.propose_many(box["review"], new, expect_rev=box["review"]["rev"],
                                                   now=box["now"], t=box["t"],
                                                   what=box["what"] or f"{by.strip()}: {len(new)} suggested cut(s)")
            by_payload = {json.dumps(e["payload"], sort_keys=True): e for e in made}
            for o in out:
                if o["edit"] is None:
                    o["edit"] = by_payload[json.dumps(o["payload"], sort_keys=True)]
    return [{"status": o["status"], "edit": o["edit"]} for o in out]


def _edit_in(review: dict, edit_id) -> dict:
    e = next((x for x in review["edits"] if x["edit_id"] == edit_id), None)
    if e is None:
        raise ReviewError(f"no edit {edit_id!r} on this run")
    return e


def decide(run_path: Path, manifest: dict, edit_id: str, accept: bool, payload: dict | None = None,
           expect_rev: int | None = None, by: str = "person") -> dict:
    """A person decides an edit. Accept a proposal as it is, or with a changed payload (stored as the person's own
    edit that corrects the proposal). Reject a proposal; rejecting an accepted edit takes it back, which brings back
    the edit accepted before it in the same slot. Returns the edit now in effect for that decision."""
    if not isinstance(accept, bool):
        raise ReviewError("accept must be true or false")
    with transaction(run_path, manifest, expect_rev) as box:
        r = box["review"]
        e = _edit_in(r, edit_id)
        kw = dict(by=by, expect_rev=r["rev"], now=box["now"], t=box["t"], what=box["what"])
        if accept and e["status"] != "proposed":
            raise ReviewError("only a proposal can be accepted; change an accepted edit by replacing it")
        if not accept and e["status"] == "rejected":
            raise ReviewError("this edit is already rejected")
        if accept and payload is not None:
            if e["source"] == "human":
                raise ReviewError("only an agent's proposal is accepted with changes")
            kind, p = _input(e["kind"], payload, manifest)
            out = _propose(box, kind=kind, payload=p, source="human", corrects=edit_id, by=by)
        elif accept:
            box["review"], out = fmt.accept(r, edit_id, **kw)
        elif e["status"] == "proposed":
            box["review"], out = fmt.reject(r, edit_id, **kw)
        else:
            kw.pop("what")              # a take back's step is always "take back"; the viewer's label is display only
            box["review"], out = fmt.retract(r, edit_id, **kw)
    return out


def decide_many(run_path: Path, manifest: dict, edit_ids: list, accept: bool, expect_rev: int | None = None,
                by: str = "person") -> int:
    """Accept or dismiss several proposals in one step (for "Cut all" and "Keep all")."""
    if not isinstance(accept, bool):
        raise ReviewError("accept must be true or false")
    if not isinstance(edit_ids, list) or not edit_ids or len(edit_ids) > 1000:
        raise ReviewError("edit_ids must be a non-empty list")
    with transaction(run_path, manifest, expect_rev) as box:
        box["review"] = fmt.decide_many(box["review"], edit_ids, accept, by=by, expect_rev=box["review"]["rev"],
                                        now=box["now"], t=box["t"],
                                        what=box["what"] or f"{'accept' if accept else 'reject'} {len(edit_ids)}")
    return len(edit_ids)


def apply_agent_edit(run_path: Path, manifest: dict, kind: str, payload: dict, by: str,
                     expect_rev: int | None = None, reason: str | None = None, job_id: str | None = None,
                     agent_version: str | None = None) -> dict:
    """An agent fills in the schema (segments, failure moment and type, drive switch). The value is a proposal,
    marked with who set it and why; a person accepts, changes or rejects it in the viewer. The same value from the
    same agent, job_id and agent_version while that proposal is still open returns it (the format's rule), so an agent
    that runs again under the same job id adds nothing, and a new job or agent version gets its own proposal."""
    if not isinstance(by, str) or not by.strip() or len(by) > 100:
        raise ReviewError("by must name the agent in 1 to 100 characters")
    kind, payload = _input(kind, payload, manifest)
    reason = reason if isinstance(reason, str) and reason.strip() else f"{by.strip()} set {kind}"
    with transaction(run_path, manifest, expect_rev) as box:
        e = _propose(box, kind=kind, payload=payload, source="agent", by=by.strip(), reason=reason,
                     job_id=job_id or _job_id(), agent_version=agent_version or _agent_version())
    return e



def add_event_note(run_path: Path, manifest: dict, key: str, text: str, by: str, source: str = "agent",
                   expect_rev: int | None = None, job_id: str | None = None,
                   agent_version: str | None = None) -> dict:
    """A note on a major event (its details). A person's note applies at once; an agent's waits as a proposal. The
    newest accepted note for an event wins and undo restores the previous one."""
    if source not in SOURCES:
        raise ReviewError(f"source must be one of {SOURCES}")
    if not isinstance(by, str) or not by.strip() or len(by) > 100:
        raise ReviewError("by must name who wrote the note in 1 to 100 characters")
    _, payload = _input("event_note", {"key": key, "text": text}, manifest)
    agent = source != "human"
    with transaction(run_path, manifest, expect_rev) as box:
        e = _propose(box, kind="event_note", payload=payload, source=source, by=by.strip(),
                     reason=f"note on {key}" if agent else None, job_id=(job_id or _job_id()) if agent else None,
                     agent_version=(agent_version or _agent_version()) if agent else None)
    return e


def replace_edit(run_path: Path, manifest: dict, edit_id: str, payload: dict, expect_rev: int | None = None,
                 by: str = "person") -> dict:
    """A person rewrites an accepted edit (for example a label): the old one is retired, the new one is in effect.
    A reading of the old label does not carry over."""
    with transaction(run_path, manifest, expect_rev) as box:
        old = next((e for e in box["review"]["edits"] if e["edit_id"] == edit_id), None)
        if old is None or old["status"] != "accepted":
            raise ReviewError(f"no accepted edit {edit_id!r} on this run")
        _, p = _input(old["kind"], payload, manifest)
        box["review"], e = fmt.replace(box["review"], edit_id, p, by=by, expect_rev=box["review"]["rev"],
                                       now=box["now"], t=box["t"], what=box["what"])
    return e


def interpret(run_path: Path, edit_id: str, train: str, meaning: str, by: str = "agent",
              expect_rev: int | None = None, replace: bool = False, source: str = "agent",
              job_id: str | None = None, agent_version: str | None = None) -> dict:
    """Record what a label means for training: keep or exclude the span, and why, as a reading edit. An agent's
    reading is a proposal a person accepts or rejects; a person's applies at once. A label that already has a reading
    in effect or an open one is refused unless replace=True."""
    if train not in TRAIN_USE:
        raise ReviewError(f"train must be one of {TRAIN_USE}")
    if not isinstance(meaning, str) or not meaning.strip() or len(meaning) > 500:
        raise ReviewError("meaning must be 1 to 500 characters")
    if not isinstance(by, str) or not by.strip() or len(by) > 100:
        raise ReviewError("by must name who reads it in 1 to 100 characters")
    if source not in SOURCES:
        raise ReviewError(f"source must be one of {SOURCES}")
    agent = source != "human"
    with transaction(run_path, expect_rev=expect_rev) as box:
        r = box["review"]
        label = next((e for e in r["edits"] if e["edit_id"] == edit_id), None)
        if label is None or label["kind"] != "label" or label["status"] != "accepted":
            raise ReviewError(f"no accepted label {edit_id!r} on this run")
        now = fmt.reading_in_effect(r, edit_id)
        open_ = [e for e in r["edits"] if e["kind"] == "reading" and e["status"] == "proposed"
                 and e["payload"]["label_id"] == edit_id]
        if (now or open_) and not replace:
            what = f"read as {now['train']!r} by {now['by']}" if now else f"has an open reading by {open_[0]['by']}"
            raise ReviewError(f"label {edit_id} was already {what}; pass replace to add another")
        e = _propose(box, kind="reading", payload={"label_id": edit_id, "train": train, "meaning": meaning.strip()},
                     source=source, by=by.strip(), reason=meaning.strip() if agent else None,
                     job_id=(job_id or _job_id()) if agent else None,
                     agent_version=(agent_version or _agent_version()) if agent else None)
    return e


def active_edits(review: dict) -> list[dict]:
    """Accepted edits that are in effect: all of the multi-valued kinds, the latest accepted per single slot."""
    return fmt.active_edits(review)


def check_review(manifest: dict, review: dict) -> None:
    """Refuse a review this logger would never have written, or one that does not fit the recorded run: fail closed
    before it shapes training."""
    with _errors():
        fmt.check_review(review)
        fmt.check_against_manifest(review, manifest)


def _who(e: dict) -> str:
    return e["by"] or ("person" if e["source"] == "human" else "agent")


def effective(manifest: dict, review: dict) -> dict:
    """The run as training should see it after the edits in effect. A format 1 review is read through the migration.
    For a review migrated from format 1 the result is the same as format 1 gave, so batch lock digests still match."""
    if fmt.is_format1(review):
        with _errors():
            review = fmt.migrate_format1(review, manifest)
    check_review(manifest, review)
    act = fmt.active_edits(review)
    single = {e["kind"]: e["payload"] for e in act if e["kind"] in ("outcome", "failure_start", "reject_run",
                                                                  "failure_type", "name")}
    out = {
        "termination_reason": manifest["termination_reason"], "failure_mode": manifest.get("failure_mode"),
        "failure_start": None, "excludes": [e["payload"] for e in act if e["kind"] == "exclude"],
        "rejected": "reject_run" in single, "notes": [e["payload"] for e in act if e["kind"] == "note"],
        "subgoals": [dict(s) for s in manifest.get("subgoals", [])], "reviewed": review["reviewed"],
        "labels": [{"id": e["edit_id"], **e["payload"], "interpretation": fmt.reading_in_effect(review, e["edit_id"])}
                   for e in act if e["kind"] == "label"],
        "drive_edits": [e["payload"] for e in act if e["kind"] == "drive_boundary"],
        "event_notes": {e["payload"]["key"]: {"text": e["payload"]["text"], "id": e["edit_id"], "by": _who(e)}
                        for e in act if e["kind"] == "event_note"},
    }
    # "segment_plan" is the key format 1 used for who set the segments; it is kept so digests stay the same
    who = {("segment_plan" if e["kind"] == "segments" else e["kind"]): _who(e) for e in act}
    out["set_by"] = {k: who[k] for k in ("outcome", "failure_start", "segment_plan", "drive_boundary") if k in who}
    if "outcome" in single:
        o = single["outcome"]
        out["termination_reason"] = o["termination_reason"] or o["outcome"]
        out["failure_mode"] = o["failure_mode"]
    if "failure_start" in single and out["termination_reason"] != "success":
        out["failure_start"] = single["failure_start"]["frame"]
    if "failure_type" in single:                # only when set, so reviews without one keep their digest
        out["failure_type"] = dict(single["failure_type"])
    if "name" in single:
        out["name"] = single["name"]["text"]
    steps = out["subgoals"]
    for e in act:
        if e["kind"] == "step_boundary":
            p = e["payload"]
            i = p["index"]
            if i + 1 < len(steps) and steps[i]["name"] == p["before"] and steps[i + 1]["name"] == p["after"]:
                f = min(max(p["frame"], steps[i]["start_frame"] + 1), steps[i + 1]["end_frame"] - 1)
                steps[i]["end_frame"] = steps[i + 1]["start_frame"] = f
        elif e["kind"] == "segment_edge":            # never past a neighbour, never below one frame
            p = e["payload"]
            i, st = p["index"], steps[p["index"]] if p["index"] < len(steps) else None
            if st is None:
                continue
            if p["edge"] == "start":
                lo = steps[i - 1]["end_frame"] if i > 0 else 0
                st["start_frame"] = min(max(p["frame"], lo), st["end_frame"] - 1)
            else:
                hi = steps[i + 1]["start_frame"] if i + 1 < len(steps) else manifest["length"]
                st["end_frame"] = max(min(p["frame"], hi), st["start_frame"] + 1)
    plan = next((e["payload"]["segments"] for e in reversed(act) if e["kind"] == "segments"), None)
    if plan is not None:                         # a person's segments replace the recorded ones
        recorded = out["subgoals"]
        named = [{"name": x["legacy_name"] if x["legacy_name"] is not None else x["subtask_id"],
                  "start": x["start"], "end": x["end"]} for x in plan]

        def match(x):
            best = max(recorded, key=lambda s: min(s["end_frame"], x["end"]) - max(s["start_frame"], x["start"])
                       if s["name"] == x["name"] else -1, default=None)
            return best if best and best["name"] == x["name"] and \
                min(best["end_frame"], x["end"]) > max(best["start_frame"], x["start"]) else None
        out["subgoals"] = [{"name": x["name"], "start_frame": x["start"], "end_frame": x["end"],
                            "status": (match(x) or {}).get("status", "completed"),
                            "reason": (match(x) or {}).get("reason")} for x in named]
    return out


def train_mask(manifest: dict, review: dict, control_modes: list[str]) -> np.ndarray:
    """Per-frame bool: False for a rejected run, excluded spans, labels whose reading in effect is "exclude", and, on
    a failed run, every frame from the accepted failure start up to the next takeover (the first expert or blend
    frame), or to the end if none follows."""
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


def migrate_file(run_path: Path, manifest: dict | None = None, task: dict | None = None) -> bool:
    """Rewrite one run's format 1 review.json as format 2 now, under the run's lock (normally this happens at the
    next change). Returns True if the file was rewritten. A run in a locked batch is refused like any change."""
    if file_format(run_path) != 1:
        return False
    with transaction(run_path, manifest) as box:
        if task is not None:
            raw = json.loads(_path(run_path).read_text())
            box["review"] = fmt.migrate_format1(raw, manifest or store.read_manifest(Path(run_path)), task)
        box["review"] = dict(box["review"])          # a new object: written even though nothing else changed
        box["force"] = True
    return True


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


def idle_spans(motion: np.ndarray, rate: float, min_s: float = 1.0, still: float = 0.02) -> list[tuple[int, int]]:
    """Stretches covered by consecutive one-second windows in which no dimension moves more than `still` of its own
    range in this run (robust range: 2nd to 98th percentile), so units and scales do not matter. Dimensions that
    never move are ignored. Returns [start, end) spans of at least min_s; spans on either side of a movement never
    merge."""
    n = len(motion)
    w = max(2, int(round(rate)))
    if n < w:
        return []
    motion = np.asarray(motion, dtype=np.float64).reshape(n, -1)
    scale = np.percentile(motion, 98, axis=0) - np.percentile(motion, 2, axis=0)
    live = scale > 1e-9
    if not live.any():
        return [(0, n)] if n >= max(1, int(round(min_s * rate))) else []
    rel = motion[:, live] / scale[live]
    quiet = [float((rel[i:i + w].max(axis=0) - rel[i:i + w].min(axis=0)).max()) < still
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
              min_s: float = 1.0, still: float = 0.02, expect_rev: int | None = None) -> list[dict]:
    """Suggest cutting idle stretches of one run (our built-in model for the "find idle frames" AI feature).
    Uses the measured state when the dataset has one, otherwise the sent commands, each scaled by its own range in
    the run, so it works for any units or action space. Suggestions are agent
    proposals of kind "exclude" with a short description; a person cuts or keeps each one."""
    n, rate = manifest["length"], manifest["action_schema"]["rate_hz"]
    motion = _episode_state(root, manifest["episode_index"], n)
    source = "measured state"
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
        who = {"base": "policy driving", "expert": "rescue driving", "blend": "shared control", "none": "nobody driving",
               "unknown": "driver not recorded"}.get(who, f"{who} driving")
        items.append({"start": a, "end": b, "reason": f"no movement for {(b - a) / rate:.1f} s {where}, {who}"})
    return add_suggestions(run_path, manifest, items, by, expect_rev=expect_rev, agent_version=_agent_version())


def suggest_cut(run_path: Path, manifest: dict, start: int, end: int, reason: str, by: str = "agent",
                job_id: str | None = None, agent_version: str | None = None) -> dict:
    """Any agent's suggestion to cut frames [start, end): shown in the viewer for a person to cut or keep.
    Returns {"status", "edit"} like add_suggestions."""
    return add_suggestions(run_path, manifest, [{"start": start, "end": end, "reason": reason}], by,
                           job_id=job_id, agent_version=agent_version)[0]


def list_labels(root: Path, only_uninterpreted: bool = False) -> list[dict]:
    """Every label in effect, with what was recorded in its span, for an agent to read. only_uninterpreted leaves out
    labels with a reading in effect or an open reading proposal."""
    out = []
    for run_path, m in store.iter_manifests(Path(root), "committed"):
        eff = effective(m, load(run_path, m))
        if not eff["labels"]:
            continue
        rate = m["action_schema"]["rate_hz"]
        modes = pq.read_table(run_path / "ticks.parquet", columns=["control_mode"]).column("control_mode").to_pylist()
        r = load(run_path, m)
        for lb in eff["labels"]:
            open_ = sum(e["kind"] == "reading" and e["status"] == "proposed" and e["payload"]["label_id"] == lb["id"]
                        for e in r["edits"])
            if only_uninterpreted and (lb["interpretation"] or open_):
                continue
            span = modes[lb["start"]:lb["end"]]
            out.append({
                "run_id": m["run_id"], "episode_index": m["episode_index"], "label_id": lb["id"],
                "start_frame": lb["start"], "end_frame": lb["end"],
                "start_s": round(lb["start"] / rate, 2), "end_s": round(lb["end"] / rate, 2),
                "text": lb["text"], "driver": lb["driver"], "segment": lb["segment"],
                "interpretation": lb["interpretation"], "open_readings": open_,
                "recorded": {
                    "control_modes": {k: span.count(k) for k in sorted(set(span))},
                    "steps": [s["name"] for s in eff["subgoals"]
                              if s["start_frame"] < lb["end"] and s["end_frame"] > lb["start"]],
                    "run_outcome": eff["termination_reason"], "failure_mode": eff["failure_mode"],
                    "task": m["task"],
                },
            })
    return out
