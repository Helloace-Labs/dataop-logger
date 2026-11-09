"""Round trip cases for review format 2, run the same way against two implementations of the format.

`LoggerImpl` is this package (`aftershock_logger.review_format`). `ContractImpl` is the reference implementation the
cloud uses; it is loaded only when the environment variable AFTERSHOCK_CONTRACTS names a checkout that holds it (the
`contracts` package with `schema.review`, `schema.review_ops` and `schema.review_format1`). Without it those tests
skip, and the logger is checked against the expected results committed in tests/data/review_format2/expected.json,
which `make_fixtures.py` there writes from the reference implementation.

Each case starts from a review (empty, or a format 1 file migrated with or without a task) and applies operations.
After every operation the case records the error class (or None), the id the operation returned, and the sha256 of
the canonical file text, so two implementations agree byte for byte or the test names the first step that differs.
Ids and times are fixed per step, so the bytes are reproducible.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data" / "review_format2"
RUN_ID = json.loads((DATA / "manifest.json").read_text())["run_id"]
SESSION_ID = json.loads((DATA / "session.json").read_text())["session_id"]   # a Drive session's record (L9)


def read(name: str):
    return json.loads((DATA / name).read_text())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def when(i: int) -> str:
    return f"2026-11-04T10:{i // 60:02d}:{i % 60:02d}Z"


class Ids:
    """new_id() values in a fixed order."""

    def __init__(self, start: int = 0xabc000000000):
        self.n = start

    def __call__(self) -> str:
        self.n += 1
        return f"{self.n:012x}"


# ------------------------------------------------------------------ the two implementations


class LoggerImpl:
    name = "logger"

    def __init__(self):
        from aftershock_logger import review_format as fmt
        self.fmt = fmt

    @contextmanager
    def ids(self, ids: Ids):
        old = self.fmt.new_id
        self.fmt.new_id = ids
        try:
            yield
        finally:
            self.fmt.new_id = old

    def manifest(self, m: dict) -> dict:
        return m

    def task(self, t: dict | None):
        return t

    def empty(self, run_id: str):
        return self.fmt.empty_review(run_id)

    def migrate(self, raw: dict, manifest: dict, task):
        return self.fmt.migrate_format1(copy.deepcopy(raw), manifest, task)

    def to_format1(self, review) -> dict:
        return self.fmt.to_format1(review)

    def dump(self, review) -> str:
        return self.fmt.dump_review(review)

    def load(self, text: str):
        return self.fmt.load_review(text)

    def check_manifest(self, review, manifest) -> None:
        self.fmt.check_against_manifest(review, manifest)

    def session_record(self, rec: dict):
        return rec

    def episodes(self, review, record) -> list | None:
        """The session's effective timeline as [kind, id, start, end, from_runs, confirmed], or None when the review
        does not fit the session record."""
        from aftershock_logger import session
        try:
            session.check_against_session(review, record)
            return [[x["kind"], x["id"], x["start"], x["end"], list(x["from_runs"]), x["confirmed"]]
                    for x in session.effective_timeline(record, review)]
        except ValueError:
            return None

    def read_run(self, text: str, manifest) -> str | None:
        """The canonical text of a file as the logger reads it for a run, or None when it refuses it."""
        try:
            return self.fmt.dump_review(self.fmt.read(text, manifest))
        except ValueError:
            return None

    def op(self, name: str):
        return getattr(self.fmt, name)

    def rid(self, out) -> str | None:
        return out if isinstance(out, str) else out["edit_id"] if isinstance(out, dict) and "edit_id" in out else None

    def rev(self, review) -> int:
        return review["rev"]

    def errors(self) -> tuple:
        return (self.fmt.ReviewError, self.fmt.FormatError)

    def digest(self, review) -> str:
        return self.fmt.review_digest(review)


class ContractImpl(LoggerImpl):
    name = "contract"

    def __init__(self, root: str):
        if root not in sys.path:
            sys.path.insert(0, root)
        from contracts.schema import review as rv, review_format1 as f1, review_ops as ops
        from contracts import RunManifest, Task
        from contracts.tests import samples
        self.rv, self.f1, self.ops, self.samples = rv, f1, ops, samples
        self.RunManifest, self.Task = RunManifest, Task

    @contextmanager
    def ids(self, ids: Ids):
        old = self.ops.new_id
        self.ops.new_id = ids
        try:
            yield
        finally:
            self.ops.new_id = old

    def manifest(self, m: dict):
        return self.RunManifest.model_validate(self.samples.manifest(**{**m, "events": [], "takeovers": []}))

    def task(self, t: dict | None):
        if t is None:
            return None
        subs = [{"description": f"{s['name']} step", **s} for s in t["subtasks"]]
        return self.Task.model_validate(self.samples.task(task_id=t["task_id"], subtasks=subs))

    def empty(self, run_id: str):
        return self.rv.empty_review(run_id)

    def migrate(self, raw: dict, manifest, task):
        return self.f1.migrate_format1(copy.deepcopy(raw), manifest, task)

    def to_format1(self, review) -> dict:
        return self.f1.to_format1(review)

    def dump(self, review) -> str:
        return self.rv.dump_review(review)

    def load(self, text: str):
        return self.rv.load_review(text)

    def check_manifest(self, review, manifest) -> None:
        self.rv.check_against_manifest(review, manifest)

    def session_record(self, rec: dict):
        from contracts.schema.drive import SessionRecord
        return SessionRecord.model_validate(rec)

    def episodes(self, review, record) -> list | None:
        from contracts.schema import drive
        try:
            drive.check_against_session(review, record)
            return [[x.kind, x.seg_id, x.start, x.end, list(x.from_runs), x.confirmed]
                    for x in drive.effective_timeline(record, review)]
        except ValueError:
            return None

    def read_run(self, text: str, manifest) -> str | None:
        try:
            r = self.rv.load_review(text)
            self.rv.check_against_manifest(r, manifest)
            return self.rv.dump_review(r)
        except (ValueError, TypeError):     # the reference raises TypeError for some malformed values
            return None

    def op(self, name: str):
        return getattr(self.ops, name)

    def rid(self, out) -> str | None:
        return out if isinstance(out, str) else getattr(out, "edit_id", None)

    def rev(self, review) -> int:
        return review.rev

    def errors(self) -> tuple:
        return (ValueError,)

    def digest(self, review) -> str:
        return self.rv.review_digest(review)


def contract_impl() -> ContractImpl:
    root = os.environ.get("AFTERSHOCK_CONTRACTS")
    if not root:
        pytest.skip("set AFTERSHOCK_CONTRACTS to a checkout with the reference contracts to compare against them")
    return ContractImpl(root)


# ------------------------------------------------------------------ cases


def _label(start=10, end=20, text="arm pauses", driver=None, segment=None):
    return {"start": start, "end": end, "text": text, "driver": driver, "segment": segment}


def _agent(**kw):
    return {"source": "agent", "by": "agent-a", "job_id": "job-1", "agent_version": "v1", "reason": "seen in video",
            **kw}


def _cases() -> dict[str, dict]:
    """name -> {start, ops}. An op is (name, kwargs); "$k" in a value is the id the k-th op returned."""
    seg = {"segments": [{"subtask_id": "s-reach", "start": 0, "end": 30, "legacy_name": None},
                        {"subtask_id": "s-grasp", "start": 30, "end": 75, "legacy_name": None}]}
    every_kind = [
        ("propose", dict(kind="outcome", payload={"outcome": "failure", "termination_reason": None,
                                                  "failure_mode": "slip"}, source="human", by="p1")),
        ("propose", dict(kind="outcome", payload={"outcome": "failure", "termination_reason": "timeout",
                                                  "failure_mode": None}, source="human", by="p1")),
        ("propose", dict(kind="failure_start", payload={"frame": 60}, source="human", by="p1")),
        ("propose", dict(kind="failure_type", payload={"failure_type_id": "ft-slip", "vocab_rev": 2}, **_agent())),
        ("propose", dict(kind="exclude", payload={"start": 0, "end": 5}, source="human", by="p1")),
        ("propose", dict(kind="segments", payload=seg, source="splitter", by="splitter", job_id="split-1",
                         agent_version="s1", reason="episode split")),
        ("propose", dict(kind="step_boundary", payload={"index": 1, "before": " Grasp ", "after": "lift",
                                                        "frame": 72}, source="human", by="p1")),
        ("propose", dict(kind="segment_edge", payload={"index": 0, "edge": "start", "frame": 2}, source="human",
                         by="p1")),
        ("propose", dict(kind="drive_boundary", payload={"index": 0, "frame": 64}, source="check", by="drive-check",
                         job_id="chk-1", agent_version="c1", reason="switch moved")),
        ("propose", dict(kind="reject_run", payload={}, source="human", by="p1")),
        ("propose", dict(kind="note", payload={"frame": 119, "text": "end"}, source="human", by="p1")),
        ("propose", dict(kind="event_note", payload={"key": "failure", "text": "slips"}, source="human", by="p1")),
        ("propose", dict(kind="label", payload=_label(), source="human", by="p1")),
        ("propose", dict(kind="reading", payload={"label_id": "$12", "train": "exclude", "meaning": "pause"},
                         **_agent())),
        ("propose", dict(kind="name", payload={"text": "run with a slip"}, source="human", by="p1")),
        ("accept", dict(edit_id="$3", by="p1")),
        ("accept", dict(edit_id="$13", by="p1")),
        ("reject", dict(edit_id="$5", by="p1")),
        ("accept", dict(edit_id="$8", by="p1")),
        ("set_reviewed", dict(reviewed=True, by="p1")),
    ]
    proposals = [
        ("propose", dict(kind="failure_start", payload={"frame": 50}, **_agent())),
        ("propose", dict(kind="failure_start", payload={"frame": 50}, **_agent())),           # same open: no change
        ("propose", dict(kind="failure_start", payload={"frame": 50}, **_agent(job_id="job-2"))),
        ("propose", dict(kind="failure_start", payload={"frame": 52}, **_agent(by="agent-b"))),
        ("propose", dict(kind="failure_start", payload={"frame": 51}, source="human", by="p1", corrects="$0")),
        ("accept", dict(edit_id="$0", by="p1")),                                              # rejected already
        ("accept", dict(edit_id="$3", by="p1")),
        ("propose", dict(kind="failure_start", payload={"frame": 9}, source="human", by="p1", corrects="$3")),
        ("replace", dict(edit_id="$3", payload={"frame": 53}, by="p1")),                     # agent edit replaced
        ("replace", dict(edit_id="$8", payload={"frame": 54}, by="p2")),                     # person's edit replaced
        ("reject", dict(edit_id="$2", by="p1")),
        ("reject", dict(edit_id="$2", by="p1")),
        ("propose", dict(kind="exclude", payload={"start": 3, "end": 3}, source="human", by="p1")),
        ("propose", dict(kind="exclude", payload={"start": 3, "end": 9}, source="agent", by="a", job_id="j",
                         agent_version="v", reason=" ")),
        ("propose", dict(kind="exclude", payload={"start": 3, "end": 9}, source="human", by="p1", job_id="j",
                         agent_version="v")),
        ("propose", dict(kind="exclude", payload={"start": 3, "end": 9}, source="robot", by="p1")),
        ("propose", dict(kind="note", payload={"frame": 1, "text": "x"}, source="human", by="")),
        ("propose", dict(kind="note", payload={"frame": 1, "text": "x"}, source="human", by="p1", expect_rev=0)),
        ("set_reviewed", dict(reviewed=True, by="p1")),
        ("set_reviewed", dict(reviewed=True, by="p1")),
        ("propose", dict(kind="note", payload={"frame": 2, "text": "after sign off"}, source="human", by="p1")),
    ]
    undo_and_readings = [
        ("propose", dict(kind="label", payload=_label(), source="human", by="p1")),
        ("propose", dict(kind="reading", payload={"label_id": "$0", "train": "exclude", "meaning": "pause"},
                         **_agent())),
        ("propose", dict(kind="reading", payload={"label_id": "$0", "train": "keep", "meaning": "fine"},
                         source="human", by="p1")),
        ("accept", dict(edit_id="$1", by="p1")),
        ("undo", dict(step_id="@0")),                                     # the reading still refers to the label
        ("replace", dict(edit_id="$0", payload=_label(10, 25, "arm pauses long"), by="p1")),
        ("undo", dict(step_id="@5")),
        ("undo", dict(step_id="@5")),                                     # once only
        ("undo", dict(step_id="@3")),
        ("undo", dict(step_id="@2")),
        ("undo", dict(step_id="@1")),
        ("undo", dict(step_id="@0")),
        ("undo", dict(step_id="ffffffffffff")),
        ("propose", dict(kind="reading", payload={"label_id": "eeeeeeeeeeee", "train": "keep", "meaning": "m"},
                         source="human", by="p1")),
    ]
    locks = [
        ("propose", dict(kind="note", payload={"frame": 3, "text": "before lock"}, source="human", by="p1")),
        ("lock", dict(by="owner", reason="pack locked")),
        ("lock", dict(by="owner")),
        ("propose", dict(kind="note", payload={"frame": 4, "text": "x"}, source="human", by="p1")),
        ("set_reviewed", dict(reviewed=True, by="p1")),
        ("undo", dict(step_id="@0")),
        ("unlock", dict(by="owner", reason=" ")),
        ("unlock", dict(by="owner", reason="fix a label")),
        ("unlock", dict(by="owner", reason="again")),
        ("propose", dict(kind="note", payload={"frame": 4, "text": "after unlock"}, source="human", by="p1")),
    ]
    migrated = [
        ("propose", dict(kind="reading", payload={"label_id": "00000000000c", "train": "keep", "meaning": "ok"},
                         **_agent())),
        ("accept", dict(edit_id="$0", by="p1")),
        ("accept", dict(edit_id="000000000004", by="p1")),
        ("replace", dict(edit_id="000000000008", payload={"segments": [
            {"subtask_id": "unresolved", "start": 0, "end": 40, "legacy_name": "wiggle"}]}, by="p1")),
        ("replace", dict(edit_id="$3", payload={"segments": [
            {"subtask_id": "s-reach", "start": 0, "end": 40, "legacy_name": None}]}, by="p1")),
        ("propose", dict(kind="step_boundary", payload={"index": 1, "before": " Grasp ", "after": "lift",
                                                        "frame": 75}, source="human", by="p1")),
        ("undo", dict(step_id="aaaaaaaaaaaa")),
        ("undo", dict(step_id="dddddddddddd")),                            # its segments edit changed since
        ("undo", dict(step_id="@4")),
        ("undo", dict(step_id="dddddddddddd")),
    ]
    take_back = [
        ("propose", dict(kind="label", payload=_label(), source="human", by="p1")),
        ("propose", dict(kind="exclude", payload={"start": 0, "end": 5}, **_agent())),
        ("accept", dict(edit_id="$1", by="p1")),
        ("propose", dict(kind="reading", payload={"label_id": "$0", "train": "exclude", "meaning": "pause"},
                         source="human", by="p1")),
        ("set_reviewed", dict(reviewed=True, by="p1")),
        ("retract", dict(edit_id="$0", by="q")),                          # the label; its reading stays with it
        ("retract", dict(edit_id="$0", by="q")),                          # a taken back edit is final
        ("accept", dict(edit_id="$0", by="p1")),
        ("reject", dict(edit_id="$0", by="p1")),
        ("replace", dict(edit_id="$0", payload=_label(10, 25), by="p1")),
        ("retract", dict(edit_id="$1", by="unknown")),                    # only the L8 upgrade writes unknown
        ("propose", dict(kind="note", payload={"frame": 5, "text": "x"}, **_agent())),
        ("retract", dict(edit_id="$11", by="p1")),                        # a proposal is rejected, not taken back
        ("propose", dict(kind="failure_start", payload={"frame": 50}, source="human", by="p1")),
        ("propose", dict(kind="failure_start", payload={"frame": 60}, source="human", by="p1")),
        ("retract", dict(edit_id="$14", by="p2")),                        # the earlier value is in effect again
        ("undo", dict(step_id="@15")),
        ("retract", dict(edit_id="$14", by="p2")),
        ("replace", dict(edit_id="$13", payload={"frame": 55}, by="p1")),
        ("retract", dict(edit_id="$18", by="p1")),                        # the replacement; $13 stays replaced
        ("retract", dict(edit_id="$13", by="p1")),                        # a replaced edit is not taken back
        ("propose", dict(kind="note", payload={"frame": 5, "text": "y"}, **_agent(job_id="unknown"))),
        ("propose", dict(kind="note", payload={"frame": 5, "text": "y"}, **_agent(agent_version="unknown"))),
    ]
    replace_refused = [
        ("propose", dict(kind="failure_start", payload={"frame": 50}, source="human", by="p1")),
        ("replace", dict(edit_id="$0", payload=_label(), by="p1")),          # refused for its payload, draws no id
        ("propose", dict(kind="note", payload={"frame": 5, "text": "x"}, source="human", by="p1")),
        ("replace", dict(edit_id="$0", payload={"frame": 55}, by="p1")),
        ("replace", dict(edit_id="$3", payload={"frame": -1}, by="p1")),      # refused again
        ("retract", dict(edit_id="$3", by="q")),                          # take back fields on the replacement
        ("propose", dict(kind="note", payload={"frame": 6, "text": "y"}, source="human", by="q")),
    ]
    migrated_take_back = [
        ("retract", dict(edit_id="00000000000e", by="q")),
        ("retract", dict(edit_id="00000000000f", by="q")),                # a format 1 take back is final too
        ("accept", dict(edit_id="000000000009", by="p1")),
        ("undo", dict(step_id="@0")),
        ("retract", dict(edit_id="00000000000c", by="q")),                # a label with a format 1 reading
    ]
    obsolete_reading = [
        ("propose", dict(kind="label", payload=_label(), source="human", by="p1")),
        ("propose", dict(kind="reading", payload={"label_id": "$0", "train": "exclude", "meaning": "pause"},
                         **_agent())),
        ("replace", dict(edit_id="$0", payload=_label(10, 25, "arm pauses long"), by="p1")),
        ("accept", dict(edit_id="$1", by="p1")),                          # its label was replaced: refused
        ("propose", dict(kind="reading", payload={"label_id": "$2", "train": "exclude", "meaning": "long pause"},
                         **_agent())),
        ("retract", dict(edit_id="$2", by="p1")),
        ("accept", dict(edit_id="$4", by="p1")),                          # its label was taken back: refused
        ("reject", dict(edit_id="$1", by="p1")),                          # rejecting it still works
        ("undo", dict(step_id="@5")),                                     # the label is back
        ("accept", dict(edit_id="$4", by="p1")),                          # so its reading can be accepted
    ]
    obsolete_reading_by_replace_or_undo = [
        ("propose", dict(kind="label", payload=_label(), source="human", by="p1")),
        ("propose", dict(kind="reading", payload={"label_id": "$0", "train": "keep", "meaning": "fine"},
                         source="human", by="p1")),
        ("retract", dict(edit_id="$1", by="p1")),                         # the reading is taken back
        ("replace", dict(edit_id="$0", payload=_label(10, 25, "arm pauses long"), by="p1")),
        ("undo", dict(step_id="@2")),                                     # would accept it again: refused
        ("propose", dict(kind="label", payload=_label(30, 40, "arm stops"), source="human", by="p1")),
        ("propose", dict(kind="reading", payload={"label_id": "$5", "train": "keep", "meaning": "fine"},
                         source="human", by="p1")),
        ("replace", dict(edit_id="$6", payload={"label_id": "$5", "train": "exclude", "meaning": "stop"}, by="p1")),
        ("retract", dict(edit_id="$5", by="p1")),
        ("replace", dict(edit_id="$7", payload={"label_id": "$5", "train": "keep", "meaning": "ok"}, by="p1")),
        ("undo", dict(step_id="@7")),                                     # would bring back $6: refused
        ("undo", dict(step_id="@8")),                                     # the label is back
        ("undo", dict(step_id="@7")),                                     # now allowed
    ]
    # A Drive session's split review (session.json): gap-0 [0,10), run A [10,40), gap-1 [40,44), gap-2 [44,55) with
    # a short stop [44,50), run B [55,80), gap-3 [80,85), run C [85,120) whose write failed, gap-4 [120,130). A
    # person's edit is in effect at once; a splitter or agent edit waits for a person.
    splitter = dict(source="splitter", by="splitter", job_id="split-1", agent_version="s1", reason="evidence")
    person = dict(source="human", by="p1")
    session_split = [
        ("propose", dict(kind="episode_boundary", payload={"at": 40, "to": 38}, **splitter)),
        ("accept", dict(edit_id="$0", by="p1")),                          # run A ends at 38: a new id, needs review
        ("propose", dict(kind="episode_boundary", payload={"at": 85, "to": 85}, **splitter)),
        ("accept", dict(edit_id="$2", by="p1")),                          # a confirmation changes nothing
        ("propose", dict(kind="episode_merge", payload={"start": 10, "end": 120}, **person)),   # swallows run C
        ("retract", dict(edit_id="$4", by="p1")),
        ("propose", dict(kind="episode_split", payload={"at": 20}, **person)),
        ("propose", dict(kind="episode_boundary", payload={"at": 44, "to": 46}, **person)),     # between two gaps
        ("undo", dict(step_id="@7")),
        ("propose", dict(kind="episode_boundary", payload={"at": 80, "to": 82}, **person)),     # B grows into gap-3
        ("replace", dict(edit_id="$9", payload={"at": 80, "to": 80}, by="p1")),               # and back: B's id
        ("propose", dict(kind="episode_merge", payload={"start": 40, "end": 40}, **person)),    # refused: start < end
        ("propose", dict(kind="episode_boundary", payload={"at": 0, "to": 4}, **person)),       # refused: at >= 1
        ("propose", dict(kind="episode_split", payload={"at": "20"}, **person)),
        ("propose", dict(kind="episode_split", payload={"at": 20, "x": 1}, **person)),
        ("propose", dict(kind="episode_split", payload={"at": True}, **person)),
        ("propose", dict(kind="episode_boundary", payload={"at": 2.0, "to": 4}, **person)),
        ("propose", dict(kind="episode_split", payload={"at": 100}, **person)),                 # splits failed run C
        ("retract", dict(edit_id="$17", by="p1")),
        ("propose", dict(kind="episode_merge", payload={"start": 20, "end": 120}, **person)),   # swallows run C
        ("undo", dict(step_id="@19")),
        ("propose", dict(kind="episode_split", payload={"at": 100}, **_agent())),            # a proposal only
        ("reject", dict(edit_id="$21", by="p1")),
        ("propose", dict(kind="episode_split", payload={"at": 131}, **splitter)),            # past the session
        ("reject", dict(edit_id="$23", by="p1")),                         # still in the review: does not fit
        ("propose", dict(kind="note", payload={"frame": 3, "text": "a run's kind"}, **person)),
    ]
    session_merge = [
        ("propose", dict(kind="episode_split", payload={"at": 60}, **person)),
        ("propose", dict(kind="episode_merge", payload={"start": 55, "end": 80}, **person)),    # run B's id again
        ("propose", dict(kind="episode_boundary", payload={"at": 10, "to": 5}, **person)),      # A takes gap frames
        ("propose", dict(kind="episode_boundary", payload={"at": 85, "to": 83}, **person)),     # into failed run C
        ("undo", dict(step_id="@3")),
        ("propose", dict(kind="episode_boundary", payload={"at": 5, "to": 0}, **person)),       # gap-0 keeps none
        ("retract", dict(edit_id="$5", by="p1")),
        ("propose", dict(kind="episode_merge", payload={"start": 5, "end": 40}, **person)),     # one episode
        ("retract", dict(edit_id="$7", by="p1")),
        ("propose", dict(kind="episode_merge", payload={"start": 85, "end": 40}, **person)),    # refused
        ("propose", dict(kind="episode_merge", payload={"start": 0, "end": 80}, **person)),     # gap-0 is no episode
        ("undo", dict(step_id="@10")),
        ("propose", dict(kind="episode_merge", payload={"start": 5, "end": 80}, **person)),     # A, the short stop, B
        ("propose", dict(kind="episode_split", payload={"at": 50}, **person)),
        ("propose", dict(kind="episode_boundary", payload={"at": 80, "to": 82}, **person)),
        ("lock", dict(by="owner", reason="pack locked")),
        ("propose", dict(kind="episode_split", payload={"at": 30}, **person)),                  # locked
    ]
    cap = [("propose", dict(kind="note", payload={"frame": i % 120, "text": f"note {i}"}, source="human", by="p1"))
           for i in range(205)] + [("undo", dict(step_id="@0")), ("undo", dict(step_id="@204"))]
    big = [("propose", dict(kind="segments", payload={"segments": [
        {"subtask_id": "s-reach", "start": k, "end": k + 1, "legacy_name": None} for k in range(j, j + 119, 2)]},
        source="human", by="p1")) for j in range(2)]
    return {
        "every_kind": {"start": "empty", "ops": every_kind},
        "proposals_and_corrections": {"start": "empty", "ops": proposals},
        "undo_and_readings": {"start": "empty", "ops": undo_and_readings},
        "lock_and_unlock": {"start": "empty", "ops": locks},
        "migrated_with_task": {"start": "format1_all_kinds.json+task", "ops": migrated},
        "migrated_without_task": {"start": "format1_all_kinds.json", "ops": migrated[:5]},
        "migrated_minimal": {"start": "format1_minimal.json", "ops": [
            ("accept", dict(edit_id="000000000015", by="p1")), ("undo", dict(step_id="@0"))]},
        "take_back": {"start": "empty", "ops": take_back},
        "replace_refused_for_its_payload": {"start": "empty", "ops": replace_refused},
        "migrated_take_back": {"start": "format1_all_kinds.json+task", "ops": migrated_take_back},
        "obsolete_reading": {"start": "empty", "ops": obsolete_reading},
        "obsolete_reading_by_replace_or_undo": {"start": "empty", "ops": obsolete_reading_by_replace_or_undo},
        "session_split_edits": {"start": "session", "ops": session_split},
        "session_merge_and_back": {"start": "session", "ops": session_merge},
        "history_cap": {"start": "empty", "ops": cap},
        "large_steps": {"start": "empty", "ops": big},
    }


CASES = _cases()


def start(impl, spec: str):
    m = impl.manifest(read("manifest.json"))
    if spec == "empty":
        return impl.empty(RUN_ID), m
    if spec == "session":
        return impl.empty(SESSION_ID), m
    name, _, task = spec.partition("+")
    return impl.migrate(read(name), m, impl.task(read("task.json")) if task else None), m


def run_case(impl, name: str) -> dict:
    """{"steps": [{"op", "error", "id", "rev", "sha"}], "final": canonical text}."""
    case = CASES[name]
    ids = Ids()
    with impl.ids(ids):
        review, m = start(impl, case["start"])
        results, steps = [], []
        for i, (op, kw) in enumerate(case["ops"]):
            def sub(v):
                if isinstance(v, dict):
                    return {k: sub(x) for k, x in v.items()}
                if isinstance(v, list):
                    return [sub(x) for x in v]
                if isinstance(v, str) and v[:1] == "$":
                    j = int(v[1:])
                    return (results[j] if j < len(results) else None) or "ffffffffffff"
                if isinstance(v, str) and v[:1] == "@":
                    j = int(v[1:])
                    return (steps[j]["step"] if j < len(steps) else None) or "ffffffffffff"
                return v
            kw = sub(dict(kw))
            kw.setdefault("expect_rev", impl.rev(review))
            if op not in ("lock", "unlock", "set_reviewed", "undo"):
                kw.setdefault("t", 1000 + i)
            kw["now"] = when(i)
            before = impl.dump(review)
            hist_before = {h["step_id"] for h in json.loads(before)["history"]}
            try:
                out = impl.op(op)(review, **kw)
                review, got = (out if isinstance(out, tuple) else (out, None))
                err = None
            except impl.errors() as e:
                got, err = None, type(e).__name__
            text = impl.dump(review)
            new_steps = [h["step_id"] for h in json.loads(text)["history"] if h["step_id"] not in hist_before]
            results.append(impl.rid(got))
            try:                       # the operations do not see the run; the caller checks before it saves
                impl.check_manifest(review, m)
                fits = True
            except impl.errors():
                fits = False
            step = {"op": op, "error": err, "id": impl.rid(got), "rev": impl.rev(review), "sha": sha(text),
                    "step": new_steps[-1] if new_steps else None, "fits_run": fits}
            if case["start"] == "session":         # and the session's effective timeline, or None when it does not fit
                step["episodes"] = impl.episodes(review, impl.session_record(read("session.json")))
            steps.append(step)
            assert impl.dump(impl.load(text)) == text                  # every result reads back byte for byte
    return {"steps": steps, "final": impl.dump(review)}


def migration_results(impl) -> dict:
    """Format 1 fixtures through the migration and back: the canonical texts."""
    out = {}
    m = impl.manifest(read("manifest.json"))
    task = read("task.json")
    reordered = dict(task, subtasks=list(reversed(task["subtasks"])))
    for f1 in ("format1_all_kinds.json", "format1_minimal.json"):
        raw = read(f1)
        for label, t in (("no task", None), ("task", task), ("task reordered", reordered)):
            r = impl.migrate(raw, m, impl.task(t))
            out[f"{f1} {label}"] = {"format2": impl.dump(r), "format1": json.dumps(impl.to_format1(r), indent=1,
                                                                                   sort_keys=True) + "\n"}
    return out


def l8_results(impl) -> dict:
    """Files the logger wrote at 4f4395e (l8_files.json, before the take back fields), as each implementation reads
    them: the upgraded text, its digest, and the text after the next change (a person's note, or the unlock of a
    locked review)."""
    out = {}
    for name, f in read("l8_files.json").items():
        review = impl.load(f["text"])
        with impl.ids(Ids(0xf9000000000)):
            if json.loads(f["text"])["locked"] is not None:
                nxt = impl.op("unlock")(review, by="owner", reason="next change", expect_rev=impl.rev(review),
                                        now=when(0))
            else:
                nxt, _ = impl.op("propose")(review, kind="note", payload={"frame": 1, "text": "next change"},
                                            source="human", by="p1", expect_rev=impl.rev(review), now=when(0))
        out[name] = {"upgraded": impl.dump(review), "digest": impl.digest(review), "next": impl.dump(nxt)}
    return out


# ------------------------------------------------------------------ random changes to valid files


MUTATION_VALUES = [None, 0, -1, 1, 2, True, "", " ", "x", "unknown", "unresolved", "aaaaaaaaaaaa", [], {}, 1.5, "a\n",
                   "human", "agent", "accepted", "proposed", "rejected", 10 ** 6, "s-reach", "Grasp", "\ud800",
                   "a\udc00b"]


TAKE_BACK_KEYS = ("status", "accepted_seq", "replaced_by", "decided_at", "decided_by", "retracted_at",
                  "retracted_by")


def mutations(rng, d, take_backs: bool = False):
    """Every path in d, each set to a random value once and dropped once: (path, "set" or "drop", value). With
    take_backs, the fields of every taken back edit (in the edits and in history snapshots) that the take back rules
    look at are also set to every value in turn."""
    paths, exhaustive = [], []

    def walk(x, path):
        paths.append(path)
        if isinstance(x, dict):
            if take_backs and x.get("retracted_at") is not None:
                exhaustive.extend(path + (k,) for k in TAKE_BACK_KEYS)
            for k in x:
                walk(x[k], path + (k,))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, path + (i,))
    walk(d, ())
    for path in paths[1:]:
        yield path, "set", rng.choice(MUTATION_VALUES)
        yield path, "drop", None
    for path in exhaustive:
        for value in MUTATION_VALUES:
            yield path, "set", value


def apply_change(d, path, how, value):
    d = copy.deepcopy(d)
    x = d
    for k in path[:-1]:
        x = x[k]
    if how == "drop":
        if isinstance(x, dict):
            del x[path[-1]]
        else:
            x.pop(path[-1])
    else:
        x[path[-1]] = value
    return d


MUTATED_CASES = {"take_back": 13, "migrated_take_back": 17}     # case -> seed; both end with edits taken back


def mutation_results(impl) -> dict:
    """Random changes to the final text of each case in MUTATED_CASES, read as a run's file: the indices of the
    changes accepted, and a digest of every result (the canonical text, or refused), so the logger is checked against
    the reference on these without it. Most changes are refused, so the shorter list of those accepted is kept."""
    import random
    out = {}
    m = impl.manifest(read("manifest.json"))
    for name, seed in MUTATED_CASES.items():
        good = json.loads(run_case(LoggerImpl(), name)["final"])
        accepted, h = [], hashlib.sha256()
        for i, (path, how, value) in enumerate(mutations(random.Random(seed), good, take_backs=True)):
            text = impl.read_run(json.dumps(apply_change(good, path, how, value)), m)
            if text is not None:
                accepted.append(i)
            h.update((sha(text) if text is not None else "refused").encode() + b"\n")
        out[name] = {"count": i + 1, "accepted": accepted, "digest": h.hexdigest()}
    return out
