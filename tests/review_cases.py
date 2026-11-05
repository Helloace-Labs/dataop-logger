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

    def op(self, name: str):
        return getattr(self.fmt, name)

    def rid(self, out) -> str | None:
        return out if isinstance(out, str) else out["edit_id"] if isinstance(out, dict) and "edit_id" in out else None

    def rev(self, review) -> int:
        return review["rev"]

    def errors(self) -> tuple:
        return (self.fmt.ReviewError, self.fmt.FormatError)


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

    def op(self, name: str):
        return getattr(self.ops, name)

    def rid(self, out) -> str | None:
        return out if isinstance(out, str) else getattr(out, "edit_id", None)

    def rev(self, review) -> int:
        return review.rev

    def errors(self) -> tuple:
        return (ValueError,)


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
        "history_cap": {"start": "empty", "ops": cap},
        "large_steps": {"start": "empty", "ops": big},
    }


CASES = _cases()


def start(impl, spec: str):
    m = impl.manifest(read("manifest.json"))
    if spec == "empty":
        return impl.empty(RUN_ID), m
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
            steps.append({"op": op, "error": err, "id": impl.rid(got), "rev": impl.rev(review), "sha": sha(text),
                          "step": new_steps[-1] if new_steps else None, "fits_run": fits})
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
