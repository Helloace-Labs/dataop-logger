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

Single-valued kinds (outcome, failure_start, reject_run, and step_boundary per index): the most recently accepted
edit wins, and undoing it brings back the one accepted before it. Every change to review.json happens under a
cross-process lock and bumps `rev`.
"""
from __future__ import annotations

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

KINDS = ("outcome", "failure_start", "exclude", "step_boundary", "reject_run", "note")
SOURCES = ("agent", "human")
STATUSES = ("proposed", "accepted", "rejected")
MULTI = ("exclude", "note")                 # kinds where every accepted edit applies
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


@contextmanager
def transaction(run_path: Path, expect_rev: int | None = None):
    """Load, change and save one run's review under an exclusive cross-process lock."""
    with open(Path(run_path) / "review.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            review = load(run_path)
            if expect_rev is not None and review["rev"] != expect_rev:
                raise ReviewConflict(f"the review changed (now at rev {review['rev']}, you had {expect_rev})")
            yield review
            review["rev"] += 1
            store.atomic_write_json(_path(run_path), review)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


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
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ReviewError("note text must be 1 to 2000 characters")
    return {"frame": frame(payload.get("frame"), "frame", 0, length - 1), "text": text}


def _slot(e: dict):
    """Edits that compete: same kind, and for step boundaries the same index."""
    return (e["kind"], e["payload"].get("index")) if e["kind"] == "step_boundary" else e["kind"]


def _accept(review: dict, edit: dict) -> None:
    review["seq"] += 1
    edit["status"], edit["decided_at"], edit["accepted_seq"] = "accepted", _now(), review["seq"]


def _clear_reviewed(review: dict) -> None:
    review["reviewed"], review["reviewed_at"] = False, None


def add_edit(run_path: Path, manifest: dict, kind: str, payload: dict, source: str = "human",
             reason: str | None = None, expect_rev: int | None = None) -> dict:
    """Add an edit. Human edits are accepted at once; agent edits wait as proposals and reopen the run's review."""
    if source not in SOURCES:
        raise ReviewError(f"source must be one of {SOURCES}")
    payload = check_payload(kind, payload, manifest)
    with transaction(run_path, expect_rev) as review:
        if source == "agent":
            for e in review["edits"]:               # proposing the same thing twice is a no-op
                if e["source"] == "agent" and e["kind"] == kind and e["payload"] == payload:
                    return e
        edit = {"id": uuid.uuid4().hex[:12], "kind": kind, "payload": payload, "source": source, "reason": reason,
                "status": "proposed", "created_at": _now(), "decided_at": None}
        review["edits"].append(edit)
        if source == "human":
            _accept(review, edit)
        else:
            _clear_reviewed(review)
    return edit


def decide(run_path: Path, manifest: dict, edit_id: str, accept: bool, payload: dict | None = None,
           expect_rev: int | None = None) -> dict:
    """Accept (optionally adjusted) or reject an edit. Rejecting an accepted edit undoes it."""
    if not isinstance(accept, bool):
        raise ReviewError("accept must be true or false")
    with transaction(run_path, expect_rev) as review:
        edit = next((e for e in review["edits"] if e["id"] == edit_id), None)
        if edit is None:
            raise ReviewError(f"no edit {edit_id!r} on this run")
        if accept:
            if payload is not None:
                edit["payload"] = check_payload(edit["kind"], payload, manifest)
                edit["adjusted"] = True
            _accept(review, edit)
        else:
            edit["status"], edit["decided_at"] = "rejected", _now()
    return edit


def set_reviewed(run_path: Path, reviewed: bool, expect_rev: int | None = None) -> dict:
    if not isinstance(reviewed, bool):
        raise ReviewError("reviewed must be true or false")
    with transaction(run_path, expect_rev) as review:
        if reviewed and any(e["status"] == "proposed" for e in review["edits"]):
            raise ReviewError("accept or dismiss every proposal before marking the run reviewed")
        review["reviewed"], review["reviewed_at"] = reviewed, _now() if reviewed else None
    return review


def active_edits(review: dict) -> list[dict]:
    """Accepted edits that are in effect: all of the multi-valued kinds, the latest accepted per single slot."""
    acc = sorted((e for e in review["edits"] if e["status"] == "accepted"), key=lambda e: e.get("accepted_seq", 0))
    latest = {}
    for e in acc:
        if e["kind"] not in MULTI:
            latest[_slot(e)] = e
    return [e for e in acc if e["kind"] in MULTI or latest.get(_slot(e)) is e]


def needs_review(manifest: dict, review: dict) -> bool:
    if any(e["status"] == "proposed" for e in review["edits"]):
        return True
    return not review["reviewed"] and effective(manifest, review)["termination_reason"] != "success"


def effective(manifest: dict, review: dict) -> dict:
    """The run as training should see it after the edits in effect."""
    act = active_edits(review)
    single = {e["kind"]: e["payload"] for e in act if e["kind"] not in MULTI and e["kind"] != "step_boundary"}
    out = {
        "termination_reason": manifest["termination_reason"], "failure_mode": manifest.get("failure_mode"),
        "failure_start": None, "excludes": [e["payload"] for e in act if e["kind"] == "exclude"],
        "rejected": "reject_run" in single, "notes": [e["payload"] for e in act if e["kind"] == "note"],
        "subgoals": [dict(s) for s in manifest.get("subgoals", [])], "reviewed": review["reviewed"],
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
    """Per-frame bool: False for a rejected run, excluded spans, and, on a failed run, every frame from the accepted
    failure start up to the next takeover (the first expert or blend frame), or to the end if none follows."""
    n = manifest["length"]
    eff = effective(manifest, review)
    mask = np.ones(n, dtype=bool)
    if eff["rejected"]:
        mask[:] = False
        return mask
    for x in eff["excludes"]:
        mask[x["start"]:x["end"]] = False
    fs = eff["failure_start"]
    if fs is not None and control_modes[fs] not in TAKEOVER_MODES:
        end = next((i for i in range(fs, n) if control_modes[i] in TAKEOVER_MODES), n)
        mask[fs:end] = False
    return mask


# ── agent proposals ────────────────────────────────────────────────────────────

def _step_budgets(runs: list[tuple[dict, dict]], q: float = 90) -> dict[str, int]:
    """Per step, the q-th percentile of how many frames it took in effectively successful runs."""
    durs: dict[str, list[int]] = {}
    for m, eff in runs:
        if eff["termination_reason"] != "success" or eff["rejected"]:
            continue
        for sg in eff["subgoals"][:-1]:             # the last step ends with the run, so its length is no budget
            if sg["status"] == "completed":
                durs.setdefault(sg["name"], []).append(sg["end_frame"] - sg["start_frame"])
    return {k: int(np.ceil(np.percentile(v, q))) for k, v in durs.items() if len(v) >= 3}


def propose(root: Path, idle_s: float = 1.0, idle_eps: float = 0.05) -> list[str]:
    """Add agent proposals to every committed run that has none yet, using the run as currently corrected.

    failure_start, on failed runs: the takeover is the first one at or after the failed step's start. When that
    step had already run longer than it took in 90% of successful runs, propose the frame where it went over that
    budget; otherwise the step's start; without step information, the frame before the first takeover.
    exclude: idle stretches of at least idle_s (no joint command moving more than idle_eps per frame) at the very
    start or end of a run. reject_run: a run in which nothing moves at all.
    """
    report = []
    loaded = []
    for run_path, m in store.iter_manifests(Path(root), "committed"):
        r = load(run_path)
        loaded.append((run_path, m, r, effective(m, r)))
    budgets = _step_budgets([(m, eff) for _, m, _, eff in loaded])
    for run_path, m, review, eff in loaded:
        if any(e["source"] == "agent" for e in review["edits"]) or eff["rejected"]:
            continue
        n = m["length"]
        rate = m["action_schema"]["rate_hz"]
        t = pq.read_table(run_path / "ticks.parquet", columns=["sent_action"])
        sent = np.asarray(t.column("sent_action").to_pylist(), dtype=np.float32).reshape(n, -1)
        moving = np.abs(np.diff(sent, axis=0)).max(axis=1) >= idle_eps if n > 1 else np.zeros(0, bool)
        if not moving.any():
            add_edit(run_path, m, "reject_run", {}, source="agent", reason="no joint command moved during the run")
            report.append(f"{m['run_id']}: reject_run (nothing moved)")
            continue

        tk = m.get("takeovers", [])
        if eff["termination_reason"] != "success" and tk:
            failed = next((s for s in eff["subgoals"] if s["status"] == "failed"), None)
            take = None
            if failed:
                take = next((x["start_frame"] for x in tk if x["start_frame"] >= failed["start_frame"]), None)
            if failed and take is not None and failed["start_frame"] < take:
                b = budgets.get(failed["name"])
                if b is not None and failed["start_frame"] + b < take:
                    frame = failed["start_frame"] + b
                    why = (f"'{failed['name']}' ran past {b / rate:.1f} s, longer than 90% of successful runs "
                           f"took for that step")
                else:
                    frame, why = failed["start_frame"], f"start of the failed step '{failed['name']}'"
            elif tk[0]["start_frame"] > 0:
                frame, why = tk[0]["start_frame"] - 1, "no step information; frame before the first takeover"
            else:
                frame = None
            if frame is not None:
                add_edit(run_path, m, "failure_start", {"frame": int(frame)}, source="agent", reason=why)
                report.append(f"{m['run_id']}: failure_start {frame} ({why})")

        k = max(1, int(round(idle_s * rate)))
        first = int(np.argmax(moving)) + 1                  # frames before the first move
        last = n - int(np.argmax(moving[::-1]))             # frames after the last move
        if first >= k:
            add_edit(run_path, m, "exclude", {"start": 0, "end": first}, source="agent",
                     reason=f"no movement for the first {first / rate:.1f} s")
            report.append(f"{m['run_id']}: exclude 0-{first} (idle start)")
        if n - last >= k:
            add_edit(run_path, m, "exclude", {"start": last, "end": n}, source="agent",
                     reason=f"no movement for the last {(n - last) / rate:.1f} s")
            report.append(f"{m['run_id']}: exclude {last}-{n} (idle end)")
    return report
