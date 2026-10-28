"""Human review of committed runs: corrections stored next to the run, never applied to the recorded data.

Each run may have `dataop/runs/<run_id>/review.json`. It holds edits (agent proposals and human edits) and whether
the run has been reviewed. The recorded frames, the manifest and its checksums never change; `effective()` folds the
accepted edits into what training should use, and the reader turns that into `dataop.train_mask`.

Edit kinds and their payloads:
  outcome        {"termination_reason": str, "failure_mode": str | None}
  failure_start  {"frame": int}               policy frames from here to the takeover are not good behaviour
  exclude        {"start": int, "end": int}   frames [start, end) are left out of training
  step_boundary  {"before": str, "after": str, "frame": int}   moves the boundary between two adjacent steps
  reject_run     {}                           the whole run is left out of training
  note           {"frame": int, "text": str}
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import store
from .schema import TERMINATION_REASONS

KINDS = ("outcome", "failure_start", "exclude", "step_boundary", "reject_run", "note")
SOURCES = ("agent", "human")
STATUSES = ("proposed", "accepted", "rejected")
# One accepted edit per kind replaces the previous one; these kinds can hold several accepted edits.
MULTI = ("exclude", "note", "step_boundary")


class ReviewError(ValueError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _path(run_path: Path) -> Path:
    return Path(run_path) / "review.json"


def empty() -> dict:
    return {"version": 1, "reviewed": False, "reviewed_at": None, "edits": []}


def load(run_path: Path) -> dict:
    p = _path(run_path)
    if not p.exists():
        return empty()
    import json
    r = json.loads(p.read_text())
    if r.get("version") != 1 or not isinstance(r.get("edits"), list):
        raise ReviewError(f"{p} is not a version 1 review file")
    return r


def save(run_path: Path, review: dict) -> None:
    store.atomic_write_json(_path(run_path), review)


def check_payload(kind: str, payload: dict, length: int, step_names: set[str] | None = None) -> dict:
    """Validate and normalise one edit's payload against the run's length."""
    if kind not in KINDS:
        raise ReviewError(f"kind must be one of {KINDS}")
    if not isinstance(payload, dict):
        raise ReviewError("payload must be an object")

    def frame(v, name, hi):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 0 <= v <= hi:
            raise ReviewError(f"{name} must be an integer frame in [0, {hi}]")
        return int(v)

    if kind == "outcome":
        tr = payload.get("termination_reason")
        if tr not in TERMINATION_REASONS:
            raise ReviewError(f"termination_reason must be one of {TERMINATION_REASONS}")
        fm = payload.get("failure_mode")
        if fm is not None and (not isinstance(fm, str) or not fm.strip()):
            raise ReviewError("failure_mode must be a non-empty string or null")
        if tr == "success" and fm is not None:
            raise ReviewError("a successful run has no failure_mode")
        return {"termination_reason": tr, "failure_mode": fm}
    if kind == "failure_start":
        return {"frame": frame(payload.get("frame"), "frame", length - 1)}
    if kind == "exclude":
        a, b = frame(payload.get("start"), "start", length), frame(payload.get("end"), "end", length)
        if a >= b:
            raise ReviewError("exclude needs start < end")
        return {"start": a, "end": b}
    if kind == "step_boundary":
        before, after = payload.get("before"), payload.get("after")
        if not isinstance(before, str) or not isinstance(after, str) or before == after:
            raise ReviewError("step_boundary needs two different step names")
        if step_names is not None and not {before, after} <= step_names:
            raise ReviewError(f"unknown step in step_boundary: {before!r}, {after!r}")
        return {"before": before, "after": after, "frame": frame(payload.get("frame"), "frame", length)}
    if kind == "reject_run":
        return {}
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ReviewError("note text must be 1 to 2000 characters")
    return {"frame": frame(payload.get("frame"), "frame", length - 1), "text": text}


def _accept(review: dict, edit: dict) -> None:
    """Mark an edit accepted; a single-valued kind retires the previously accepted edit of that kind."""
    if edit["kind"] not in MULTI:
        for e in review["edits"]:
            if e is not edit and e["kind"] == edit["kind"] and e["status"] == "accepted":
                e["status"], e["decided_at"], e["superseded_by"] = "rejected", _now(), edit["id"]
    edit["status"], edit["decided_at"] = "accepted", _now()


def add_edit(run_path: Path, manifest: dict, kind: str, payload: dict, source: str = "human",
             reason: str | None = None) -> dict:
    """Add an edit. Human edits are accepted at once; agent edits wait as proposals."""
    if source not in SOURCES:
        raise ReviewError(f"source must be one of {SOURCES}")
    names = {s["name"] for s in manifest.get("subgoals", [])}
    payload = check_payload(kind, payload, manifest["length"], names)
    review = load(run_path)
    if source == "agent":
        for e in review["edits"]:                   # proposing the same thing twice is a no-op
            if e["source"] == "agent" and e["kind"] == kind and e["payload"] == payload:
                return e
    edit = {"id": uuid.uuid4().hex[:12], "kind": kind, "payload": payload, "source": source, "reason": reason,
            "status": "proposed", "created_at": _now(), "decided_at": None}
    review["edits"].append(edit)
    if source == "human":
        _accept(review, edit)
    save(run_path, review)
    return edit


def decide(run_path: Path, manifest: dict, edit_id: str, accept: bool, payload: dict | None = None) -> dict:
    """Accept (optionally adjusted) or reject an edit."""
    review = load(run_path)
    edit = next((e for e in review["edits"] if e["id"] == edit_id), None)
    if edit is None:
        raise ReviewError(f"no edit {edit_id!r} on this run")
    if accept:
        if payload is not None:
            names = {s["name"] for s in manifest.get("subgoals", [])}
            edit["payload"] = check_payload(edit["kind"], payload, manifest["length"], names)
            edit["adjusted"] = True
        _accept(review, edit)
    else:
        edit["status"], edit["decided_at"] = "rejected", _now()
    save(run_path, review)
    return edit


def set_reviewed(run_path: Path, reviewed: bool = True) -> dict:
    review = load(run_path)
    review["reviewed"], review["reviewed_at"] = bool(reviewed), _now() if reviewed else None
    save(run_path, review)
    return review


def effective(manifest: dict, review: dict) -> dict:
    """The run as training should see it after accepted edits."""
    acc = [e for e in review["edits"] if e["status"] == "accepted"]
    last = {e["kind"]: e["payload"] for e in acc if e["kind"] not in MULTI}
    out = {
        "termination_reason": manifest["termination_reason"], "failure_mode": manifest.get("failure_mode"),
        "failure_start": None, "excludes": [e["payload"] for e in acc if e["kind"] == "exclude"],
        "rejected": "reject_run" in last, "notes": [e["payload"] for e in acc if e["kind"] == "note"],
        "subgoals": [dict(s) for s in manifest.get("subgoals", [])], "reviewed": review["reviewed"],
    }
    if "outcome" in last:
        out["termination_reason"] = last["outcome"]["termination_reason"]
        out["failure_mode"] = last["outcome"]["failure_mode"]
    if "failure_start" in last:
        out["failure_start"] = last["failure_start"]["frame"]
    for e in acc:
        if e["kind"] == "step_boundary":
            p = e["payload"]
            for s in out["subgoals"]:
                if s["name"] == p["before"]:
                    s["end_frame"] = max(s["start_frame"], p["frame"])
                if s["name"] == p["after"]:
                    s["start_frame"] = min(s["end_frame"], p["frame"])
    return out


def train_mask(manifest: dict, review: dict, control_modes: list[str]) -> np.ndarray:
    """Per-frame bool: False for a rejected run, excluded spans, and policy frames from the accepted failure start
    up to the first expert frame after it (the policy was already going wrong there)."""
    n = manifest["length"]
    eff = effective(manifest, review)
    mask = np.ones(n, dtype=bool)
    if eff["rejected"]:
        mask[:] = False
        return mask
    for x in eff["excludes"]:
        mask[x["start"]:x["end"]] = False
    fs = eff["failure_start"]
    if fs is not None:
        i = fs
        while i < n and control_modes[i] in ("base", "blend"):
            mask[i] = False
            i += 1
    return mask


# ── agent proposals ────────────────────────────────────────────────────────────

def _step_budgets(root: Path, q: float = 90) -> dict[str, int]:
    """Per step, the q-th percentile of how many frames the step took in successful runs."""
    durs: dict[str, list[int]] = {}
    for _, m in store.iter_manifests(Path(root), "committed"):
        if m["termination_reason"] != "success":
            continue
        steps = m.get("subgoals", [])
        for sg in steps[:-1]:                   # the last step ends with the run, so its length is no budget
            if sg["status"] == "completed":
                durs.setdefault(sg["name"], []).append(sg["end_frame"] - sg["start_frame"])
    return {k: int(np.ceil(np.percentile(v, q))) for k, v in durs.items() if len(v) >= 3}


def propose(root: Path, idle_s: float = 1.0, idle_eps: float = 0.05) -> list[str]:
    """Add agent proposals to every committed run that has none yet. Returns a report line per proposal.

    failure_start, on failed runs with a takeover: when the failed step had already run longer than it took in 90%
    of successful runs, the frame where it went over that budget; otherwise the start of the failed step; otherwise
    the frame before the takeover.
    exclude: idle stretches of at least idle_s (no joint command moving more than idle_eps per frame) at the very
    start or end of a run.
    """
    report = []
    budgets = _step_budgets(root)
    for run_path, m in store.iter_manifests(Path(root), "committed"):
        review = load(run_path)
        if any(e["source"] == "agent" for e in review["edits"]):
            continue
        n = m["length"]
        rate = m["action_schema"]["rate_hz"]
        tk = m.get("takeovers", [])
        if m["termination_reason"] != "success" and tk and tk[0]["start_frame"] > 0:
            take = tk[0]["start_frame"]
            failed = next((s for s in m.get("subgoals", []) if s["status"] == "failed"), None)
            if failed and failed["start_frame"] < take:
                b = budgets.get(failed["name"])
                if b is not None and failed["start_frame"] + b < take:
                    frame = failed["start_frame"] + b
                    why = (f"'{failed['name']}' ran past {b / rate:.1f} s, longer than 90% of successful runs "
                           f"took for that step")
                else:
                    frame, why = failed["start_frame"], f"start of the failed step '{failed['name']}'"
            else:
                frame, why = take - 1, "no step information; frame before the takeover"
            add_edit(run_path, m, "failure_start", {"frame": int(frame)}, source="agent", reason=why)
            report.append(f"{m['run_id']}: failure_start {frame} ({why})")

        t = pq.read_table(run_path / "ticks.parquet", columns=["sent_action"])
        sent = np.asarray(t.column("sent_action").to_pylist(), dtype=np.float32).reshape(n, -1)
        moving = np.abs(np.diff(sent, axis=0)).max(axis=1) >= idle_eps if n > 1 else np.zeros(0, bool)
        k = max(1, int(round(idle_s * rate)))
        first = int(np.argmax(moving)) + 1 if moving.any() else n          # frames before the first move
        last = n - int(np.argmax(moving[::-1])) if moving.any() else 0     # frames after the last move
        if moving.any() and first >= k:
            add_edit(run_path, m, "exclude", {"start": 0, "end": first}, source="agent",
                     reason=f"no movement for the first {first / rate:.1f} s")
            report.append(f"{m['run_id']}: exclude 0-{first} (idle start)")
        if moving.any() and n - last >= k:
            add_edit(run_path, m, "exclude", {"start": last, "end": n}, source="agent",
                     reason=f"no movement for the last {(n - last) / rate:.1f} s")
            report.append(f"{m['run_id']}: exclude {last}-{n} (idle end)")
    return report
