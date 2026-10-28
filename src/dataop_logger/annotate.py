"""Read and write a run's schema data: what agents (and scripts) use to fill in the same fields people edit.

  runs_json(root)    every run's task segments, drive mode spans, failure moment and type, and video files
  annotate(...)      set task segments, the failure moment and type, or the takeover time, by name from the
                     dataset's lists; applied at once, marked with who set it, undoable in the viewer
"""
from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from . import events as ev
from . import lists as ls
from . import review as rv
from . import store


def _run(root: Path, run_id: str):
    for run_path, m in store.iter_manifests(Path(root), "committed"):
        if m["run_id"] == run_id:
            return run_path, m
    raise rv.ReviewError(f"no committed run {run_id!r}")


def _state(run_path: Path, m: dict):
    t = pq.read_table(run_path / "ticks.parquet", columns=["control_mode", "expert_source"])
    r = rv.load(run_path)
    evs, segs, drive = ev.major_events(m, r, t.column("control_mode").to_pylist(),
                                       t.column("expert_source").to_pylist(), with_segments=True)
    return r, evs, segs, drive


def runs_json(root: Path, run_id: str | None = None) -> dict:
    from .viewer import _video_table
    root = Path(root)
    videos = _video_table(root)
    lists = ls.load(root)
    out = []
    for run_path, m in store.iter_manifests(root, "committed"):
        if run_id and m["run_id"] != run_id:
            continue
        rate = m["action_schema"]["rate_hz"]
        r, evs, segs, drive = _state(run_path, m)
        eff = rv.effective(m, r)
        fail = next((e for e in evs if e["kind"] == "failure"), None)
        s = lambda f: round(f / rate, 3)
        out.append({
            "run_id": m["run_id"], "episode_index": m["episode_index"], "task": m["task"], "fps": rate,
            "frames": m["length"], "duration_s": s(m["length"]), "outcome": eff["termination_reason"],
            "failure": None if not fail else {
                "at_s": s(fail["frame"]), "frame": fail["frame"], "estimated": fail.get("estimated", False),
                "type": ls.canonical(lists, "failure_types", eff["failure_mode"]),
                "set_by": eff.get("set_by", {})},
            "segments": [{"name": ls.canonical(lists, "task_segments", x["name"]), "start_s": s(x["start"]),
                          "end_s": s(x["end"]), "start_frame": x["start"], "end_frame": x["end"]} for x in segs],
            "segments_set_by": eff.get("set_by", {}).get("segment_plan", "recorded"),
            "drive": [{"mode": d["driver"], "by": d["source"], "start_s": s(d["start"]), "end_s": s(d["end"]),
                       "start_frame": d["start"], "end_frame": d["end"]} for d in drive],
            "videos": [{"camera": v["key"], "path": str(root / v["src"]), "starts_at_s": v["t0"]}
                       for v in videos.get(m["episode_index"], [])],
        })
    return {"dataset": str(root), "lists": {k: [{"name": e["name"], "description": e.get("description", "")}
                                                 for e in lists[k]] for k in ls.KINDS},
            "runs": out}


def _known(lists: dict, kind: str, name: str) -> str:
    e = next((x for x in lists[kind] if x["name"] == name or name in x.get("aliases", [])), None)
    if e is None:
        raise ls.ListError(f"{name!r} is not in this dataset's {kind.replace('_', ' ')}; "
                           f"add it first with: dataop-logger lists <root> add {kind} {name}")
    return e["name"]


def annotate(root: Path, run_id: str, by: str, segments: list | None = None, failure_at: float | None = None,
             failure_type: str | None = None, takeover_at: float | None = None) -> list[str]:
    root = Path(root)
    run_path, m = _run(root, run_id)
    rate, n = m["action_schema"]["rate_hz"], m["length"]
    lists = ls.load(root)
    frame = lambda x: max(0, min(n, int(round(float(x) * rate))))
    done = []
    if segments is not None:
        plan = [{"name": _known(lists, "task_segments", x["name"]), "start": frame(x["start_s"]),
                 "end": frame(x["end_s"])} for x in segments]
        with rv.recording(f"{by}: task segments"):
            rv.apply_agent_edit(run_path, m, "segment_plan", {"segments": plan}, by)
        done.append(f"segments: {', '.join(p['name'] for p in plan)}")
    if takeover_at is not None:
        _, _, _, drive = _state(run_path, m)
        i = next((k for k in range(len(drive) - 1) if drive[k]["mode"] == "base" and drive[k + 1]["mode"] != "base"),
                 None)
        if i is None:
            raise rv.ReviewError("this run has no switch from policy to another drive mode")
        with rv.recording(f"{by}: takeover at {takeover_at:.1f} s"):
            rv.apply_agent_edit(run_path, m, "drive_boundary", {"index": i, "frame": frame(takeover_at)}, by)
        done.append(f"takeover at {takeover_at:.2f} s")
    if failure_type is not None:
        name = _known(lists, "failure_types", failure_type)
        tr = rv.effective(m, rv.load(run_path))["termination_reason"]
        with rv.recording(f"{by}: failure type {name}"):
            rv.apply_agent_edit(run_path, m, "outcome",
                                {"termination_reason": "failure" if tr == "success" else tr, "failure_mode": name}, by)
        done.append(f"failure type: {name}")
    if failure_at is not None:
        with rv.recording(f"{by}: policy fails at {failure_at:.1f} s"):
            rv.apply_agent_edit(run_path, m, "failure_start", {"frame": min(frame(failure_at), n - 1)}, by)
        done.append(f"policy fails at {failure_at:.2f} s")
    return done
