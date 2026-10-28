"""validate, repair, inspect and remap: checks a dataset root's sidecar against the LeRobot data on disk.

Validation rebuilds every invariant from the files alone; it never trusts the writer.
"""
from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.types as pa_types
import pyarrow.parquet as pq

from . import adapter, store
from .fingerprint import fingerprint
from .schema import CONTROL_MODES, TERMINAL_FLAG, TERMINATION_REASONS, ActionSchema

TICK_COLUMNS = ("frame_index", "control_mode", "expert_source", "blend_weight", "base_action", "base_valid",
                "expert_action", "expert_valid", "sent_action", "safety_modified", "inference_id", "chunk_offset",
                "obs_ns", "obs_skew_ns", "sent_ns", "reward", "reward_valid", "terminated", "truncated")


def load_lerobot_episodes(root: Path, action_key: str = "action", orphans: list | None = None) -> dict[int, dict]:
    """Episode index -> {frame_index, action (raw, canonicalized later per schema), task}.

    Only episodes LeRobot fully registered in meta/episodes are returned. Data rows for other episodes (a save
    that failed after writing data but before metadata) are reported through `orphans`, never returned.
    """
    data = Path(root) / "data"
    if not data.exists():
        return {}
    meta = adapter.episode_metadata(root)
    dset = pads.dataset(str(data), format="parquet")
    has_index = "index" in dset.schema.names
    t = dset.to_table(columns=["episode_index", "frame_index", "task_index", action_key] + (["index"] if has_index else []))
    tasks = adapter.episode_tasks(root)
    ep = t.column("episode_index").to_numpy()
    fi = t.column("frame_index").to_numpy()
    ti = t.column("task_index").to_numpy()
    act = np.asarray(t.column(action_key).to_pylist(), dtype=np.float32)
    gidx = t.column("index").to_numpy() if has_index else None
    out = {}
    for e in np.unique(ep):
        m = ep == e
        order = np.argsort(fi[m], kind="stable")
        tis = set(ti[m].tolist())
        task = tasks.get(tis.pop()) if len(tis) == 1 else None      # None: mixed or unknown task
        em = meta.get(int(e))
        span_ok = em is not None and (gidx is None or np.array_equal(np.sort(gidx[m]), np.arange(em["from"], em["to"])))
        videos_ok = em is not None and len(em["videos"]) == len(em["video_keys"]) and \
            all(v.exists() and v.stat().st_size > 0 for v in em["videos"])
        if em is None or em["length"] != int(m.sum()) or em["to"] - em["from"] != em["length"] or \
                (em["tasks"] and task not in em["tasks"]) or not span_ok or not videos_ok:
            if orphans is not None:
                orphans.append(int(e))
            continue
        out[int(e)] = {"frame_index": fi[m][order], "action": act[m][order], "task": task}
    return out


def _ep_fingerprint(schema: ActionSchema, ep: dict) -> str | None:
    if ep["task"] is None:
        return None
    try:
        acts = schema.canon_rows(ep["action"], len(ep["frame_index"]))
    except ValueError:
        return None
    return fingerprint(schema.id, ep["task"], acts)


def _arr(t, col, n, dim):
    a = np.asarray(t.column(col).to_pylist(), dtype=np.float32)
    if a.shape != (n, dim):
        raise ValueError(f"{col} shape {a.shape}, expected {(n, dim)}")
    return a


def _check_run(run_path: Path, m: dict, episodes: dict, errors: list) -> bool:
    """Append errors for one run. Returns False when the run could not be read at all."""
    rid = m["run_id"]
    err = lambda s: errors.append(f"{rid}: {s}")
    bad = integrity_problems(run_path, m)
    if bad:
        for b in bad:
            err(b)
        return False
    try:
        schema = ActionSchema.from_dict(m["action_schema"])
    except Exception as e:
        err(f"bad action schema ({e})")
        return False
    if schema.id != m["action_schema"].get("id"):
        err("action schema id does not match its content")
    if m["termination_reason"] not in TERMINATION_REASONS:
        err(f"unknown termination_reason {m['termination_reason']!r}")
    try:
        t = pq.read_table(run_path / "ticks.parquet")
    except Exception as e:
        err(f"ticks.parquet unreadable ({e})")
        return False
    missing = [c for c in TICK_COLUMNS if c not in t.column_names]
    if missing:
        err(f"ticks.parquet is missing columns {missing}")
        return False
    n, dim = t.num_rows, schema.dim
    if n == 0:
        err("run has no ticks")
        return False
    if n != m["length"]:
        err(f"ticks has {n} rows, manifest says {m['length']}")
    col = {c: t.column(c).to_pylist() for c in TICK_COLUMNS if not c.endswith("_action")}
    try:
        base, expert, sent = (_arr(t, c, n, dim) for c in ("base_action", "expert_action", "sent_action"))
    except ValueError as e:
        err(str(e))
        return False
    if col["frame_index"] != list(range(n)):
        err(f"frame_index is not 0..{n - 1}")
    for name, a in (("base_action", base), ("expert_action", expert), ("sent_action", sent)):
        if not np.all(np.isfinite(a)):
            err(f"{name} has NaN or Inf")
    bv, ev = np.array(col["base_valid"], bool), np.array(col["expert_valid"], bool)
    if np.any(base[~bv] != 0) or np.any(expert[~ev] != 0):
        err("an invalid base_action/expert_action row is not zero-filled")
    for i in range(n):
        mode, src, w = col["control_mode"][i], col["expert_source"][i], col["blend_weight"][i]
        if mode not in CONTROL_MODES:
            err(f"tick {i}: unknown control_mode {mode!r}")
        if mode in ("expert", "blend") and not src:
            err(f"tick {i}: {mode} without expert_source")
        if mode == "base" and not bv[i]:
            err(f"tick {i}: control_mode base without a valid base_action")
        if mode == "expert" and not ev[i]:
            err(f"tick {i}: control_mode expert without a valid expert_action")
        if mode == "blend" and not (bv[i] and ev[i]):
            err(f"tick {i}: blend without both base_action and expert_action")
        if ev[i] and not src:
            err(f"tick {i}: expert_action without expert_source")
        want_w = {"expert": 1.0, "base": 0.0, "none": 0.0}.get(mode)
        if (want_w is not None and w != want_w) or not 0.0 <= w <= 1.0:
            err(f"tick {i}: blend_weight {w} invalid for {mode}")
        iid, off = col["inference_id"][i], col["chunk_offset"][i]
        if (iid == -1) != (not bv[i]) or (iid != -1 and (iid < 0 or off < 0)) or (iid == -1 and off != -1):
            err(f"tick {i}: inference_id/chunk_offset inconsistent with base_valid")
        if not np.isfinite(col["reward"][i]):
            err(f"tick {i}: reward is not finite")
        if col["obs_skew_ns"][i] < -1:
            err(f"tick {i}: obs_skew_ns < -1")
    for c in ("obs_ns", "sent_ns", "obs_skew_ns", "inference_id", "chunk_offset", "frame_index"):
        if not pa_types.is_integer(t.schema.field(c).type):
            err(f"{c} is not an integer column")
    if any(v < 0 for v in col["obs_ns"] + col["sent_ns"]):
        err("negative timestamps")
    obs, snt = np.array(col["obs_ns"], np.int64), np.array(col["sent_ns"], np.int64)
    if np.any(obs > snt):
        err("obs_ns is after sent_ns on some tick")
    if n > 1 and (np.any(np.diff(obs) < 0) or np.any(np.diff(snt) < 0)):
        err("timestamps are not monotonic")
    term, trunc = np.array(col["terminated"], bool), np.array(col["truncated"], bool)
    if term[:-1].any() or trunc[:-1].any():
        err("terminated/truncated before the last tick")
    want = TERMINAL_FLAG.get(m["termination_reason"])
    if want and not (term[-1] if want == "terminated" else trunc[-1]) or (term[-1] and trunc[-1]):
        err(f"final tick flags do not match termination_reason {m['termination_reason']!r}")
    if m.get("rl_ready") is not None and m["rl_ready"] != all(col["reward_valid"]):
        err("rl_ready disagrees with reward_valid")
    rows = [dict(control_mode=col["control_mode"][i], expert_source=col["expert_source"][i]) for i in range(n)]
    from .logger import derive_takeovers
    if derive_takeovers(rows) != m["takeovers"]:
        err("takeovers summary does not match the per-tick control modes")
    from .logger import derive_subgoals
    from .schema import SUBGOAL_STATUSES
    if "subgoals" in m and derive_subgoals(m["events"], n) != m["subgoals"]:
        err("subgoals summary does not match the subgoal events")
    for ev_ in m["events"]:
        if ev_.get("kind") == "subgoal" and (not ev_.get("source") or ev_.get("text") not in SUBGOAL_STATUSES):
            err(f"event {ev_.get('event_id')} is a subgoal without a name or a valid status")
        if not -1 <= ev_.get("frame_index", -2) < n:
            err(f"event {ev_.get('event_id')} frame_index out of range")
        if not isinstance(ev_.get("ns"), int) or ev_["ns"] < 0:
            err(f"event {ev_.get('event_id')} has a bad timestamp")
    pc = m.get("predictions_count")
    pf = run_path / "predictions.parquet"
    if pc is not None:
        if not pf.exists():
            err("predictions_count is set but predictions.parquet is missing")
        else:
            ptab = pq.read_table(pf)
            for c in ("inference_id", "obs_frame_index", "infer_start_ns", "infer_end_ns", "horizon", "valid_len"):
                if c not in ptab.column_names or not pa_types.is_integer(ptab.schema.field(c).type):
                    err(f"predictions.parquet column {c} is missing or not integer")
                    return True
            p = ptab.to_pydict()
            ids = p["inference_id"]
            if len(ids) != pc or len(set(ids)) != len(ids):
                err("predictions.parquet count or inference_id uniqueness is wrong")
            vl = dict(zip(ids, p["valid_len"]))
            for k in range(len(ids)):
                chunk = np.asarray(p["action_chunk"][k], dtype=np.float32)
                if chunk.size != p["horizon"][k] * dim or not 0 < p["valid_len"][k] <= p["horizon"][k]:
                    err(f"prediction {ids[k]} has a bad chunk shape or valid_len")
                if not np.all(np.isfinite(chunk)):
                    err(f"prediction {ids[k]} has NaN or Inf values")
                if ids[k] < 0 or not 0 <= p["infer_start_ns"][k] <= p["infer_end_ns"][k]:
                    err(f"prediction {ids[k]} has a negative id or bad inference times")
                if not -1 <= p["obs_frame_index"][k] < n:
                    err(f"prediction {ids[k]} obs_frame_index out of range")
            for i in range(n):
                iid = col["inference_id"][i]
                if iid != -1 and (iid not in vl or col["chunk_offset"][i] >= vl[iid]):
                    err(f"tick {i}: refers to a missing prediction or an offset past valid_len")
    elif pf.exists():
        err("predictions.parquet exists but the manifest has no predictions_count")
    if fingerprint(schema.id, m["task"], sent) != m["fingerprint"]:
        err("fingerprint does not match the logged actions")
    ep = episodes.get(m["episode_index"])
    if ep is None:
        err(f"LeRobot episode {m['episode_index']} not found on disk")
        return True
    if len(ep["frame_index"]) != n or not np.array_equal(ep["frame_index"], np.arange(n)):
        err(f"LeRobot episode {m['episode_index']} has {len(ep['frame_index'])} frames, run has {n}")
        return True
    if ep["task"] != m["task"]:
        err(f"LeRobot episode task {ep['task']!r} differs from run task {m['task']!r}")
    try:
        disk = schema.canon_rows(ep["action"], n)
    except ValueError as e:
        err(f"LeRobot actions: {e}")
        return True
    if disk.tobytes() != sent.tobytes():
        err(f"sent_action differs from LeRobot action (max abs diff {float(np.max(np.abs(disk - sent)))})")
    return True


def validate(root: Path, action_key: str = "action") -> dict:
    root = Path(root)
    errors, warnings, orphans = [], [], []
    try:
        episodes = load_lerobot_episodes(root, action_key, orphans)
    except Exception as e:
        return {"ok": False, "errors": [f"cannot read LeRobot data: {e}"], "warnings": [], "runs": 0,
                "episodes_on_disk": 0, "unbound_episodes": [], "quarantined": []}
    bound = defaultdict(list)
    staged = []
    for run_path, m, problem in store.iter_run_dirs(root):
        if problem:
            errors.append(f"{run_path.name}: {problem}")
            continue
        if m["status"] == "staged":
            staged.append(m["run_id"] + (f" (save_error: {m['save_error']})" if m.get("save_error") else ""))
            continue
        bound[m["episode_index"]].append(m["run_id"])
        try:
            _check_run(run_path, m, episodes, errors)
        except Exception as e:                                  # report, keep checking the other runs
            errors.append(f"{m['run_id']}: could not be checked ({type(e).__name__}: {e})")
    for e, ids in bound.items():
        if len(ids) > 1:
            errors.append(f"episode {e} is bound to several runs: {ids}")
    if staged:
        errors.append(f"staged runs not committed (run validate --repair): {staged}")
    if orphans:
        errors.append(f"LeRobot data rows for episodes missing from or inconsistent with meta/episodes "
                      f"(an incomplete save?): {orphans}")
    q = store.quarantine_dir(root)
    quarantined = sorted(p.name for p in q.iterdir()) if q.exists() else []
    unbound = sorted(set(episodes) - set(bound))
    if unbound:
        warnings.append(f"{len(unbound)} LeRobot episodes have no dataop run: {unbound[:20]}")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "runs": sum(map(len, bound.values())),
            "episodes_on_disk": len(episodes), "unbound_episodes": unbound, "quarantined": quarantined}


def integrity_problems(run_path: Path, m: dict) -> list[str]:
    """Compare the sidecar files with the digests recorded when they were written."""
    integ = m.get("integrity")
    if not isinstance(integ, dict) or not all(v is None or isinstance(v, str) for v in integ.values()):
        return ["integrity record is malformed"]
    out = []
    try:
        if store.sha256((run_path / "ticks.parquet").read_bytes()) != integ.get("ticks_sha256"):
            out.append("ticks.parquet does not match its recorded digest")
    except OSError as e:
        out.append(f"ticks.parquet unreadable ({e})")
    if store.json_sha256(m.get("events", [])) != integ.get("events_sha256"):
        out.append("events do not match their recorded digest")
    pf = run_path / "predictions.parquet"
    want = integ.get("predictions_sha256")
    if want is not None or pf.exists():
        try:
            if store.sha256(pf.read_bytes()) != want:
                out.append("predictions.parquet does not match its recorded digest")
        except OSError as e:
            out.append(f"predictions.parquet unreadable ({e})")
    return out


def _matches(m: dict, episodes: dict, exclude: set) -> list[int]:
    schema = ActionSchema.from_dict(m["action_schema"])
    return [e for e, ep in episodes.items() if e not in exclude and len(ep["frame_index"]) == m["length"]
            and ep["task"] == m["task"] and _ep_fingerprint(schema, ep) == m["fingerprint"]]


def repair(root: Path, delete_unmatched: bool = False, action_key: str = "action") -> list[str]:
    """Non-destructive by default: commit exact unique matches, quarantine the rest, never touch unreadable runs."""
    root = Path(root)
    episodes = load_lerobot_episodes(root, action_key)
    committed = {m["episode_index"] for _, m in store.iter_manifests(root, "committed")}
    report = []
    for run_path, m, problem in list(store.iter_run_dirs(root)):
        if problem:
            report.append(f"left untouched {run_path.name}: {problem}")
            continue
        if m["status"] != "staged":
            continue
        try:
            bad = integrity_problems(run_path, m)
            if bad:
                report.append(f"left untouched {m['run_id']}: {'; '.join(bad)}")
                continue
            matches = _matches(m, episodes, committed)
        except Exception as e:
            report.append(f"left untouched {m['run_id']}: {e}")
            continue
        if len(matches) == 1:
            m.update(status="committed", episode_index=matches[0])
            m.pop("save_error", None)
            store.atomic_write_json(run_path / "manifest.json", m)
            committed.add(matches[0])
            report.append(f"committed {m['run_id']} as episode {matches[0]}")
        elif delete_unmatched and not matches:
            shutil.rmtree(run_path)
            report.append(f"deleted unmatched {m['run_id']}")
        else:
            why = "no matching episode" if not matches else f"several matching episodes {matches}"
            store.quarantine(root, run_path, why)
            report.append(f"quarantined {m['run_id']}: {why}")
    store.rebuild_index(root)
    return report


def inspect(root: Path) -> dict:
    root = Path(root)
    outcomes, minutes, takeovers, runs = defaultdict(int), defaultdict(float), 0, 0
    failure_modes, failed_subgoals = defaultdict(int), defaultdict(int)
    for run_path, m in store.iter_manifests(root, "committed"):
        runs += 1
        outcomes[m["termination_reason"]] += 1
        takeovers += len(m.get("takeovers", []))
        if m.get("failure_mode"):
            failure_modes[m["failure_mode"]] += 1
        for sg in m.get("subgoals", []):
            if sg["status"] == "failed":
                failed_subgoals[sg["name"]] += 1
        rate = m["action_schema"]["rate_hz"]
        modes = pq.read_table(run_path / "ticks.parquet", columns=["control_mode"]).column("control_mode").to_pylist()
        for mode in modes:
            minutes[mode] += 1 / rate / 60
    return {"runs": runs, "outcomes": dict(outcomes), "takeovers": takeovers,
            "failure_modes": dict(failure_modes), "failed_subgoals": dict(failed_subgoals),
            "minutes_by_control_mode": {k: round(v, 2) for k, v in minutes.items()}}


def remap(old_root: Path, new_root: Path, action_key: str = "action", repo_id: str | None = None,
          revision: str | None = None) -> list[str]:
    """Copy runs into a filtered or renumbered dataset, binding each by a unique task + fingerprint match."""
    new_eps = load_lerobot_episodes(Path(new_root), action_key)
    taken, report = set(), []
    for run_path, m in store.iter_manifests(Path(old_root), "committed"):
        matches = _matches(m, new_eps, taken)
        if len(matches) != 1:
            report.append(f"skipped {m['run_id']}: {len(matches)} matches")
            continue
        dest = store.runs_dir(Path(new_root)) / m["run_id"]
        shutil.copytree(run_path, dest)
        m2 = dict(m, episode_index=matches[0], dataset={"repo_id": repo_id, "revision": revision, "root": str(new_root)},
                  remapped_from={"dataset": m.get("dataset"), "episode_index": m["episode_index"]})
        store.atomic_write_json(dest / "manifest.json", m2)
        taken.add(matches[0])
        report.append(f"{m['run_id']}: episode {m['episode_index']} -> {matches[0]}")
    store.rebuild_index(Path(new_root))
    return report


def to_json(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True)
