"""Import: bind dataop runs to the episodes of an existing LeRobot v3.0 dataset and fill the sidecar from what the
dataset (and optional side files) already hold. LeRobot files are never rewritten.

Sources: a local LeRobot v3.0 folder (read in place), or a Hugging Face Hub repo id (downloaded by LeRobot, then
copied with --into, since the Hub cache is not ours to change). A v2.1 folder is refused with LeRobot's own
conversion command.

A mapping (JSON) says where each sidecar field comes from. Every entry is optional; a field with no source is recorded
as unknown, never invented. Per-run `origin.fields` says where each field came from.

  {"control":      {"column": "intervention"}                        bool column: true = expert, false = policy
                   | {"column": "mode", "values": {"auto": "base", "teleop": "expert"}},
   "expert_source": "human",                                         who the expert was (default "expert")
   "base_action":  {"column": "complementary_info.policy_action"}
                   | {"file": "{root}_raw/ep{episode:03d}.npy"}       .npy (rows from frame 0) or .parquet/.csv
                     with "covers": "policy_frames": one row per policy-driven frame, in order (count must match)
                   | "sent_on_policy_frames",                         the command sent while the policy drove
   "expert_action": {"column": ...} | {"file": ...} | "sent_on_expert_frames" (default when control is mapped),
   "outcome":      {"column": "next.success"}                        true on the last frame = success
                   | {"file": "{root}_results_*.json", "key": "episode", "field": "result",
                      "values": {"vla_success": "success", "watcher_rescue": "failure"}},
   "failure_mode": {"file": ..., "key": "episode", "field": "outcome", "skip": ["success"]},
   "reward":       {"column": "next.reward"},
   "segments":     {"column": "task_segment"},                       per-frame segment name (empty = none)
   "policy":       {"family": "smolvla", "source": "...", "revision": null},
   "action_space": "joint_position", "action_units": "deg"}

Path patterns: {root} is the dataset folder path, {name} its folder name, {episode} the episode index.
"""
from __future__ import annotations

import glob
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads

from . import batches as B
from . import store
from .fingerprint import fingerprint
from .schema import KNOWN_SPACES, SCHEMA_VERSION, TERMINAL_FLAG, TERMINATION_REASONS, ActionSchema

PRESETS = {
    "plain": {},
    # LeRobot's human-in-the-loop recordings (gym_manipulator / HIL-SERL)
    "lerobot-hil": {"control": {"column": "complementary_info.is_intervention"}, "expert_source": "human",
                    "base_action": "sent_on_policy_frames", "outcome": {"column": "next.success"},
                    "reward": {"column": "next.reward"}},
    # sessions recorded by Watcher: an intervention column, the base model's raw output and a results file
    "watcher": {"control": {"column": "intervention"}, "expert_source": "skill:rescue",
                "base_action": {"file": "{root}_raw/ep{episode:03d}.npy", "covers": "policy_frames"},
                "outcome": {"file": "{root}_results_*.json", "key": "episode", "field": "result",
                            "values": {"vla_success": "success", "watcher_rescue": "failure", "fail": "failure"}},
                "failure_mode": {"file": "{root}_results_*.json", "key": "episode", "field": "outcome",
                                 "skip": ["success"]}},
}
KEYS = {"control", "expert_source", "base_action", "expert_action", "outcome", "failure_mode", "reward", "segments",
        "policy", "action_space", "action_units"}


class ImportError_(ValueError):
    pass


def _info(root: Path) -> dict:
    p = root / "meta" / "info.json"
    if not p.exists():
        raise ImportError_(f"{root} is not a LeRobot dataset (no meta/info.json)")
    info = json.loads(p.read_text())
    v = str(info.get("codebase_version", ""))
    if v.startswith("v2"):
        raise ImportError_(f"{root} is a LeRobot {v} dataset. Convert it with LeRobot first: python -m "
                           "lerobot.scripts.convert_dataset_v21_to_v30 --repo-id <repo_id> (then import the result)")
    if v != "v3.0":
        raise ImportError_(f"{root} has codebase_version {v!r}; import reads LeRobot v3.0 datasets")
    return info


def resolve_source(source: str, into: str | None) -> tuple[Path, bool]:
    """(local root, must_copy). A path that exists is local; anything else is treated as a Hub repo id."""
    p = Path(source).expanduser()
    if p.exists():
        return p, False
    if "/" not in source or source.startswith((".", "/")):
        raise ImportError_(f"{source} is neither a local folder nor a Hub repo id (owner/name)")
    if not into:
        raise ImportError_("a Hub dataset is imported into a copy: pass --into <folder>")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(source)                       # downloads into the Hugging Face cache
    return Path(ds.root), True


def _mapping(preset: str | None, mapping: dict | None) -> dict:
    if preset and preset not in PRESETS:
        raise ImportError_(f"unknown preset {preset!r}; one of {sorted(PRESETS)}")
    m = {**PRESETS.get(preset or "plain", {}), **(mapping or {})}
    bad = set(m) - KEYS
    if bad:
        raise ImportError_(f"unknown mapping keys {sorted(bad)}; allowed: {sorted(KEYS)}")
    return m


def _pattern(pat: str, root: Path, episode: int | None = None) -> str:
    return pat.format(root=str(root), name=root.name, episode=episode if episode is not None else 0)


def _per_episode_file(spec: dict, root: Path, ep: int, n: int, dim: int,
                      policy_frames: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """(values [n, dim], valid [n]) from a per-episode file, or None when the file is missing."""
    path = Path(_pattern(spec["file"], root, ep))
    if not path.exists():
        return None
    if path.suffix == ".npy":
        a = np.load(path, allow_pickle=False).astype(np.float32)
        if spec.get("covers") == "policy_frames":
            if policy_frames is None or len(policy_frames) != len(a):
                raise ImportError_(f"{path} has {len(a)} rows but the episode has "
                                   f"{0 if policy_frames is None else len(policy_frames)} policy-driven frames")
            rows = policy_frames
        else:
            rows = np.arange(len(a)) + int(spec.get("start", 0))
    else:
        import pandas as pd
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        rows = df["frame_index"].to_numpy()
        a = np.asarray([np.asarray(v, np.float32).ravel() for v in df[spec.get("column", "action")]], np.float32)
    if a.ndim != 2 or a.shape[1] != dim:
        raise ImportError_(f"{path}: expected rows of {dim} values, got shape {a.shape}")
    if len(rows) and (rows.min() < 0 or rows.max() >= n):
        raise ImportError_(f"{path}: rows reach frame {int(rows.max())}, the episode has {n} frames")
    out, valid = np.zeros((n, dim), np.float32), np.zeros(n, bool)
    out[rows], valid[rows] = a, True
    return out, valid


def _results(spec: dict, root: Path) -> dict:
    """key value -> record, from every file matching the pattern (JSON list, JSON object of lists, or JSONL)."""
    recs = {}
    for f in sorted(glob.glob(_pattern(spec["file"], root))):
        text = Path(f).read_text()
        try:
            data = json.loads(text)
            items = data if isinstance(data, list) else data.get("runs", data.get("episodes", []))
        except json.JSONDecodeError:
            items = [json.loads(line) for line in text.splitlines() if line.strip()]
        for x in items:
            if isinstance(x, dict) and spec.get("key", "episode") in x:
                recs[int(x[spec.get("key", "episode")])] = x
    return recs


def _columns(root: Path, wanted: list[str]) -> dict:
    dset = pads.dataset(str(root / "data"), format="parquet")
    have = [c for c in wanted if c in dset.schema.names]
    t = dset.to_table(columns=["episode_index", "frame_index", "timestamp", *have])
    return {"names": set(dset.schema.names), "table": t}


def plan(root: Path, preset: str | None = "plain", mapping: dict | None = None, episodes: list[int] | None = None,
         action_key: str = "action") -> dict:
    """Work out, without writing, what import would do: per field where it comes from, and per episode bind / skip."""
    from .validate import load_lerobot_episodes
    root = Path(root)
    info = _info(root)
    mp = _mapping(preset, mapping)
    feat = info["features"].get(action_key)
    if not feat:
        raise ImportError_(f"the dataset has no {action_key!r} feature")
    names = feat.get("names") or [f"a{i}" for i in range(int(np.prod(feat["shape"])))]
    if isinstance(names, dict):
        names = next(iter(names.values()))
    dim = int(np.prod(feat["shape"]))
    space = mp.get("action_space", "custom:imported")
    if space not in KNOWN_SPACES and not str(space).startswith("custom:"):
        raise ImportError_(f"action_space {space!r} is not one of {sorted(KNOWN_SPACES)} or custom:<name>")
    schema = ActionSchema(names=tuple(str(x) for x in names), units=tuple([mp.get("action_units", "unknown")] * dim),
                          space=space, rate_hz=float(info["fps"]))
    cols_wanted = [s["column"] for s in (mp.get(k) for k in ("control", "base_action", "expert_action", "outcome",
                                                             "reward", "segments"))
                   if isinstance(s, dict) and "column" in s]
    cols = _columns(root, cols_wanted)
    missing_cols = [c for c in cols_wanted if c not in cols["names"]]
    eps = load_lerobot_episodes(root, action_key)
    bound = {m["episode_index"] for _, m in store.iter_manifests(root, "committed")}
    want = sorted(eps) if episodes is None else [e for e in sorted(set(episodes))]
    fields = {}
    for k in ("control", "base_action", "expert_action", "outcome", "failure_mode", "reward", "segments"):
        s = mp.get(k)
        fields[k] = ("unknown" if s is None else s if isinstance(s, str) else
                     f"column {s['column']}" + (" (missing)" if s["column"] in missing_cols else "")
                     if "column" in s else f"file {s['file']}")
    if mp.get("expert_action") is None and mp.get("control"):
        fields["expert_action"] = "sent_on_expert_frames"
    todo, skipped = [], []
    for e in want:
        if e not in eps:
            skipped.append({"episode": e, "why": "not a complete LeRobot episode"})
        elif e in bound:
            skipped.append({"episode": e, "why": "already has a dataop run"})
        elif eps[e]["task"] is None:
            skipped.append({"episode": e, "why": "the episode's task is missing or mixed"})
        else:
            todo.append(e)
    return {"root": str(root), "fps": info["fps"], "schema": schema, "mapping": mp, "fields": fields,
            "missing_columns": missing_cols, "episodes": todo, "skipped": skipped, "lerobot": eps, "cols": cols}


def _episode_rows(p: dict, e: int) -> dict:
    t = p["cols"]["table"]
    epcol = t.column("episode_index").to_numpy()
    m = epcol == e
    order = np.argsort(t.column("frame_index").to_numpy()[m], kind="stable")
    return {c: np.asarray(t.column(c).to_pylist(), dtype=object)[m][order] for c in t.column_names}


def _build_run(p: dict, e: int, results: dict) -> tuple[dict, pa.Table, list[str]]:
    """Manifest, ticks table and notes for one episode."""
    mp, schema, root = p["mapping"], p["schema"], Path(p["root"])
    ep = p["lerobot"][e]
    n, dim = len(ep["frame_index"]), schema.dim
    sent = schema.canon_rows(ep["action"], n)
    rows = _episode_rows(p, e)
    notes = []

    # who drove
    ctl = mp.get("control")
    if ctl and ctl["column"] in rows:
        raw = rows[ctl["column"]]
        vals = ctl.get("values")
        if vals:
            mode = np.array([vals.get(str(_scalar(v)), "unknown") for v in raw], dtype=object)
        else:
            mode = np.array(["expert" if bool(_scalar(v)) else "base" for v in raw], dtype=object)
    else:
        mode = np.array(["unknown"] * n, dtype=object)

    def actions(spec, which):
        if spec is None:
            return np.zeros((n, dim), np.float32), np.zeros(n, bool)
        if spec == "sent_on_policy_frames":
            v = mode == "base"
            return np.where(v[:, None], sent, 0).astype(np.float32), v
        if spec == "sent_on_expert_frames":
            v = mode == "expert"
            return np.where(v[:, None], sent, 0).astype(np.float32), v
        if "column" in spec:
            if spec["column"] not in rows:
                return np.zeros((n, dim), np.float32), np.zeros(n, bool)
            a = np.asarray([np.asarray(x, np.float32).ravel() for x in rows[spec["column"]]], np.float32)
            if a.shape != (n, dim):
                raise ImportError_(f"episode {e}: column {spec['column']} has shape {a.shape}, expected {(n, dim)}")
            return a, np.isfinite(a).all(axis=1)
        got = _per_episode_file(spec, root, e, n, dim, np.flatnonzero(mode == "base") if ctl else None)
        if got is None:
            notes.append(f"{which}: no file for episode {e}")
            return np.zeros((n, dim), np.float32), np.zeros(n, bool)
        return got

    base, bv = actions(mp.get("base_action"), "base_action")
    exp_spec = mp.get("expert_action", "sent_on_expert_frames" if ctl else None)
    expert, xv = actions(exp_spec, "expert_action")
    base[~bv], expert[~xv] = 0, 0
    # the validator's rules: policy frames need the policy's action, expert frames need the expert's
    for i in range(n):
        if mode[i] == "base" and not bv[i]:
            mode[i] = "unknown"
        if mode[i] == "expert" and not xv[i]:
            mode[i] = "unknown"
    if (mode == "unknown").any() and ctl:
        notes.append(f"{int((mode == 'unknown').sum())} frames: who drove is not known")
    src = mp.get("expert_source", "expert")
    expert_source = [src if (mode[i] == "expert" or xv[i]) else None for i in range(n)]

    # outcome and failure type
    reason, failure_mode = "unknown", None
    oc = mp.get("outcome")
    if isinstance(oc, dict) and "column" in oc and oc["column"] in rows:
        # LeRobot marks success on the step that reached it (next.success); any such step makes the run a success
        reason = "success" if any(bool(_scalar(v)) for v in rows[oc["column"]]) else "failure"
    elif isinstance(oc, dict) and "file" in oc:
        rec = results["outcome"].get(e)
        if rec is not None:
            v = rec.get(oc["field"])
            reason = oc.get("values", {}).get(str(v), v if v in TERMINATION_REASONS else "unknown")
    fm = mp.get("failure_mode")
    if isinstance(fm, dict) and reason != "success":
        rec = results["failure_mode"].get(e)
        v = rec and rec.get(fm["field"])
        if v and str(v) not in fm.get("skip", []):
            failure_mode = str(v)

    # reward
    reward, rvalid = np.zeros(n, np.float32), np.zeros(n, bool)
    rw = mp.get("reward")
    if isinstance(rw, dict) and rw.get("column") in rows:
        reward = np.asarray([float(_scalar(v)) for v in rows[rw["column"]]], np.float32)
        rvalid = np.isfinite(reward)
        reward[~rvalid] = 0

    # task segments from a per-frame name column: subgoal events, as the logger writes them
    events = []
    sg = mp.get("segments")
    if isinstance(sg, dict) and sg.get("column") in rows:
        names = [str(_scalar(v) or "") for v in rows[sg["column"]]]
        i = 0
        while i < n:
            j = i
            while j < n and names[j] == names[i]:
                j += 1
            if names[i]:
                for f, status in ((i - 1, "started"), (j - 1, "completed")):
                    events.append({"event_id": len(events), "ns": 0, "frame_index": f, "kind": "subgoal",
                                   "source": names[i], "reason": None, "text": status})
            i = j
    ts = np.asarray([float(x) for x in rows["timestamp"]])
    ns = np.maximum.accumulate(np.round((ts - ts[0]) * 1e9).astype(np.int64)) if n else np.zeros(0, np.int64)
    for ev in events:
        ev["ns"] = int(ns[max(0, ev["frame_index"])])
    term = np.zeros(n, bool)
    trunc = np.zeros(n, bool)
    flag = TERMINAL_FLAG.get(reason)
    if flag == "terminated":
        term[-1] = True
    elif flag == "truncated":
        trunc[-1] = True
    fixed = lambda a: pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), pa.float32()), dim)
    table = pa.table({
        "frame_index": pa.array(np.arange(n, dtype=np.int64)),
        "control_mode": pa.array([str(x) for x in mode]),
        "expert_source": pa.array(expert_source, pa.string()),
        "blend_weight": pa.array(np.where(mode == "expert", 1.0, 0.0).astype(np.float32)),
        "base_action": fixed(base), "base_valid": pa.array(bv),
        "expert_action": fixed(expert), "expert_valid": pa.array(xv),
        "sent_action": fixed(sent), "safety_modified": pa.array(np.zeros(n, bool)),
        # imported data has no inference record; the base action is known without one
        "inference_id": pa.array(np.full(n, -1, np.int64)),
        "chunk_offset": pa.array(np.full(n, -1, np.int32)),
        "obs_ns": pa.array(ns), "obs_skew_ns": pa.array(np.full(n, -1, np.int64)), "sent_ns": pa.array(ns),
        "reward": pa.array(reward), "reward_valid": pa.array(rvalid),
        "terminated": pa.array(term), "truncated": pa.array(trunc),
    })
    from .logger import derive_subgoals, derive_takeovers
    tick_rows = [{"control_mode": str(mode[i]), "expert_source": expert_source[i]} for i in range(n)]
    policy = mp.get("policy")
    manifest = {
        "schema_version": SCHEMA_VERSION, "run_id": str(uuid.uuid4()), "status": "committed",
        "episode_index": int(e), "length": n, "task": ep["task"], "started_at": None, "ended_at": None,
        "termination_reason": reason, "task_success": None if reason == "unknown" else reason == "success",
        "labels": [], "failure_mode": failure_mode,
        "fingerprint": fingerprint(schema.id, ep["task"], sent),
        "events": events, "takeovers": derive_takeovers(tick_rows), "subgoals": derive_subgoals(events, n),
        "policy": {"family": None, "source": None, "revision": None, "code_revision": None, "config_digest": None,
                   "chunk_size": None, "temporal_ensembling": False, "device": None, "precision": None,
                   **(policy or {})} if policy else None,
        "env": {"robot_id": None, "calibration_digest": None, "operator": None, "extras": {},
                "software": {"lerobot": None, "dataop_logger": None}},  # filled below
        "action_schema": {**schema.to_dict(), "id": schema.id},
        "clock": {"source": "dataset timestamp", "obs_timestamp_kind": "from_dataset_timestamp"},
        "dataset": {"repo_id": None, "revision": None, "root": str(root)},
        "predictions_count": None,
        "rl_ready": bool(rvalid.all()) if rw else None,
        "batch": None,
        "origin": {"kind": "import", "fields": p["fields"], "notes": notes},
    }
    return manifest, table, notes


def _scalar(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return v[0] if len(v) else None
    return v


def import_dataset(source: str, into: str | None = None, preset: str | None = "plain", mapping: dict | None = None,
                   batch: str | None = None, episodes: list[int] | None = None, dry_run: bool = False,
                   by: str = "import") -> dict:
    """Bind runs to a LeRobot dataset's episodes. Returns a report; with dry_run nothing is written."""
    from .logger import _DATAOP_VERSION as __version__
    src, must_copy = resolve_source(source, into)
    _info(src)
    root = src
    if into:
        root = Path(into).expanduser()
        if not dry_run:
            if root.exists() and any(root.iterdir()):
                raise ImportError_(f"{root} exists and is not empty; pick a new folder for the copy")
            shutil.copytree(src, root, dirs_exist_ok=True, ignore=shutil.ignore_patterns("dataop"))
    elif must_copy:
        raise ImportError_("a Hub dataset is imported into a copy: pass --into <folder>")
    p = plan(src if dry_run else root, preset, mapping, episodes)
    p["root"] = str(src)                               # side-file patterns ({root}, {name}) refer to the source dataset
    report = {"dataset": str(root), "source": str(src), "fields": p["fields"], "missing_columns": p["missing_columns"],
              "to_bind": len(p["episodes"]), "skipped": p["skipped"], "dry_run": dry_run, "runs": []}
    results = {k: _results(p["mapping"][k], Path(p["root"])) if isinstance(p["mapping"].get(k), dict)
               and "file" in p["mapping"][k] else {} for k in ("outcome", "failure_mode")}
    for e in p["episodes"]:
        try:
            m, table, notes = _build_run(p, e, results)
        except ImportError_ as err:                   # one bad episode is reported, the rest still import
            report["skipped"].append({"episode": e, "why": str(err)})
            continue
        m["dataset"]["root"] = str(root)
        m["origin"].update(source=str(source), preset=preset, mapping_sha256=store.json_sha256(p["mapping"]),
                           imported_by=by, dataop_logger=__version__)
        report["runs"].append({"episode": e, "run_id": None if dry_run else m["run_id"], "outcome": m["termination_reason"],
                               "frames": m["length"], "policy_frames": int(sum(x == "base" for x in
                                                                             table.column("control_mode").to_pylist())),
                               "notes": notes})
        if dry_run:
            continue
        run_path = store.runs_dir(root) / m["run_id"]
        ticks = store.table_bytes(table)
        store.atomic_write_bytes(run_path / "ticks.parquet", ticks)
        m["integrity"] = {"ticks_sha256": store.sha256(ticks), "events_sha256": store.json_sha256(m["events"]),
                          "predictions_sha256": None}
        store.atomic_write_json(run_path / "manifest.json", m)
    if not dry_run and report["runs"]:
        store.rebuild_index(root)
        name = batch or f"import {Path(source).name}"
        B.add_runs(root, [r["run_id"] for r in report["runs"]], name, p["mapping"].get("policy"), by=by)
        report["batch"] = name
    return report
