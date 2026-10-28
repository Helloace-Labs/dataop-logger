"""Generate a training set from locked batches. A plain, deterministic script: no AI runs here.

  export(sources=[(root, batch)], recipe, into, ...) -> report

Every recipe works from one snapshot per run (the recorded ticks with the review applied, including moved drive-mode
boundaries) and decides, per frame, whether to keep it and which action to write:

  filtered   successful runs only; action = the command that was sent (plain behaviour cloning)
  dagger     frames with a valid expert label; action = expert_action. --with-success adds the policy's own
             successful runs as recorded. --base adds every episode of a demo dataset unchanged.
  residual   full-action supervised residual: frames with both base and expert actions;
             action = expert_action - base_action, plus feature observation.base_action (what the head corrects).
             --zero-on-success adds successful runs' policy frames where sent == base and no safety change, with
             target zero, so the head learns to leave good behaviour alone.

Shared rules: only frames the review keeps (dataop train_mask); optional task-segment and outcome filters; every run
is cut into maximal contiguous kept spans and each span becomes its own LeRobot episode (temporal sampling never
jumps a gap); transition features (`next.*`) are dropped because span ends are not true episode ends. The output is
written with LeRobot's own writer (videos re-encoded, stats recomputed). `val` > 0 writes a second dataset
`<into>_val` with whole runs chosen by a stable hash of the run id.

A hook (`module:function` or a .py file path with `:function`) may change the selection for task-specific rules:
it is called per run with a dict (manifest, effective review, ticks as numpy arrays, keep, action) and returns
{"keep": bool[n]?, "action": float32[n, dim]?, "reason": str?}. The record stores the hook's file digest and a digest
of what it returned.

`dataop_export.json` in the output records the source batches and their locks, each run's review digest and data
digests, the dataset lists digest, source LeRobot file digests, recipe and parameters, versions, per-run counts and
reasons, and a digest of the written data.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from . import batches as B
from . import events as ev
from . import lists as ls
from . import review as rv
from . import store

RECIPES = ("filtered", "dagger", "residual")
RESERVED = ("index", "episode_index", "frame_index", "timestamp", "task_index")


class ExportError(ValueError):
    pass


# ── one run as the review sees it ──────────────────────────────────────────────

def snapshot(run_path: Path, m: dict) -> dict:
    """Ticks with the review applied: effective drive modes, segments, outcome and the training mask."""
    t = pq.read_table(run_path / "ticks.parquet")
    n, dim = m["length"], len(m["action_schema"]["names"])
    arr = lambda c: np.asarray(t.column(c).to_pylist(), dtype=np.float32).reshape(n, dim)
    modes, sources = t.column("control_mode").to_pylist(), t.column("expert_source").to_pylist()
    review = rv.load(run_path)
    rv.check_review(m, review)
    eff = rv.effective(m, review)
    _, segs, _ = ev.major_events(m, review, modes, sources, with_segments=True)
    eff_modes = ev.effective_modes(m, review, modes, sources)
    return {"manifest": m, "review": review, "effective": eff, "n": n, "dim": dim,
            "base": arr("base_action"), "expert": arr("expert_action"), "sent": arr("sent_action"),
            "base_valid": np.asarray(t.column("base_valid").to_pylist(), bool),
            "expert_valid": np.asarray(t.column("expert_valid").to_pylist(), bool),
            "safety_modified": np.asarray(t.column("safety_modified").to_pylist(), bool),
            "mode": np.asarray(eff_modes, dtype=object), "segments": segs,
            "train_mask": rv.train_mask(m, review, eff_modes)}


# ── recipes: per frame keep + action, with a reason for every dropped frame ────

def select(s: dict, recipe: str, opts: dict, lists: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    n, eff = s["n"], s["effective"]
    outcome = "rejected" if eff["rejected"] else eff["termination_reason"]
    keep = s["train_mask"].copy()
    why = {"review": int((~keep).sum())}
    success = outcome == "success"
    action = s["sent"].copy()
    if recipe == "filtered":
        if not success:
            keep[:] = False
            why = {f"run {outcome}": n}
    elif recipe == "dagger":
        if success and opts.get("with_success"):
            pass                                          # the policy's own good run, as recorded
        else:
            lab = s["expert_valid"]
            why["no expert label"] = int((keep & ~lab).sum())
            keep &= lab
            action = np.where(lab[:, None], s["expert"], 0).astype(np.float32)
    elif recipe == "residual":
        both = s["base_valid"] & s["expert_valid"]
        target = np.where(both[:, None], s["expert"] - s["base"], 0).astype(np.float32)
        if success and opts.get("zero_on_success"):
            good = (s["mode"] == "base") & s["base_valid"] & ~s["safety_modified"] & \
                   np.all(s["sent"] == s["base"], axis=1)
            target = np.where(good[:, None], 0, target).astype(np.float32)
            both = both | good
        why["no base and expert action"] = int((keep & ~both).sum())
        keep &= both
        action = target
    else:
        raise ExportError(f"recipe must be one of {RECIPES}")
    outs = opts.get("outcomes")
    if outs and outcome not in outs:
        why = {f"run {outcome}": n}
        keep[:] = False
    names = opts.get("segments")
    if names:
        want = {ls.canonical(lists, "task_segments", x) for x in names}
        inside = np.zeros(n, bool)
        for g in s["segments"]:
            if ls.canonical(lists, "task_segments", g["name"]) in want:
                inside[g["start"]:g["end"]] = True
        why["outside the chosen task segments"] = int((keep & ~inside).sum())
        keep &= inside
    return keep, action, {k: v for k, v in why.items() if v}


def spans(keep: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Maximal contiguous runs of kept frames, [start, end)."""
    out, i, n = [], 0, len(keep)
    while i < n:
        if keep[i]:
            j = i
            while j < n and keep[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _load_hook(spec: str | None):
    if not spec:
        return None, None
    mod, _, fn = spec.rpartition(":")
    if not mod or not fn:
        raise ExportError("a hook is module:function or path/to/file.py:function")
    if mod.endswith(".py"):
        path = Path(mod).expanduser().resolve()
        sp = importlib.util.spec_from_file_location(f"dataop_hook_{path.stem}", path)
        module = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(module)
    else:
        module = importlib.import_module(mod)
        path = Path(module.__file__)
    return getattr(module, fn), store.sha256(path.read_bytes())


# ── reading source frames (generic: any features, video or not) ───────────────

class _Source:
    def __init__(self, root: Path, drop: tuple = ()):
        self.root = Path(root)
        self.drop = set(drop)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.feats = self.info["features"]
        self.videos = [k for k, v in self.feats.items() if v.get("dtype") == "video"]
        from .adapter import episode_metadata
        self.meta = episode_metadata(self.root)
        import pandas as pd
        self.eps = pd.concat([pd.read_parquet(f) for f in sorted((self.root / "meta" / "episodes").rglob("*.parquet"))])
        self.dset = pads.dataset(str(self.root / "data"), format="parquet")

    def out_features(self) -> dict:
        out = {}
        for k, v in self.feats.items():
            if k in RESERVED or k.startswith("next.") or k in self.drop:
                continue
            out[k] = {kk: vv for kk, vv in v.items() if kk != "info"}
        return out

    def episode(self, e: int) -> dict:
        cols = [k for k in self.feats if k not in RESERVED and not k.startswith("next.") and k not in self.drop
                and self.feats[k].get("dtype") != "video"]
        t = self.dset.to_table(columns=["frame_index", *cols], filter=pads.field("episode_index") == e).sort_by("frame_index")
        data = {c: t.column(c).to_pylist() for c in cols}
        row = self.eps[self.eps["episode_index"] == e].iloc[0]
        n = int(row["length"])
        for k in self.videos:
            if k not in self.drop:
                data[k] = self._frames(k, row, n)
        return {"n": n, "data": data}

    def _frames(self, key: str, row, n: int) -> list:
        import av
        c, fi = int(row[f"videos/{key}/chunk_index"]), int(row[f"videos/{key}/file_index"])
        t0 = float(row[f"videos/{key}/from_timestamp"])
        path = self.root / self.info["video_path"].format(video_key=key, chunk_index=c, file_index=fi)
        out = []
        with av.open(str(path)) as f:
            s = f.streams.video[0]
            f.seek(int(max(0.0, t0 - 1) / s.time_base), stream=s)
            for fr in f.decode(s):
                if float(fr.pts * s.time_base) < t0 - 1e-3:
                    continue
                out.append(fr.to_ndarray(format="rgb24"))
                if len(out) == n:
                    break
        if len(out) != n:
            raise ExportError(f"{path}: decoded {len(out)} frames for an episode of {n}")
        return out

    def frame(self, ep: dict, i: int, keys: list[str]) -> dict:
        out = {}
        for k in keys:
            v = ep["data"][k][i]
            dt = self.feats[k].get("dtype")
            if dt == "image" and isinstance(v, dict):
                import io
                from PIL import Image
                v = np.asarray(Image.open(io.BytesIO(v["bytes"])).convert("RGB"))
            elif dt not in ("video", "image", "string"):
                v = np.asarray(v, dtype=dt).reshape(self.feats[k]["shape"])
            out[k] = v
        return out


def _digest_files(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.name).encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def tree_digest(root: Path, skip: tuple = ()) -> str:
    """sha256 over every file under root (relative path and bytes, in path order), except the names in skip."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name not in skip):
        h.update(str(p.relative_to(root)).encode() + b"\0")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _source_digest(root: Path) -> dict:
    """The LeRobot files the export read: all metadata, data and videos (the dataop sidecar is digested per run)."""
    root = Path(root)
    return {k: tree_digest(root / k) if (root / k).exists() else None for k in ("meta", "data", "videos")}


def _split(run_id: str, val: float) -> str:
    """Stable train/val assignment: sha256 of the run id as a number in [0, 1)."""
    x = int(hashlib.sha256(run_id.encode()).hexdigest()[:12], 16) / float(16 ** 12)
    return "val" if x < val else "train"


# ── export ─────────────────────────────────────────────────────────────────────

def export(sources: list[tuple[str, str]], recipe: str, into: str, *, with_success: bool = False,
           zero_on_success: bool = False, base: str | None = None, segments: list[str] | None = None,
           outcomes: list[str] | None = None, val: float = 0.0, min_span: int = 1, hook: str | None = None,
           allow_unlocked: bool = False, dry_run: bool = False, repo_id: str | None = None, by: str = "human",
           drop_features: list[str] | None = None) -> dict:
    if recipe not in RECIPES:
        raise ExportError(f"recipe must be one of {RECIPES}")
    if not 0.0 <= val < 1.0:
        raise ExportError("val must be in [0, 1)")
    if base and recipe == "residual":
        raise ExportError("--base adds demos to dagger or filtered exports; a residual set has no base demos")
    out_root = Path(into).expanduser()
    targets = {"train": out_root, **({"val": out_root.with_name(out_root.name + "_val")} if val > 0 else {})}
    if not dry_run:
        for p in targets.values():
            if p.exists():
                raise ExportError(f"{p} already exists; export never overwrites, pick a new folder")
    opts = {"with_success": with_success, "zero_on_success": zero_on_success, "segments": segments,
            "outcomes": outcomes, "min_span": min_span}
    hook_fn, hook_sha = _load_hook(hook)
    drop = tuple(drop_features or ())
    if "action" in drop:
        raise ExportError("the action feature cannot be dropped")

    # 1. resolve sources and check they can be combined
    plan_runs, src_records, first, planned_locks = [], [], None, {}
    for root, ref in sources:
        root = Path(root).expanduser()
        d = B.load(root)
        b = B.find(d, ref)
        planned_locks[(str(root), b["batch_id"])] = b["locked"] if b["status"] == "locked" else None
        if b["status"] != "locked" and not allow_unlocked:
            raise ExportError(f"batch {b['name']!r} in {root} is {b['status']}; lock it first (or allow unlocked)")
        src = _Source(root, drop)
        sig = (src.info["fps"], src.info.get("robot_type"),
               json.dumps(src.out_features(), sort_keys=True))
        if first is None:
            first = (sig, src)
        elif sig != first[0]:
            raise ExportError(f"{root} has a different fps, robot type or features from {first[1].root} "
                              f"(features only in one: {sorted(set(src.out_features()) ^ set(first[1].out_features()))}); "
                              "export them separately or leave extras out with drop_features")
        lists = ls.load(root)
        manifests = {p.name: (p, m) for p, m in store.iter_manifests(root, "committed")}
        members = sorted(r for r, x in d["members"].items() if x == b["batch_id"])
        for rid in members:
            p, m = manifests[rid]
            s = snapshot(p, m)
            if b["locked"]:
                lk = b["locked"]["runs"].get(rid)
                if lk is None or lk["review_digest"] != B.review_digest(m, s["review"]):
                    raise ExportError(f"run {rid} changed after batch {b['name']!r} was locked; run validate")
            plan_runs.append((root, src, lists, b, p, m, s))
        src_records.append({"root": str(root), "batch_id": b["batch_id"], "batch": b["name"], "status": b["status"],
                            "locked": b["locked"] and {"at": b["locked"]["at"], "by": b["locked"]["by"],
                                                       "forced": b["locked"].get("forced", False)},
                            "lists_sha256": store.json_sha256(lists),
                            "lerobot": None if dry_run else _source_digest(root)})
    if first is None:
        raise ExportError("no sources given")
    base_src = None
    if base:
        base_src = _Source(Path(base).expanduser(), drop)
        if (base_src.info["fps"], base_src.info.get("robot_type"), json.dumps(base_src.out_features(), sort_keys=True)) \
                != first[0]:
            raise ExportError(f"the base dataset {base} has a different fps, robot type or features "
                              f"(features only in one: {sorted(set(base_src.out_features()) ^ set(first[1].out_features()))}; "
                              "leave extras out with drop_features)")

    # 2. decide frames per run
    runs_out, hook_results = [], []
    for root, src, lists, b, p, m, s in plan_runs:
        keep, action, why = select(s, recipe, opts, lists)
        if hook_fn is not None:
            res = hook_fn({"manifest": m, "effective": s["effective"], "ticks": {k: s[k] for k in (
                "base", "expert", "sent", "base_valid", "expert_valid", "safety_modified", "mode")},
                "keep": keep.copy(), "action": action.copy(), "recipe": recipe}) or {}
            if "keep" in res:
                k2 = np.asarray(res["keep"], bool)
                if k2.shape != keep.shape:
                    raise ExportError(f"hook returned keep of shape {k2.shape} for run {m['run_id']}")
                why["hook" + (f": {res['reason']}" if res.get("reason") else "")] = int((keep & ~k2).sum())
                keep = k2
            if "action" in res:
                a2 = np.asarray(res["action"], np.float32)
                if a2.shape != action.shape or not np.all(np.isfinite(a2[keep])):
                    raise ExportError(f"hook returned bad actions for run {m['run_id']}")
                action = a2
            hook_results.append(store.sha256(keep.tobytes() + action.tobytes()))
        sp = spans(keep, min_span)
        dropped_short = int(keep.sum()) - sum(b_ - a for a, b_ in sp)
        if dropped_short:
            why["span shorter than min_span"] = dropped_short
        split = _split(m["run_id"], val) if val > 0 else "train"
        runs_out.append({"root": str(root), "batch": b["name"], "run_id": m["run_id"], "episode": m["episode_index"],
                         "outcome": "rejected" if s["effective"]["rejected"] else s["effective"]["termination_reason"],
                         "split": split, "frames": m["length"], "kept": sum(b_ - a for a, b_ in sp), "spans": sp,
                         "dropped": why, "review_rev": s["review"]["rev"],
                         "review_digest": B.review_digest(m, s["review"]),
                         "ticks_sha256": m["integrity"]["ticks_sha256"], "manifest_sha256": store.json_sha256(m),
                         "_action": action, "_src": src, "_task": m["task"]})
    report = {"recipe": recipe, "options": {**opts, "base": base, "val": val, "hook": hook, "drop_features": list(drop)},
              "sources": src_records, "dry_run": dry_run,
              "episodes": sum(len(r["spans"]) for r in runs_out) + (len(base_src.eps) if base_src is not None else 0),
              "frames": sum(r["kept"] for r in runs_out) +
                        (int(base_src.eps["length"].sum()) if base_src is not None else 0),
              "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs_out]}
    if dry_run:
        return report

    # 3. write with LeRobot's own writer
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    src0 = first[1]
    feats = src0.out_features()
    keys = [k for k in feats]
    dim = int(np.prod(feats["action"]["shape"]))
    if recipe == "residual":
        feats["observation.base_action"] = {"dtype": "float32", "shape": [dim], "names": feats["action"].get("names")}
    written = {}
    try:
        for split, path in targets.items():
            ds = LeRobotDataset.create(repo_id or f"dataop/{path.name}", fps=int(src0.info["fps"]), features=feats,
                                       root=path, robot_type=src0.info.get("robot_type"), use_videos=bool(src0.videos))
            n_ep = 0
            if split == "train" and base_src is not None:
                for e in sorted(int(x) for x in base_src.eps["episode_index"]):
                    ep = base_src.episode(e)
                    task = base_src.meta[e]["tasks"][0] if base_src.meta[e]["tasks"] else "task"
                    for i in range(ep["n"]):
                        fr = base_src.frame(ep, i, keys)
                        if recipe == "residual":
                            fr["observation.base_action"] = np.zeros(dim, np.float32)
                        ds.add_frame({**fr, "task": task})
                    ds.save_episode(parallel_encoding=False)
                    n_ep += 1
            for r in runs_out:
                if r["split"] != split or not r["spans"]:
                    continue
                ep = r["_src"].episode(r["episode"])
                snap = next(x for x in plan_runs if x[5]["run_id"] == r["run_id"])[6]
                for a, b_ in r["spans"]:
                    for i in range(a, b_):
                        fr = r["_src"].frame(ep, i, keys)
                        fr["action"] = r["_action"][i].reshape(feats["action"]["shape"]).astype(np.float32)
                        if recipe == "residual":
                            fr["observation.base_action"] = snap["base"][i].astype(np.float32)
                        ds.add_frame({**fr, "task": r["_task"]})
                    ds.save_episode(parallel_encoding=False)
                    n_ep += 1
            ds.finalize()
            written[split] = {"path": str(path), "episodes": n_ep,
                              "tree_sha256": tree_digest(path, skip=("dataop_export.json",))}
    except BaseException:
        for p in targets.values():                      # never leave a half-written training set behind
            shutil.rmtree(p, ignore_errors=True)
        raise
    export_id = uuid.uuid4().hex[:12]
    from .logger import _DATAOP_VERSION, _lerobot_version
    record = {**report, "export_id": export_id, "at": B._now(), "by": by, "written": written,
              "versions": {"dataop_logger": _DATAOP_VERSION, "lerobot": _lerobot_version()},
              "hook": hook and {"spec": hook, "file_sha256": hook_sha,
                                "results_sha256": store.json_sha256(hook_results)},
              "split": val > 0 and {"method": "sha256(run_id)[:12] / 16^12 < val", "val": val}}
    regs = []
    for root, ref in sources:
        b = B.find(B.load(Path(root).expanduser()), ref)
        regs.append((Path(root).expanduser(), b["batch_id"], planned_locks[(str(Path(root).expanduser()), b["batch_id"])]))
    try:                                            # one decision for every source: current only if all kept their lock
        stale = B.add_export(regs, {"export_id": export_id, "at": record["at"], "by": by, "recipe": recipe,
                                    "path": str(out_root), "episodes": written["train"]["episodes"],
                                    "frames": report["frames"], "options": report["options"]})
    except BaseException:
        for split, path in targets.items():         # keep the written data, but say it was never registered
            store.atomic_write_json(path / "dataop_export.json", {**record, "this_split": split, "registered": False,
                                                                  "stale": True})
        raise
    record["stale"], record["registered"] = stale, True
    for split, path in targets.items():             # the same record in every split, naming which one this is
        store.atomic_write_json(path / "dataop_export.json", {**record, "this_split": split})
    report.update(export_id=export_id, written=written, stale=stale)
    return report
