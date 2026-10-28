"""dataop-logger view: a local, read-only page for browsing logged runs next to their videos."""
from __future__ import annotations

import functools
import re
import secrets
import http.server
import json
import os
import threading
import webbrowser
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from urllib.parse import unquote

from . import batches as B
from . import export as X
from . import importer as I
from . import workspace as W
from . import events as ev
from . import lists as ls
from . import review as rv
from . import store


def _segments(modes, sources) -> list[list]:
    """Run-length encode the control mode per frame: [[start, end, mode, source], ...]."""
    out = []
    for i, (m, s) in enumerate(zip(modes, sources)):
        if out and out[-1][2] == m and out[-1][3] == s:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1, m, s])
    return out


def _video_table(root: Path) -> dict[int, dict]:
    d = root / "meta" / "episodes"
    info_p = root / "meta" / "info.json"
    if not d.exists() or not info_p.exists():
        return {}
    info = json.loads(info_p.read_text())
    vpath = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    ds = pads.dataset(str(d), format="parquet")
    keys = sorted({c.split("/")[1] for c in ds.schema.names if c.startswith("videos/") and c.endswith("/chunk_index")})
    cols = ["episode_index"] + [f"videos/{k}/{f}" for k in keys for f in ("chunk_index", "file_index", "from_timestamp")]
    t = ds.to_table(columns=cols).to_pydict()
    out = {}
    for i, e in enumerate(t["episode_index"]):
        out[int(e)] = [{"key": k.removeprefix("observation.images."),
                        "src": vpath.format(video_key=k, chunk_index=int(t[f"videos/{k}/chunk_index"][i]),
                                            file_index=int(t[f"videos/{k}/file_index"][i])),
                        "t0": float(t[f"videos/{k}/from_timestamp"][i])}
                       for k in keys if t[f"videos/{k}/chunk_index"][i] is not None]
    return out


RUN_ID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _modes(run_path: Path) -> tuple[list, list]:
    t = pq.read_table(run_path / "ticks.parquet", columns=["control_mode", "expert_source"])
    return t.column("control_mode").to_pylist(), t.column("expert_source").to_pylist()


def _review_state(run_path: Path, m: dict) -> dict:
    modes, sources = _modes(run_path)
    try:
        r = rv.load(run_path)
        return {"edits": r["edits"], "rev": r["rev"], "effective": rv.effective(m, r), "undoable": rv.undoable(r),
                **dict(zip(("major", "segments", "drive"), ev.major_events(m, r, modes, sources, with_segments=True))),
                "error": None}
    except rv.ReviewError as e:                        # shown on the run; editing it is disabled
        return {"edits": [], "rev": -1, "effective": rv.effective(m, rv.empty()), "undoable": [],
                **dict(zip(("major", "segments", "drive"), ev.major_events(m, rv.empty(), modes, sources, with_segments=True))),
                "error": str(e)}


def collect(root: Path) -> dict:
    root = Path(root)
    videos = _video_table(root)
    runs = []
    for run_path, m in store.iter_manifests(root, "committed"):
        if not RUN_ID.match(str(m["run_id"])) or run_path.name != m["run_id"]:
            continue                                   # validate reports it; never render an unexpected id
        t = pq.read_table(run_path / "ticks.parquet",
                          columns=["control_mode", "expert_source", "safety_modified", "obs_ns", "sent_ns"])
        modes = t.column("control_mode").to_pylist()
        sources = t.column("expert_source").to_pylist()
        lat = (t.column("sent_ns").to_numpy() - t.column("obs_ns").to_numpy()) / 1e6     # ms, frame -> command
        policy, env = m.get("policy") or {}, m.get("env") or {}
        runs.append({
            "run_id": m["run_id"], "episode": m["episode_index"], "task": m["task"], "length": m["length"],
            "fps": m["action_schema"]["rate_hz"], "outcome": m["termination_reason"],
            "task_success": m.get("task_success"), "labels": m.get("labels", []),
            "failure_mode": m.get("failure_mode"), "subgoals": m.get("subgoals", []),
            "takeovers": m.get("takeovers", []),
            "events": [{"frame": e["frame_index"], "kind": e["kind"], "source": e.get("source"),
                        "reason": e.get("reason"), "text": e.get("text")} for e in m.get("events", [])
                       if e["kind"] != "subgoal"],
            "segments": _segments(modes, sources),
            "safety_modified": int(sum(t.column("safety_modified").to_pylist())),
            "policy": policy.get("source"), "policy_revision": policy.get("revision"),
            "robot": env.get("robot_id"), "operator": env.get("operator"),
            "conditions": env.get("extras", {}),
            # frames whose command time equals the frame time carry no timing (imports, synthetic clocks)
            "latency_ms": {"p50": round(float(np.percentile(lat[lat > 0], 50)), 1) if (lat > 0).any() else None,
                           "p95": round(float(np.percentile(lat[lat > 0], 95)), 1) if (lat > 0).any() else None,
                           "measured": bool((lat > 0).any()), "timed": int((lat > 0).sum()), "frames": int(len(lat))},
            "started_at": m.get("started_at"),
            "time_estimated": bool((m.get("origin") or {}).get("time")),
            "imported": (m.get("origin") or {}).get("kind") == "import",
            "videos": videos.get(m["episode_index"], []),
            "review": _review_state(run_path, m),
        })
    runs.sort(key=lambda r: r["episode"])
    modes = sorted({r["failure_mode"] for r in runs if r["failure_mode"]} |
                   {e["payload"].get("failure_mode") for r in runs for e in r["review"]["edits"]
                    if e["kind"] == "outcome" and e["payload"].get("failure_mode")})
    try:
        lists = ls.load(root)
    except ls.ListError as e:
        lists = {"version": 1, "rev": -1, "failure_types": [], "task_segments": [], "error": str(e)}
    return {"dataset": root.name, "root": str(root), "runs": runs, "failure_modes": modes, "lists": lists,
            "editable": False, "datasets": W.datasets(), **_batch_state(root)}


def _pickable(root: Path, current: bool = False) -> dict:
    """One dataset's batches and runs, for choosing what goes into a training set."""
    d = B.load(root)
    runs = []
    for p, m in store.iter_manifests(root, "committed"):
        if m["run_id"] not in d["members"]:
            continue
        modes = pq.read_table(p / "ticks.parquet", columns=["control_mode"]).column("control_mode").to_pylist()
        try:
            r = rv.load(p)
            e = rv.effective(m, r)
            outcome, reviewed = ("rejected" if e["rejected"] else e["termination_reason"]), bool(r.get("reviewed"))
        except rv.ReviewError:
            outcome, reviewed = m["termination_reason"], False
        runs.append({"run_id": m["run_id"], "episode": m["episode_index"], "batch_id": d["members"][m["run_id"]],
                     "outcome": outcome, "reviewed": reviewed, "length": m["length"],
                     "fps": m["action_schema"]["rate_hz"],
                     "drove": "teleop" if "base" not in modes and "expert" in modes
                              else "rescued" if "expert" in modes else "policy" if "base" in modes else "unknown"})
    runs.sort(key=lambda x: x["episode"])
    return {"root": str(root), "name": root.name, "current": current, "runs": runs,
            "batches": [{"batch_id": b["batch_id"], "name": b["name"], "status": b["status"]}
                        for b in reversed(d["batches"])]}


def _batch_state(root: Path) -> dict:
    """Batches newest first with their runs, and the revision the page sends back with batch changes."""
    try:
        bs = B.summary(root)
        return {"batches": bs, "batches_rev": B._read(root)["rev"], "batches_error": None,
                "training_sets": X.training_sets(root)}
    except (B.BatchError, OSError) as e:
        return {"batches": [], "batches_rev": -1, "batches_error": str(e), "training_sets": []}


def build(root: Path, editable: bool = False) -> Path:
    root = Path(root)
    data = collect(root)
    data["editable"] = editable
    here = Path(__file__).parent
    js = (here / "viewer.js").read_text().replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/"))
    nonce = secrets.token_urlsafe(16)
    html = (here / "viewer.html").read_text().replace("__NONCE__", nonce).replace(
        '<script src="viewer.js"></script>', f'<script nonce="{nonce}">\n' + js + "</script>")
    out = store.dataop_dir(root) / "view" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


API = "/dataop/api/review"
_JOB: dict = {}                     # the one long job the page started (an import or a training set), if any


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "batch"
MAX_BODY = 64 * 1024


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with HTTP Range support, so browsers can seek inside the episode videos, plus the review API."""

    @property
    def dataset_root(self) -> Path:          # the dataset the page shows; the viewer can switch it (ws_open)
        return self.server.dataop_state["root"]

    def translate_path(self, path):          # static files (videos) come from whichever dataset is open
        self.directory = str(self.dataset_root)
        return super().translate_path(path)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        # Never inside another site's frame (clickjacking); the page's own CSP is in a meta tag with a nonce.
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    editable: bool = True

    def do_GET(self):
        if self.path.split("?")[0] in ("/dataop/view/", "/dataop/view/index.html"):
            with _WRITE_LOCK:                          # rebuilt on every load so a refresh shows saved edits
                build(self.dataset_root, editable=self.editable)
        return super().do_GET()

    def do_POST(self):
        # Only the page we served may write: a custom header forces a CORS preflight that this server never
        # answers, and Host/Origin must be this server, which blocks other sites and DNS rebinding.
        host = f"127.0.0.1:{self.server.server_address[1]}"
        origin = self.headers.get("Origin")
        if (self.path != API or self.headers.get("X-Dataop") != "1" or self.headers.get("Host") != host
                or (origin is not None and origin != f"http://{host}")
                or not (self.headers.get("Content-Type") or "").startswith("application/json")):
            return self._json(403, {"error": "forbidden"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 < n <= MAX_BODY:
            return self._json(413, {"error": "body missing or too large"})
        try:
            req = json.loads(self.rfile.read(n))
            if not isinstance(req, dict):
                raise ValueError("request must be an object")
            op = req.get("op")
            shown = self.headers.get("X-Dataop-Root")
            if shown is not None and unquote(shown) != str(self.dataset_root) and not str(op).startswith("ws_") \
                    and op != "job_status":
                return self._json(409, {"error": "this page shows another dataset; reload the page",
                                        "reload": True})
            if isinstance(op, str) and (op.startswith(("ws_", "import_", "train_")) or op == "job_status"):
                return self._json(200, self._long(op, req))
            if isinstance(op, str) and op.startswith("lists_"):      # the dataset's lists, not one run
                root, kind, rev = self.dataset_root, req.get("kind"), req.get("rev")
                if isinstance(rev, bool) or not isinstance(rev, int):
                    raise rv.ReviewError("rev (the lists revision the page shows) is required")
                if op == "lists_add":
                    d = ls.add(root, kind, req.get("name"), req.get("description") or "", expect_rev=rev)
                elif op == "lists_rename":
                    d = ls.rename(root, kind, req.get("old"), req.get("new"), expect_rev=rev)
                elif op == "lists_describe":
                    d = ls.describe(root, kind, req.get("name"), req.get("description") or "", expect_rev=rev)
                elif op == "lists_delete":
                    d = ls.delete(root, kind, req.get("name"), expect_rev=rev)
                else:
                    raise rv.ReviewError("unknown lists operation")
                return self._json(200, {"lists": d})
            if isinstance(op, str) and op.startswith("batch_"):      # a batch of runs, not one run
                rev = req.get("rev")
                if isinstance(rev, bool) or not isinstance(rev, int):
                    raise rv.ReviewError("rev (the batches revision the page shows) is required")
                if op == "batch_status":
                    status = req.get("status")
                    cur = B.find(B.load(self.dataset_root), req.get("batch"))
                    if (cur["status"] == "locked") != (status == "unlock"):
                        raise rv.ReviewError("unlock a locked batch before changing it" if cur["status"] == "locked"
                                             else "this batch is not locked")
                    B.set_status(self.dataset_root, req.get("batch"),
                                 "reviewing" if status == "unlock" else status, by="person",
                                 reason=req.get("reason"), expect_rev=rev)
                elif op == "batch_new":
                    B.new(self.dataset_root, req.get("name"), by="person", expect_rev=rev)
                else:
                    raise rv.ReviewError("unknown batch operation")
                return self._json(200, _batch_state(self.dataset_root))
            run_path, m = _find_run(self.dataset_root, req.get("run_id"))
            rev = req.get("rev")
            if isinstance(rev, bool) or not isinstance(rev, int):
                raise rv.ReviewError("rev (the review revision the page shows) is required")
            op, summary = req.get("op"), None
            what = req.get("what") if isinstance(req.get("what"), str) and 0 < len(req.get("what")) <= 120 else op
            if op == "undo":
                entry = rv.undo(run_path, m, req.get("entry_id"), expect_rev=rev)
                state = _review_state(run_path, m)
                state["undone"] = {"entry": entry["id"], "what": entry["what"]}
                return self._json(200, state)
            with rv.recording(what) as rec:             # one undoable step (Cmd/Ctrl+Z in the page)
                summary = self._apply(op, req, run_path, m, rev)
        except rv.ReviewConflict as e:
            return self._json(409, {"error": str(e), "state": _review_state(run_path, m)})
        except (X.ExportError, I.ImportError_) as e:
            return self._json(400, {"error": str(e)})
        except B.BatchError as e:
            return self._json(409 if "changed" in str(e) else 400, {"error": str(e), **_batch_state(self.dataset_root)})
        except ls.ListError as e:
            code = 409 if "changed" in str(e) else 400
            try:
                cur = ls.load(self.dataset_root)
            except ls.ListError:
                cur = None
            return self._json(code, {"error": str(e), "lists": cur})
        except (rv.ReviewError, ValueError, KeyError, TypeError) as e:
            return self._json(400, {"error": str(e)})
        state = _review_state(run_path, m)
        if op == "find_idle":
            state["found"] = summary
        if rec["entry"]:
            state["step"] = {"entry": rec["entry"], "what": rec["what"]}
        return self._json(200, state)

    def _long(self, op, req):
        """Datasets (ws_*), import (import_*) and training sets (train_*). Import and training-set writes run in the
        background (download, video re-encode); the page polls job_status."""
        root = self.dataset_root
        if op == "job_status":
            with _WRITE_LOCK:
                return {"job": dict(_JOB) if _JOB else None, **_batch_state(root)}
        if op == "ws_list":
            return {"datasets": W.datasets(), "current": str(root)}
        if op == "ws_open":
            want = str(Path(str(req.get("root") or "")).expanduser().resolve())
            if want not in {d["root"] for d in W.datasets()}:
                raise rv.ReviewError("that dataset is not in this viewer's list; import it first")
            with _WRITE_LOCK:
                self.server.dataop_state["root"] = Path(want)
            return {"current": want}
        if op == "import_scan":                          # LeRobot datasets on this machine, with a guessed type
            out = []
            for c in W.candidates([root.parent]):
                try:
                    c["preset"] = I.detect(Path(c["root"])) if c["version"] == "v3.0" else None
                except I.ImportError_:
                    c["preset"] = None
                out.append(c)
            return {"candidates": out, "presets": I.PRESET_LABELS}
        if op.startswith("import_"):
            source = str(req.get("source") or "").strip()
            if not source:
                raise I.ImportError_("choose a dataset folder or type a Hub repo id (owner/name)")
            local = Path(source).expanduser().exists()
            preset = req.get("preset") or "auto"
            into = req.get("into") or (None if local else str(root.parent / source.replace("/", "_")))
            batch = (req.get("batch") or "").strip() or None
            if op == "import_preview":
                if not local:
                    return {"preview": {"hub": True, "into": into, "source": source}}
                if preset == "auto":
                    preset = I.detect(Path(source).expanduser())
                rep = I.import_dataset(source, into=into, preset=preset, batch=batch, dry_run=True, by="person")
                oc = {}
                for r in rep["runs"]:
                    oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
                return {"preview": {"hub": False, "source": source, "into": into, "preset": preset,
                                    "episodes": rep["to_bind"], "frames": sum(r["frames"] for r in rep["runs"]),
                                    "policy_frames": sum(r["policy_frames"] for r in rep["runs"]),
                                    "outcomes": oc, "fields": rep["fields"], "skipped": len(rep["skipped"]),
                                    "skipped_why": sorted({x["why"] for x in rep["skipped"]})[:3],
                                    "batch": batch or f"import {Path(source).name}"}}
            if op != "import_start":
                raise rv.ReviewError("unknown import operation")

            def work():
                rep = I.import_dataset(source, into=into, preset=preset, batch=batch, by="person")
                W.remember(Path(rep["dataset"]))
                return {"root": str(Path(rep["dataset"]).resolve()), "runs": len(rep["runs"]),
                        "skipped": len(rep["skipped"]), "batch": rep.get("batch"),
                        "same": Path(rep["dataset"]).resolve() == root}
            return self._start("import", {"source": source}, work)
        # training sets: the page sends the chosen runs (from any dataset in the list), grouped here by batch
        if op == "train_list":
            return {"training_sets": X.training_sets(root)}
        if op == "train_runs":
            out = []
            for d in W.datasets():
                try:
                    out.append(_pickable(Path(d["root"]), current=Path(d["root"]) == root))
                except (B.BatchError, OSError, ValueError) as e:
                    out.append({"root": d["root"], "name": d["name"], "error": str(e), "batches": [], "runs": []})
            return {"datasets": out}
        recipe = req.get("recipe")
        if recipe not in X.RECIPES:
            raise X.ExportError(f"choose a training set type ({', '.join(X.RECIPES)})")
        picks = req.get("runs")
        if not isinstance(picks, list) or not picks or not all(
                isinstance(x, dict) and isinstance(x.get("root"), str) and isinstance(x.get("run_id"), str) for x in picks):
            raise X.ExportError("choose at least one run")
        known = {d["root"] for d in W.datasets()}
        by_root = {}
        for x in picks:
            r = str(Path(x["root"]).expanduser().resolve())
            if r not in known:
                raise X.ExportError(f"{x['root']} is not in this viewer's dataset list")
            by_root.setdefault(r, set()).add(x["run_id"])
        srcs, names = [], []
        for r, ids in sorted(by_root.items()):
            d = B.load(Path(r))
            missing = [i for i in ids if i not in d["members"]]
            if missing:
                raise X.ExportError(f"{len(missing)} chosen runs in {Path(r).name} are in no batch; add them to one first")
            for bid in sorted({d["members"][i] for i in ids}):
                srcs.append((r, bid))
                names.append(B.find(d, bid)["name"])
        drop = X.extra_features(sorted(by_root))
        opts = {"with_success": bool(req.get("with_success")), "zero_on_success": bool(req.get("zero_on_success")),
                "allow_unlocked": True, "runs": sorted({i for ids in by_root.values() for i in ids}),
                "drop_features": drop or None}
        import datetime as _dt
        label = "_".join(_slug(x) for x in names[:3]) + ("_and_more" if len(names) > 3 else "")
        into = req.get("into") or str(root.parent / f"{label}_{recipe}_{_dt.datetime.now():%Y%m%d_%H%M}")
        if op == "train_preview":
            rep = X.export(srcs, recipe, into, dry_run=True, by="person", **opts)
            why = {}
            for r in rep["runs"]:
                for k, v in r["dropped"].items():
                    why[k] = why.get(k, 0) + v
            return {"preview": {"into": into, "episodes": rep["episodes"], "frames": rep["frames"],
                                "dropped": sorted(why.items(), key=lambda kv: -kv[1])[:3],
                                "runs_used": sum(1 for r in rep["runs"] if r["kept"]), "runs": len(rep["runs"]),
                                "teleop_frames": sum(r["kept"] for r in rep["runs"] if r.get("as") == "teleop"),
                                "exists": Path(into).exists(), "batches": names, "left_out_columns": drop}}
        if op != "train_start":
            raise rv.ReviewError("unknown training set operation")
        if Path(into).exists():
            raise X.ExportError(f"{into} already exists; choose another folder")

        def work():
            rep = X.export(srcs, recipe, into, by="person", **opts)
            return {"into": into, "episodes": rep["written"]["train"]["episodes"], "frames": rep["frames"],
                    "stale": rep["stale"]}
        return self._start("train", {"recipe": recipe, "into": into}, work)

    def _start(self, kind, info, fn):
        with _WRITE_LOCK:
            if _JOB and _JOB.get("state") == "running":
                raise rv.ReviewError(f"a {_JOB['kind'] == 'import' and 'import' or 'training set'} is still running")
            _JOB.clear()
            _JOB.update(kind=kind, state="running", error=None, result=None, **info)

        def run():
            try:
                res = fn()
                with _WRITE_LOCK:
                    _JOB.update(state="done", result=res)
            except Exception as e:                      # shown in the dialog; nothing half-written is left
                with _WRITE_LOCK:
                    _JOB.update(state="failed", error=str(e) if isinstance(e, (X.ExportError, I.ImportError_))
                                else f"{type(e).__name__}: {e}")
        threading.Thread(target=run, daemon=True).start()
        return {"job": dict(_JOB)}

    def _apply(self, op, req, run_path, m, rev):
        summary = None
        if op == "add":
            if req.get("kind") not in ("label", "exclude", "failure_start", "step_boundary", "segment_edge", "drive_boundary", "outcome",
                                   "segment_plan"):
                raise rv.ReviewError("the viewer adds labels, cuts and the policy-failure moment only")
            payload = req.get("payload", {})
            rv.add_edit(run_path, m, req.get("kind"), payload, source="human", expect_rev=rev)
        elif op == "note":
            rv.add_event_note(run_path, m, req.get("key"), req.get("text"), by="person", source="human", expect_rev=rev)
        elif op == "decide":
            rv.decide(run_path, m, req.get("edit_id"), req.get("accept"), req.get("payload"), expect_rev=rev)
        elif op == "decide_many":
            rv.decide_many(run_path, m, req.get("edit_ids"), req.get("accept"), expect_rev=rev)
        elif op == "find_idle":                      # AI feature with the built-in model
            found = rv.find_idle(self.dataset_root, run_path, m, expect_rev=rev)
            summary = {k: sum(f["status"] == k for f in found)
                       for k in ("created", "already_suggested", "already_cut", "previously_kept")}
        elif op == "replace":
            rv.replace_edit(run_path, m, req.get("edit_id"), req.get("payload"), expect_rev=rev)
        elif op == "reviewed":
            rv.set_reviewed(run_path, req.get("reviewed"), by="person", expect_rev=rev)
        else:
            raise rv.ReviewError("op must be add, note, decide, decide_many, replace, find_idle, reviewed or undo")
        return summary

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        span = _parse_range(rng, size)
        if span is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        start, end = span
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        n = getattr(self, "_remaining", None)
        if n is None:
            return super().copyfile(source, outputfile)
        while n > 0:
            chunk = source.read(min(64 * 1024, n))
            if not chunk:
                break
            outputfile.write(chunk)
            n -= len(chunk)
        self._remaining = None


_WRITE_LOCK = threading.Lock()


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """One RFC 7233 byte range (first-last, first-, or -suffix) as inclusive (start, end); None if unsatisfiable."""
    m = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", header)
    if not m or (not m.group(1) and not m.group(2)) or size == 0:
        return None
    a, b = m.group(1), m.group(2)
    if not a:                                          # suffix: the last b bytes
        n = int(b)
        return (max(0, size - n), size - 1) if n > 0 else None
    start = int(a)
    end = min(int(b), size - 1) if b else size - 1
    if start >= size or end < start:
        return None
    return start, end


def _find_run(root: Path, run_id) -> tuple[Path, dict]:
    if not isinstance(run_id, str):
        raise rv.ReviewError("run_id is required")
    for run_path, m in store.iter_manifests(root, "committed"):
        if m["run_id"] == run_id:
            return run_path, m
    raise rv.ReviewError(f"no committed run {run_id!r}")


def serve(root: Path, port: int = 8765, open_browser: bool = True, editable: bool = True) -> None:
    root = Path(root).resolve()
    W.remember(root)
    build(root, editable=editable)
    handler_cls = type("Handler", (_RangeHandler,), {"editable": editable})
    if not editable:
        handler_cls.do_POST = lambda self: self._json(403, {"error": "this viewer is read-only"})
    handler = functools.partial(handler_cls, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    httpd.dataop_state = {"root": root}
    url = f"http://127.0.0.1:{port}/dataop/view/"
    print(f"Serving {root.name} at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
