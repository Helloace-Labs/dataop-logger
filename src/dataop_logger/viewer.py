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

import pyarrow.dataset as pads
import pyarrow.parquet as pq

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


def _review_state(run_path: Path, m: dict) -> dict:
    try:
        r = rv.load(run_path)
        return {"edits": r["edits"], "rev": r["rev"], "effective": rv.effective(m, r), "error": None}
    except rv.ReviewError as e:                        # shown on the run; editing it is disabled
        return {"edits": [], "rev": -1, "effective": rv.effective(m, rv.empty()), "error": str(e)}


def collect(root: Path) -> dict:
    root = Path(root)
    videos = _video_table(root)
    runs = []
    for run_path, m in store.iter_manifests(root, "committed"):
        if not RUN_ID.match(str(m["run_id"])) or run_path.name != m["run_id"]:
            continue                                   # validate reports it; never render an unexpected id
        t = pq.read_table(run_path / "ticks.parquet", columns=["control_mode", "expert_source", "safety_modified"])
        modes = t.column("control_mode").to_pylist()
        sources = t.column("expert_source").to_pylist()
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
            "policy": (m.get("policy") or {}).get("source"),
            "env": (m.get("env") or {}).get("extras", {}),
            "started_at": m.get("started_at"),
            "videos": videos.get(m["episode_index"], []),
            "review": _review_state(run_path, m),
        })
    runs.sort(key=lambda r: r["episode"])
    modes = sorted({r["failure_mode"] for r in runs if r["failure_mode"]} |
                   {e["payload"].get("failure_mode") for r in runs for e in r["review"]["edits"]
                    if e["kind"] == "outcome" and e["payload"].get("failure_mode")})
    return {"dataset": root.name, "runs": runs, "failure_modes": modes, "editable": False}


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
MAX_BODY = 64 * 1024


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with HTTP Range support, so browsers can seek inside the episode videos, plus the review API."""

    dataset_root: Path = Path(".")

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
            run_path, m = _find_run(self.dataset_root, req.get("run_id"))
            rev = req.get("rev")
            if isinstance(rev, bool) or not isinstance(rev, int):
                raise rv.ReviewError("rev (the review revision the page shows) is required")
            op, summary = req.get("op"), None
            if op == "add":
                if req.get("kind") not in ("label", "exclude"):
                    raise rv.ReviewError("the viewer adds labels and cuts only")
                payload = req.get("payload", {})
                rv.add_edit(run_path, m, req.get("kind"), payload, source="human", expect_rev=rev)
            elif op == "decide":
                rv.decide(run_path, m, req.get("edit_id"), req.get("accept"), req.get("payload"), expect_rev=rev)
            elif op == "find_idle":                      # AI feature with the built-in model
                found = rv.find_idle(self.dataset_root, run_path, m, expect_rev=rev)
                summary = {k: sum(f["status"] == k for f in found)
                           for k in ("created", "already_suggested", "already_cut", "previously_kept")}
            elif op == "replace":
                rv.replace_edit(run_path, m, req.get("edit_id"), req.get("payload"), expect_rev=rev)
            else:
                raise rv.ReviewError("op must be add, decide, replace or find_idle")
        except rv.ReviewConflict as e:
            return self._json(409, {"error": str(e), "state": _review_state(run_path, m)})
        except (rv.ReviewError, ValueError, KeyError, TypeError) as e:
            return self._json(400, {"error": str(e)})
        state = _review_state(run_path, m)
        if op == "find_idle":
            state["found"] = summary
        return self._json(200, state)

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
    build(root, editable=editable)
    handler_cls = type("Handler", (_RangeHandler,), {"dataset_root": root, "editable": editable})
    if not editable:
        handler_cls.do_POST = lambda self: self._json(403, {"error": "this viewer is read-only"})
    handler = functools.partial(handler_cls, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
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
