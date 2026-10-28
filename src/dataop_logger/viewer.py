"""dataop-logger view: a local, read-only page for browsing logged runs next to their videos."""
from __future__ import annotations

import functools
import http.server
import json
import os
import threading
import webbrowser
from pathlib import Path

import pyarrow.dataset as pads
import pyarrow.parquet as pq

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


def collect(root: Path) -> dict:
    root = Path(root)
    videos = _video_table(root)
    runs = []
    for run_path, m in store.iter_manifests(root, "committed"):
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
        })
    runs.sort(key=lambda r: r["episode"])
    return {"dataset": root.name, "runs": runs}


def build(root: Path) -> Path:
    root = Path(root)
    data = collect(root)
    html = (Path(__file__).with_name("viewer.html").read_text()
            .replace("/*__DATA__*/null", json.dumps(data).replace("</", "<\\/")))
    out = store.dataop_dir(root) / "view" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Static files with HTTP Range support, so browsers can seek inside the episode videos."""

    def log_message(self, *a):
        pass

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start, _, end = rng.removeprefix("bytes=").partition("-")
        start = int(start) if start else 0
        end = min(int(end) if end else size - 1, size - 1)
        if start >= size:
            self.send_error(416)
            return None
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


def serve(root: Path, port: int = 8765, open_browser: bool = True) -> None:
    root = Path(root).resolve()
    build(root)
    handler = functools.partial(_RangeHandler, directory=str(root))
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
