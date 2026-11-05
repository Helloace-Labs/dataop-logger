"""Files under <dataset root>/dataop: atomic writes, a single-writer lock, manifests and the derived index."""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DIR = "dataop"


def dataop_dir(root: Path) -> Path:
    return Path(root) / DIR


def runs_dir(root: Path) -> Path:
    return dataop_dir(root) / "runs"


def quarantine_dir(root: Path) -> Path:
    return dataop_dir(root) / "quarantine"


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Temp file in the destination directory, fsync, rename, fsync the directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: dict) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())


def table_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def atomic_write_table(path: Path, table: pa.Table) -> None:
    atomic_write_bytes(path, table_bytes(table))


def sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def json_sha256(obj) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def read_manifest(run_path: Path) -> dict:
    return json.loads((run_path / "manifest.json").read_text())


REQUIRED_KEYS = ("schema_version", "run_id", "status", "episode_index", "length", "task", "termination_reason",
                 "fingerprint", "action_schema", "takeovers", "events", "integrity")


def iter_run_dirs(root: Path):
    """Every run directory as (path, manifest or None, error or None); never raises on a bad run."""
    d = runs_dir(root)
    if not d.exists():
        return
    for run_path in sorted(p for p in d.iterdir() if p.is_dir()):
        try:
            m = read_manifest(run_path)
        except FileNotFoundError:
            yield run_path, None, "no manifest.json (crashed while staging?)"
            continue
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            yield run_path, None, f"manifest.json is unreadable ({e})"
            continue
        missing = [k for k in REQUIRED_KEYS if k not in m]
        if missing:
            yield run_path, None, f"manifest.json is missing {missing}"
        elif m["status"] not in ("staged", "committed"):
            yield run_path, None, f"unknown status {m['status']!r}"
        else:
            yield run_path, m, None


def iter_manifests(root: Path, status: str | None = None):
    """Readable, well-formed manifests only. Use iter_run_dirs to see the broken ones too."""
    for run_path, m, err in iter_run_dirs(root):
        if m is not None and (status is None or m["status"] == status):
            yield run_path, m


def rebuild_index(root: Path) -> int:
    lines = []
    for _, m in iter_manifests(root, "committed"):
        lines.append(json.dumps({k: m.get(k) for k in ("run_id", "episode_index", "length", "task",
                                                       "termination_reason", "task_success")}, sort_keys=True))
    lines.sort(key=lambda s: json.loads(s)["episode_index"])
    atomic_write_bytes(dataop_dir(root) / "index.jsonl", ("\n".join(lines) + ("\n" if lines else "")).encode())
    return len(lines)


def quarantine(root: Path, run_path: Path, reason: str) -> Path:
    dest = quarantine_dir(root) / run_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_path), str(dest))
    note = dest / "QUARANTINED.txt"
    note.write_text(reason + "\n")
    return dest


class WriterLock:
    """One writer per dataset root (v0). Held for the life of a RunLogger."""

    def __init__(self, root: Path):
        if "://" in str(root):
            raise ValueError(f"remote dataset roots are not supported in v0: {root}")
        dataop_dir(root).mkdir(parents=True, exist_ok=True)
        self.path = dataop_dir(root) / ".lock"
        self._f = open(self.path, "w")
        try:
            fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._f.close()
            raise RuntimeError(f"another dataop-logger writer holds {self.path}") from None
        self._f.write(str(os.getpid()))
        self._f.flush()

    def release(self) -> None:
        if not self._f.closed:
            fcntl.flock(self._f, fcntl.LOCK_UN)
            self._f.close()
