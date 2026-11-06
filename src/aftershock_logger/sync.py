"""Datasets whose reviews are synced from the Aftershock cloud (logger task L7).

A station that uploads a dataset's runs to Aftershock also pulls each run's review back from the cloud and writes it
into `dataop/runs/<run_id>/review.json`. Before it writes any review it writes `dataop/sync.json`, which marks the
dataset as synced:

    {"sync_format": 1, "source": "aftershock", "workspace_id": "...", "station_id": "...", "since": "<UTC time>"}

The cloud is the source of truth for review, and offline edit reconciliation is not specified in v1, so a synced
dataset's reviews are read only here: every review change (`review.transaction`, which every edit, decision, undo,
sign off and `migrate-reviews` goes through) is refused, and the viewer serves the dataset read only. Reading,
exporting and recording are unchanged. A marker that exists but cannot be read, or is not in this shape, counts as
synced (fail closed), so a damaged marker never reopens local edits that the next sync would not keep.

The marker holds no key or secret. It is only removed by hand; a station that syncs again writes it again, and leaves
any local review it finds changed since its last sync (or since its upload) as it is until a person decides.

**How the syncing writer writes a review.** So that a reader never sees half a file and no logger change can slip in
between, the writer follows the logger's own locking: it holds `dataop/batches.lock` shared, then the run's
`review.lock` exclusive (the order `review.transaction` uses), writes the new text to a temp file in the run's folder
whose name starts with `.review.json.` and ends with `.tmp`, syncs it, renames it over `review.json` and syncs the
folder. The text is the review's canonical form (`review_format.dump_review`), exactly as the cloud stores it, so its
digest is the cloud's. A temp file left by an interrupted write is never read as the review.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import store

MARKER = "sync.json"
SYNC_FORMAT = 1
KEYS = {"sync_format", "source", "workspace_id", "station_id", "since"}
MAX_MARKER = 64 * 1024


class SyncedError(ValueError):
    """A change was refused because the dataset's reviews are synced from Aftershock."""


def marker_path(root: Path) -> Path:
    return store.dataop_dir(Path(root)) / MARKER


def state(root: Path) -> dict | None:
    """None when the dataset is not synced (no marker at all). Otherwise {"station_id", "workspace_id", "since",
    "error"}: `error` is None for a well formed marker, and says what is wrong with one that cannot be read, in which
    case the dataset still counts as synced."""
    p = marker_path(root)
    try:
        p.lstat()
    except FileNotFoundError:
        return None
    except OSError as e:
        return _broken(f"{MARKER} cannot be looked up ({type(e).__name__})")
    try:
        raw = p.read_bytes()
        if len(raw) > MAX_MARKER:
            return _broken(f"{MARKER} is too large")
        d = json.loads(raw)
    except OSError as e:
        return _broken(f"{MARKER} cannot be read ({type(e).__name__})")
    except ValueError:
        return _broken(f"{MARKER} is not valid JSON")
    if (not isinstance(d, dict) or set(d) != KEYS or d.get("sync_format") != SYNC_FORMAT
            or d.get("source") != "aftershock"
            or not all(isinstance(d.get(k), str) and d[k].strip() for k in ("workspace_id", "station_id", "since"))):
        return _broken(f"{MARKER} is not a sync marker this logger knows")
    return {"station_id": d["station_id"], "workspace_id": d["workspace_id"], "since": d["since"], "error": None}


def _broken(why: str) -> dict:
    return {"station_id": None, "workspace_id": None, "since": None, "error": why}


def describe(s: dict) -> str:
    if s["error"]:
        return f"this dataset is marked as synced from Aftershock, but its marker is damaged ({s['error']})"
    return f"this dataset's reviews are synced from Aftershock by station {s['station_id']}"


def check_editable(root: Path) -> None:
    """Refuse a local review change on a synced dataset."""
    s = state(root)
    if s is not None:
        raise SyncedError(f"{describe(s)}, so they are read only here. Change them in Aftershock's Review page; the "
                          "station syncs the result back")
