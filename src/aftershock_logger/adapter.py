"""The only place that touches LeRobot internals. Tested against lerobot 0.6.x (DatasetWriter API)."""
from __future__ import annotations

SUPPORTED = "lerobot>=0.6,<0.7"


def has_pending_frames(dataset) -> bool:
    return bool(dataset.has_pending_frames())


def pending_episode_index(dataset) -> int:
    """Index the episode being recorded will get when saved."""
    writer = getattr(dataset, "writer", None)
    buf = getattr(writer, "episode_buffer", None) if writer is not None else None
    if isinstance(buf, dict) and "episode_index" in buf:
        return int(buf["episode_index"])
    return int(dataset.meta.total_episodes)


def buffer_size(dataset) -> int | None:
    """Frames in the unsaved episode buffer, or None when this LeRobot version does not expose it."""
    writer = getattr(dataset, "writer", None)
    buf = getattr(writer, "episode_buffer", None) if writer is not None else None
    if isinstance(buf, dict) and "size" in buf:
        return int(buf["size"])
    return None


def total_episodes(dataset) -> int:
    return int(dataset.meta.total_episodes)


def saved_episode(dataset, episode_index: int) -> dict | None:
    """The episode LeRobot just registered, from its in-memory record.

    While recording, LeRobot keeps meta/episodes open in a parquet writer (no footer until it is closed), so the
    file on disk cannot be read until finalize(). Returns None when the record is not available.
    """
    e = getattr(dataset.meta, "latest_episode", None)
    if not e or int(e["episode_index"][0]) != episode_index:
        return None
    return {"length": int(e["length"][0]), "from": int(e["dataset_from_index"][0]),
            "to": int(e["dataset_to_index"][0])}


def dataset_identity(dataset) -> dict:
    return {"repo_id": getattr(dataset, "repo_id", None), "revision": getattr(dataset, "revision", None),
            "root": str(dataset.root)}


class EpisodeBufferError(RuntimeError):
    """The unsaved LeRobot episode buffer is malformed or unreadable."""


def buffer_snapshot(dataset, action_key: str = "action") -> dict | None:
    """Episode index, size, buffered actions (float32 vectors) and tasks of the unsaved episode.

    Returns None if this LeRobot version does not expose the buffer; raises EpisodeBufferError if it is malformed.
    """
    import numpy as np
    writer = getattr(dataset, "writer", None)
    buf = getattr(writer, "episode_buffer", None) if writer is not None else None
    if not isinstance(buf, dict) or "size" not in buf:
        return None
    try:
        actions = [np.ascontiguousarray(np.asarray(a, dtype=np.float32).reshape(-1)) for a in buf.get(action_key, [])]
        tasks = list(buf.get("task", []))
        ep = buf.get("episode_index")
        return {"episode_index": None if ep is None else int(ep), "size": int(buf["size"]),
                "actions": actions, "tasks": tasks}
    except Exception as e:
        raise EpisodeBufferError(f"cannot read the LeRobot episode buffer: {type(e).__name__}: {e}") from e


def reset_episode_buffer(dataset) -> None:
    """Clear the unsaved episode unconditionally, even when a failed add_frame left size at 0."""
    try:
        dataset.clear_episode_buffer()
    except Exception:
        writer = getattr(dataset, "writer", None)
        if writer is not None and hasattr(writer, "_create_episode_buffer"):
            writer.episode_buffer = writer._create_episode_buffer()
        else:
            raise


def relative_index_map(dataset) -> dict[int, int] | None:
    """Absolute dataset row -> position in this (possibly episode-filtered) view; None means identity."""
    return getattr(dataset, "absolute_to_relative_idx", None)


def selected_episodes(dataset) -> list[int] | None:
    eps = getattr(dataset, "episodes", None)
    return None if eps is None else [int(e) for e in eps]


def episode_tasks(root) -> dict[int, str]:
    """task_index -> task string from meta/tasks.parquet."""
    from pathlib import Path
    import pyarrow.parquet as pq
    f = Path(root) / "meta" / "tasks.parquet"
    if not f.exists():
        return {}
    t = pq.read_table(f).to_pydict()
    return {int(i): s for i, s in zip(t["task_index"], t["task"])}


def episode_metadata(root) -> dict[int, dict]:
    """Authoritative LeRobot episode table (meta/episodes): only episodes LeRobot fully registered."""
    from pathlib import Path
    import pyarrow.dataset as pads
    d = Path(root) / "meta" / "episodes"
    if not d.exists():
        return {}
    import json
    ds = pads.dataset(str(d), format="parquet")
    vkeys = sorted({c.split("/")[1] for c in ds.schema.names if c.startswith("videos/") and c.endswith("/chunk_index")})
    cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index", "tasks"]
    cols += [f"videos/{k}/{f}" for k in vkeys for f in ("chunk_index", "file_index")]
    times = [f"videos/{k}/{f}" for k in vkeys for f in ("from_timestamp", "to_timestamp")
             if f"videos/{k}/{f}" in ds.schema.names]
    t = ds.to_table(columns=cols + times).to_pydict()
    info = json.loads((Path(root) / "meta" / "info.json").read_text()) if (Path(root) / "meta" / "info.json").exists() else {}
    vpath = info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    out = {}
    for i, e in enumerate(t["episode_index"]):
        videos = [Path(root) / vpath.format(video_key=k, chunk_index=int(t[f"videos/{k}/chunk_index"][i]),
                                           file_index=int(t[f"videos/{k}/file_index"][i])) for k in vkeys
                  if t[f"videos/{k}/chunk_index"][i] is not None]
        out[int(e)] = {"length": int(t["length"][i]), "from": int(t["dataset_from_index"][i]),
                       "to": int(t["dataset_to_index"][i]), "tasks": list(t["tasks"][i] or []), "videos": videos,
                       "video_keys": vkeys,
                       "video_frames": episode_video_frames({c: t[c][i] for c in times}, info["fps"])
                       if times and info.get("fps") else {}}
    return out


_LEROBOT_DEFAULT_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def dataset_fps(dataset) -> float:
    return float(dataset.meta.fps)


def create_like(dataset, root, repo_id: str):
    """A new, empty LeRobot dataset at `root` with the same features (every camera, observation.state, action), fps,
    robot type and video setting as `dataset`. A Drive session's gap dataset (logger L9) is made this way, so a gap
    can later become a trainable episode with the same features as the runs."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    meta = dataset.meta
    features = {k: dict(v) for k, v in meta.features.items() if k not in _LEROBOT_DEFAULT_FEATURES}
    return LeRobotDataset.create(repo_id, fps=int(meta.fps), features=features, root=root,
                                 robot_type=getattr(meta, "robot_type", None), use_videos=bool(meta.video_keys))


# ---------------------------------------------------------------- LeRobot's streaming video encoder (issue #479)
#
# With streaming_encoding, LeRobot hands every camera frame to one encoder thread per camera through a bounded queue
# (encoder_queue_maxsize). When that queue is full, LeRobot drops the frame with only a warning, so the video ends up
# with fewer frames than the dataset has rows and every later frame is misaligned. The logger is the only producer
# during a run, so it waits for room in every queue before it hands a frame over, and checks LeRobot's own count of
# dropped frames after each one.


def _streaming_encoder(dataset):
    enc = getattr(getattr(dataset, "writer", None), "_streaming_encoder", None)
    if enc is None or not getattr(enc, "_episode_active", False):
        return None
    return enc


def encoder_full(dataset) -> list[tuple[str, int, int]]:
    """The cameras whose encoder queue has no room for one more frame, as (camera, frames waiting, queue size). Empty
    when the dataset does not stream video, before a run's first frame (LeRobot opens the queues then), and when this
    LeRobot version keeps its queues elsewhere."""
    enc = _streaming_encoder(dataset)
    queues = getattr(enc, "_frame_queues", None) or {}
    out = []
    for key, q in sorted(queues.items()):
        size = getattr(q, "maxsize", 0)
        try:
            n = q.qsize()
        except Exception:
            continue
        if size > 0 and n >= size:
            out.append((key, n, size))
    return out


def encoder_dropped(dataset) -> dict[str, int]:
    """Frames LeRobot's streaming encoder dropped in the episode being recorded, per camera (only cameras with drops)."""
    enc = _streaming_encoder(dataset)
    dropped = getattr(enc, "_dropped_frames", None) or {}
    try:
        return {k: int(v) for k, v in sorted(dropped.items()) if int(v) > 0}
    except Exception:
        return {}


def _one(v):
    """A meta/episodes value: LeRobot's in-memory record keeps a list (or array) of one value per column."""
    if v is not None and not isinstance(v, (str, bytes, int, float)) and hasattr(v, "__len__"):
        return v[0] if len(v) else None
    return v


def episode_video_frames(episode: dict, fps: float) -> dict[str, int]:
    """Per camera, the frames an episode's video holds, from LeRobot's episode record (its videos/<camera>/from_timestamp
    and to_timestamp, which LeRobot takes from the encoded file's duration). `episode` is a meta/episodes row, as
    columns to values (a list of one value, as LeRobot's latest_episode keeps it, or a plain value)."""
    out = {}
    for k, v in episode.items():
        if not (k.startswith("videos/") and k.endswith("/to_timestamp")):
            continue
        cam = k[len("videos/"):-len("/to_timestamp")]
        a, b = _one(episode.get(f"videos/{cam}/from_timestamp")), _one(v)
        if a is None or b is None:
            continue
        out[cam] = round((float(b) - float(a)) * float(fps))
    return out
