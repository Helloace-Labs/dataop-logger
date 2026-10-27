"""Canonical episode fingerprint: remapping evidence, not identity (run_id is the identity)."""
from __future__ import annotations

import hashlib
import struct

import numpy as np

from .schema import SCHEMA_VERSION


def fingerprint(schema_id: str, task: str, actions: np.ndarray) -> str:
    """SHA-256 over schema version, action schema id, task, length and little-endian float32 actions."""
    a = np.ascontiguousarray(np.asarray(actions, dtype="<f4"))
    h = hashlib.sha256()
    h.update(b"dataop-fingerprint\0")
    h.update(struct.pack("<I", SCHEMA_VERSION))
    h.update(schema_id.encode() + b"\0")
    t = task.encode("utf-8")
    h.update(struct.pack("<Q", len(t)) + t)
    h.update(struct.pack("<QQ", a.shape[0] if a.ndim else 0, a.shape[1] if a.ndim > 1 else 0))
    h.update(a.tobytes())
    return h.hexdigest()
