"""Per-run action schema: names, units and space of every logged action, with a derived id."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

SCHEMA_VERSION = 1
KNOWN_SPACES = {"joint_position", "joint_velocity", "ee_delta", "ee_pose"}
CONTROL_MODES = ("base", "expert", "blend", "none")
TERMINATION_REASONS = ("success", "failure", "timeout", "operator_abort", "system_error")
SUBGOAL_STATUSES = ("started", "completed", "failed")
TERMINAL_FLAG = {"success": "terminated", "failure": "terminated",          # final-tick flag each reason implies
                 "timeout": "truncated", "operator_abort": "truncated", "system_error": "truncated"}


@dataclass(frozen=True)
class ActionSchema:
    """Bounds are descriptive (recorded, not enforced). Values are logged denormalized, in robot command space."""
    names: tuple[str, ...]
    units: tuple[str, ...]
    space: str
    rate_hz: float
    frame: str = "robot"
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None
    dtype: str = "float32"
    normalization_note: str = "values are logged denormalized, in robot command space"

    def __post_init__(self):
        n = len(self.names)
        if n == 0:
            raise ValueError("action schema needs at least one action name")
        if any(not isinstance(x, str) or not x for x in self.names) or len(set(self.names)) != n:
            raise ValueError("action names must be unique, non-empty strings")
        if len(self.units) != n:
            raise ValueError(f"units has {len(self.units)} entries, names has {n}")
        if any(not isinstance(u, str) or not u for u in self.units):
            raise ValueError("every unit must be a non-empty string")
        for k in ("frame", "normalization_note"):
            v = getattr(self, k)
            if not isinstance(v, str) or not v:
                raise ValueError(f"{k} must be a non-empty string")
        if self.space not in KNOWN_SPACES and not (self.space.startswith("custom:") and len(self.space) > 7):
            raise ValueError(f"unknown action space {self.space!r}; use one of {sorted(KNOWN_SPACES)} or 'custom:<name>'")
        if not (isinstance(self.rate_hz, (int, float)) and np.isfinite(self.rate_hz) and self.rate_hz > 0):
            raise ValueError("rate_hz must be a positive finite number")
        if (self.low is None) != (self.high is None):
            raise ValueError("give both low and high bounds, or neither")
        if self.low is not None:
            if len(self.low) != n or len(self.high) != n:
                raise ValueError(f"bounds must have {n} entries")
            if any(not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi) for lo, hi in zip(self.low, self.high)):
                raise ValueError("bounds must be finite with low <= high")
        if self.dtype != "float32":
            raise ValueError("v0 logs float32 actions only")

    @property
    def dim(self) -> int:
        return len(self.names)

    def to_dict(self) -> dict:
        """The SPEC v0.3 shape: bounds as {low, high} or null."""
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}
        low, high = d.pop("low"), d.pop("high")
        d["bounds"] = None if low is None else {"low": low, "high": high}
        return d

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def id(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSchema":
        d = {k: v for k, v in d.items() if k != "id"}
        b = d.pop("bounds", None)
        if b is not None:
            d["low"], d["high"] = b["low"], b["high"]
        for k in ("names", "units", "low", "high"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return cls(**d)

    def coerce(self, value, what: str) -> np.ndarray:
        """One canonical conversion to a contiguous float32 vector of the schema's size."""
        if hasattr(value, "detach"):                       # torch tensor
            value = value.detach().cpu().numpy()
        a = np.array(value, dtype=np.float32, order="C", copy=True).reshape(-1)   # owned: never shares caller storage
        if a.shape != (self.dim,):
            raise ValueError(f"{what} has {a.size} values, the action schema has {self.dim} ({', '.join(self.names)})")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{what} has NaN or Inf values")
        a.setflags(write=False)                            # the run keeps this exact copy for pre-save checks
        return a

    def canon_rows(self, actions, n_rows: int) -> np.ndarray:
        """Reshape actions read back from disk to [n_rows, dim] (LeRobot stores 1-D features as scalars)."""
        a = np.asarray(actions, dtype=np.float32)
        if a.size != n_rows * self.dim:
            raise ValueError(f"expected {n_rows} x {self.dim} action values, found {a.size}")
        return np.ascontiguousarray(a.reshape(n_rows, self.dim))


@dataclass
class PolicyInfo:
    family: str                                   # "smolvla", "act", "pi05", ...
    source: str                                   # hub repo id or local path
    revision: str | None = None                   # hub revision or artifact sha256
    code_revision: str | None = None
    config_digest: str | None = None
    chunk_size: int | None = None
    temporal_ensembling: bool = False
    device: str | None = None
    precision: str | None = None


@dataclass
class EnvInfo:
    robot_id: str | None = None
    calibration_digest: str | None = None
    operator: str | None = None                   # a pseudonym, never a real name
    extras: dict = field(default_factory=dict)    # lighting, site, anything else
