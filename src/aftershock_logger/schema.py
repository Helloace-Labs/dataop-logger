"""Per-run action schema: names, units and space of every logged action, with a derived id."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

SCHEMA_VERSION = 1
KNOWN_SPACES = {"joint_position", "joint_velocity", "ee_delta", "ee_pose"}
LIVE_CONTROL_MODES = ("base", "expert", "blend", "none")          # what the logger records
LIVE_TERMINATION_REASONS = ("success", "failure", "timeout", "operator_abort", "system_error")
# Imported data may not say who drove or how a run ended; it is stored as "unknown", never guessed.
CONTROL_MODES = LIVE_CONTROL_MODES + ("unknown",)
TERMINATION_REASONS = LIVE_TERMINATION_REASONS + ("unknown",)
OBS_TIMESTAMP_KINDS = ("capture", "bundle_completion", "from_dataset_timestamp")
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


# ── Optional task, planner and prompt fields (logger task L2) ────────────────────────────────────────────────
#
# Three cases stay apart: "unknown" (the value existed but was not recorded), null (known to be none, for example
# no planner ran) and a value. A field an older logger never wrote reads as "unknown", never as null.

UNKNOWN = "unknown"
UNKNOWN_INT = -1             # prompt_rev and control_gen are int64 columns on disk; -1 there means "unknown"
PLANNED_MANIFEST_KEYS = ("task_id", "task_rev", "planner")
PLANNED_TICK_COLUMNS = ("prompt", "prompt_rev", "subtask_cmd", "control_gen")


@dataclass(frozen=True)
class PlannerInfo:
    """The planner that picked the subtasks during a run. A changed planner (new rules, new model) gets a new id."""
    name: str
    id: str

    def __post_init__(self):
        for k in ("name", "id"):
            v = getattr(self, k)
            if not isinstance(v, str) or not v:
                raise ValueError(f"planner {k} must be a non-empty string")


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def task_id_value(v) -> str:
    """A task id is a non-empty string, or "unknown" when the caller did not give one."""
    if not isinstance(v, str) or not v:
        raise ValueError("task_id must be a non-empty string, or 'unknown' when not recorded")
    return v


def task_rev_value(v):
    """A task revision is an int >= 0, or "unknown"."""
    if v == UNKNOWN and isinstance(v, str):
        return UNKNOWN
    if not _is_int(v) or v < 0:
        raise ValueError("task_rev must be an int >= 0, or 'unknown' when not recorded")
    return int(v)


def planner_value(v):
    """{name, id}, None (no planner ran) or "unknown" (not recorded). Accepts a PlannerInfo or a dict."""
    if v is None:
        return None
    if isinstance(v, str):
        if v != UNKNOWN:
            raise ValueError("planner must be a PlannerInfo or {name, id}, None (no planner ran) or 'unknown'; "
                             "a bare name is not enough, since the id tells two versions of a planner apart")
        return UNKNOWN
    if isinstance(v, PlannerInfo):
        return {"name": v.name, "id": v.id}
    if isinstance(v, dict) and set(v) == {"name", "id"}:
        return asdict(PlannerInfo(v["name"], v["id"]))
    raise ValueError("planner must be a PlannerInfo or {name, id}, None (no planner ran) or 'unknown'")


def manifest_problems(m: dict) -> list[str]:
    """Problems with the optional L2 manifest fields and the policy value. Absent keys are fine (older loggers)."""
    out = []
    for key, check in (("task_id", task_id_value), ("task_rev", task_rev_value), ("planner", planner_value)):
        if key in m:
            try:
                check(m[key])
            except ValueError as e:
                out.append(str(e))
    p = m.get("policy")
    if not (p is None or p == UNKNOWN or isinstance(p, dict)):
        out.append("policy must be a policy record, null (no policy ran) or 'unknown'")
    return out


def manifest_fields(m: dict) -> dict:
    """task_id, task_rev, planner and policy as Aftershock reads them. A key an older logger never wrote reads as
    "unknown". Before L2 the importer wrote policy null when it did not record the policy, so a null policy in a
    manifest with no task_rev key reads as "unknown"; once task_rev is present, null means no policy ran."""
    legacy = "task_rev" not in m
    out = {k: m.get(k, UNKNOWN) for k in PLANNED_MANIFEST_KEYS}
    out["policy"] = UNKNOWN if legacy and m.get("policy") is None else m.get("policy")
    return out


def tick_values(prompt, prompt_rev, subtask_cmd, control_gen) -> dict:
    """Check one tick's optional fields and return them as stored (ints use -1 for unknown on disk)."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be the non-empty text the policy received, or 'unknown'")
    out = {"prompt": prompt}
    for name, v in (("prompt_rev", prompt_rev), ("control_gen", control_gen)):
        if isinstance(v, str) and v == UNKNOWN:
            out[name] = UNKNOWN_INT
        elif _is_int(v) and v >= 0:
            out[name] = int(v)
        else:
            raise ValueError(f"{name} must be an int >= 0, or 'unknown' when not recorded")
    if subtask_cmd is not None and (not isinstance(subtask_cmd, str) or not subtask_cmd):
        raise ValueError("subtask_cmd must be a subtask id, None (no planner command) or 'unknown'")
    out["subtask_cmd"] = subtask_cmd
    return out


class TickSequence:
    """Rules across one run's ticks, on stored values (ints -1 when unknown), checked one tick at a time.

    prompt_rev and control_gen never go down where known; ticks with the same known prompt_rev carry the same known
    prompt; with planner null (no planner ran) no tick names a subtask command."""

    def __init__(self, planner):
        self.planner = planner
        self.last = {"prompt_rev": None, "control_gen": None}
        self.rev_prompt: dict[int, str] = {}

    def problem(self, r: dict) -> str | None:
        for k in ("prompt_rev", "control_gen"):
            v = r[k]
            if v != UNKNOWN_INT and self.last[k] is not None and v < self.last[k]:
                return f"{k} went down from {self.last[k]} to {v}"
        rev, p = r["prompt_rev"], r["prompt"]
        if rev != UNKNOWN_INT and p != UNKNOWN and self.rev_prompt.get(rev, p) != p:
            return f"prompt changed without a new prompt_rev (still {rev})"
        if self.planner is None and r["subtask_cmd"] not in (None, UNKNOWN):
            return f"subtask_cmd {r['subtask_cmd']!r} given, but the run has no planner (planner is None)"
        return None

    def accept(self, r: dict) -> None:
        for k in ("prompt_rev", "control_gen"):
            if r[k] != UNKNOWN_INT:
                self.last[k] = r[k]
        if r["prompt_rev"] != UNKNOWN_INT and r["prompt"] != UNKNOWN:
            self.rev_prompt.setdefault(r["prompt_rev"], r["prompt"])


def tick_sequence_problems(rows: list[dict], planner) -> list[str]:
    seq, out = TickSequence(planner), []
    for i, r in enumerate(rows):
        bad = seq.problem(r)
        if bad:
            out.append(f"tick {i}: {bad}")
        seq.accept(r)
    return out
