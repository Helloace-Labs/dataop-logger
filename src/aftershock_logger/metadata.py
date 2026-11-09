"""Run metadata: what is learned about a run after it was recorded (logger task L4).

In Watch mode the customer's own code calls the logger, so nobody may have said that a planner picked the prompts.
A prompt that changes during a run shows one anyway. The manifest is written once and never edited, so what is
learned later lives in a separate, versioned record next to it:

    dataop/runs/<run_id>/metadata.json
    {"metadata_format": 1, "run_id", "rev", "planner_detected", "planner_confirmed", "history"}

  planner_detected  = {"found": detection, "source": "rule" | "agent" | "human", "by", "job_id", "agent_version",
                       "reason", "proposed_at", "suggestions": [suggestion], "status": "open" | "confirmed" | "removed"}
                      | null
  detection         = {"subtasks": [{"text", "sha256", "cut"}], "first_seen_frame", "distinct_prompts"}
  suggestion        = {"name", "by", "source": "agent" | "human", "job_id", "agent_version", "reason", "at"}
  planner_confirmed = {"name", "planner_id", "by", "at"} | null
  history           = [{"rev", "op", "by", "source", "job_id", "agent_version", "reason", "at",
                        "changes": [{"field", "before", "after"}]}]

It is the same record, field for field, as Aftershock's cloud model. No file means rev 0 with both fields null. Each
write raises `rev` by one and adds one history step that keeps every field it changed whole, before and after, so
replaying the history gives the current fields and any earlier revision (`metadata_at`), and every revision holds the
pair rule: a confirmation sits on a detection whose status is `confirmed`.

**Writes.** `propose`, `confirm` (a rename when already confirmed) and `remove` take the `rev` the caller read; a
stale one is refused with the current one. A detection is always computed from the run's own per tick `prompt`
(`detect_planner`), so nobody can propose a planner the recording does not show. A later proposal never changes a
detection or a person's decision: with a name it only adds a suggestion. Which operation is allowed on which
detection status is `TRANSITIONS`. Every write holds `dataop/batches.lock` exclusive (the dataset lock), and writes
the file through a temp file and a rename, so a reader sees the old record or the new one, never half of one. A run
in a locked batch is refused, as review edits are. A dataset synced from Aftershock (`sync.json`, logger L7) is read
only here: only the sync from the cloud writes it (`write_synced`). The manifest is never touched.

**The effective planner.** A run's planner is `planner_confirmed` when a person set one, else the manifest's
`planner` (`effective_planner`). Batches (pack identity, `mixed_policy`) and export records use it, so a confirmed
run moves to the batch of the confirmed pair; a run in a locked batch stays until the batch is unlocked.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from . import store

FILE = "metadata.json"
FORMAT = 1
MAX_FILE = 64 * 1024 * 1024
MAX_DETECTED_SUBTASKS = 32
MAX_PROMPT_CHARS = 2000
MAX_SUGGESTIONS = 20
DETECTION_RULE = "planner-detect"
RULE_VERSION = "logger-l4-1"            # the version a rule proposal records as its agent_version
OPS = ("propose", "suggest", "confirm", "rename", "remove", "sync")
SOURCES = ("rule", "agent", "human")
STATUSES = ("open", "confirmed", "removed")
FIELDS = ("planner_detected", "planner_confirmed")
UNKNOWN = "unknown"

# Which detection statuses each operation accepts (None: the run has no detection yet).
TRANSITIONS = {
    "propose": (None,),
    "suggest": ("open", "confirmed", "removed"),
    "confirm": ("open", "removed"),
    "rename": ("confirmed",),
    "remove": ("open", "confirmed"),
}

_PLANNER_ID = re.compile(r"(rules|llm|ext)-[0-9a-z][0-9a-z_.-]{0,62}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_NONSPACE = re.compile(r"\S")


class MetadataError(ValueError):
    """A run metadata record that cannot be read, or a write that was refused."""


class StaleRevError(MetadataError):
    def __init__(self, current: int, given):
        super().__init__(f"the run's metadata changed (now at rev {current}, you had {given}); read it again")
        self.current = current


class ConflictError(MetadataError):
    """The operation does not apply to the detection as it is (TRANSITIONS), or a limit was reached."""


class NoPlannerError(MetadataError):
    """The run's known prompt never changes, so there is nothing to propose."""


# ---------------------------------------------------------------- the detection rule

def _sha(p: str) -> str:
    return hashlib.sha256(p.encode("utf-8", "surrogatepass")).hexdigest()


def detect_planner(prompts) -> dict | None:
    """The no-model detection rule, the same as Aftershock's server rule.

    `prompts` is a run's per tick `prompt` in frame order. "unknown", empty and None are not counted. Two or more
    distinct known prompts mean a planner: the subtasks are the distinct prompts in first seen order (the first
    MAX_DETECTED_SUBTASKS, each cut at MAX_PROMPT_CHARS with the whole prompt's sha256 and a `cut` flag),
    `distinct_prompts` counts them all, and `first_seen_frame` is the first frame whose known prompt differs from the
    first known prompt. One known prompt, or none, gives None."""
    seen: dict[str, None] = {}
    first_change = None
    for i, p in enumerate(prompts):
        if not isinstance(p, str) or p == "" or p == UNKNOWN:
            continue
        if p not in seen:
            if seen and first_change is None:
                first_change = i
            seen[p] = None
    if len(seen) < 2:
        return None
    subs = [{"text": p[:MAX_PROMPT_CHARS], "sha256": _sha(p), "cut": len(p) > MAX_PROMPT_CHARS}
            for p in list(seen)[:MAX_DETECTED_SUBTASKS]]
    return {"subtasks": subs, "first_seen_frame": first_change, "distinct_prompts": len(seen)}


def run_prompts(run_path: Path) -> list:
    """The run's per tick prompts. A run from a logger without the prompt column reads as all unknown."""
    import pyarrow.parquet as pq
    t = pq.read_table(Path(run_path) / "ticks.parquet")
    if "prompt" not in t.column_names:
        return [UNKNOWN] * t.num_rows
    return t.column("prompt").to_pylist()


def detect(run_path: Path) -> dict | None:
    """The detection for one run, from its own ticks."""
    return detect_planner(run_prompts(run_path))


# ---------------------------------------------------------------- the record and its checks

def path(root: Path, run_id: str) -> Path:
    return store.runs_dir(Path(root)) / run_id / FILE


def empty(run_id: str) -> dict:
    return {"metadata_format": FORMAT, "run_id": run_id, "rev": 0, "planner_detected": None,
            "planner_confirmed": None, "history": []}


def _is_int(v) -> bool:
    return type(v) is int


def _text(v, lo: int, hi: int, nonspace: bool = True) -> bool:
    return isinstance(v, str) and lo <= len(v) <= hi and (not nonspace or bool(_NONSPACE.search(v)))


def _trace(v) -> bool:            # by, job_id, agent_version
    return _text(v, 1, 100)


def _reason(v) -> bool:
    return _text(v, 1, 2000)


def _time(v) -> bool:
    return isinstance(v, str) and len(v) >= 1


def _keys(d, want: tuple, where: str, out: list) -> bool:
    if not isinstance(d, dict):
        out.append(f"{where} is not an object")
        return False
    if set(d) != set(want):
        extra, missing = sorted(set(d) - set(want)), sorted(set(want) - set(d))
        out.append(f"{where} has " + "; ".join(x for x in (f"unknown keys {extra}" if extra else "",
                                                           f"missing keys {missing}" if missing else "") if x))
        return False
    return True


def _opt(v, check) -> bool:
    return v is None or check(v)


def _detection_problems(f, where: str, out: list) -> None:
    if not _keys(f, ("subtasks", "first_seen_frame", "distinct_prompts"), where, out):
        return
    subs = f["subtasks"]
    if not isinstance(subs, list) or not 2 <= len(subs) <= MAX_DETECTED_SUBTASKS:
        out.append(f"{where}.subtasks is a list of 2 to {MAX_DETECTED_SUBTASKS}")
        subs = []
    for i, s in enumerate(subs):
        w = f"{where}.subtasks[{i}]"
        if not _keys(s, ("text", "sha256", "cut"), w, out):
            continue
        if not _text(s["text"], 1, MAX_PROMPT_CHARS, nonspace=False):
            out.append(f"{w}.text is 1 to {MAX_PROMPT_CHARS} characters")
        if not (isinstance(s["sha256"], str) and _SHA256.fullmatch(s["sha256"])):
            out.append(f"{w}.sha256 is 64 lowercase hex")
        if not isinstance(s["cut"], bool):
            out.append(f"{w}.cut is true or false")
    if not (_is_int(f["first_seen_frame"]) and f["first_seen_frame"] >= 0):
        out.append(f"{where}.first_seen_frame is an int of at least 0")
    n = f["distinct_prompts"]
    if not (_is_int(n) and n >= 2):
        out.append(f"{where}.distinct_prompts is an int of at least 2")
        return
    shas = [s.get("sha256") for s in subs if isinstance(s, dict)]
    if len(set(shas)) != len(shas):
        out.append(f"{where}: detected subtasks are distinct")
    if subs and n < len(subs):
        out.append(f"{where}: distinct_prompts counts at least the stored subtasks")
    if subs and n > len(subs) and len(subs) != MAX_DETECTED_SUBTASKS:
        out.append(f"{where}: subtasks are capped only at {MAX_DETECTED_SUBTASKS}")


def _suggestion_problems(s, where: str, out: list) -> None:
    if not _keys(s, ("name", "by", "source", "job_id", "agent_version", "reason", "at"), where, out):
        return
    if not _text(s["name"], 1, 100):
        out.append(f"{where}.name is 1 to 100 characters, not only spaces")
    if not _trace(s["by"]):
        out.append(f"{where}.by is 1 to 100 characters, not only spaces")
    if s["source"] not in ("agent", "human"):
        out.append(f"{where}.source is agent or human")
    for k, check in (("job_id", _trace), ("agent_version", _trace), ("reason", _reason)):
        if not _opt(s[k], check):
            out.append(f"{where}.{k} is null or a non-blank string within its length")
    if not _time(s["at"]):
        out.append(f"{where}.at is a time")
    agent = s["source"] == "agent"
    if any(agent != (s[k] is not None) for k in ("job_id", "agent_version", "reason")):
        out.append(f"{where}: an agent's suggestion carries job_id, agent_version and reason; a person's does not")


def _detected_problems(d, where: str, out: list) -> None:
    keys = ("found", "source", "by", "job_id", "agent_version", "reason", "proposed_at", "suggestions", "status")
    if not _keys(d, keys, where, out):
        return
    _detection_problems(d["found"], f"{where}.found", out)
    if d["source"] not in SOURCES:
        out.append(f"{where}.source is rule, agent or human")
    if not _trace(d["by"]):
        out.append(f"{where}.by is 1 to 100 characters, not only spaces")
    for k, check in (("job_id", _trace), ("agent_version", _trace), ("reason", _reason)):
        if not _opt(d[k], check):
            out.append(f"{where}.{k} is null or a non-blank string within its length")
    if not _time(d["proposed_at"]):
        out.append(f"{where}.proposed_at is a time")
    sg = d["suggestions"]
    if not isinstance(sg, list) or len(sg) > MAX_SUGGESTIONS:
        out.append(f"{where}.suggestions is a list of at most {MAX_SUGGESTIONS}")
    else:
        for i, s in enumerate(sg):
            _suggestion_problems(s, f"{where}.suggestions[{i}]", out)
    if d["status"] not in STATUSES:
        out.append(f"{where}.status is open, confirmed or removed")
    if d["source"] == "agent" and (d["job_id"] is None or d["agent_version"] is None or d["reason"] is None):
        out.append(f"{where}: an agent's detection carries job_id, agent_version and reason")
    if d["source"] == "rule" and (d["by"] != DETECTION_RULE or d["agent_version"] is None):
        out.append(f"{where}: the rule's detection is by {DETECTION_RULE} with its version")
    if d["source"] == "human" and (d["job_id"] or d["agent_version"] or d["reason"]):
        out.append(f"{where}: a person's detection carries no agent trace")


def _confirmed_problems(c, where: str, out: list) -> None:
    if not _keys(c, ("name", "planner_id", "by", "at"), where, out):
        return
    if not _text(c["name"], 1, 100):
        out.append(f"{where}.name is 1 to 100 characters, not only spaces")
    if not (isinstance(c["planner_id"], str) and _PLANNER_ID.fullmatch(c["planner_id"])):
        out.append(f"{where}.planner_id names a planner (rules-, llm- or ext- and lower case letters and digits)")
    if not _trace(c["by"]):
        out.append(f"{where}.by is 1 to 100 characters, not only spaces")
    if not _time(c["at"]):
        out.append(f"{where}.at is a time")


def _field_problems(field: str, v, where: str, out: list) -> None:
    if v is None:
        return
    (_detected_problems if field == "planner_detected" else _confirmed_problems)(v, where, out)


def _pair_problem(d, c) -> str | None:
    if c is not None and (not isinstance(d, dict) or d.get("status") != "confirmed"):
        return "a confirmation needs a detection whose status is confirmed"
    if isinstance(d, dict) and d.get("status") == "confirmed" and c is None:
        return "a confirmed detection has a confirmation"
    return None


def problems(rec, run_id: str | None = None) -> list[str]:
    """Everything wrong with a record, as Aftershock's cloud model would refuse it, plus that it belongs to `run_id`.
    An empty list means the record is valid."""
    out: list[str] = []
    keys = ("metadata_format", "run_id", "rev", "planner_detected", "planner_confirmed", "history")
    if not _keys(rec, keys, "metadata", out):
        return out
    if not (_is_int(rec["metadata_format"]) and rec["metadata_format"] == FORMAT):
        return [f"metadata_format is {rec['metadata_format']!r}; this logger reads format {FORMAT}"]
    if not (isinstance(rec["run_id"], str) and rec["run_id"]):
        out.append("run_id is a non-empty string")
    elif run_id is not None and rec["run_id"] != run_id:
        out.append(f"run_id {rec['run_id']!r} does not match its run folder {run_id!r}")
    if not (_is_int(rec["rev"]) and rec["rev"] >= 0):
        out.append("rev is an int of at least 0")
    for f in FIELDS:
        _field_problems(f, rec[f], f, out)
    hist = rec["history"]
    if not isinstance(hist, list):
        out.append("history is a list")
        return out
    for i, step in enumerate(hist):
        w = f"history[{i}]"
        if not _keys(step, ("rev", "op", "by", "source", "job_id", "agent_version", "reason", "at", "changes"), w, out):
            continue
        if not (_is_int(step["rev"]) and step["rev"] >= 1):
            out.append(f"{w}.rev is an int of at least 1")
        if step["op"] not in OPS:
            out.append(f"{w}.op is one of {list(OPS)}")
        if step["source"] not in SOURCES:
            out.append(f"{w}.source is rule, agent or human")
        if not _trace(step["by"]):
            out.append(f"{w}.by is 1 to 100 characters, not only spaces")
        for k, check in (("job_id", _trace), ("agent_version", _trace), ("reason", _reason)):
            if not _opt(step[k], check):
                out.append(f"{w}.{k} is null or a non-blank string within its length")
        if not _time(step["at"]):
            out.append(f"{w}.at is a time")
        ch = step["changes"]
        if not isinstance(ch, list) or not ch:
            out.append(f"{w}.changes is a list of at least one change")
            continue
        for j, c in enumerate(ch):
            wc = f"{w}.changes[{j}]"
            if not _keys(c, ("field", "before", "after"), wc, out):
                continue
            if c["field"] not in FIELDS:
                out.append(f"{wc}.field is planner_detected or planner_confirmed")
                continue
            _field_problems(c["field"], c["before"], f"{wc}.before", out)
            _field_problems(c["field"], c["after"], f"{wc}.after", out)
    if out:
        return out
    # the record's own rules, once every part has its shape
    if rec["rev"] != len(hist):
        out.append("rev equals the number of history steps")
    if [s["rev"] for s in hist] != list(range(1, len(hist) + 1)):
        out.append("history steps are numbered 1, 2, ... in order")
    bad = _pair_problem(rec["planner_detected"], rec["planner_confirmed"])
    if bad:
        out.append(bad)
    state = {f: None for f in FIELDS}
    for step in hist:
        fields = [c["field"] for c in step["changes"]]
        if len(set(fields)) != len(fields):
            out.append(f"history step {step['rev']} changes a field twice")
        for c in step["changes"]:
            if state[c["field"]] != c["before"]:
                out.append(f"history step {step['rev']}: {c['field']} before does not match the previous value")
            state[c["field"]] = c["after"]
        bad = _pair_problem(state["planner_detected"], state["planner_confirmed"])
        if bad:
            out.append(f"history step {step['rev']}: {bad}")
    if state != {f: rec[f] for f in FIELDS}:
        out.append("replaying history does not give the current fields")
    return out


def _parse(raw: bytes, run_id: str) -> dict:
    if len(raw) > MAX_FILE:
        raise MetadataError(f"{FILE} is larger than {MAX_FILE} bytes")
    try:
        rec = json.loads(raw)
    except ValueError as e:
        raise MetadataError(f"{FILE} is not valid JSON ({e})") from None
    bad = problems(rec, run_id)
    if bad:
        raise MetadataError(f"{FILE}: " + "; ".join(bad[:5]))
    return rec


def load(root: Path, run_id: str) -> dict:
    """The run's record; rev 0 with both fields null when it has none. A file that cannot be read or is not a valid
    record raises MetadataError, never reads as empty."""
    p = path(root, run_id)
    try:
        raw = p.read_bytes()
    except FileNotFoundError:
        return empty(run_id)
    except OSError as e:
        raise MetadataError(f"{FILE} cannot be read ({type(e).__name__})") from None
    return _parse(raw, run_id)


def load_or_none(root: Path, run_id: str) -> dict | None:
    """The record, or None when it is missing or broken (validate reports a broken one)."""
    try:
        return load(root, run_id)
    except MetadataError:
        return None


def decided(rec: dict | None) -> bool:
    """Whether a person ever confirmed this run's planner: then its batch follows its effective planner."""
    return bool(rec) and any(c["field"] == "planner_confirmed" for s in rec["history"] for c in s["changes"])


def effective_planner(manifest_planner, rec: dict | None):
    """A run's planner: the confirmed one ({name, id}) when a person set one, else the manifest's value as recorded
    ({name, id}, None for no planner, or "unknown")."""
    if rec is not None and rec.get("planner_confirmed") is not None:
        c = rec["planner_confirmed"]
        return {"name": c["name"], "id": c["planner_id"]}
    return manifest_planner


def metadata_at(rec: dict, rev: int) -> dict:
    """The two fields as they were at `rev` (0 to the record's rev), rebuilt from history, as fresh copies."""
    if not _is_int(rev) or not 0 <= rev <= rec["rev"]:
        raise MetadataError(f"rev is 0 to {rec['rev']}")
    state = {f: None for f in FIELDS}
    for step in rec["history"][:rev]:
        for c in step["changes"]:
            state[c["field"]] = c["after"]
    return copy.deepcopy(state)


def external_planner_id(name: str) -> str:
    """The id a confirmation gives a planner that runs in the customer's own code: `ext-` and the name in lower case,
    every run of other characters as one `-`. The same name gives the same id, as in Aftershock."""
    slug = re.sub(r"[^0-9a-z]+", "-", name.strip().lower()).strip("-")[:60]
    if not slug:
        raise MetadataError("a planner name needs at least one letter or digit")
    return f"ext-{slug}"


# ---------------------------------------------------------------- writes

def _now() -> str:
    from .review import _now as now
    return now()


def _check_trace(source: str, job_id, agent_version, reason, what: str) -> None:
    if source == "agent":
        if not (_trace(job_id) and _trace(agent_version) and _reason(reason)):
            raise MetadataError(f"an agent's {what} needs job_id, agent_version and reason")
    elif source == "human" and (job_id is not None or agent_version is not None):
        raise MetadataError(f"a person's {what} carries no job_id or agent_version")


def _committed(root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or not run_id or "/" in run_id or run_id in (".", ".."):
        raise MetadataError(f"no committed run {run_id!r}")
    run_path = store.runs_dir(root) / run_id
    try:
        m = store.read_manifest(run_path)
    except (OSError, ValueError):
        raise MetadataError(f"no committed run {run_id!r}") from None
    if m.get("status") != "committed" or m.get("run_id") != run_id:
        raise MetadataError(f"no committed run {run_id!r}")
    return run_path


def _save(root: Path, rec: dict) -> None:
    bad = problems(rec, rec["run_id"])
    if bad:                                    # never write a record the cloud model would refuse
        raise MetadataError("refusing to write an invalid record: " + "; ".join(bad[:5]))
    store.atomic_write_json(path(root, rec["run_id"]), rec)


def _step(rec: dict, op: str, by: str, source: str, job_id, agent_version, reason, at: str, new: dict) -> dict:
    """The record after one write that sets the fields in `new`. Only changed fields go into the step, whole."""
    changes = [{"field": f, "before": copy.deepcopy(rec[f]), "after": copy.deepcopy(new[f])}
               for f in FIELDS if f in new and new[f] != rec[f]]
    out = copy.deepcopy(rec)
    out["rev"] = rec["rev"] + 1
    for c in changes:
        out[c["field"]] = copy.deepcopy(c["after"])
    out["history"].append({"rev": out["rev"], "op": op, "by": by, "source": source, "job_id": job_id,
                           "agent_version": agent_version, "reason": reason, "at": at, "changes": changes})
    return out


def _write(root: Path, run_id: str, base_rev, change) -> dict:
    """Run `change(record, run_path)` under the dataset lock and save what it returns (None: nothing to write)."""
    from . import batches as B
    from . import sync
    root = Path(root)
    sync.check_editable(root)
    if not _is_int(base_rev):
        raise MetadataError("base_rev is the record's rev as you read it (0 when it has none)")
    run_path = _committed(root, run_id)
    with B._transaction(root) as d:
        bid = d["members"].get(run_id)
        b = next((x for x in d["batches"] if x["batch_id"] == bid), None)
        if b is not None and b["status"] == "locked":
            raise B.BatchError(f"this run is in batch {b['name']!r}, which is locked for export; unlock the batch to "
                               "change its planner")
        cur = load(root, run_id)
        if base_rev != cur["rev"]:
            raise StaleRevError(cur["rev"], base_rev)
        new = change(cur, run_path)
        if new is None:
            return cur
        _save(root, new)
    return new


def propose(root: Path, run_id: str, base_rev: int, *, by: str, source: str = "human", name: str | None = None,
            job_id: str | None = None, agent_version: str | None = None, reason: str | None = None) -> dict:
    """Propose that a planner ran in this run. The detection comes from the run's own prompts; a run whose known
    prompt never changes raises NoPlannerError. With no detection yet it writes one with status open (a `name`
    becomes its first suggestion). When the run already has a detection, a `name` only adds a suggestion beside the
    others; the same name from the same proposer again, or no name, returns the record unchanged.

    source "rule" is this logger's own rule (by is planner-detect, agent_version is RULE_VERSION, no name); "agent"
    needs job_id, agent_version and reason; "human" takes no agent trace."""
    if source not in SOURCES:
        raise MetadataError("source is rule, agent or human")
    if source == "rule":
        if name is not None:
            raise MetadataError("the rule does not suggest names")
        by, agent_version = DETECTION_RULE, agent_version or RULE_VERSION
    if not _trace(by):
        raise MetadataError("by is 1 to 100 characters, not only spaces")
    if name is not None and not _text(name, 1, 100):
        raise MetadataError("a name is 1 to 100 characters, not only spaces")
    if reason is not None and not _reason(reason):
        raise MetadataError("a reason is 1 to 2000 characters, not only spaces")
    _check_trace(source, job_id, agent_version, reason, "proposal")

    def change(cur: dict, run_path: Path):
        at = _now()
        sug = None if name is None else {
            "name": name, "by": by, "source": source, "job_id": job_id if source == "agent" else None,
            "agent_version": agent_version if source == "agent" else None,
            "reason": reason if source == "agent" else None, "at": at}
        d = cur["planner_detected"]
        if d is None:
            found = detect(run_path)
            if found is None:
                raise NoPlannerError(f"run {run_id}: the known prompt never changes, so no planner shows")
            det = {"found": found, "source": source, "by": by, "job_id": job_id, "agent_version": agent_version,
                   "reason": reason, "proposed_at": at, "suggestions": [sug] if sug else [], "status": "open"}
            return _step(cur, "propose", by, source, job_id, agent_version, reason, at, {"planner_detected": det})
        if sug is None or any(s["name"] == name and s["by"] == by for s in d["suggestions"]):
            return None
        if len(d["suggestions"]) >= MAX_SUGGESTIONS:
            raise ConflictError(f"the detection already has {MAX_SUGGESTIONS} suggestions")
        det = {**copy.deepcopy(d), "suggestions": copy.deepcopy(d["suggestions"]) + [sug]}
        return _step(cur, "suggest", by, source, job_id, agent_version, reason, at, {"planner_detected": det})

    return _write(root, run_id, base_rev, change)


def confirm(root: Path, run_id: str, base_rev: int, *, name: str, by: str, planner_id: str | None = None,
            reason: str | None = None) -> dict:
    """A person confirms and names the run's planner: an `open` or `removed` detection becomes `confirmed`
    (op confirm); a `confirmed` one with another name or id is a rename (op rename); the same again returns the
    record unchanged. `planner_id` defaults to external_planner_id(name), a planner in the customer's own code."""
    if not _text(name, 1, 100):
        raise MetadataError("a name is 1 to 100 characters, not only spaces")
    if not _trace(by):
        raise MetadataError("by is 1 to 100 characters, not only spaces")
    if reason is not None and not _reason(reason):
        raise MetadataError("a reason is 1 to 2000 characters, not only spaces")
    pid = planner_id if planner_id is not None else external_planner_id(name)
    if not (isinstance(pid, str) and _PLANNER_ID.fullmatch(pid)):
        raise MetadataError(f"planner_id {pid!r} is not a planner id (rules-, llm- or ext- and lower case letters "
                            "and digits)")

    def change(cur: dict, run_path: Path):
        d = cur["planner_detected"]
        status = d["status"] if d is not None else None
        c = cur["planner_confirmed"]
        if status == "confirmed" and c is not None and (c["name"], c["planner_id"]) == (name, pid):
            return None
        op = "rename" if status == "confirmed" else "confirm"
        if status not in TRANSITIONS[op]:
            raise ConflictError(f"run {run_id} has no detected planner to confirm; propose one first")
        at = _now()
        new = {"planner_detected": {**copy.deepcopy(d), "status": "confirmed"},
               "planner_confirmed": {"name": name, "planner_id": pid, "by": by, "at": at}}
        return _step(cur, op, by, "human", None, None, reason, at, new)

    return _write(root, run_id, base_rev, change)


def remove(root: Path, run_id: str, base_rev: int, *, by: str, reason: str | None = None) -> dict:
    """A person removes an `open` or `confirmed` detection: its status becomes `removed` and any confirmation is
    cleared, so the run's effective planner is the manifest's again. A person may confirm it again later."""
    if not _trace(by):
        raise MetadataError("by is 1 to 100 characters, not only spaces")
    if reason is not None and not _reason(reason):
        raise MetadataError("a reason is 1 to 2000 characters, not only spaces")

    def change(cur: dict, run_path: Path):
        d = cur["planner_detected"]
        status = d["status"] if d is not None else None
        if status not in TRANSITIONS["remove"]:
            raise ConflictError(f"run {run_id} has no open or confirmed planner to remove"
                                + (" (it is already removed)" if status == "removed" else ""))
        at = _now()
        new = {"planner_detected": {**copy.deepcopy(d), "status": "removed"}, "planner_confirmed": None}
        return _step(cur, "remove", by, "human", None, None, reason, at, new)

    return _write(root, run_id, base_rev, change)


def write_synced(root: Path, rec: dict) -> dict:
    """The sync from Aftershock writes the cloud's record into a synced dataset (one with a well formed sync.json).
    The record must be valid and belong to a committed run; one older than the file on disk (a lower rev) is
    refused with StaleRevError, so a late answer never rolls a newer record back. Under the dataset lock, through a
    temp file and a rename, like every write. A run in a locked batch keeps its record and stays in its batch until
    the batch is unlocked."""
    from . import batches as B
    from . import sync
    root = Path(root)
    s = sync.state(root)
    if s is None or s["error"]:
        raise MetadataError("only a dataset marked as synced from Aftershock (a well formed dataop/sync.json) takes "
                            "the cloud's record")
    run_id = rec.get("run_id") if isinstance(rec, dict) else None
    bad = problems(rec, run_id if isinstance(run_id, str) else None)
    if bad:
        raise MetadataError("the cloud's record is not valid: " + "; ".join(bad[:5]))
    _committed(root, run_id)
    with B._transaction(root):
        cur = load_or_none(root, run_id)
        if cur is not None and rec["rev"] < cur["rev"]:
            raise StaleRevError(cur["rev"], rec["rev"])
        if cur != rec:
            store.atomic_write_json(path(root, run_id), rec)
    return rec


# ---------------------------------------------------------------- validate

def check_run(root: Path, run_path: Path, manifest: dict) -> list[str]:
    """validate's problems with one committed run's metadata record. A run without one has none."""
    rid = manifest["run_id"]
    p = Path(run_path) / FILE
    if not p.exists():
        return []
    try:
        rec = _parse(p.read_bytes(), rid)
    except OSError as e:
        return [f"{rid}: {FILE} cannot be read ({type(e).__name__})"]
    except MetadataError as e:
        return [f"{rid}: {e}"]
    d = rec["planner_detected"]
    if d is not None:
        try:
            found = detect(run_path)
        except Exception as e:                          # unreadable ticks are reported by the run checks
            return [f"{rid}: {FILE}: the detection could not be checked against the run's prompts ({type(e).__name__})"]
        if found != d["found"]:
            return [f"{rid}: {FILE}: the detected planner does not match the run's own prompts"]
    return []


def planner_summary(root: Path, manifest: dict) -> dict:
    """What `runs` and `inspect` show: the effective planner, whether a person confirmed it, the detection's status
    (null when none), the record's rev, and the record's problem when it cannot be read (the manifest's planner is
    shown then)."""
    from .schema import manifest_fields
    rid = manifest["run_id"]
    try:
        rec, err = load(root, rid), None
    except MetadataError as e:
        rec, err = None, str(e)
    d = rec["planner_detected"] if rec else None
    out = {"planner": effective_planner(manifest_fields(manifest)["planner"], rec),
           "planner_confirmed": bool(rec and rec["planner_confirmed"] is not None),
           "planner_detected": d["status"] if d else None, "metadata_rev": rec["rev"] if rec else None}
    if err:
        out["metadata_error"] = err
    return out
