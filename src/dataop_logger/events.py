"""Major events of a run, the same three kinds for every task:

  drive    who drives from this moment (policy, rescue, shared, nobody, or a team's own drive mode)
  segment  which task segment starts here (the recorded steps, corrected by people's labels)
  failure  the moment the policy fails (marked by a person or agent, otherwise estimated)

plus the run's start and end. Each event has a stable key, a short generic title and one note: the details. A note
comes from recorded data by default and can be replaced by a person, the user's own agent or an AI feature
(review kind "event_note").
"""
from __future__ import annotations

from . import review as rv

DRIVER = {"base": "policy", "expert": "rescue", "blend": "shared", "none": "nobody", "unknown": "not recorded"}


def _spans(values: list) -> list[tuple[int, int, object]]:
    out = []
    for i, v in enumerate(values):
        if out and out[-1][2] == v:
            out[-1] = (out[-1][0], i + 1, v)
        else:
            out.append((i, i + 1, v))
    return out


def major_events(manifest: dict, review: dict, modes: list[str], sources: list, with_segments: bool = False):
    """[{key, kind, frame, title, note, note_by, estimated?}] in time order. Uses the review as corrected."""
    eff = rv.effective(manifest, review)
    n, rate = manifest["length"], manifest["action_schema"]["rate_hz"]
    events = []

    def add(kind, frame, key, title, note=None, **more):
        events.append({"key": key, "kind": kind, "frame": frame, "title": title, "note": note, "note_by": "recorded"
                       if note else None, **more})

    # drive mode: recorded control-mode spans, with people's moved switch points, then drive-mode labels on top
    rec = [[a, b, v] for a, b, v in _spans(list(zip(modes, sources)))]
    spans = [list(x) for x in rec]
    for p in eff.get("drive_edits", []):
        i = p["index"]
        if i + 1 < len(spans):
            f = min(max(p["frame"], spans[i][0] + 1), spans[i + 1][1] - 1)
            spans[i][1] = spans[i + 1][0] = f
    modes, sources = [None] * n, [None] * n
    for a, b, (m, src) in spans:
        for f in range(a, b):
            modes[f], sources[f] = m, src
    drive_spans = [{"index": i, "mode": v[0], "driver": DRIVER.get(v[0], v[0]), "source": v[1], "start": a, "end": b,
                    "rec_start": rec[i][0], "rec_end": rec[i][1]} for i, (a, b, v) in enumerate(spans)]
    driver = [DRIVER.get(m, m) for m in modes]
    who = [s for s in sources]
    for lb in sorted(eff["labels"], key=lambda x: x["start"]):
        if lb.get("driver"):
            for f in range(lb["start"], min(lb["end"], n)):
                driver[f], who[f] = lb["driver"], None
    requests = [e for e in manifest.get("events", []) if e.get("kind") == "takeover"]
    for a, _, d in _spans(list(zip(driver, who))):
        name, src = d
        req = next((e for e in requests if a - rate <= e["frame_index"] <= a), None) if name != "policy" else None
        note = "; ".join(x for x in [src and f"by {src}", req and req.get("reason"), req and req.get("text")] if x) or None
        add("drive", a, f"drive@{a}", f"{name.capitalize()} drives", note)

    # task segments: each segment as recorded or as a person replanned them; neighbours with the same name stay
    # separate segments
    seg_spans = []
    for i, s in enumerate(sorted(eff["subgoals"], key=lambda x: x["start_frame"])):
        a, b = s["start_frame"], min(s["end_frame"], n)
        if b <= a:
            continue
        seg_spans.append({"name": s["name"], "start": a, "end": b, "status": s.get("status"), "from_label": False,
                          "index": i})
        note = ("failed" + (f": {s['reason']}" if s.get("reason") else "")) if s.get("status") == "failed" else None
        add("segment", a, f"segment:{s['name']}@{a}", f"{s['name']}", note)

    # the moment the policy fails
    if eff["termination_reason"] != "success" and not eff["rejected"]:
        failed = next((s for s in eff["subgoals"] if s["status"] == "failed"), None)
        frame, estimated = eff["failure_start"], False
        if frame is None:
            estimated = True
            first_help = next((i for i, m in enumerate(modes) if m in ("expert", "blend") and i > 0
                               and modes[i - 1] == "base"), None)
            frame = first_help if first_help is not None else (failed["end_frame"] - 1 if failed else None)
        if frame is not None:
            detail = [eff["failure_mode"] and eff["failure_mode"].replace("_", " "),
                      failed and f"{failed['name']} failed" + (f": {failed['reason']}" if failed.get("reason") else "")]
            add("failure", frame, "failure", "Policy fails", "; ".join(x for x in detail if x) or None,
                estimated=estimated, window=[max(0, frame - int(round(2 * rate))), frame])

    outcome = "rejected" if eff["rejected"] else eff["termination_reason"]
    add("end", n, "end", f"Run ends: {outcome}")

    notes = {k: v for k, v in eff.get("event_notes", {}).items()}
    for e in events:
        if e["key"] in notes:
            e["note"], e["note_by"] = notes[e["key"]]["text"], notes[e["key"]]["by"]
    order = {"drive": 0, "segment": 1, "failure": 2, "end": 3}
    events = sorted(events, key=lambda e: (e["frame"], order[e["kind"]]))
    return (events, seg_spans, drive_spans) if with_segments else events
