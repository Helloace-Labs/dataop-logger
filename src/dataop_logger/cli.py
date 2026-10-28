"""dataop-logger validate | repair | inspect | remap | view | labels | interpret | events | note | find-idle | suggest-cut"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import validate as V


def main(argv=None) -> int:
    from .review import ReviewError
    try:
        return _main(argv)
    except ReviewError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dataop-logger")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="check the sidecar against the LeRobot data on disk")
    v.add_argument("root")
    v.add_argument("--repair", action="store_true", help="commit exact matches, quarantine the rest")
    v.add_argument("--delete-unmatched", action="store_true", help="with --repair, delete staged runs with no match")
    v.add_argument("--json", action="store_true")
    i = sub.add_parser("inspect", help="runs, outcomes, takeovers and minutes per control mode")
    i.add_argument("root")
    r = sub.add_parser("remap", help="bind runs to a filtered or renumbered copy of the dataset")
    r.add_argument("old_root")
    r.add_argument("new_root")
    r.add_argument("--repo-id", default=None, help="destination dataset repo id, recorded in manifests")
    r.add_argument("--revision", default=None, help="destination dataset revision, recorded in manifests")
    w = sub.add_parser("view", help="open a local page to browse runs next to their videos")
    w.add_argument("root")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--no-open", action="store_true", help="do not open a browser")
    w.add_argument("--build-only", action="store_true", help="write dataop/view/index.html and exit")
    w.add_argument("--read-only", action="store_true", help="browse only; no review edits")
    lb = sub.add_parser("labels", help="list people's span labels as JSON, with context, for an agent to interpret")
    lb.add_argument("root")
    lb.add_argument("--uninterpreted", action="store_true", help="only labels no agent has interpreted yet")
    it = sub.add_parser("interpret", help="record what a label means for training (an agent's reading of it)")
    it.add_argument("root")
    it.add_argument("run_id")
    it.add_argument("label_id")
    it.add_argument("--train", required=True, choices=["keep", "exclude"])
    it.add_argument("--meaning", required=True, help="one sentence: what the person meant")
    it.add_argument("--by", default="agent", help="name of the agent")
    it.add_argument("--replace", action="store_true", help="change an existing interpretation (kept in history)")
    evp = sub.add_parser("events", help="major events of each run (drive mode, task segments, policy failure) as JSON")
    evp.add_argument("root")
    evp.add_argument("--run", default=None, help="only this run id")
    nt = sub.add_parser("note", help="for your own agent: write the details of one major event")
    nt.add_argument("root")
    nt.add_argument("run_id")
    nt.add_argument("key", help="the event key from `dataop-logger events`, e.g. failure or drive@384")
    nt.add_argument("--text", required=True)
    nt.add_argument("--by", default="agent", help="name of the agent")
    q = sub.add_parser("find-idle", help="AI feature: suggest cutting idle stretches (built-in finder)")
    q.add_argument("root")
    q.add_argument("--run", default=None, help="only this run id")
    sc = sub.add_parser("suggest-cut", help="for your own agent: suggest cutting frames [start, end) of a run")
    sc.add_argument("root")
    sc.add_argument("run_id")
    sc.add_argument("start", type=int)
    sc.add_argument("end", type=int)
    sc.add_argument("--reason", required=True, help="short description shown to the person")
    sc.add_argument("--by", default="agent", help="name of the agent making the suggestion")
    a = p.parse_args(argv)

    if a.cmd == "validate":
        if a.repair:
            for line in V.repair(a.root, delete_unmatched=a.delete_unmatched):
                print(line)
        res = V.validate(a.root)
        if a.json:
            print(V.to_json(res))
        else:
            print(f"{'OK' if res['ok'] else 'FAILED'}: {res['runs']} runs, {res['episodes_on_disk']} LeRobot episodes")
            for e in res["errors"]:
                print(f"error: {e}")
            for w in res["warnings"]:
                print(f"warning: {w}")
        return 0 if res["ok"] else 1
    if a.cmd == "inspect":
        print(V.to_json(V.inspect(a.root)))
        return 0
    if a.cmd == "remap":
        for line in V.remap(a.old_root, a.new_root, repo_id=a.repo_id, revision=a.revision):
            print(line)
        return 0
    if a.cmd == "view":
        from . import viewer
        if a.build_only:
            print(viewer.build(a.root))
        else:
            viewer.serve(a.root, port=a.port, open_browser=not a.no_open, editable=not a.read_only)
        return 0
    if a.cmd == "labels":
        from . import review
        print(V.to_json(review.list_labels(a.root, only_uninterpreted=a.uninterpreted)))
        return 0
    if a.cmd == "interpret":
        from . import review, store
        for run_path, m in store.iter_manifests(Path(a.root), "committed"):
            if m["run_id"] == a.run_id:
                review.interpret(run_path, a.label_id, a.train, a.meaning, by=a.by, replace=a.replace)
                print(f"{a.label_id}: {a.train} ({a.meaning})")
                return 0
        print(f"no committed run {a.run_id}", file=sys.stderr)
        return 1
    if a.cmd in ("events", "note"):
        import pyarrow.parquet as pq
        from . import events, review, store
        out, found = [], False
        for run_path, m in store.iter_manifests(Path(a.root), "committed"):
            want = a.run if a.cmd == "events" else a.run_id
            if want and m["run_id"] != want:
                continue
            found = True
            if a.cmd == "note":
                review.add_event_note(run_path, m, a.key, a.text, by=a.by)
                print(f"{m['run_id']}: note on {a.key}")
                continue
            t = pq.read_table(run_path / "ticks.parquet", columns=["control_mode", "expert_source"])
            evs = events.major_events(m, review.load(run_path), t.column("control_mode").to_pylist(),
                                      t.column("expert_source").to_pylist())
            rate = m["action_schema"]["rate_hz"]
            out.append({"run_id": m["run_id"], "episode_index": m["episode_index"], "task": m["task"],
                        "events": [{**e, "time_s": round(e["frame"] / rate, 2)} for e in evs]})
        if not found:
            print("no matching committed run", file=sys.stderr)
            return 1
        if a.cmd == "events":
            print(V.to_json(out))
        return 0
    if a.cmd in ("find-idle", "suggest-cut"):
        from . import review, store
        found = False
        for run_path, m in store.iter_manifests(Path(a.root), "committed"):
            want = a.run if a.cmd == "find-idle" else a.run_id
            if want and m["run_id"] != want:
                continue
            found = True
            if a.cmd == "find-idle":
                for f in review.find_idle(a.root, run_path, m):
                    p = f["edit"]["payload"]
                    print(f"{m['run_id']}: {f['status'].replace('_', ' ')}: {p['start']}-{p['end']} ({f['edit']['reason']})")
            else:
                f = review.suggest_cut(run_path, m, a.start, a.end, a.reason, by=a.by)
                print(f"{m['run_id']}: {f['status'].replace('_', ' ')}: {a.start}-{a.end} ({f['edit']['id']})")
        if not found:
            print("no matching committed run", file=sys.stderr)
            return 1
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
