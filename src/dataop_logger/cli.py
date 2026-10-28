"""dataop-logger validate | repair | inspect | remap | view | labels | interpret | events | note | find-idle | suggest-cut |
batches"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import validate as V


def main(argv=None) -> int:
    from .batches import BatchError
    from .lists import ListError
    from .review import ReviewError
    try:
        return _main(argv)
    except (ReviewError, ListError, BatchError) as e:
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
    rp = sub.add_parser("runs", help="each run's task segments, drive mode, failure moment and type, and video files (JSON)")
    rp.add_argument("root")
    rp.add_argument("--run", default=None, help="only this run id")
    an = sub.add_parser("annotate", help="for agents: fill in a run's task segments, failure moment and type, or takeover")
    an.add_argument("root")
    an.add_argument("run_id")
    an.add_argument("--segments", default=None,
                    help='JSON list, e.g. [{"name": "grasp", "start_s": 0, "end_s": 4.2}, ...] using the dataset list names')
    an.add_argument("--failure-at", type=float, default=None, help="seconds into the run where the policy starts to fail")
    an.add_argument("--failure-type", default=None, help="a name from the dataset's failure types")
    an.add_argument("--takeover-at", type=float, default=None, help="seconds where control really switched from the policy")
    an.add_argument("--by", required=True, help="name of the agent")
    lp = sub.add_parser("lists", help="the dataset's agreed failure types and task segments; add, rename, describe, delete")
    lp.add_argument("root")
    lp.add_argument("action", nargs="?", choices=["show", "add", "rename", "describe", "delete"], default="show")
    lp.add_argument("kind", nargs="?", choices=["failure_types", "task_segments"])
    lp.add_argument("name", nargs="?")
    lp.add_argument("new_name", nargs="?", help="for rename")
    lp.add_argument("--description", default="")
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
    bp = sub.add_parser("batches", help="batches of runs (one per policy version or DAgger round): list, open, close, lock")
    bp.add_argument("root")
    bs = bp.add_subparsers(dest="op", required=True)
    bs.add_parser("list", help="every batch with its counts (JSON)")
    b1 = bs.add_parser("show", help="one batch with its runs, exports and history (JSON)")
    b1.add_argument("batch", help="batch id or name")
    b1 = bs.add_parser("new", help="open a new batch; new runs join it and the batch collecting until now is closed")
    b1.add_argument("name")
    b1.add_argument("--purpose", default="")
    for op, hlp in (("close", "stop adding new runs; review continues"), ("reopen", "add new runs again"),
                    ("lock", "freeze the review so a training set can be exported"),
                    ("unlock", "allow review edits again; earlier exports are marked stale")):
        b1 = bs.add_parser(op, help=hlp)
        b1.add_argument("batch")
        if op == "unlock":
            b1.add_argument("--reason", required=True)
        if op == "lock":
            b1.add_argument("--force", action="store_true", help="lock even with unreviewed runs or open suggestions")
    b1 = bs.add_parser("move", help="move runs into another batch")
    b1.add_argument("batch")
    b1.add_argument("run_ids", nargs="+")
    b1 = bs.add_parser("edit", help="rename, describe, or record the model trained from this batch (lineage)")
    b1.add_argument("batch")
    b1.add_argument("--name")
    b1.add_argument("--purpose")
    b1.add_argument("--model", help="source or source@revision of the model trained from this batch's export")
    for x in bs.choices.values():
        x.add_argument("--by", default="human", help="who is making the change (a person's or an agent's name)")
    ip = sub.add_parser("import", help="bind runs to an existing LeRobot v3.0 dataset (local folder or Hub repo id)")
    ip.add_argument("source", help="dataset folder, or a Hugging Face Hub repo id (needs --into)")
    ip.add_argument("--into", help="copy the dataset here first and import into the copy")
    ip.add_argument("--preset", default="plain", help="field mapping preset: plain, lerobot-hil or watcher")
    ip.add_argument("--map", help="JSON file with a field mapping (overrides the preset's entries)")
    ip.add_argument("--batch", help="batch the imported runs join (default: 'import <folder name>')")
    ip.add_argument("--episodes", help="only these episodes, e.g. 0-9,12")
    ip.add_argument("--dry-run", action="store_true", help="show what would be imported; write nothing")
    ip.add_argument("--by", default="import")
    xp = sub.add_parser("export", help="generate a training set (LeRobot dataset) from locked batches; no AI involved")
    xp.add_argument("--batch", action="append", required=True, metavar="ROOT:BATCH",
                    help="dataset folder and batch name or id, e.g. data/round3:'round 3'; repeat to combine")
    xp.add_argument("--recipe", required=True, choices=["filtered", "dagger", "residual"])
    xp.add_argument("--into", required=True, help="new folder for the training set")
    xp.add_argument("--with-success", action="store_true", help="dagger: add the policy's own successful runs")
    xp.add_argument("--zero-on-success", action="store_true",
                    help="residual: add successful policy frames with target zero")
    xp.add_argument("--base", help="dagger/filtered: also include every episode of this demo dataset")
    xp.add_argument("--segments", help="only frames in these task segments, comma separated")
    xp.add_argument("--outcomes", help="only runs with these outcomes, comma separated (success,failure,...)")
    xp.add_argument("--val", type=float, default=0.0, help="share of runs for a second dataset <into>_val")
    xp.add_argument("--min-span", type=int, default=1, help="drop kept stretches shorter than this many frames")
    xp.add_argument("--hook", help="module:function or file.py:function for task-specific selection")
    xp.add_argument("--drop-features", help="leave these features out, comma separated (to combine datasets)")
    xp.add_argument("--allow-unlocked", action="store_true", help="export a batch that is not locked (recorded)")
    xp.add_argument("--repo-id", help="repo id written into the new dataset's metadata")
    xp.add_argument("--dry-run", action="store_true", help="count episodes and frames per run; write nothing")
    xp.add_argument("--by", default="human")
    a = p.parse_args(argv)

    if a.cmd == "export":
        from . import export as X
        from .batches import BatchError
        srcs = []
        for spec in a.batch:
            root, sep, ref = spec.partition(":")
            if not sep or not root or not ref:
                print(f"error: --batch takes ROOT:BATCH, got {spec!r}", file=sys.stderr)
                return 2
            srcs.append((root, ref))
        split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None
        try:
            rep = X.export(srcs, a.recipe, a.into, with_success=a.with_success, zero_on_success=a.zero_on_success,
                           base=a.base, segments=split(a.segments), outcomes=split(a.outcomes), val=a.val,
                           min_span=a.min_span, hook=a.hook, allow_unlocked=a.allow_unlocked, dry_run=a.dry_run,
                           repo_id=a.repo_id, by=a.by, drop_features=split(a.drop_features))
        except (X.ExportError, BatchError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(V.to_json(rep))
        return 0
    if a.cmd == "import":
        import json
        from . import importer
        eps = None
        if a.episodes:
            eps = []
            for part in a.episodes.split(","):
                lo, _, hi = part.partition("-")
                eps += list(range(int(lo), int(hi or lo) + 1))
        try:
            rep = importer.import_dataset(a.source, into=a.into, preset=a.preset,
                                          mapping=json.loads(Path(a.map).read_text()) if a.map else None,
                                          batch=a.batch, episodes=eps, dry_run=a.dry_run, by=a.by)
        except importer.ImportError_ as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(V.to_json(rep))
        return 0
    if a.cmd == "batches":
        from . import batches as B
        root = Path(a.root)
        if a.op in ("list", "show"):
            s = B.summary(root)
            print(V.to_json(s if a.op == "list" else next(x for x in s if x["batch_id"] == B.find(B.load(root), a.batch)["batch_id"])))
            return 0
        if a.op == "new":
            b = B.new(root, a.name, a.purpose, by=a.by)
        elif a.op in ("close", "reopen", "lock", "unlock"):
            status = {"close": "reviewing", "reopen": "collecting", "lock": "locked", "unlock": "reviewing"}[a.op]
            cur = B.find(B.load(root), a.batch)
            if a.op == "unlock" and cur["status"] != "locked":
                raise B.BatchError(f"batch {cur['name']!r} is not locked")
            if a.op != "unlock" and cur["status"] == "locked":
                raise B.BatchError(f"batch {cur['name']!r} is locked; unlock it first")
            b = B.set_status(root, a.batch, status, by=a.by, reason=getattr(a, "reason", None),
                             force=getattr(a, "force", False))
        elif a.op == "move":
            b = B.move(root, a.run_ids, a.batch, by=a.by)
        else:
            b = B.edit(root, a.batch, by=a.by, name=a.name, purpose=a.purpose, model=a.model)
        print(f"{b['name']} ({b['batch_id']}): {b['status']}")
        return 0

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
    if a.cmd == "runs":
        from . import annotate
        print(V.to_json(annotate.runs_json(a.root, a.run)))
        return 0
    if a.cmd == "annotate":
        import json as _json
        from . import annotate
        segs = _json.loads(a.segments) if a.segments else None
        for line in annotate.annotate(a.root, a.run_id, a.by, segments=segs, failure_at=a.failure_at,
                                      failure_type=a.failure_type, takeover_at=a.takeover_at):
            print(line)
        return 0
    if a.cmd == "lists":
        from . import lists
        if a.action == "show":
            print(V.to_json(lists.load(a.root)))
        elif a.action == "add":
            lists.add(a.root, a.kind, a.name, a.description)
            print(f"added {a.name} to {a.kind}")
        elif a.action == "rename":
            lists.rename(a.root, a.kind, a.name, a.new_name)
            print(f"renamed {a.name} to {a.new_name} in {a.kind}")
        elif a.action == "describe":
            lists.describe(a.root, a.kind, a.name, a.description)
            print(f"described {a.name} in {a.kind}")
        else:
            lists.delete(a.root, a.kind, a.name)
            print(f"deleted {a.name} from {a.kind}")
        return 0
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
