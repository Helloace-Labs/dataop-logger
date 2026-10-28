"""dataop-logger validate | repair | inspect | remap | view | labels | interpret | propose"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import validate as V


def main(argv=None) -> int:
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
    q = sub.add_parser("propose", help="add agent review proposals (failure start, idle trims) to committed runs")
    q.add_argument("root")
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
                review.interpret(run_path, a.label_id, a.train, a.meaning)
                print(f"{a.label_id}: {a.train} ({a.meaning})")
                return 0
        print(f"no committed run {a.run_id}", file=sys.stderr)
        return 1
    if a.cmd == "propose":
        from . import review
        for line in review.propose(a.root):
            print(line)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
