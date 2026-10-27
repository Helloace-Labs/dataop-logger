"""dataop-logger validate | repair | inspect | remap"""
from __future__ import annotations

import argparse
import sys

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
    return 2


if __name__ == "__main__":
    sys.exit(main())
