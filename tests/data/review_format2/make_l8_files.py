"""Write l8_files.json: review files exactly as the logger wrote them at 4f4395e (logger L8), before edits had
retracted_at and retracted_by.

    python tests/data/review_format2/make_l8_files.py        # from a clone with this repo's git history

The logger at 4f4395e is taken from git history into a temporary folder and run there, so the files are its real
output, not a copy of its shape. Ids and times are fixed, so the output is reproducible. Each entry holds:

  text          the file text that logger wrote
  migrated      whether the review was migrated from format 1 (legacy_subtask_map set)
  batch_digest  the batch lock digest that logger recorded for the run (batches.review_digest, the sha256 of the
                effective view), which must still verify after the take back fields were added
"""
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
L8_COMMIT = "4f4395e"


def build() -> dict:
    """Run inside the extracted 4f4395e source (see main)."""
    from aftershock_logger import batches as B, review_format as fmt

    n = [0xf8000000000]

    def ids():
        n[0] += 1
        return f"{n[0]:012x}"
    fmt.new_id = ids
    manifest = json.loads((HERE / "manifest.json").read_text())
    clock = iter(f"2026-11-04T09:{i // 60:02d}:{i % 60:02d}Z" for i in range(1000))

    def human(r, kind, payload):
        return fmt.propose(r, kind=kind, payload=payload, source="human", by="p1", expect_rev=r["rev"],
                           now=next(clock))

    out = {}

    # not migrated: a take back of a label and of a cut, a replaced edit, a rejected proposal, an agent proposal
    r = fmt.empty_review(manifest["run_id"])
    r, label = human(r, "label", {"start": 10, "end": 20, "text": "arm pauses", "driver": None, "segment": None})
    r, cut = fmt.propose(r, kind="exclude", payload={"start": 0, "end": 5}, source="agent", by="idle-finder",
                         job_id="local-1", agent_version="aftershock-logger 0.3.0", reason="no motion",
                         expect_rev=r["rev"], now=next(clock))
    r, _ = fmt.accept(r, cut["edit_id"], by="p1", expect_rev=r["rev"], now=next(clock))
    r, fs = human(r, "failure_start", {"frame": 50})
    r, _ = fmt.replace(r, fs["edit_id"], {"frame": 52}, by="p1", expect_rev=r["rev"], now=next(clock))
    r, note = fmt.propose(r, kind="note", payload={"frame": 3, "text": "check"}, source="agent", by="agent-a",
                          job_id="j1", agent_version="v1", reason="seen", expect_rev=r["rev"], now=next(clock))
    r, _ = fmt.reject(r, note["edit_id"], by="p1", expect_rev=r["rev"], now=next(clock))
    r, _ = fmt.retract(r, label["edit_id"], by="q", expect_rev=r["rev"], now=next(clock))
    r, _ = fmt.retract(r, cut["edit_id"], by="p1", expect_rev=r["rev"], now=next(clock))
    out["take_back"] = r

    # the same edits without any take back
    r = fmt.empty_review(manifest["run_id"])
    r, label = human(r, "label", {"start": 10, "end": 20, "text": "arm pauses", "driver": None, "segment": None})
    r, fs = human(r, "failure_start", {"frame": 50})
    r, _ = fmt.replace(r, fs["edit_id"], {"frame": 52}, by="p1", expect_rev=r["rev"], now=next(clock))
    out["no_take_back"] = r

    # migrated from format 1 (which holds two format 1 take backs), then a take back made by L8 after the migration
    r = fmt.migrate_format1(json.loads((HERE / "format1_all_kinds.json").read_text()), manifest, None)
    r, _ = fmt.retract(r, "00000000000e", by="q", expect_rev=r["rev"], now=next(clock))
    out["migrated_take_back"] = r

    # migrated with no change since
    out["migrated_no_take_back"] = fmt.migrate_format1(json.loads((HERE / "format1_all_kinds.json").read_text()),
                                                       manifest, None)

    # no edits and no history: the upgrade leaves it as it is
    r = fmt.empty_review(manifest["run_id"])
    r = fmt.lock(r, by="owner", expect_rev=0, now=next(clock), reason="pack locked")
    out["lock_only"] = r

    return {name: {"text": fmt.dump_review(r), "migrated": r["legacy_subtask_map"] is not None,
                   "batch_digest": B.review_digest(manifest, r)} for name, r in out.items()}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--inner":
        sys.path.insert(0, sys.argv[2])
        print(json.dumps(build(), indent=1, sort_keys=True))
        return 0
    tar = subprocess.run(["git", "archive", L8_COMMIT, "src/aftershock_logger"], cwd=REPO, capture_output=True,
                         check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=BytesIO(tar)) as t:
            t.extractall(tmp, filter="data")
        res = subprocess.run([sys.executable, "-I", __file__, "--inner", os.path.join(tmp, "src")],
                             capture_output=True, text=True, check=True)
    (HERE / "l8_files.json").write_text(res.stdout)
    print(f"wrote {len(json.loads(res.stdout))} files written by the logger at {L8_COMMIT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
