"""Logger task L4: the run metadata record for a planner found later (detection and confirmation), versioned, outside
the manifest.

`dataop/runs/<run_id>/metadata.json` holds what was learned after recording: planner_detected (from the run's own
prompts), planner_confirmed (a person's) and a history that keeps every change whole. The effective planner (the
confirmed one, else the manifest's) drives batch identity, mixed_policy and export records.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aftershock_logger import metadata as md  # noqa: E402
from aftershock_logger import batches as B  # noqa: E402
from aftershock_logger import store  # noqa: E402

VECTORS = Path(md.__file__).with_name("planner_detect_vectors.json")


# ---------------------------------------------------------------- the rule (no LeRobot needed)

def test_the_rule_gives_every_answer_in_the_shared_vectors():
    doc = json.loads(VECTORS.read_text())
    assert doc["vectors_format"] == 1 and doc["rule"] == md.DETECTION_RULE and len(doc["cases"]) >= 10
    for case in doc["cases"]:
        assert md.detect_planner(case["prompts"]) == case["expect"], case["name"]


def test_the_rule_on_a_changing_prompt_and_on_none():
    found = md.detect_planner(["unknown", "reach", "reach", "grasp", "reach", "lift"])
    assert [s["text"] for s in found["subtasks"]] == ["reach", "grasp", "lift"]
    assert found["first_seen_frame"] == 3 and found["distinct_prompts"] == 3
    assert md.detect_planner(["reach"] * 5) is None
    assert md.detect_planner(["unknown"] * 5) is None
    assert md.detect_planner([None, "", "unknown"]) is None
    long = "z" * 2001
    got = md.detect_planner([long, "a"])["subtasks"][0]
    assert got["cut"] and len(got["text"]) == md.MAX_PROMPT_CHARS and got["sha256"] == md._sha(long)


# ---------------------------------------------------------------- record checks (no LeRobot needed)

def _det(status="open", source="rule"):
    return {"found": md.detect_planner(["a", "b"]), "source": source, "by": md.DETECTION_RULE if source == "rule"
            else "ana", "job_id": None, "agent_version": md.RULE_VERSION if source == "rule" else None,
            "reason": None, "proposed_at": "2026-11-09T10:00:00Z", "suggestions": [], "status": status}


def _record(run_id="r1"):
    d0 = _det()
    d1 = {**d0, "status": "confirmed"}
    c1 = {"name": "their rules", "planner_id": "ext-their-rules", "by": "ana", "at": "2026-11-09T10:01:00Z"}
    step = lambda rev, op, by, src, changes: {"rev": rev, "op": op, "by": by, "source": src, "job_id": None,
                                              "agent_version": md.RULE_VERSION if src == "rule" else None,
                                              "reason": None, "at": "2026-11-09T10:00:00Z", "changes": changes}
    return json.loads(json.dumps({"metadata_format": 1, "run_id": run_id, "rev": 2, "planner_detected": d1, "planner_confirmed": c1,
            "history": [step(1, "propose", md.DETECTION_RULE, "rule",
                             [{"field": "planner_detected", "before": None, "after": d0}]),
                        step(2, "confirm", "ana", "human",
                             [{"field": "planner_detected", "before": d0, "after": d1},
                              {"field": "planner_confirmed", "before": None, "after": c1}])]}))


def test_a_valid_record_has_no_problems_and_rebuilds_every_revision():
    rec = _record()
    assert md.problems(rec, "r1") == []
    assert md.metadata_at(rec, 0) == {"planner_detected": None, "planner_confirmed": None}
    assert md.metadata_at(rec, 1)["planner_detected"]["status"] == "open"
    assert md.metadata_at(rec, 1)["planner_confirmed"] is None
    assert md.metadata_at(rec, 2) == {"planner_detected": rec["planner_detected"],
                                      "planner_confirmed": rec["planner_confirmed"]}
    got = md.metadata_at(rec, 2)
    got["planner_detected"]["status"] = "removed"            # a fresh copy: the record is unchanged
    assert rec["planner_detected"]["status"] == "confirmed"
    assert md.effective_planner(md.UNKNOWN, rec) == {"name": "their rules", "id": "ext-their-rules"}
    assert md.effective_planner(None, md.empty("r1")) is None


@pytest.mark.parametrize("breaks, why", [
    (lambda r: r.update(metadata_format=2), "format"),
    (lambda r: r.update(run_id="r2"), "does not match its run folder"),
    (lambda r: r.update(rev=3), "rev equals the number of history steps"),
    (lambda r: r.update(extra=1), "unknown keys"),
    (lambda r: r["planner_confirmed"].update(planner_id="Their Rules"), "planner_id names a planner"),
    (lambda r: r["planner_confirmed"].update(name="   "), "name is 1 to 100"),
    (lambda r: r.update(planner_confirmed=None), "a confirmed detection has a confirmation"),
    (lambda r: r["planner_detected"]["found"]["subtasks"].append(
        dict(r["planner_detected"]["found"]["subtasks"][0])), "distinct"),
    (lambda r: r["planner_detected"].update(by="someone"), "by planner-detect"),
    (lambda r: r["history"][1]["changes"][0]["before"].update(status="removed"), "before does not match"),
    (lambda r: r["history"][1].update(op="approve"), "op is one of"),
    (lambda r: r["history"][0].update(rev=True), "rev is an int"),
    (lambda r: r["planner_detected"]["found"].update(distinct_prompts=5), "capped only at"),
])
def test_a_broken_record_is_refused(breaks, why):
    rec = _record()
    breaks(rec)
    bad = md.problems(rec, "r1")
    assert any(why in p for p in bad), bad


def test_every_revision_keeps_the_pair_rule():
    rec = _record()
    # confirm in a step of its own before the detection is confirmed: the last state is fine, rev 2 is not
    c = rec["history"][1]["changes"]
    rec["history"][1]["changes"] = [c[1]]
    rec["history"].append({**rec["history"][1], "rev": 3, "changes": [c[0]]})
    rec["rev"] = 3
    assert any("history step 2: a confirmation needs a detection whose status is confirmed" in p
               for p in md.problems(rec, "r1"))


# ---------------------------------------------------------------- on a dataset

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")

from aftershock_logger import ActionSchema, PlannerInfo, PolicyInfo, RunLogger  # noqa: E402
from aftershock_logger import cli  # noqa: E402
from aftershock_logger import export as X  # noqa: E402
from aftershock_logger import sync as S  # noqa: E402
from aftershock_logger import validate as V  # noqa: E402
from aftershock_logger.annotate import runs_json  # noqa: E402

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCHEMA = ActionSchema(names=NAMES, units=("deg",) * 5 + ("pct",), space="joint_position", rate_hz=30)
TASK = "pick up the tube"
CHANGING = ["reach the tube", "reach the tube", "grasp the cap", "grasp the cap", "lift", "lift"]


def make_ds(tmp_path, name="ds"):
    feats = {k: {"dtype": "float32", "shape": (6,), "names": list(NAMES)} for k in ("observation.state", "action")}
    return lerobot_dataset.LeRobotDataset.create(f"test/{name}", fps=30, features=feats, root=tmp_path / name,
                                                 use_videos=False)


def record(log, prompts, **run_kw):
    rev, last = -1, None
    with log.run(task=TASK, policy=PolicyInfo("smolvla", "train/smolvla_100"), **run_kw) as run:
        for i, p in enumerate(prompts):
            if p not in (None, "unknown") and p != last:
                rev, last = rev + 1, p
            t = time.monotonic_ns()
            a = np.arange(6, dtype=np.float32) + i
            kw = {} if p is None else {"prompt": p, "prompt_rev": rev if p != "unknown" else "unknown"}
            run.add_frame({"observation.state": a, "task": TASK}, sent_action=a, base_action=a, control_mode="base",
                          inference_id=i, chunk_offset=0, obs_ns=t, sent_ns=t + 1000, **kw)
        run.end(termination_reason="success")
    return run


@pytest.fixture
def ds(tmp_path):
    """Three runs recorded by the customer's own code in Watch mode: nobody said a planner ran (unknown). Two have a
    changing prompt, one keeps a single prompt."""
    d = make_ds(tmp_path)
    with RunLogger(d, SCHEMA) as log:
        runs = [record(log, CHANGING), record(log, CHANGING), record(log, ["reach the tube"] * 4)]
    d.finalize()
    d.ids = [r.run_id for r in runs]
    return d


def batch_of(root, rid):
    return B.batch_of(root, rid)


def manifest_bytes(root, rids):
    return {r: (store.runs_dir(root) / r / "manifest.json").read_bytes() for r in rids}


def test_no_file_reads_as_rev_0_and_still_validates(ds):
    rid = ds.ids[0]
    assert md.load(ds.root, rid) == md.empty(rid)
    assert not md.path(ds.root, rid).exists()
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    run = next(r for r in runs_json(ds.root)["runs"] if r["run_id"] == rid)
    assert (run["planner"], run["planner_confirmed"], run["planner_detected"], run["metadata_rev"]) == \
        ("unknown", False, None, 0)


def test_detection_from_a_run_and_none_for_one_prompt(ds):
    found = md.detect(store.runs_dir(ds.root) / ds.ids[0])
    assert [s["text"] for s in found["subtasks"]] == ["reach the tube", "grasp the cap", "lift"]
    assert found["first_seen_frame"] == 2 and found["distinct_prompts"] == 3
    assert md.detect(store.runs_dir(ds.root) / ds.ids[2]) is None
    with pytest.raises(md.NoPlannerError):
        md.propose(ds.root, ds.ids[2], 0, by="ana")
    assert not md.path(ds.root, ds.ids[2]).exists()


def test_every_operation_bumps_rev_with_one_whole_step(ds):
    rid = ds.ids[0]
    before = manifest_bytes(ds.root, ds.ids)
    r1 = md.propose(ds.root, rid, 0, by="x", source="rule")
    assert r1["rev"] == 1 and r1["planner_detected"]["status"] == "open"
    assert r1["planner_detected"]["by"] == md.DETECTION_RULE and r1["planner_detected"]["found"]["distinct_prompts"] == 3
    r2 = md.propose(ds.root, rid, 1, by="helper", source="agent", name="rules v2", job_id="job-1",
                    agent_version="a1", reason="three prompts in order")
    assert r2["rev"] == 2 and r2["history"][-1]["op"] == "suggest"
    assert [s["name"] for s in r2["planner_detected"]["suggestions"]] == ["rules v2"]
    r3 = md.confirm(ds.root, rid, 2, name="their rules", by="ana")
    assert r3["rev"] == 3 and r3["history"][-1]["op"] == "confirm"
    assert r3["planner_confirmed"]["planner_id"] == "ext-their-rules"
    r4 = md.confirm(ds.root, rid, 3, name="their rules v2", by="ana")
    assert r4["rev"] == 4 and r4["history"][-1]["op"] == "rename"
    r5 = md.remove(ds.root, rid, 4, by="ana", reason="it was a prompt sweep, not a planner")
    assert r5["rev"] == 5 and r5["planner_detected"]["status"] == "removed" and r5["planner_confirmed"] is None
    r6 = md.confirm(ds.root, rid, 5, name="their rules v2", by="ben")       # a person reverses the removal
    assert r6["rev"] == 6 and r6["history"][-1]["op"] == "confirm"
    for rec, prev in zip((r1, r2, r3, r4, r5, r6), (md.empty(rid), r1, r2, r3, r4, r5)):
        step = rec["history"][-1]
        assert len(rec["history"]) == rec["rev"] == prev["rev"] + 1
        for c in step["changes"]:                            # every changed field whole, before and after
            assert c["before"] == prev[c["field"]] and c["after"] == rec[c["field"]]
        assert {c["field"] for c in step["changes"]} == {f for f in md.FIELDS if rec[f] != prev[f]}
    on_disk = md.load(ds.root, rid)
    assert on_disk == r6 and md.problems(on_disk, rid) == []
    for rec in (r1, r2, r3, r4, r5):                         # every earlier revision rebuilds from the history
        assert md.metadata_at(on_disk, rec["rev"]) == {f: rec[f] for f in md.FIELDS}
    assert manifest_bytes(ds.root, ds.ids) == before         # the manifests never change
    assert V.validate(ds.root)["ok"]


def test_a_later_proposal_never_changes_a_detection_or_a_decision(ds):
    rid = ds.ids[0]
    md.propose(ds.root, rid, 0, by="ana", name="rules")
    rec = md.confirm(ds.root, rid, 1, name="their rules", by="ana")
    again = md.propose(ds.root, rid, 2, by="x", source="rule")                  # no name: nothing to add
    assert again == rec and md.load(ds.root, rid)["rev"] == 2
    same = md.propose(ds.root, rid, 2, by="ana", name="rules")                  # same name, same proposer
    assert same == rec
    other = md.propose(ds.root, rid, 2, by="bot", source="agent", name="other", job_id="j", agent_version="v",
                       reason="looks like a different planner")
    assert other["rev"] == 3 and other["planner_confirmed"] == rec["planner_confirmed"]
    det = {k: v for k, v in other["planner_detected"].items() if k != "suggestions"}
    assert det == {k: v for k, v in rec["planner_detected"].items() if k != "suggestions"}
    assert md.confirm(ds.root, rid, 3, name="their rules", by="ana")["rev"] == 3   # the same again: unchanged


def test_stale_rev_and_transitions_are_refused(ds):
    rid = ds.ids[0]
    with pytest.raises(md.ConflictError):
        md.confirm(ds.root, rid, 0, name="x", by="ana")                    # nothing detected yet
    with pytest.raises(md.ConflictError):
        md.remove(ds.root, rid, 0, by="ana")
    md.propose(ds.root, rid, 0, by="ana")
    with pytest.raises(md.StaleRevError) as e:
        md.confirm(ds.root, rid, 0, name="x", by="ana")
    assert e.value.current == 1
    md.remove(ds.root, rid, 1, by="ana")
    with pytest.raises(md.ConflictError, match="already removed"):
        md.remove(ds.root, rid, 2, by="ana")
    with pytest.raises(md.MetadataError, match="agent"):
        md.propose(ds.root, rid, 2, by="bot", source="agent", name="x")       # an agent needs its trace
    with pytest.raises(md.MetadataError, match="planner id"):
        md.confirm(ds.root, rid, 2, name="x", by="ana", planner_id="Bad Id")
    with pytest.raises(md.MetadataError, match="no committed run"):
        md.propose(ds.root, "../x", 0, by="ana")
    assert md.load(ds.root, rid)["rev"] == 2


def test_a_crash_during_a_write_leaves_the_old_file_whole(ds, monkeypatch):
    rid = ds.ids[0]
    md.propose(ds.root, rid, 0, by="ana")
    old = md.path(ds.root, rid).read_bytes()

    def boom(*a, **k):
        raise OSError("power cut")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="power cut"):
        md.confirm(ds.root, rid, 1, name="their rules", by="ana")
    monkeypatch.undo()
    assert md.path(ds.root, rid).read_bytes() == old
    assert not [p for p in md.path(ds.root, rid).parent.iterdir() if p.name.endswith(".tmp")]
    assert md.load(ds.root, rid)["rev"] == 1


def test_a_confirmed_run_moves_to_the_batch_of_the_confirmed_pair(ds):
    a, b, c = ds.ids
    home = batch_of(ds.root, a)
    assert home["batch_id"] == batch_of(ds.root, b)["batch_id"] == batch_of(ds.root, c)["batch_id"]
    assert home["identity"]["planner"] == "unknown"
    md.propose(ds.root, a, 0, by="ana")
    assert batch_of(ds.root, a)["batch_id"] == home["batch_id"]         # a proposal moves nothing
    md.confirm(ds.root, a, 1, name="their rules", by="ana")
    new = batch_of(ds.root, a)
    assert new["batch_id"] != home["batch_id"] and new["identity"]["planner"] == "ext-their-rules"
    assert new["identity"] == {**home["identity"], "planner": "ext-their-rules"} and not new["mixed_policy"]
    md.propose(ds.root, b, 0, by="ana")
    md.confirm(ds.root, b, 1, name="Their Rules!", by="ana")            # the same id: the same batch
    assert batch_of(ds.root, b)["batch_id"] == new["batch_id"]
    assert batch_of(ds.root, c)["batch_id"] == home["batch_id"]
    s = {x["batch_id"]: x for x in B.summary(ds.root)}
    assert sorted(s[new["batch_id"]]["runs"]) == sorted([a, b]) and not s[new["batch_id"]]["mixed_policy"]
    assert not s[home["batch_id"]]["mixed_policy"]
    with pytest.raises(B.BatchError, match="confirmed"):                # a confirmed run stays with its pair
        B.move(ds.root, [a], home["batch_id"])
    md.remove(ds.root, a, 2, by="ana")                                  # back to the manifest's identity
    assert batch_of(ds.root, a)["batch_id"] == home["batch_id"]
    out = runs_json(ds.root)["runs"]
    got = {r["run_id"]: (r["planner"], r["planner_confirmed"], r["planner_detected"]) for r in out}
    assert got == {a: ("unknown", False, "removed"),
                   b: ({"name": "Their Rules!", "id": "ext-their-rules"}, True, "confirmed"),
                   c: ("unknown", False, None)}
    info = {json.dumps(x["planner"]): x for x in V.inspect(ds.root)["planners"]}
    assert info['"unknown"']["runs"] == 2 and info['"unknown"']["confirmed"] == 0
    assert info[json.dumps({"name": "Their Rules!", "id": "ext-their-rules"})]["confirmed"] == 1
    assert V.validate(ds.root)["ok"], V.validate(ds.root)["errors"]


def test_a_run_in_a_locked_batch_stays_until_unlocked(ds):
    a = ds.ids[0]
    md.propose(ds.root, a, 0, by="ana")
    home = batch_of(ds.root, a)
    B.set_status(ds.root, home["batch_id"], "locked", force=True)
    with pytest.raises(B.BatchError, match="locked"):                   # a local change waits for an unlock
        md.confirm(ds.root, a, 1, name="their rules", by="ana")
    assert md.load(ds.root, a)["rev"] == 1
    # the cloud's confirmation arrives by sync while the batch is locked: the record changes, the run does not move
    store.atomic_write_json(S.marker_path(ds.root), {"sync_format": 1, "source": "aftershock", "workspace_id": "w",
                                                     "station_id": "st", "since": "2026-11-09T10:00:00Z"})
    cur = md.load(ds.root, a)
    d1 = {**cur["planner_detected"], "status": "confirmed"}
    c1 = {"name": "their rules", "planner_id": "ext-their-rules", "by": "person:owner", "at": "2026-11-09T11:00:00Z"}
    cloud = {**copy.deepcopy(cur), "rev": 2, "planner_detected": d1, "planner_confirmed": c1,
             "history": cur["history"] + [{"rev": 2, "op": "confirm", "by": "person:owner", "source": "human",
                                           "job_id": None, "agent_version": None, "reason": None,
                                           "at": "2026-11-09T11:00:00Z", "changes": [
                                               {"field": "planner_detected", "before": cur["planner_detected"],
                                                "after": d1},
                                               {"field": "planner_confirmed", "before": None, "after": c1}]}]}
    md.write_synced(ds.root, cloud)
    assert md.load(ds.root, a) == cloud
    assert batch_of(ds.root, a)["batch_id"] == home["batch_id"]
    assert B.load(ds.root)["members"][a] == home["batch_id"]
    res = V.validate(ds.root)
    assert res["ok"], res["errors"]
    assert any("stays in locked batch" in w for w in res["warnings"])
    with pytest.raises(md.StaleRevError):                               # an older answer never rolls it back
        md.write_synced(ds.root, cur)
    with pytest.raises(S.SyncedError):                                  # synced: only the sync writes it
        md.remove(ds.root, a, 2, by="ana")
    B.set_status(ds.root, home["batch_id"], "reviewing", reason="planner confirmed")
    moved = batch_of(ds.root, a)
    assert moved["batch_id"] != home["batch_id"] and moved["identity"]["planner"] == "ext-their-rules"
    assert all(e["stale"] for e in B.find(B.load(ds.root), home["batch_id"])["exports"])


def test_write_synced_needs_a_synced_dataset_and_a_valid_record(ds):
    a = ds.ids[0]
    rec = md.empty(a)
    with pytest.raises(md.MetadataError, match="synced"):
        md.write_synced(ds.root, rec)
    store.atomic_write_json(S.marker_path(ds.root), {"sync_format": 1, "source": "aftershock", "workspace_id": "w",
                                                     "station_id": "st", "since": "2026-11-09T10:00:00Z"})
    with pytest.raises(md.MetadataError, match="not valid"):
        md.write_synced(ds.root, {**rec, "rev": 1})


def test_validate_reports_a_bad_file(ds):
    a = ds.ids[0]
    md.propose(ds.root, a, 0, by="ana")
    good = md.path(ds.root, a).read_bytes()
    for text, why in (('{"metadata_format": 1', "not valid JSON"),
                      (json.dumps({**json.loads(good), "run_id": "other"}), "does not match its run folder"),
                      (json.dumps({**json.loads(good), "rev": 4}), "rev equals"),
                      (json.dumps({**json.loads(good), "metadata_format": 9}), "format")):
        md.path(ds.root, a).write_text(text)
        res = V.validate(ds.root)
        assert not res["ok"] and any(why in e for e in res["errors"]), (why, res["errors"])
        run = next(r for r in runs_json(ds.root)["runs"] if r["run_id"] == a)
        assert run["metadata_error"] and run["planner"] == "unknown"     # the manifest's planner, never a guess
    # a detection that the run's own prompts do not show
    rec = json.loads(good)
    rec["planner_detected"]["found"]["first_seen_frame"] = 1
    rec["history"][0]["changes"][0]["after"]["found"]["first_seen_frame"] = 1
    md.path(ds.root, a).write_text(json.dumps(rec))
    assert any("does not match the run's own prompts" in e for e in V.validate(ds.root)["errors"])
    md.path(ds.root, a).write_bytes(good)
    assert V.validate(ds.root)["ok"]


def test_export_records_the_effective_planner(ds, tmp_path):
    a, b, c = ds.ids
    md.propose(ds.root, a, 0, by="ana")
    md.confirm(ds.root, a, 1, name="their rules", by="ana")
    bid = batch_of(ds.root, a)["batch_id"]
    rep = X.export([(str(ds.root), bid)], "filtered", str(tmp_path / "out"), allow_unlocked=True, dry_run=True)
    run = next(r for r in rep["runs"] if r["run_id"] == a)
    assert run["planner"] == {"name": "their rules", "id": "ext-their-rules"} and run["metadata_rev"] == 2
    home = batch_of(ds.root, c)["batch_id"]
    rep = X.export([(str(ds.root), home)], "filtered", str(tmp_path / "out2"), allow_unlocked=True, dry_run=True)
    assert all("planner" not in r and "metadata_rev" not in r for r in rep["runs"])   # no record: as before


def test_the_command_line(ds, capsys):
    a, _, c = ds.ids
    assert cli.main(["planner", "detect", str(ds.root)]) == 0
    out = {r["run_id"]: r for r in json.loads(capsys.readouterr().out)}
    assert out[a]["found"]["distinct_prompts"] == 3 and out[c]["found"] is None
    assert cli.main(["planner", "propose", str(ds.root), a, "--rev", "0", "--source", "rule"]) == 0
    assert cli.main(["planner", "confirm", str(ds.root), a, "--rev", "0", "--name", "x", "--by", "ana"]) == 2
    assert "rev 1" in capsys.readouterr().err
    assert cli.main(["planner", "confirm", str(ds.root), a, "--rev", "1", "--name", "their rules", "--by", "ana"]) == 0
    assert "confirmed as 'their rules'" in capsys.readouterr().out
    assert cli.main(["planner", "show", str(ds.root), a]) == 0
    assert json.loads(capsys.readouterr().out)["rev"] == 2
    assert cli.main(["planner", "propose", str(ds.root), c, "--rev", "0", "--by", "ana"]) == 2
    assert "never changes" in capsys.readouterr().err
