"""Review format 2 (logger task L8): the logger reads and writes the review format it shares with the Aftershock
cloud, byte for byte, and reads its own format 1 files through a lossless migration.

Fixtures are in tests/data/review_format2 (synthetic runs, see review_cases.py). Three optional environment variables
add checks that need material outside this repo; each test that needs one skips without it:

  AFTERSHOCK_CONTRACTS     a checkout with the reference implementation: every case and random operation sequences
                           are run against both implementations, and random changes to valid files must be refused
                           by both or accepted by both
  AFTERSHOCK_REVIEW_DATA   a folder to search for real dataop/runs/<run_id>/review.json files; each one is copied to
                           a temporary folder (the originals are only read) and migrated, back and forth
"""
import copy
import json
import os
import random
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import review_cases as rc  # noqa: E402
from aftershock_logger import batches as B, review as rv, review_format as fmt, store  # noqa: E402

EXPECTED = rc.read("expected.json")
MANIFEST = rc.read("manifest.json")
TASK = rc.read("task.json")


def dumps(o) -> str:
    return json.dumps(o, indent=1, sort_keys=True) + "\n"


# ------------------------------------------------------------------ against the committed reference results


@pytest.mark.parametrize("name", sorted(rc.CASES))
def test_operations_give_the_reference_bytes(name):
    got = rc.run_case(rc.LoggerImpl(), name)["steps"]
    want = EXPECTED["cases"][name]
    for i, (g, w) in enumerate(zip(got, want)):
        assert g == w, f"{name}: step {i} ({g['op']}) differs"
    assert len(got) == len(want)


def test_migration_gives_the_reference_bytes_both_ways():
    got = rc.migration_results(rc.LoggerImpl())
    assert got == EXPECTED["migration"]
    for f1 in ("format1_all_kinds.json", "format1_minimal.json"):
        raw = rc.read(f1)
        for label in ("no task", "task", "task reordered"):
            res = got[f"{f1} {label}"]
            # lossless: back to format 1 is the logger's own text for the same review
            assert res["format1"] == dumps(fmt.normalize_format1(raw))
            # byte stable: the reference text reads back to the same bytes
            assert fmt.dump_review(fmt.load_review(res["format2"])) == res["format2"]
        # names only, never positions: a reordered task gives the same bytes
        assert got[f"{f1} task"]["format2"] == got[f"{f1} task reordered"]["format2"]
    # the full fixture has every optional key, so its round trip is its exact file text
    raw_text = (rc.DATA / "format1_all_kinds.json").read_text()
    assert got["format1_all_kinds.json task"]["format1"] == raw_text


def test_migration_maps_by_recorded_name_and_never_guesses():
    raw = rc.read("format1_all_kinds.json")
    with_task = fmt.migrate_format1(raw, MANIFEST, TASK)
    entries = {(e["step_index"], e["recorded_name"]): e["subtask_id"] for e in with_task["legacy_subtask_map"]["entries"]}
    assert entries == {(0, "reach"): "s-reach", (1, " Grasp "): "s-grasp", (2, "lift"): "s-lift",
                       (None, "Grasp"): "s-grasp", (None, "reach"): "s-reach", (None, "wiggle"): "unresolved"}
    segs = next(e for e in with_task["edits"] if e["kind"] == "segments")["payload"]["segments"]
    assert [(s["subtask_id"], s["legacy_name"], s["start"], s["end"]) for s in segs] == [
        ("s-reach", "reach", 0, 28), ("s-grasp", "Grasp", 28, 72), ("unresolved", "wiggle", 72, 110)]
    without = fmt.migrate_format1(raw, MANIFEST, None)
    assert {e["subtask_id"] for e in without["legacy_subtask_map"]["entries"]} == {"unresolved"}
    assert without["legacy_subtask_map"]["task_id"] is None
    # agent edits from format 1 never recorded a job or a version
    agent = [e for e in without["edits"] if e["source"] == "agent"]
    assert agent and all((e["job_id"], e["agent_version"]) == ("unknown", "unknown") for e in agent)
    label = next(e for e in without["edits"] if e["edit_id"] == "00000000000c")
    assert label["legacy_reading"]["by"] == "reader-agent" and len(label["legacy_reading_history"]) == 1


@pytest.mark.parametrize("change", [
    lambda r: r.__setitem__("extra", 1),
    lambda r: r["edits"][0].__setitem__("kind", "failure_type"),
    lambda r: r["edits"][0].__setitem__("source", "check"),
    lambda r: r["edits"][0].__setitem__("adjusted", False),
    lambda r: r["edits"][0].__setitem__("edit_id", "x"),
    lambda r: r["edits"][0].pop("reason"),
    lambda r: r["edits"][0]["payload"].__setitem__("outcome", "failure"),
    lambda r: r["edits"][1]["payload"].__setitem__("frame", 500),
    lambda r: r["edits"][4]["payload"].__setitem__("before", "Grasp"),
    lambda r: r["edits"][7]["payload"]["segments"][0].__setitem__("label", "x"),
    lambda r: r["edits"][11]["interpretation"].__setitem__("note", "x"),
    lambda r: r["history"][0].__setitem__("extra", 1),
    lambda r: r.__setitem__("version", 2),
])
def test_anything_format1_never_wrote_is_refused(change):
    raw = rc.read("format1_all_kinds.json")
    change(raw)
    with pytest.raises(fmt.MigrationError):
        fmt.migrate_format1(raw, MANIFEST, None)


def test_to_format1_refuses_what_format1_cannot_say():
    r = fmt.migrate_format1(rc.read("format1_all_kinds.json"), MANIFEST, None)
    r2, _ = fmt.propose(r, kind="name", payload={"text": "x"}, source="human", by="p", expect_rev=r["rev"], now="t")
    with pytest.raises(fmt.MigrationError):
        fmt.to_format1(r2)
    r3 = fmt.lock(r, by="owner", expect_rev=r["rev"], now="t")
    with pytest.raises(fmt.MigrationError):
        fmt.to_format1(r3)


def test_effective_of_a_migrated_review_is_what_format1_gave():
    """Batch locks record a digest of effective(); a migrated review must give the same one."""
    want = rc.read("effective_format1.json")
    for f1, eff in want.items():
        raw = rc.read(f1)
        assert rv.effective(MANIFEST, raw) == eff
        assert rv.effective(MANIFEST, fmt.migrate_format1(raw, MANIFEST, None)) == eff
        assert rv.effective(MANIFEST, fmt.migrate_format1(raw, MANIFEST, TASK)) == eff


# ------------------------------------------------------------------ the logger's own operations


def test_logger_only_operations_write_ordinary_format2_reviews():
    r = fmt.empty_review(rc.RUN_ID)
    r, made = fmt.propose_many(r, [dict(kind="exclude", payload={"start": i, "end": i + 2}, source="agent", by="a",
                                        job_id="j", agent_version="v", reason="idle") for i in (0, 10, 10)],
                               expect_rev=0, now="t1", what="find idle")
    assert r["rev"] == 1 and len(r["history"]) == 1 and made[1] is not made[0] and made[1] == made[2]
    r = fmt.decide_many(r, [made[0]["edit_id"], made[1]["edit_id"]], True, by="p", expect_rev=1, now="t2",
                        what="cut all")
    r, gone = fmt.retract(r, made[0]["edit_id"], by="q", expect_rev=2, now="t3")
    # the acceptance stays as it was; the take back says who did it and when, and is its own step
    assert (gone["status"], gone["accepted_seq"], gone["decided_by"], gone["decided_at"]) == ("rejected", 1, "p", "t2")
    assert (gone["retracted_by"], gone["retracted_at"]) == ("q", "t3")
    assert r["history"][-1]["at"] == "t3" and r["history"][-1]["what"] == "take back"
    assert [e["edit_id"] for e in fmt.active_edits(r)] == [made[1]["edit_id"]]
    with pytest.raises(fmt.ReviewError):
        fmt.retract(r, made[0]["edit_id"], by="p", expect_rev=3, now="t4")
    assert fmt.dump_review(fmt.load_review(fmt.dump_review(r))) == fmt.dump_review(r)


def test_taking_back_a_label_takes_its_reading_out_of_effect():
    r = fmt.empty_review(rc.RUN_ID)
    r, lb = fmt.propose(r, kind="label", payload={"start": 1, "end": 3, "text": "x", "driver": None, "segment": None},
                        source="human", by="p", expect_rev=0, now="t")
    r, rd = fmt.propose(r, kind="reading", payload={"label_id": lb["edit_id"], "train": "exclude", "meaning": "m"},
                        source="human", by="p", expect_rev=1, now="t")
    assert fmt.reading_in_effect(r, lb["edit_id"])["train"] == "exclude"
    r, _ = fmt.retract(r, lb["edit_id"], by="p", expect_rev=2, now="t")
    assert [e["kind"] for e in fmt.active_edits(r)] == ["reading"] and fmt.reading_in_effect(r, lb["edit_id"]) is None
    r, _ = fmt.undo(r, r["history"][-1]["step_id"], expect_rev=3, now="t")      # undo brings both back
    assert fmt.reading_in_effect(r, lb["edit_id"])["train"] == "exclude"


# ------------------------------------------------------------------ format 1 files on disk


def _dataset(tmp_path):
    """A committed run of the synthetic fixture: manifest and a format 1 review only, which is all review needs."""
    run = tmp_path / "ds" / "dataop" / "runs" / rc.RUN_ID
    run.mkdir(parents=True)
    # the keys every manifest has, so the run lists as committed; review reads only run_id, length and subgoals
    (run / "manifest.json").write_text(dumps({**MANIFEST, "schema_version": 1, "status": "committed",
                                              "episode_index": 0, "task": "demo", "fingerprint": "f",
                                              "action_schema": {}, "takeovers": [], "events": [], "integrity": {}}))
    shutil.copy(rc.DATA / "format1_all_kinds.json", run / "review.json")
    return run


def test_a_format1_file_is_read_in_place_and_written_as_format2_on_the_next_change(tmp_path):
    run = _dataset(tmp_path)
    before = (run / "review.json").read_bytes()
    r = rv.load(run)
    assert r["review_format"] == 2 and rv.file_format(run) == 1
    assert (run / "review.json").read_bytes() == before                # reading never writes
    e = rv.add_edit(run, MANIFEST, "note", {"frame": 3, "text": "first format 2 edit"})
    assert rv.file_format(run) == 2 and e["by"] == "person" and e["status"] == "accepted"
    text = (run / "review.json").read_text()
    after = fmt.load_review(text)
    assert text == fmt.dump_review(after) and after["rev"] == 24
    # everything format 1 said is still there: drop the new edit and its step, and format 1 comes back exactly
    back = copy.deepcopy(after)
    back["edits"] = [x for x in back["edits"] if x["edit_id"] != e["edit_id"]]
    back["history"] = back["history"][:-1]
    back["rev"], back["seq"] = 23, 14
    back["reviewed"], back["reviewed_at"], back["reviewed_by"] = True, rc.read("format1_all_kinds.json")["reviewed_at"], \
        "reviewer-1"
    assert dumps(fmt.to_format1(back)) == before.decode()


def test_a_no_op_leaves_a_format1_file_alone(tmp_path):
    run = _dataset(tmp_path)
    before = (run / "review.json").read_bytes()
    prior = rv.add_edit(run, MANIFEST, "exclude", {"start": 112, "end": 120}, source="agent", reason="idle")
    assert prior["edit_id"] == "000000000004" and (run / "review.json").read_bytes() == before


def test_migrate_reviews_command(tmp_path, capsys):
    from aftershock_logger.cli import main
    run = _dataset(tmp_path)
    before = (run / "review.json").read_bytes()
    root = str(tmp_path / "ds")
    assert main(["migrate-reviews", root, "--dry-run"]) == 0
    assert (run / "review.json").read_bytes() == before
    assert main(["migrate-reviews", root]) == 0
    assert rv.file_format(run) == 2
    assert fmt.dump_review(fmt.migrate_format1(json.loads(before), MANIFEST, None)) == (run / "review.json").read_text()
    assert main(["migrate-reviews", root]) == 0 and "0 review file(s) migrated" in capsys.readouterr().out


def test_a_broken_format1_file_is_refused_and_left_alone(tmp_path):
    run = _dataset(tmp_path)
    raw = rc.read("format1_all_kinds.json")
    raw["edits"][0]["source"] = "robot"
    (run / "review.json").write_text(dumps(raw))
    with pytest.raises(rv.ReviewError):
        rv.load(run)
    with pytest.raises(rv.ReviewError):
        rv.add_edit(run, MANIFEST, "note", {"frame": 3, "text": "x"})
    assert json.loads((run / "review.json").read_text()) == raw


def _named_run(tmp_path, steps, review=None):
    m = dict(MANIFEST, subgoals=steps)
    run = tmp_path / "ds" / "dataop" / "runs" / rc.RUN_ID
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(dumps(m))
    if review is not None:
        (run / "review.json").write_text(dumps(review))
    return run, m


def test_named_segments_keep_their_names_status_and_reason_and_get_no_invented_id(tmp_path):
    steps = [{"name": "Reach over", "start_frame": 0, "end_frame": 40, "status": "completed", "reason": None},
             {"name": "Pick tube", "start_frame": 40, "end_frame": 100, "status": "failed", "reason": "slipped"},
             {"name": "抓取", "start_frame": 100, "end_frame": 120, "status": "completed", "reason": None}]
    run, m = _named_run(tmp_path, steps)
    plan = [{"name": "Reach over", "start": 0, "end": 42}, {"name": "Pick tube ", "start": 42, "end": 100},
            {"name": "抓取", "start": 100, "end": 120}]
    rv.add_edit(run, m, "segment_plan", {"segments": plan})
    segs = next(e for e in rv.load(run, m)["edits"] if e["kind"] == "segments")["payload"]["segments"]
    assert [(s["subtask_id"], s["legacy_name"]) for s in segs] == [
        ("unresolved", "Reach over"), ("unresolved", "Pick tube"), ("unresolved", "抓取")]
    eff = rv.effective(m, rv.load(run, m))["subgoals"]
    assert [(s["name"], s["start_frame"], s["status"], s["reason"]) for s in eff] == [
        ("Reach over", 0, "completed", None), ("Pick tube", 42, "failed", "slipped"), ("抓取", 100, "completed", None)]
    assert rv.load(run, m)["legacy_subtask_map"] is None                     # a review never migrated has no map


def test_named_segments_in_a_migrated_review_keep_its_map(tmp_path):
    run, m = _named_run(tmp_path, MANIFEST["subgoals"], rc.read("format1_all_kinds.json"))
    # the viewer sends the whole plan by name; one segment changed, one new name
    plan = [{"name": "reach", "start": 0, "end": 28}, {"name": "Grasp", "start": 28, "end": 72},
            {"name": "wiggle", "start": 72, "end": 100}, {"name": "Place it", "start": 100, "end": 110}]
    rv.add_edit(run, m, "segment_plan", {"segments": plan})
    r = rv.load(run, m)
    segs = r["edits"][-1]["payload"]["segments"]
    assert [(s["subtask_id"], s["legacy_name"]) for s in segs] == [
        ("unresolved", "reach"), ("unresolved", "Grasp"), ("unresolved", "wiggle"), ("unresolved", "Place it")]
    assert r["legacy_subtask_map"]["entries"][-1] == {"step_index": None, "recorded_name": "Place it",
                                                      "subtask_id": "unresolved"}
    assert fmt.dump_review(fmt.load_review(fmt.dump_review(r))) == fmt.dump_review(r)
    # with a task, names keep the ids the map gave them
    run2, _ = _named_run(tmp_path / "t", MANIFEST["subgoals"], rc.read("format1_all_kinds.json"))
    rv.migrate_file(run2, m, TASK)
    rv.add_edit(run2, m, "segment_plan", {"segments": plan})
    segs = rv.load(run2, m)["edits"][-1]["payload"]["segments"]
    assert [s["subtask_id"] for s in segs] == ["s-reach", "s-grasp", "unresolved", "unresolved"]


# ------------------------------------------------------------------ against the reference implementation, live


@pytest.mark.parametrize("name", sorted(rc.CASES))
def test_operations_match_the_reference_live(name):
    impl = rc.contract_impl()
    assert rc.run_case(rc.LoggerImpl(), name) == rc.run_case(impl, name)


def test_migration_matches_the_reference_live():
    impl = rc.contract_impl()
    assert rc.migration_results(rc.LoggerImpl()) == rc.migration_results(impl) == EXPECTED["migration"]


def _random_ops(rng: random.Random, review: dict, n: int) -> list:
    """Plausible and implausible operations, chosen from what the review holds at each point."""
    ops = []
    kinds = {
        "exclude": lambda: {"start": rng.randrange(0, 60), "end": rng.randrange(60, 121)},
        "failure_start": lambda: {"frame": rng.choice([0, 50, 119, 120])},
        "note": lambda: {"frame": rng.randrange(0, 121), "text": rng.choice(["a", " ", "slip"])},
        "label": lambda: {"start": rng.randrange(0, 50), "end": rng.randrange(40, 121), "text": rng.choice(["x", " "]),
                          "driver": rng.choice([None, "policy"]), "segment": None},
        "outcome": lambda: {"outcome": rng.choice(["success", "failure", "unknown"]),
                            "termination_reason": rng.choice([None, "success", "timeout"]),
                            "failure_mode": rng.choice([None, "slip"])},
        "drive_boundary": lambda: {"index": rng.choice([0, 1, 200]), "frame": rng.randrange(0, 121)},
        "event_note": lambda: {"key": rng.choice(["failure", "a\nb"]), "text": "t"},
        "name": lambda: {"text": "run"},
    }
    for i in range(n):
        ids = ["$" + str(j) for j in range(i)] or ["ffffffffffff"]
        roll = rng.random()
        if roll < 0.45:
            kind = rng.choice(sorted(kinds))
            src = rng.choice(["human", "human", "agent", "check"])
            kw = dict(kind=kind, payload=kinds[kind](), source=src, by=rng.choice(["p1", "agent-a"]))
            if src != "human":
                kw.update(job_id=rng.choice(["j1", "j2"]), agent_version="v1", reason=rng.choice(["why", None]))
            if src == "human" and rng.random() < 0.15:
                kw["corrects"] = rng.choice(ids)
            if kind == "label" and rng.random() < 0.3:
                kw = dict(kind="reading", payload={"label_id": rng.choice(ids), "train": "exclude", "meaning": "m"},
                          source=src, by="p1", **({} if src == "human" else dict(job_id="j", agent_version="v",
                                                                                reason="r")))
            ops.append(("propose", kw))
        elif roll < 0.6:
            ops.append((rng.choice(["accept", "reject", "retract"]), dict(edit_id=rng.choice(ids), by="p1")))
        elif roll < 0.7:
            ops.append(("replace", dict(edit_id=rng.choice(ids), payload={"frame": rng.randrange(0, 125)},
                                        by="p1")))
        elif roll < 0.85:
            ops.append(("undo", dict(step_id="@" + str(rng.randrange(0, max(i, 1))))))
        elif roll < 0.9:
            ops.append(("set_reviewed", dict(reviewed=rng.random() < 0.7, by="p1")))
        elif roll < 0.95:
            ops.append(("lock", dict(by="owner", reason=None)))
        else:
            ops.append(("unlock", dict(by="owner", reason="r")))
    return ops


@pytest.mark.parametrize("seed", range(40))
def test_random_operation_sequences_match_the_reference(seed):
    impl = rc.contract_impl()
    rng = random.Random(seed)
    start = rng.choice(["empty", "format1_all_kinds.json", "format1_all_kinds.json+task", "format1_minimal.json"])
    name = f"random-{seed}"
    rc.CASES[name] = {"start": start, "ops": _random_ops(rng, {}, 60)}
    try:
        got, want = rc.run_case(rc.LoggerImpl(), name), rc.run_case(impl, name)
    finally:
        del rc.CASES[name]
    for i, (g, w) in enumerate(zip(got["steps"], want["steps"])):
        assert g == w, f"seed {seed}: step {i} differs"
    assert got["final"] == want["final"]


def _mutations(rng: random.Random, d):
    """Every path in d, each changed one way at random."""
    paths = []

    def walk(x, path):
        paths.append(path)
        if isinstance(x, dict):
            for k in x:
                walk(x[k], path + (k,))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, path + (i,))
    walk(d, ())
    values = [None, 0, -1, 1, 2, True, "", " ", "x", "unknown", "unresolved", "aaaaaaaaaaaa", [], {}, 1.5, "a\n",
              "human", "agent", "accepted", "proposed", "rejected", 10 ** 6, "s-reach", "Grasp", "\ud800",
              "a\udc00b"]
    for path in paths[1:]:
        yield path, "set", rng.choice(values)
        yield path, "drop", None


def _apply(d, path, how, value):
    d = copy.deepcopy(d)
    x = d
    for k in path[:-1]:
        x = x[k]
    if how == "drop":
        if isinstance(x, dict):
            del x[path[-1]]
        else:
            x.pop(path[-1])
    else:
        x[path[-1]] = value
    return d


def test_random_changes_are_refused_or_accepted_like_the_reference():
    impl = rc.contract_impl()
    rng = random.Random(7)
    m = impl.manifest(MANIFEST)
    texts = [EXPECTED["migration"]["format1_all_kinds.json task"]["format2"],
             rc.run_case(rc.LoggerImpl(), "every_kind")["final"],
             rc.run_case(rc.LoggerImpl(), "lock_and_unlock")["final"]]
    checked = 0
    for text in texts:
        good = json.loads(text)
        for path, how, value in _mutations(rng, good):
            bad = _apply(good, path, how, value)
            body = json.dumps(bad)
            try:
                mine = fmt.dump_review(fmt.read(body, MANIFEST))
            except (fmt.FormatError, ValueError):
                mine = None
            try:
                ref = impl.load(body)
                impl.check_manifest(ref, m)
                theirs = impl.dump(ref)
            except (ValueError, TypeError):     # the reference raises TypeError for some malformed values
                theirs = None
            assert mine == theirs, f"{path} {how} {value!r}: logger {'accepts' if mine else 'refuses'}"
            checked += 1
    assert checked > 1000


def test_random_changes_to_format1_migrate_like_the_reference():
    impl = rc.contract_impl()
    rng = random.Random(11)
    m = impl.manifest(MANIFEST)
    good = rc.read("format1_all_kinds.json")
    for path, how, value in _mutations(rng, good):
        bad = _apply(good, path, how, value)
        try:
            mine = fmt.dump_review(fmt.migrate_format1(copy.deepcopy(bad), MANIFEST, TASK))
        except (fmt.FormatError, ValueError, KeyError, TypeError, AttributeError):
            mine = None
        try:
            theirs = impl.dump(impl.migrate(bad, m, impl.task(TASK)))
        except (ValueError, KeyError, TypeError, AttributeError):
            theirs = None
        assert mine == theirs, f"{path} {how} {value!r}: logger {'accepts' if mine else 'refuses'}"


def test_agent_values_name_a_version_and_a_rerun_adds_nothing(tmp_path):
    run, m = _named_run(tmp_path, MANIFEST["subgoals"])
    e = rv.apply_agent_edit(run, m, "failure_start", {"frame": 50}, "agent-a")
    assert e["agent_version"] != "unknown" and e["job_id"]
    again = rv.apply_agent_edit(run, m, "failure_start", {"frame": 50}, "agent-a", job_id=e["job_id"])
    assert again["edit_id"] == e["edit_id"] and len(rv.load(run, m)["edits"]) == 1      # same job: no repeat
    newer = rv.apply_agent_edit(run, m, "failure_start", {"frame": 50}, "agent-a", job_id=e["job_id"],
                                agent_version="m-3")
    other = rv.apply_agent_edit(run, m, "failure_start", {"frame": 50}, "agent-a", job_id="another-job")
    assert len({e["edit_id"], newer["edit_id"], other["edit_id"]}) == 3                    # a new version or job counts
    mine = rv.apply_agent_edit(run, m, "failure_start", {"frame": 51}, "agent-a", job_id="j7", agent_version="m-2")
    assert (mine["job_id"], mine["agent_version"]) == ("j7", "m-2")


def test_migrate_reviews_skips_locked_batches(tmp_path, capsys, monkeypatch):
    from aftershock_logger import batches
    from aftershock_logger.cli import main
    run = _dataset(tmp_path)
    before = (run / "review.json").read_bytes()

    monkeypatch.setattr(batches, "batch_of", lambda root, run_id: {"status": "locked", "name": "b"})
    assert main(["migrate-reviews", str(tmp_path / "ds")]) == 0
    out = capsys.readouterr()
    assert "1 skipped in locked batches" in out.out and "skipped, batch locked" in out.err
    assert (run / "review.json").read_bytes() == before

    def broken(root, run_id):
        raise batches.BatchError("batches.json is unreadable")
    monkeypatch.setattr(batches, "batch_of", broken)           # a broken batch file is refused, not skipped
    assert main(["migrate-reviews", str(tmp_path / "ds")]) == 1
    out = capsys.readouterr()
    assert "1 refused" in out.out and "batches.json is unreadable" in out.err
    assert (run / "review.json").read_bytes() == before


# ------------------------------------------------------------------ against the reference, live: logger-only steps


def test_logger_only_steps_load_in_the_reference_and_its_operations_continue_after_them():
    impl = rc.contract_impl()
    m = impl.manifest(MANIFEST)
    with impl.ids(rc.Ids(0xdd0000000000)):
        r = fmt.migrate_format1(rc.read("format1_all_kinds.json"), MANIFEST, None)
        steps = []
        r, made = fmt.propose_many(r, [dict(kind="exclude", payload={"start": i, "end": i + 3}, source="agent", by="a",
                                            job_id="j", agent_version="v", reason="idle") for i in (30, 40)],
                                   expect_rev=r["rev"], now="t1", what="find idle")
        steps.append(r)
        r = fmt.decide_many(r, [x["edit_id"] for x in made], True, by="p", expect_rev=r["rev"], now="t2",
                            what="cut all")
        steps.append(r)
        r, _ = fmt.retract(r, made[0]["edit_id"], by="p", expect_rev=r["rev"], now="t3")
        steps.append(r)
        r, _ = fmt.retract(r, "00000000000c", by="p", expect_rev=r["rev"], now="t4")     # a label with a reading
        steps.append(r)
        for x in steps:
            text = fmt.dump_review(x)
            ref = impl.load(text)
            impl.check_manifest(ref, m)
            assert impl.dump(ref) == text
        # the shared operations go on from there the same way on both sides
        ref = impl.load(fmt.dump_review(r))
        ops = [("undo", dict(step_id=r["history"][-1]["step_id"])), ("undo", dict(step_id=r["history"][-2]["step_id"])),
               ("undo", dict(step_id=r["history"][-3]["step_id"])),
               ("accept", dict(edit_id=made[1]["edit_id"], by="p")),
               ("replace", dict(edit_id=made[1]["edit_id"], payload={"start": 41, "end": 45}, by="p")),
               ("accept", dict(edit_id="000000000004", by="p")), ("set_reviewed", dict(reviewed=True, by="p"))]
        fmt_ids, ref_ids = rc.Ids(0xee0000000000), rc.Ids(0xee0000000000)
        for i, (op, kw) in enumerate(ops):
            with rc.LoggerImpl().ids(fmt_ids), impl.ids(ref_ids):
                out_l = getattr(fmt, op)(r, expect_rev=r["rev"], now=f"u{i}", **kw)    # each one applies
                out_r = impl.op(op)(ref, expect_rev=ref.rev, now=f"u{i}", **kw)
            r = out_l[0] if isinstance(out_l, tuple) else out_l
            ref = out_r[0] if isinstance(out_r, tuple) else out_r
            assert fmt.dump_review(r) == impl.dump(ref), f"{op} differs"


def _string_paths(d, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _string_paths(v, path + (k,))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _string_paths(v, path + (i,))
    elif isinstance(d, str):
        yield path


def test_lone_surrogates_are_refused_or_accepted_like_the_reference():
    impl = rc.contract_impl()
    m = impl.manifest(MANIFEST)
    texts = [EXPECTED["migration"]["format1_all_kinds.json task"]["format2"],
             rc.run_case(rc.LoggerImpl(), "every_kind")["final"], rc.run_case(rc.LoggerImpl(), "lock_and_unlock")["final"]]
    seen = {True: 0, False: 0}
    for text in texts:
        good = json.loads(text)
        for path in _string_paths(good):
            for value in ("\ud800", "a\udc00", "\udbff\udfff"):
                body = json.dumps(_apply(good, path, "set", value))
                try:
                    mine = fmt.dump_review(fmt.read(body, MANIFEST))
                except (fmt.FormatError, ValueError):
                    mine = None
                try:
                    ref = impl.load(body)
                    impl.check_manifest(ref, m)
                    theirs = impl.dump(ref)
                except (ValueError, TypeError):
                    theirs = None
                assert mine == theirs, f"{path} {value!r}: logger {'accepts' if mine else 'refuses'}"
                seen[mine is not None] += 1
    assert seen[True] and seen[False]           # both kinds of field were exercised


# ------------------------------------------------------------------ take back fields, text and unknown


def _review_with_take_back():
    """A review with a label taken back by q and a cut still in effect: (review, label id, cut id)."""
    r = fmt.empty_review(rc.RUN_ID)
    r, lb = fmt.propose(r, kind="label", payload={"start": 1, "end": 3, "text": "x", "driver": None, "segment": None},
                        source="human", by="p", expect_rev=0, now="t1")
    r, cut = fmt.propose(r, kind="exclude", payload={"start": 4, "end": 9}, source="human", by="p", expect_rev=1,
                         now="t2")
    r, _ = fmt.retract(r, lb["edit_id"], by="q", expect_rev=2, now="t3")
    return r, lb["edit_id"], cut["edit_id"]


def test_a_take_back_records_who_and_when_and_is_final():
    r, lb, cut = _review_with_take_back()
    gone = next(e for e in r["edits"] if e["edit_id"] == lb)
    assert (gone["status"], gone["accepted_seq"], gone["decided_at"], gone["decided_by"]) == ("rejected", 1, "t1", "p")
    assert (gone["retracted_at"], gone["retracted_by"]) == ("t3", "q")
    assert (r["history"][-1]["what"], r["history"][-1]["at"]) == ("take back", "t3")
    # accept, reject, replace and retract all refuse a taken back edit
    for call in (lambda: fmt.accept(r, lb, by="p", expect_rev=3, now="t4"),
                 lambda: fmt.reject(r, lb, by="p", expect_rev=3, now="t4"),
                 lambda: fmt.replace(r, lb, {"start": 1, "end": 4, "text": "y", "driver": None, "segment": None},
                                     by="p", expect_rev=3, now="t4"),
                 lambda: fmt.retract(r, lb, by="p", expect_rev=3, now="t4")):
        with pytest.raises(fmt.ReviewError):
            call()
    # unknown is written only by the upgrade of an L8 file, never by retract
    for by, now in (("unknown", "t4"), ("p", "unknown")):
        with pytest.raises(fmt.ReviewError, match="unknown"):
            fmt.retract(r, cut, by=by, expect_rev=3, now=now)
    # undo puts the edit back exactly as accepted
    back, _ = fmt.undo(r, r["history"][-1]["step_id"], expect_rev=3, now="t5")
    assert next(e for e in back["edits"] if e["edit_id"] == lb)["retracted_at"] is None


def test_format1_cannot_say_who_took_an_edit_back():
    r = fmt.migrate_format1(rc.read("format1_all_kinds.json"), MANIFEST, None)
    assert all((e["retracted_at"], e["retracted_by"]) == (None, None) for e in fmt.all_edit_versions(r))
    fmt.to_format1(r)
    r2, _ = fmt.retract(r, "00000000000e", by="q", expect_rev=r["rev"], now="t")
    with pytest.raises(fmt.MigrationError, match="retracted_by"):
        fmt.to_format1(r2)


def _edit_variants():
    """(name, change to a taken back edit, accepted by check_edit)."""
    def put(**kw):
        return lambda e: e.update(kw)
    return [
        ("as written", put(), True),
        ("only retracted_at", put(retracted_by=None), False),
        ("only retracted_by", put(retracted_at=None), False),
        ("still accepted", put(status="accepted"), False),
        ("never accepted", put(accepted_seq=None), False),
        ("also replaced", put(replaced_by="aaaaaaaaaaaa"), False),
        ("both unknown", put(retracted_at="unknown", retracted_by="unknown"), True),
        ("unknown at only", put(retracted_at="unknown"), False),
        ("unknown by only", put(retracted_by="unknown"), False),
        ("by too long", put(retracted_by="x" * 101), False),
        ("blank at", put(retracted_at=""), False),
    ]


@pytest.mark.parametrize("name,change,ok", _edit_variants(), ids=[v[0] for v in _edit_variants()])
def test_take_back_fields_are_set_together_and_only_on_a_take_back(name, change, ok):
    r, lb, _ = _review_with_take_back()
    e = copy.deepcopy(next(x for x in r["edits"] if x["edit_id"] == lb))
    change(e)
    if ok:
        fmt.check_edit(e)
    else:
        with pytest.raises(fmt.FormatError):
            fmt.check_edit(e)


def _unrecorded(e):
    e.update(retracted_at=None, retracted_by=None)


def test_a_take_back_that_names_nobody_is_kept_only_in_a_migrated_review():
    r, lb, _ = _review_with_take_back()
    bad = copy.deepcopy(r)
    _unrecorded(next(x for x in bad["edits"] if x["edit_id"] == lb))
    with pytest.raises(fmt.FormatError, match="names nobody"):
        fmt.check_review(bad)
    bad = copy.deepcopy(r)                                  # only in a history snapshot
    _unrecorded(bad["history"][-1]["changes"][0]["after"])
    with pytest.raises(fmt.FormatError, match="names nobody"):
        fmt.check_review(bad)
    # a migrated review keeps format 1 take backs (000000000009 and 00000000000f) as format 1 wrote them
    m = fmt.migrate_format1(rc.read("format1_all_kinds.json"), MANIFEST, None)
    assert [e["edit_id"] for e in m["edits"] if fmt._unrecorded_take_back(e)] == ["000000000009", "00000000000f"]


def _migrated_unknown_variants():
    """Reviews that differ only in where unknown take back fields are: (name, review, accepted)."""
    m = fmt.migrate_format1(rc.read("format1_all_kinds.json"), MANIFEST, None)
    on_edit = copy.deepcopy(m)
    next(e for e in on_edit["edits"] if e["edit_id"] == "000000000009").update(retracted_at="unknown",
                                                                               retracted_by="unknown")
    on_snapshot = copy.deepcopy(m)                          # a format 1 style take back only in a history `after`
    after = on_snapshot["history"][0]["changes"][0]["after"]
    after.update(status="rejected", retracted_at="unknown", retracted_by="unknown")
    outside, lb, _ = _review_with_take_back()               # not migrated: an L8 take back reads as unknown
    outside = copy.deepcopy(outside)
    next(e for e in outside["edits"] if e["edit_id"] == lb).update(retracted_at="unknown", retracted_by="unknown")
    outside_snapshot = copy.deepcopy(outside)
    outside_snapshot["history"][-1]["changes"][0]["after"].update(retracted_at="unknown", retracted_by="unknown")
    return [("migrated, format 1 take back", m, True), ("migrated, unknown on the edit", on_edit, False),
            ("migrated, unknown on a history snapshot only", on_snapshot, False),
            ("not migrated, unknown on the edit", outside, True),
            ("not migrated, unknown on the edit and its snapshot", outside_snapshot, True)]


def test_unknown_take_back_fields_are_refused_in_a_migrated_review():
    for name, review, ok in _migrated_unknown_variants():
        if ok:
            fmt.check_review(review)
            assert fmt.dump_review(fmt.read(fmt.dump_review(review), MANIFEST)) == fmt.dump_review(review), name
        else:
            with pytest.raises(fmt.FormatError, match="migrated"):
                fmt.check_review(review)


def test_unknown_take_back_fields_are_refused_in_a_migrated_review_like_the_reference():
    impl = rc.contract_impl()
    m = impl.manifest(MANIFEST)
    for name, review, ok in _migrated_unknown_variants():
        text = fmt.dump_review(review)
        try:
            ref = impl.load(text)
            impl.check_manifest(ref, m)
            theirs = impl.dump(ref)
        except ValueError:
            theirs = None
        assert (theirs is not None) == ok and (theirs is None or theirs == text), name


def test_job_id_and_agent_version_unknown_only_together_on_agent_edits_of_a_migrated_review():
    m = fmt.migrate_format1(rc.read("format1_all_kinds.json"), MANIFEST, None)
    agent = next(e for e in m["edits"] if e["source"] == "agent")
    fmt.check_edit(agent)
    for change in (dict(job_id="j1"), dict(agent_version="v1"), dict(source="check")):
        with pytest.raises(fmt.FormatError, match="unknown only together"):
            fmt.check_edit({**agent, **change})
    # outside a migrated review, in the edits or only in a history snapshot
    r = fmt.empty_review(rc.RUN_ID)
    r, e = fmt.propose(r, kind="note", payload={"frame": 1, "text": "x"}, source="agent", by="a", job_id="j",
                       agent_version="v", reason="r", expect_rev=0, now="t")
    for where in ("edits", "snapshot"):
        bad = copy.deepcopy(r)
        x = bad["edits"][0] if where == "edits" else bad["history"][0]["changes"][0]["after"]
        x.update(job_id="unknown", agent_version="unknown")
        with pytest.raises(fmt.FormatError, match="migrated from format 1"):
            fmt.check_review(bad)


def test_a_caller_cannot_pass_unknown_as_job_id_or_agent_version(tmp_path):
    r = fmt.empty_review(rc.RUN_ID)
    for kw in (dict(job_id="unknown", agent_version="v"), dict(job_id="j", agent_version="unknown")):
        item = dict(kind="note", payload={"frame": 1, "text": "x"}, source="agent", by="a", reason="r", **kw)
        with pytest.raises(fmt.ReviewError, match="unknown"):
            fmt.propose(r, expect_rev=0, now="t", **item)
        with pytest.raises(fmt.ReviewError, match="unknown"):
            fmt.propose_many(r, [item], expect_rev=0, now="t", what="w")
    run, m = _named_run(tmp_path, MANIFEST["subgoals"])
    with pytest.raises(rv.ReviewError, match="unknown"):
        rv.add_edit(run, m, "note", {"frame": 1, "text": "x"}, source="agent", agent_version="unknown")


S = "a\udc00"     # a lone surrogate


def test_no_text_field_holds_a_lone_surrogate():
    r, lb, _ = _review_with_take_back()
    for change in (lambda d: d["edits"][0].update(by=S), lambda d: d["edits"][0].update(reason=S),
                   lambda d: d["history"][0].update(what=S), lambda d: d["edits"][0].update(retracted_by=S),
                   lambda d: d["lock_log"].append({"action": "unlock", "by": "o", "at": "t", "reason": S})):
        bad = copy.deepcopy(r)
        change(bad)
        with pytest.raises(fmt.FormatError):
            fmt.check_review(bad)
        with pytest.raises(fmt.FormatError):
            fmt.dump_review(bad)                            # the writer refuses it too, checked or not
        with pytest.raises(fmt.FormatError):
            fmt.load_review(json.dumps(bad))
    raw = rc.read("format1_all_kinds.json")                # the format 1 read
    raw["edits"][0]["by"] = S
    with pytest.raises(fmt.MigrationError):
        fmt.migrate_format1(raw, MANIFEST, None)
    assert fmt.dump_review(fmt.load_review(fmt.dump_review(r))) == fmt.dump_review(r)      # valid text still reads


def _surrogate_calls(r, lb, cut):
    """Operation calls with a lone surrogate in one input, several of which would change nothing otherwise."""
    agent = dict(kind="note", payload={"frame": 1, "text": "x"}, source="agent", by="a", job_id="j",
                 agent_version="v", reason="r")
    return [
        ("propose", dict(agent, reason=S)),                             # a repeat of an open proposal
        ("propose", dict(agent, now=S)),
        ("propose", dict(agent, by=S)),
        ("propose", dict(agent, job_id=S)),
        ("propose", dict(kind="note", payload={"frame": 1, "text": S}, source="human", by="p")),
        ("set_reviewed", dict(reviewed=False, by="p", now=S)),          # not signed off: nothing to take back
        ("set_reviewed", dict(reviewed=False, by=S)),
        ("accept", dict(edit_id=cut, by=S)),
        ("reject", dict(edit_id=cut, by=S)),
        ("replace", dict(edit_id=cut, payload={"start": 4, "end": 8}, by=S)),
        ("retract", dict(edit_id=cut, by=S)),
        ("retract", dict(edit_id=cut, by="p", now=S)),
        ("lock", dict(by=S)),
        ("lock", dict(by="o", reason=S)),
    ]


def _with_open_proposal():
    r, lb, cut = _review_with_take_back()
    r, _ = fmt.propose(r, kind="note", payload={"frame": 1, "text": "x"}, source="agent", by="a", job_id="j",
                       agent_version="v", reason="r", expect_rev=r["rev"], now="t")
    return r, lb, cut


def test_operations_refuse_a_lone_surrogate_even_when_they_change_nothing():
    r, lb, cut = _with_open_proposal()
    for op, kw in _surrogate_calls(r, lb, cut):
        kw = {"now": "t9", **kw}
        with pytest.raises(fmt.ReviewError):
            getattr(fmt, op)(r, expect_rev=r["rev"], **kw)
    item = dict(kind="note", payload={"frame": 1, "text": "x"}, source="agent", by="a", job_id="j",
                agent_version="v", reason="r")
    for kw in (dict(now=S, what="w"), dict(now="t", what=S)):
        with pytest.raises(fmt.ReviewError):
            fmt.propose_many(r, [item], expect_rev=r["rev"], **kw)
        with pytest.raises(fmt.ReviewError):
            fmt.decide_many(r, [], True, by="p", expect_rev=r["rev"], **kw)
    locked = fmt.lock(r, by="o", expect_rev=r["rev"], now="t")
    with pytest.raises(fmt.ReviewError):
        fmt.unlock(locked, by=S, reason="r", expect_rev=locked["rev"], now="t")


def test_operations_refuse_a_lone_surrogate_like_the_reference():
    impl = rc.contract_impl()
    r, lb, cut = _with_open_proposal()
    ref = impl.load(fmt.dump_review(r))
    for op, kw in _surrogate_calls(r, lb, cut):
        kw = {"now": "t9", **kw}
        outs = []
        for call, review in ((getattr(fmt, op), r), (impl.op(op), ref)):
            try:
                call(review, expect_rev=r["rev"], **copy.deepcopy(kw))
                outs.append("ok")
            except ValueError as e:
                outs.append(type(e).__name__)
        assert outs[0] == outs[1] == "ReviewError", (op, kw, outs)


# ------------------------------------------------------------------ files the logger wrote at 4f4395e (logger L8)


L8 = rc.read("l8_files.json")


def _l8_run(tmp_path, name):
    run, m = _named_run(tmp_path / name, MANIFEST["subgoals"])
    (run / "review.json").write_text(L8[name]["text"])
    return run, m


def test_the_l8_fixtures_are_what_that_logger_wrote():
    for name, f in L8.items():
        raw = json.loads(f["text"])
        assert all("retracted_at" not in e and "retracted_by" not in e for e in raw["edits"]), name
        assert (raw["legacy_subtask_map"] is not None) == f["migrated"]
    whats = [h["what"] for h in json.loads(L8["take_back"]["text"])["history"]]
    assert whats[-2:] == ["take back (taken back by q)", "take back (taken back by p1)"]


@pytest.mark.parametrize("name", sorted(L8))
def test_an_l8_file_reads_as_the_reference_reads_it_through_both_read_paths(tmp_path, name):
    f, want = L8[name], EXPECTED["l8_files"][name]
    text = f["text"]
    up = fmt.load_review(text)
    assert fmt.dump_review(up) == want["upgraded"]
    assert fmt.dump_review(fmt.read(text, MANIFEST)) == want["upgraded"]
    run, m = _l8_run(tmp_path, name)
    assert fmt.dump_review(rv.load(run, m)) == want["upgraded"]
    assert fmt.dump_review(rv.load(run)) == want["upgraded"]                       # manifest read from disk
    assert (run / "review.json").read_text() == text                              # reading never writes
    # the digest is that of the upgraded text, which differs from the file's own sha256 once it has an edit
    assert fmt.review_digest(up) == want["digest"]
    has_edits = bool(json.loads(text)["edits"] or json.loads(text)["history"])
    assert (want["digest"] != rc.sha(text)) == has_edits
    # each edit not taken back reads with both null; a take back made by L8 outside a migrated review with unknown
    for e in fmt.all_edit_versions(up):
        took_back = e["status"] == "rejected" and e["accepted_seq"] is not None and e["replaced_by"] is None
        want_mark = "unknown" if took_back and not f["migrated"] else None
        assert (e["retracted_at"], e["retracted_by"]) == (want_mark, want_mark), (name, e["edit_id"])


@pytest.mark.parametrize("name", sorted(L8))
def test_the_next_change_to_an_l8_file_writes_both_keys(tmp_path, name):
    want = EXPECTED["l8_files"][name]
    review = fmt.load_review(L8[name]["text"])
    with rc.LoggerImpl().ids(rc.Ids(0xf9000000000)):
        if review["locked"] is not None:
            nxt = fmt.unlock(review, by="owner", reason="next change", expect_rev=review["rev"], now=rc.when(0))
        else:
            nxt, _ = fmt.propose(review, kind="note", payload={"frame": 1, "text": "next change"}, source="human",
                                 by="p1", expect_rev=review["rev"], now=rc.when(0))
    assert fmt.dump_review(nxt) == want["next"]
    # through the run: the file on disk becomes the canonical text, with both keys on every edit and snapshot
    run, m = _l8_run(tmp_path, name)
    if review["locked"] is None:
        rv.add_edit(run, m, "note", {"frame": 1, "text": "next change"})
        on_disk = json.loads((run / "review.json").read_text())
        assert all({"retracted_at", "retracted_by"} <= set(e) for e in fmt.all_edit_versions(on_disk))
        assert fmt.dump_review(fmt.load_review((run / "review.json").read_text())) == \
            (run / "review.json").read_text()


@pytest.mark.parametrize("name", sorted(L8))
def test_an_l8_file_matches_the_reference_live(name):
    impl = rc.contract_impl()
    assert rc.l8_results(rc.LoggerImpl())[name] == rc.l8_results(impl)[name] == EXPECTED["l8_files"][name]


def test_an_l8_edit_with_only_one_take_back_key_is_refused(tmp_path):
    for key in ("retracted_at", "retracted_by"):
        raw = json.loads(L8["take_back"]["text"])
        raw["edits"][0][key] = None
        text = fmt.dumps(raw)
        with pytest.raises(fmt.FormatError):
            fmt.load_review(text)
        with pytest.raises(fmt.FormatError):
            fmt.read(text, MANIFEST)
        run, m = _l8_run(tmp_path / key, "take_back")
        (run / "review.json").write_text(text)
        with pytest.raises(rv.ReviewError):
            rv.load(run, m)


@pytest.mark.parametrize("name", sorted(L8))
def test_batches_locked_by_the_l8_logger_still_verify(tmp_path, name):
    """A batch lock records the sha256 of the effective view, not of the review text. Taken back edits are rejected
    and not in that view, so the digest that logger recorded still matches after the upgrade."""
    run, m = _l8_run(tmp_path, name)
    assert B.review_digest(m, rv.load(run, m)) == L8[name]["batch_digest"]


# ------------------------------------------------------------------ real files, read only


def _real_reviews() -> list[Path]:
    root = os.environ.get("AFTERSHOCK_REVIEW_DATA")
    if not root:
        pytest.skip("set AFTERSHOCK_REVIEW_DATA to a folder with recorded datasets to migrate their review files")
    found = sorted(Path(root).expanduser().glob("**/dataop/runs/*/review.json"))
    if not found:
        pytest.skip("no review.json files under AFTERSHOCK_REVIEW_DATA")
    return found


def _format1_logger():
    """review.py as it was before review format 2 (from this repo's history), or None without git history."""
    import importlib.util
    import subprocess
    repo = Path(__file__).resolve().parents[1]
    try:
        src = subprocess.run(["git", "show", f"{FORMAT1_COMMIT}:src/aftershock_logger/review.py"], cwd=repo,
                             capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    path = Path(os.environ.get("TMPDIR", "/tmp")) / f"review_format1_{os.getpid()}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("aftershock_logger._review_before_format2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    path.unlink()
    return mod


FORMAT1_COMMIT = "a051fa0"     # the last commit whose logger wrote format 1


def test_effective_matches_the_format1_logger_from_history():
    old = _format1_logger()
    if old is None:
        pytest.skip("no git history to read the format 1 logger from")
    for f1 in ("format1_all_kinds.json", "format1_minimal.json"):
        raw = rc.read(f1)
        r1 = copy.deepcopy(raw)
        r1.setdefault("rev", 0)
        r1.setdefault("seq", 0)
        for h in r1.get("history", []):
            h.setdefault("t", 1)
        assert rv.effective(MANIFEST, raw) == old.effective(MANIFEST, r1)


def test_real_review_files_migrate_losslessly_on_copies(tmp_path):
    root = os.environ.get("AFTERSHOCK_CONTRACTS")
    impl = rc.ContractImpl(root) if root else None
    for i, src in enumerate(_real_reviews()):
        run = tmp_path / "ds" / "dataop" / "runs" / src.parent.name
        run.mkdir(parents=True)
        shutil.copy(src, run / "review.json")                       # the original is only read
        shutil.copy(src.parent / "manifest.json", run / "manifest.json")
        text = (run / "review.json").read_text()
        raw, m = json.loads(text), json.loads((run / "manifest.json").read_text())
        if not fmt.is_format1(raw):
            assert fmt.dump_review(fmt.read(text, m)) == text
            continue
        r = fmt.migrate_format1(json.loads(text), m, None)
        assert dumps(fmt.to_format1(r)) == dumps(fmt.normalize_format1(raw))
        if fmt.normalize_format1(raw) == raw:
            assert dumps(fmt.to_format1(r)) == text
        assert fmt.dump_review(fmt.load_review(fmt.dump_review(r))) == fmt.dump_review(r)
        assert {e["subtask_id"] for e in r["legacy_subtask_map"]["entries"]} <= {"unresolved"}   # no guesses
        if impl is not None:
            assert impl.dump(impl.migrate(raw, impl.RunManifest.model_validate(m), None)) == fmt.dump_review(r)
        # batch locks record a digest of effective(): the logger before this change must give the same one
        old = _format1_logger()
        if old is not None:
            r1 = copy.deepcopy(raw)
            r1.setdefault("rev", 0)
            r1.setdefault("seq", 0)
            for h in r1.get("history", []):
                h.setdefault("t", 1)            # that reader wanted t on every step; effective() never reads history
            assert B.review_digest(m, rv.load(run, m)) == store.json_sha256(old.effective(m, r1))
        assert rv.migrate_file(run, m)
        assert B.review_digest(m, rv.load(run, m)) == B.review_digest(m, r) and (run / "review.json").read_text() == fmt.dump_review(r)
        assert dumps(fmt.to_format1(rv.load(run, m))) == dumps(fmt.normalize_format1(raw))
