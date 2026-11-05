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
from aftershock_logger import review as rv, review_format as fmt  # noqa: E402

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
    r, gone = fmt.retract(r, made[0]["edit_id"], by="p", expect_rev=2, now="t3")
    assert gone["status"] == "rejected" and gone["accepted_seq"] == 1 and gone["decided_by"] == "p"
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


def test_viewer_segments_get_subtask_ids_and_keep_their_names_through_the_lists():
    from aftershock_logger import lists as ls
    assert rv.subtask_id_for("Pick tube") == "pick-tube" and rv.subtask_id_for(" grasp ") == "grasp"
    assert rv.subtask_id_for("unresolved").startswith("segment-") and rv.subtask_id_for("??").startswith("segment-")
    assert fmt.SLUG.fullmatch(rv.subtask_id_for("x" * 90))
    d = {"failure_types": [], "task_segments": [{"name": "Pick tube", "description": "", "aliases": ["Grab"]}]}
    assert ls.canonical(d, "task_segments", "pick-tube") == "Pick tube"
    assert ls.canonical(d, "task_segments", "grab") == "Pick tube"
    kind, p = rv._input("segment_plan", {"segments": [{"name": "Pick tube", "start": 0, "end": 5}]}, MANIFEST)
    assert (kind, p) == ("segments", {"segments": [{"subtask_id": "pick-tube", "start": 0, "end": 5,
                                                    "legacy_name": None}]})


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
            ops.append((rng.choice(["accept", "reject"]), dict(edit_id=rng.choice(ids), by="p1")))
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
              "human", "agent", "accepted", "proposed", "rejected", 10 ** 6, "s-reach", "Grasp"]
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


# ------------------------------------------------------------------ real files, read only


def _real_reviews() -> list[Path]:
    root = os.environ.get("AFTERSHOCK_REVIEW_DATA")
    if not root:
        pytest.skip("set AFTERSHOCK_REVIEW_DATA to a folder with recorded datasets to migrate their review files")
    found = sorted(Path(root).expanduser().glob("**/dataop/runs/*/review.json"))
    if not found:
        pytest.skip("no review.json files under AFTERSHOCK_REVIEW_DATA")
    return found


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
        rv.effective(m, rv.load(run, m))
        assert rv.migrate_file(run, m) and (run / "review.json").read_text() == fmt.dump_review(r)
        assert dumps(fmt.to_format1(rv.load(run, m))) == dumps(fmt.normalize_format1(raw))
