"""Logger task L7: a dataset whose reviews are synced from Aftershock is read only here (src/aftershock_logger/sync.py).

A station writes `dataop/sync.json` before it writes any cloud review into `review.json`. From then on every local
review change is refused, the viewer serves the dataset read only, and reading and exporting work as before. A
damaged marker still counts as synced (fail closed).
"""
import functools
import http.client
import http.server
import json
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_logger import _review_fixture  # noqa: E402

from dataop_logger import review as rv, review_format as fmt, store, sync, viewer  # noqa: E402
from dataop_logger.cli import main  # noqa: E402

MARKER = {"sync_format": 1, "source": "aftershock", "workspace_id": "ws-1", "station_id": "station-1",
          "since": "2026-11-05T12:00:00Z"}


def mark(root, value=MARKER) -> Path:
    p = Path(root) / "dataop" / "sync.json"
    p.write_text(value if isinstance(value, str) else json.dumps(value))
    return p


def cloud_write(run_path: Path, review: dict) -> None:
    """What the station's sync does: the canonical text, a temp file named .review.json.*.tmp, a rename."""
    tmp = run_path / ".review.json.sync-test.tmp"
    tmp.write_bytes(fmt.dump_review(review).encode())
    os.replace(tmp, run_path / "review.json")


def test_without_a_marker_the_dataset_is_not_synced_and_edits_work(tmp_path):
    ds, run, m = _review_fixture(tmp_path)
    assert sync.state(ds.root) is None
    sync.check_editable(ds.root)
    rv.add_edit(run.run_path, m, "label", {"start": 1, "end": 4, "text": "x"})
    assert rv.load(run.run_path, m)["rev"] == 1


def test_a_synced_dataset_refuses_every_local_review_change_and_keeps_the_file(tmp_path):
    ds, run, m = _review_fixture(tmp_path)
    first = rv.add_edit(run.run_path, m, "label", {"start": 1, "end": 4, "text": "x"})
    step = rv.load(run.run_path, m)["history"][-1]["step_id"]
    mark(ds.root)
    assert sync.state(ds.root) == {"station_id": "station-1", "workspace_id": "ws-1",
                                   "since": "2026-11-05T12:00:00Z", "error": None}
    before = (run.run_path / "review.json").read_bytes()
    changes = [
        lambda: rv.add_edit(run.run_path, m, "label", {"start": 2, "end": 5, "text": "y"}),
        lambda: rv.add_edit(run.run_path, m, "exclude", {"start": 0, "end": 3}, source="agent", reason="idle"),
        lambda: rv.replace_edit(run.run_path, m, first["edit_id"], {"start": 1, "end": 6, "text": "z"}),
        lambda: rv.set_reviewed(run.run_path, True),
        lambda: rv.undo(run.run_path, m, step),
        lambda: rv.add_event_note(run.run_path, m, "takeover:0", "late", by="person", source="human"),
    ]
    for change in changes:
        with pytest.raises(rv.ReviewError, match="synced from Aftershock by station station-1"):
            change()
    assert (run.run_path / "review.json").read_bytes() == before
    # Reading is unchanged: the review, its effective view and the training mask.
    r = rv.load(run.run_path, m)
    assert r["rev"] == 1 and rv.effective(m, r)["labels"][0]["text"] == "x"
    # The command line reports the refusal and exits 2, as for a locked batch.
    assert main(["interpret", str(ds.root), run.run_id, first["edit_id"], "--train", "exclude",
                 "--meaning", "blocked"]) == 2
    assert (run.run_path / "review.json").read_bytes() == before


def test_migrate_reviews_refuses_a_synced_format_1_file(tmp_path):
    ds, run, m = _review_fixture(tmp_path)
    f1 = {"version": 1, "rev": 1, "seq": 0, "reviewed": False, "edits": [], "history": []}
    (run.run_path / "review.json").write_text(json.dumps(f1))
    mark(ds.root)
    with pytest.raises(rv.ReviewError, match="synced"):
        rv.migrate_file(run.run_path, m)
    assert json.loads((run.run_path / "review.json").read_text()) == f1


@pytest.mark.parametrize("bad", ["{", "[]", json.dumps({**MARKER, "sync_format": 2}),
                                 json.dumps({**MARKER, "extra": 1}), json.dumps({**MARKER, "station_id": " "}),
                                 json.dumps({k: v for k, v in MARKER.items() if k != "since"}),
                                 json.dumps({**MARKER, "source": "elsewhere"}), "dir", "link"])
def test_a_damaged_marker_still_counts_as_synced(tmp_path, bad):
    ds, run, m = _review_fixture(tmp_path)
    p = Path(ds.root) / "dataop" / "sync.json"
    if bad == "dir":
        p.mkdir()
    elif bad == "link":
        p.symlink_to(tmp_path / "nowhere.json")
    else:
        mark(ds.root, bad)
    s = sync.state(ds.root)
    assert s is not None and s["error"]
    with pytest.raises(rv.ReviewError, match="marker is damaged"):
        rv.add_edit(run.run_path, m, "label", {"start": 1, "end": 4, "text": "x"})
    assert not (run.run_path / "review.json").exists()


def test_a_cloud_review_written_by_the_sync_reads_whole_with_its_rev_and_lock(tmp_path):
    ds, run, m = _review_fixture(tmp_path)
    mark(ds.root)
    review = fmt.empty_review(run.run_id)
    review, _ = fmt.propose(review, kind="name", payload={"text": "run one"}, source="human", by="user_1",
                            expect_rev=0, now="2026-11-05T12:01:00Z")
    review = fmt.lock(review, by="user_1", expect_rev=1, now="2026-11-05T12:02:00Z")
    cloud_write(run.run_path, review)
    (run.run_path / ".review.json.sync-left.tmp").write_bytes(b'{"review_format": 2, "rev"')   # an interrupted write
    r = rv.load(run.run_path, m)
    assert r["rev"] == 2 and r["locked"]["by"] == "user_1"
    assert fmt.review_digest(r) == fmt.review_digest(review)
    from dataop_logger import validate as V
    assert V.validate(ds.root)["ok"]
    state = viewer._review_state(run.run_path, m)
    assert state["rev"] == 2 and state["locked"]["by"] == "user_1" and state["error"] is None


def _serve(root: Path, editable: bool = True):
    h = type("H", (viewer._RangeHandler,), {"editable": editable})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(h, directory=str(root)))
    srv.dataop_state = {"root": Path(root)}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(srv, body):
    port = srv.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("POST", viewer.API, json.dumps(body), {"Content-Type": "application/json", "X-Dataop": "1",
                                                      "Host": f"127.0.0.1:{port}"})
    r = c.getresponse()
    return r.status, json.loads(r.read())


def test_the_viewer_serves_a_synced_dataset_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAOP_HOME", str(tmp_path / "dataop-home"))     # the viewer's dataset list
    ds, run, m = _review_fixture(tmp_path)
    srv = _serve(Path(ds.root))
    try:
        lbl = {"run_id": run.run_id, "rev": 0, "op": "add", "kind": "label",
               "payload": {"start": 1, "end": 4, "text": "x"}}
        assert _post(srv, lbl)[0] == 200                # not synced yet: the page edits
        mark(ds.root)
        before = (run.run_path / "review.json").read_bytes()
        for body in ({**lbl, "rev": 1}, {"run_id": run.run_id, "rev": 1, "op": "reviewed", "reviewed": True},
                     {"op": "lists_add", "rev": 0, "kind": "failure_types", "name": "drop"},
                     {"op": "batch_new", "rev": 0, "name": "new"}, {"op": "import_scan"},
                     {"op": "train_list"}):
            code, out = _post(srv, body)
            assert code == 403 and out["synced"] is True and "read only" in out["error"], body
        assert (run.run_path / "review.json").read_bytes() == before
        assert _post(srv, {"op": "ws_list"})[0] == 200      # switching to another dataset still works
        c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1])
        c.request("GET", "/dataop/view/")
        page = c.getresponse().read().decode()
        data = viewer.collect(Path(ds.root))
        assert data["synced"]["station_id"] == "station-1" and "synced from Aftershock" in data["synced"]["message"]
        assert '"synced": {' in page and "DATA.editable && !DATA.synced" in page
    finally:
        srv.shutdown()


def test_an_unsynced_dataset_page_has_no_synced_state(tmp_path, monkeypatch):
    monkeypatch.setenv("DATAOP_HOME", str(tmp_path / "dataop-home"))
    ds, run, m = _review_fixture(tmp_path)
    assert viewer.collect(Path(ds.root))["synced"] is None
    assert viewer._review_state(run.run_path, m)["locked"] is None
    assert store.dataop_dir(Path(ds.root)).is_dir()
