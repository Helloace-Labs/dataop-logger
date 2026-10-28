# Round 18 code review and functional test — `5fba9d4`

## Verdict

**Request changes.** The main workflows work on the real 40-run SO-101 dataset, and the exported DAgger data checked out frame-for-frame. One concurrency bug is release-blocking: an export can finish after its batch is unlocked and edited, then register itself as current (`stale: false`). I also found three medium-severity integrity/provenance issues and one low-severity dry-run determinism issue.

I tested an exact `git archive 5fba9d4` with `/Users/chadster/lerobot/.venv/bin/python` and `PYTHONPATH=src`; the source worktree had advanced beyond the requested commit during the review.

## Findings

### High — An in-flight export can be registered as current after unlock and review edits

`export.export()` snapshots reviews, releases all batch/review locks, writes the dataset, then calls `B.add_export()` (`export.py:279-355, 368-418`). `add_export()` does not require the batch still to be locked or its lock snapshot to match. Meanwhile unlock only marks exports already in the registry stale (`batches.py:292-298`).

I put a barrier in the writer, started an export from a locked batch, unlocked the batch and changed a review, then let the writer finish. The result was:

```json
{
  "batch_status": "reviewing",
  "new_export_stale": false,
  "digests_match": false
}
```

The same race is reachable through the viewer: `export_start`, then `batch_status: unlock` while the background job is writing, then any review edit.

Fix: make export registration conditional on the exact lock snapshot used to plan it. Under `batches.lock`, compare status plus every locked run digest (or an immutable lock/export token) before appending. If the batch was unlocked or changed, append the record as stale or refuse registration and clearly mark the output stale. The backend must enforce this even if the viewer disables Unlock while a job is running. Add a barrier-based concurrency test.

### Medium — A malformed `batches.json` fails open and bypasses a locked batch

`batch_of()` catches `BatchError` and returns `None` (`batches.py:164-171`), so `review.transaction()` treats an unreadable registry as “not locked.” `validate` does not inspect `batches.json` at all.

Exact check on the locked test batch (the file was restored in `finally`):

```python
saved = batches_path.read_bytes()
try:
    batches_path.write_text("{")
    review.set_reviewed(run_path, review.load(run_path)["reviewed"])
    print("lock bypassed")
finally:
    batches_path.write_bytes(saved)
```

It printed `lock bypassed`; immediately afterward `dataop-logger validate /tmp/dataop_r18 --json` still returned `ok: true`.

Fix: fail closed—propagate registry read/validation errors from `batch_of`/`check_editable`. Extend `validate` to check batch schema, unique IDs/names, complete/unique membership, referenced committed runs, and locked review rev/digests.

### Medium — “Reviewed” remains true after later review changes

Through the viewer API I marked a run reviewed, then moved its drive boundary and set its failure start. Both responses still returned `effective.reviewed: true`:

```text
reviewed true -> rev 1
drive_boundary 901 -> 895 -> rev 2, reviewed true
failure_start = 890 -> rev 3, reviewed true
```

That permits a normal lock to count a run as reviewed even though its labels changed after the review sign-off. The UI also continues to show “Reviewed.”

Fix: any review-relevant mutation after `set_reviewed(true)` should clear `reviewed`, `reviewed_at`, and `reviewed_by` in the same transaction (at least add/replace/decide/undo/interpret and agent annotations). Add a test that marks reviewed, changes an accepted edit, and verifies lock refuses until it is marked reviewed again.

### Medium — The provenance record does not digest the complete output

Round 17 required an output digest. The record currently stores only a digest of output `data/*.parquet` (`export.py:399-412`). It does not digest output videos, `meta/info.json`, episode metadata, tasks or stats. A corrupted/replaced video therefore leaves the recorded digest unchanged. With `--val`, the record is also written only to the train root, not `<into>_val`.

```bash
jq '.written.train' /tmp/r18_viewer_dagger/dataop_export.json
# {path, episodes, data_sha256} only
```

Fix: record a canonical tree digest (relative path + bytes) for all output files except the record itself, per split, and write the provenance record (or a record plus cross-linked identity) into both train and validation roots. Include all source metadata files in the source digest too.

### Low — Import dry runs generate random run IDs

Two identical Watcher dry runs had identical plans and counts but different `runs[0].run_id` values:

```text
19eb58f8-e35c-46c1-92ae-b0b6f9c41d23
6872a1cc-7fcf-4cc3-a9c4-03fe590d2f0d
```

`_build_run()` allocates `uuid.uuid4()` even in dry-run (`importer.py:350, 401-415`). This makes agent diffs noisy and violates deterministic planning.

Fix: omit `run_id` from dry-run output, use a stable preview ID from source/episode/fingerprint, or allocate UUIDs only during the write phase.

## Functional results

Representative commands:

```bash
PY=/Users/chadster/lerobot/.venv/bin/python
PYTHONPATH=src $PY -m dataop_logger.cli import /tmp/dataop_r18 --dry-run
PYTHONPATH=src $PY -m dataop_logger.cli import /tmp/dataop_r18 --preset watcher \
  --map /tmp/r18_watcher_map.json --batch 'r18 watcher'
PYTHONPATH=src $PY -m dataop_logger.cli validate /tmp/dataop_r18 --json
PYTHONPATH=src $PY -m dataop_logger.cli export --batch '/tmp/dataop_r18:r18 watcher' \
  --recipe dagger --into /tmp/r18_viewer_dagger
PYTHONPATH=src $PY -m pytest -q tests
```

- Import: 40/40 episodes, 22,988 frames; 19,869 policy and 3,119 rescue frames; 31 success and 9 failure. Every imported `sent_action`, policy-frame `.npy` base action and rescue-frame expert action matched exactly (max absolute error 0). Re-import skipped all 40; validation was clean.
- Batches: list/new/close worked; normal lock refused 40 unreviewed runs; `lock --force` worked; a locked review edit was refused; unlock required a reason and marked existing exports stale. A simultaneous lock correctly waited for an in-flight review transaction and captured its final rev/digest.
- Recipe dry runs: filtered = 31 episodes / 13,878 frames; DAgger = 9 / 3,119; DAgger with successes = 40 / 16,997; residual = 0; residual with zero-on-success = 31 / 13,878. The zero plain residual count is correct for this Watcher session: raw base actions cover policy frames only, so rescue frames have no simultaneous base target. Tube-pick correction relabeling therefore still belongs in the agreed task hook/user script.
- Real DAgger export: loaded with `LeRobotDataset` as 9 episodes / 3,119 frames. Actions and states exactly matched the selected source frames; episode starts/ends and frame indices were correct; no `next.*` features remained. Across 54 sampled top/wrist frames, worst re-encode PSNR was 33.77 dB. A second real export had byte-identical Parquet and video files; dry-run JSON was identical.
- Reader: after moving a takeover from frame 901 to 895 and setting failure start 890, `DataopDataset` produced `train_mask = false` for 890–894 and true from effective takeover frame 895, as intended.
- Viewer API on port 8772: exercised `batch_status`, `reviewed`, `export_preview`, `export_start`, and polling `export_status`; the background DAgger write completed successfully.
- Tests: **84 passed in 5.89s**, including the accepted round 16 fixes.
