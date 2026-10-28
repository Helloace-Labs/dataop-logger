# Round 18 decisions (code review and functional test of batches, import and export)

All findings were accepted and fixed, with tests.

| Finding | Decision |
| --- | --- |
| H export finishing after an unlock was registered as current | Export remembers the exact lock it was planned from. Registration happens under the batches lock and marks the export stale unless the batch still holds that same lock. The record file and the viewer job say so. Test: an unlock injected between writing and registration. |
| M broken `batches.json` unlocked a locked batch | The lock check fails closed: an unreadable registry refuses review edits. `validate` now checks `batches.json` (schema, unique ids and names, one collecting batch, members are committed runs, locked runs still match their lock digests). |
| M "reviewed" survived later changes | Any change to a run's edits clears its reviewed mark in the same transaction, so a batch cannot lock until the run is looked at again. |
| M output digest covered only data parquet | Each split records a digest of its whole folder (paths and bytes, except the record itself). The record is written into the train and the val folder, with `this_split`. Source digests cover all of `meta/`, `data/` and `videos/`. |
| L dry-run import printed random run ids | Dry runs print no run id. |
