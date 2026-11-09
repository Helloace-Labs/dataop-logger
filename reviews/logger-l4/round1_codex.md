request changes

1. **High — Local writes can slip past sync ownership.**  
   `src/aftershock_logger/metadata.py`, `_write`, lines 486–507. The sync check happens before acquiring the dataset lock. A waiting local writer can proceed after another writer creates the sync marker. Recheck editability inside the lock, and coordinate marker creation with that lock. Add a test that starts sync while a local write waits.

2. **High — Broken metadata links silently erase the effective confirmation.**  
   `src/aftershock_logger/metadata.py`, `load` and `check_run`, lines 383–393 and 641–646. A dangling link reads as empty metadata, and validation skips it. Exports then use the manifest planner without reporting the missing confirmation. Distinguish an absent directory entry from an unreadable link. Add tests requiring validation, export, and local writes to refuse the broken link.

3. **High — A decided proposal incorrectly returns success.**  
   `src/aftershock_logger/metadata.py`, `propose`, lines 540–553; `tests/test_run_metadata.py`, lines 232–239. A proposal without a name succeeds after confirmation or removal. The transition rules require a conflict in those states. The test currently asserts this incorrect success. Reject these calls, preserve named suggestions, and test both decided states.

4. **High — Successful writes can create unreadable records.**  
   `src/aftershock_logger/metadata.py`, `_parse`, `_save`, and `write_synced`, lines 370–372, 466–470, and 625–635. Readers reject files above 64 MiB, but both writers can save them successfully. Full history snapshots can eventually cross that limit. Subsequent reads fail, and batch following can silently skip the record. Support valid histories consistently. Any retained size limit must be checked before replacing the existing file. Add a boundary test for both write paths.

5. **Medium — Synced confirmations leave stored batch flags wrong.**  
   `src/aftershock_logger/batches.py`, `_follow` and `_transaction`, lines 390–393 and 521–524. Sync can change a locked run’s effective planner while membership stays frozen. Since no membership changes, `_derive` skips that batch and its stored `mixed_policy` can remain false. Recompute this flag for batches affected by metadata changes while preserving membership. Extend the locked-batch test to inspect the saved batch record.

6. **Medium — Malformed hashes crash validation.**  
   `src/aftershock_logger/metadata.py`, `_detection_problems`, lines 191–208; `src/aftershock_logger/validate.py`, line 313. A list or object in `sha256` records a type error, then raises `TypeError` when passed to `set`. This escapes the metadata error handling and aborts validation. Check shapes before checking uniqueness. Add malformed-value tests that require a normal validation error and continued checking of other runs.

Read-only review. Tests were not run.