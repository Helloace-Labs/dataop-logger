request changes

1. **Migration protection: correct and complete for the reported review race.**  
   Files: `src/aftershock_logger/batches.py`, `src/aftershock_logger/review.py`.  
   Review edits hold the shared batches lock through the write. Both review edits and batch locking acquire the batches lock first. No lock-order cycle was found in the inspected paths, within one process or across processes.

2. **High — Named recovery still accepts closed batches.**  
   Files: `src/aftershock_logger/batches.py:345`, `tests/test_batches_files.py:402`.  
   Fix 2 is incomplete. If a named batch closes during recording, `_choose_for_commit` rejects it. `_place` catches that rejection, then selects the same reviewing batch by name. A reader before `join` still bypasses the collecting-only rule.  
   Fix: keep logger recovery separate from legacy grouping. Preserve the rejection or choose a collecting batch through the shared chooser. Add a named-run test that closes its batch before commit and reads before `join`.

3. **Import invalidation: correct and complete.**  
   File: `src/aftershock_logger/batches.py`.  
   `add_runs` marks source exports stale under the same transaction as the move. Failed moves leave those changes uncommitted. No adjacent regression was found.

4. **Medium — Moving back loses invalidation in other datasets.**  
   File: `src/aftershock_logger/export.py:535`.  
   Fix 4 detects runs currently in another batch. Moving a run out and back makes `moved_runs` zero again. The source export remains permanently stale, but another source reports the same training set current when reviews are unchanged.  
   Fix: also read the matching export’s stale flag from every recorded source and combine those flags. Add a move-out-and-back test that checks status from both datasets.

5. **Pack fields: correct and complete for new records and legacy migration.**  
   File: `src/aftershock_logger/batches.py`.  
   The fields come from identity, and policy null stays distinct from unknown. No adjacent regression was found.

Static review only. Tests were read, not run.