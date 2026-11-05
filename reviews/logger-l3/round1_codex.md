request changes

1. **High — Migration can bypass locked review protection.**  
   Files: `src/aftershock_logger/batches.py`, `_load` and `batch_of`; `src/aftershock_logger/review.py`, `transaction`.  
   A review reader can check for the batch directory before migration creates it, then check the journal and legacy file after migration removes them. It sees an empty registry and permits editing a locked run. Acquire a shared batch lock before the review lock and hold it through the edit. Add a test that pauses migration between these reads.

2. **High — A concurrent read can assign a new run to a closed batch.**  
   File: `src/aftershock_logger/batches.py`, `_place` and `join`.  
   Between manifest commit and `join`, another reader can place the run into an older reviewing batch with matching identity. `join` then returns without opening the required collecting batch. Use the same collecting-batch selection for normal commits and recovery. Keep legacy grouping separate. Test a read between commit and joining.

3. **High — Import-driven moves leave exports marked current.**  
   Files: `src/aftershock_logger/batches.py`, `add_runs`; `src/aftershock_logger/export.py`, `training_sets`.  
   `add_runs` can move an existing run between unlocked batches without invalidating the source exports. Since reviews stay unchanged, training-set status also stays current. Share the invalidation logic with `move`. Add a test that exports an unlocked batch, moves a run through `add_runs`, and checks its status.

4. **High — Move invalidation does not reach other dataset roots.**  
   Files: `src/aftershock_logger/batches.py`, `move`; `src/aftershock_logger/export.py`, `training_sets`.  
   For an export combining roots, moving a run only marks the local export record stale. Another source root still reports the training set current because its status check compares review digests without checking membership. Check membership and invalidation across every recorded source. Test the result from both roots after a move.

5. **Medium — Saved pack records omit required fields.**  
   Files: `src/aftershock_logger/batches.py`, `_new` and `_record`; `docs/DESIGN_batches_import_export.md`, section 1 storage schema.  
   Records lack `kind` and the separate `station`, `task` and `planner` fields. Nesting identity values does not supply those fields to pack consumers. Update serialization, migration and the documented schema. Add assertions for both policy runs and teleoperation.

Static review only. Tests were read, not run.