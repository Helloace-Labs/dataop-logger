# Round 2 decisions (logger-l3)

The round 1 confirmation (`round1_confirm_codex.md`) confirmed fixes 1, 3 and 5 and found a new High on fix 2 and a
Medium on fix 4. A new High in a confirmation round counts as a full round, so this is round 2.

2. **High, a named run whose batch closed before its join could still be placed in the closed batch. Changed.**
   `src/aftershock_logger/batches.py`, `_place`: a run committed by a logger with batches is now placed only through
   `_choose_for_commit`. If its named batch is no longer collecting, it is placed as a run recorded without a name
   (the collecting batch when its identity matches, otherwise a new collecting batch); it never falls through to the
   name lookup used for older runs. Test: `test_a_named_run_whose_batch_closed_before_its_join_is_not_placed_in_the_closed_batch`.
4. **Medium, a run moved out and back made other datasets report the training set current. Changed.**
   `src/aftershock_logger/export.py`, `training_sets`: a set is stale when its export record is stale in any of its
   source datasets, so an unlock or a move in one dataset reaches every other. Test:
   `test_a_run_moved_out_and_back_keeps_a_combined_training_set_out_of_date_everywhere`.

Both new tests fail on the round 1 code and pass now. Tests: 229 passed, 1 skipped.
