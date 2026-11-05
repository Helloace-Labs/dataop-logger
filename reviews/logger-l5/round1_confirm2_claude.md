## Review by Claude (fallback, Codex at limit), confirmation round 2

Reviewed commit: `71760cc97c2010ea4b351e72352f780d5c96bff3` (diff `bb917a2..71760cc`)

**Verdict: approve.** 0 High, 0 Medium, 0 Low. L6, L7 and L8 from the first confirmation round are resolved, and the fixes add no new findings.

- **L6 resolved.** `default_spill_dir()` is now a per-user persistent folder (`$XDG_DATA_HOME/aftershock-logger/spill`, else `~/.local/share/aftershock-logger/spill`), never the system temp folder. When the default is used, the close message says it is usually on the same disk as the dataset and recommends a persistent folder on a different disk. README, SPEC and the `RunLogger` docstring say the same. Both tests cover it.
- **L7 resolved.** After a spill, `close()` sets `spilled` and `quarantine_path` on the run's `ControlledStop` and calls `_settle()`, so the message says "spilled to <path>" and no longer says "NOT written". `__exit__` no longer lets a close `RunError` replace an exception that is leaving the block. A `ControlledStop` gets it as `close_error` and in its message, a warning repeats it, and the original exception keeps propagating. A plain `close()` still raises. `_settle()` leaves `close_error` out of the message when the run was spilled, which is fine because the logger refuses new runs while one is unsaved, so the close error can only be about that same run.
- **L8 resolved.** `test_write_partial_returns_this_runs_record_and_refuses_another` covers the matching and the mismatched existing record. `test_an_interrupt_during_the_quarantine_write_blocks_the_logger` checks `broken`, `_unsaved_runs`, refusal of a new run and a working retry. `test_close_does_not_misreport_an_idle_writer` covers round 1 L5. The rewritten spill tests assert the stop propagates, its message and attributes, and the warning.

No public-repo hygiene problems in the diff.

