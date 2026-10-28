# Round 19 decisions (verification of round 18)

Codex confirmed the round 18 fixes (race, review reset, tree digests, split records, dry-run ids). Both new findings were accepted and fixed, with tests.

| Finding | Decision |
| --- | --- |
| M multi-source export could be current in one batch and stale in another | `add_export` takes every source at once, locks all dataset roots in path order, decides stale once for all of them and writes the same result to every registry. The output record is written after registration; if registration fails, the record says `registered: false`. Test: two roots, one unlocked between writing and registration. |
| M malformed nested batch fields crashed validate | `_read` checks every batch field and type (ids, name, status, exports, history, policy, parent, lock and its runs, status agreeing with the lock). `validate` turns any malformed registry into an error. Tests for four kinds of breakage. |

Import also now resolves side-file patterns against the source dataset for every preset, not only `watcher`.
