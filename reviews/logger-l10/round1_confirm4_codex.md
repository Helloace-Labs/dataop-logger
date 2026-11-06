request changes.

Item 8’s lock change is correct: path resolution happens before `_stop_lock`. The seen marker, status assignment and event append remain together under the lock. However, the move introduces one nearby regression.

**Medium — Path errors can now escape into the run block.**

File and section: `src/aftershock_logger/live.py`, `controlled_stop`, lines 536–540.

The moved path work lost `_stop_update`’s outer `except Exception` protection. On supported Python 3.10–3.12, `Path.resolve()` can raise `RuntimeError` for a symlink loop. The new handler does not catch it. If a path becomes a loop before resolution, this tap error can replace the intended `ControlledStop` and skip final stop publication.

Keep resolution outside the lock, but restore broad exception containment around the moved work. Handle unexpected exceptions through `_fail` and return. Add coverage for resolution raising `RuntimeError`.

The lock fix is complete, but error isolation needs correction. No other nearby regression found. Tests and scripts were not run.