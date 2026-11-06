request changes.

The ordering race is fixed, but item 8 is not complete.

**Medium — Filesystem work remains inside the publication lock.**

Files and sections: `src/aftershock_logger/live.py`, `controlled_stop` and `_stop_update`.

`controlled_stop` holds `_stop_lock` while `_stop_update` calls `Path.resolve()` on the quarantine path and root. These calls access the filesystem. A slow lookup now also blocks the tap thread in `stop_seen`, delaying status and event writes. This violates the decision to keep filesystem work outside the critical section.

Compute the relative path before acquiring the lock. Keep the seen marker, status assignment and event append together inside the lock.

No other nearby regression found. Tests and scripts were not run.