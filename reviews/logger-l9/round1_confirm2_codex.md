approve

Checked only the four requested fixes in `2bdf831`.

- **1. Failed-run tick counts:** Correct and complete. `src/aftershock_logger/session.py:908` checks quarantined frame indices against exactly `0..n-1`, enforcing count and order. No adjacent regression found.
- **2. Close after a terminal tick:** Correct and complete. `src/aftershock_logger/session.py:262` clears both final flags before sealing as `system_error`. The episode can then be staged and saved. No adjacent regression found.
- **3. Finalize retry:** Correct and complete within scope. `src/aftershock_logger/session.py:279` and `:604` retain retry state, refuse new frames while ending, and retry using the first call’s end values. No adjacent regression found.
- **5. Nested type errors:** Correct and complete. `src/aftershock_logger/session.py:733`, `:766` and `:827` guard dependent lookups and isolate errors per session. No adjacent regression found.

No findings. Read-only review. No tests, scripts or other code were run.