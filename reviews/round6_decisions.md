# Round 6 decisions

All findings were accepted and fixed. 49 tests pass, including new ones for each finding. A Playwright walk through the Review tab covers accepting an adjusted proposal, undo, step-boundary drag, mark reviewed, and a two-window conflict.

| Finding | Decision |
| --- | --- |
| H1 failure_start masks | `train_mask` now masks `[failure_start, next expert or blend frame)`, or to the run end when no takeover follows. `none` frames no longer end the stretch, and `blend` counts as a takeover. A failure start on a takeover frame masks nothing. `effective()` ignores `failure_start` when the effective outcome is success. Table-driven test covers every mode. |
| M1 lost updates | Every mutation runs in `review.transaction()`, which holds an `fcntl` lock on `review.lock` for load, change and save. `review.json` has a `rev` counter. The API requires the `rev` the page rendered and returns 409 with the current state when it is stale. The page refreshes itself and tells the reviewer. A test runs 4 processes adding 60 edits and checks that all of them survive. |
| M2 step_boundary | Edits name a step pair by index (`index`, `index + 1`). The pair must share a boundary in the recorded steps. The frame must leave each step at least one frame, and both sides move to exactly that frame. |
| M3 reviewed with open proposals | `set_reviewed(True)` refuses while proposals are open, and the button is disabled. A new agent proposal clears `reviewed`. `needs_review` is computed on the server and used by the page. |
| M4 undo | Older accepted edits are no longer rejected when superseded. The latest accepted edit per slot (`accepted_seq`) is the one in effect, so undoing it brings the previous one back. Tested A, B, undo B, undo A. |
| M5 proposals ignore corrections | `propose()` folds in each run's accepted corrections. It skips runs corrected to success and builds step budgets from effective successes and effective steps. The failure start uses the first takeover at or after the failed step's start. A run where nothing moves gets a `reject_run` proposal. `inspect` adds an `after_review` section. |
| M6 injection through ids | Ids are escaped at every HTML interpolation. The viewer skips and `validate` reports any run whose `run_id` is not a UUID matching its directory, and `review.load` rejects malformed edit ids. The page has a CSP with a fresh per-load script nonce, so injected inline handlers cannot run. |
| L1 missing saved record | Fails closed: no record, a wrong index or a malformed record keeps the run staged with `save_error` and stops the logger. Fault-injection test. |
| L2 contradictory summaries | `Run.end` rejects `failure_mode` on success, and `validate` checks it. Steps run one at a time (`subgoal_problem`), enforced while recording and on disk. |
| L3 truthy coercion | `accept` and `reviewed` must be JSON booleans; `rev` must be an int. |
| L4 Range | Strict single-range parser with suffix ranges, and 416 with `Content-Range: bytes */size` for anything else. Unit test. |
