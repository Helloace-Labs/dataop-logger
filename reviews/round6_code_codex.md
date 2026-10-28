# Round 6 code review: dataop-logger

## Verdict

The requested test suite passes:

```text
/Users/chadster/lerobot/.venv/bin/python -m pytest -q tests
37 passed in 4.31s
```

The `meta.latest_episode` post-save change is correct for the LeRobot 0.6 writer: it avoids reading an open parquet file, and the 12-episode regression test exercises the metadata-buffer rollover. The POST endpoint's custom header plus exact Host/Origin checks also correctly block ordinary cross-origin forms/fetches and DNS rebinding, and the viewer consistently escapes the user-controlled task, label, environment, failure-mode, subgoal, and event text I checked.

One high-severity issue remains. The accepted `failure_start` correction can silently select the wrong training frames around `none`/`blend` transitions and continues masking a run after its effective outcome is corrected to success. There are also medium-severity review durability, supersession, proposal, step-boundary, and stored-XSS issues.

## High severity

### H1. `failure_start` produces incorrect training masks for mixed control modes and corrected outcomes

**File:** `src/dataop_logger/review.py:164-207`, `src/dataop_logger/reader.py:35-46`, `src/dataop_logger/reader.py:84-106`

**Failure scenario:** `train_mask()` walks forward only while the current mode is `base` or `blend`. That is not the promised interval from failure start up to the takeover:

- For `base, none, base, expert` with failure start 0, it masks only frame 0. The `none` frame stops the loop, so the later bad policy frame is kept even though takeover has not happened.
- For `base, blend, expert`, it masks both base and blend. A blend is itself a takeover (`derive_takeovers()` treats it that way), so the blend frame and its expert label should remain trainable.
- An accepted outcome edit to `success` does not retire or ignore an accepted `failure_start`. I reproduced an effectively successful run still masking frames 1–2.

The reader gates every DAgger/residual target on this array, so the first case silently trains on behavior the reviewer excluded and the second/third cases silently discard valid supervision.

**Concrete fix:** Compute the effective outcome first and ignore/retire `failure_start` when it is `success`. For a failed run, find the first takeover frame at or after `failure_start` whose mode is `expert` or `blend`, then mask the half-open interval `[failure_start, takeover)`, regardless of intervening `none` frames. If there is no later takeover, define and test whether the mask extends to the run end. Add table-driven tests beginning on and crossing every mode (`base`, `none`, `blend`, `expert`) and an outcome-corrected-to-success test through `DataopDataset`, not only `train_mask()`.

## Medium severity

### M1. Concurrent review writers can silently lose a human correction

**File:** `src/dataop_logger/review.py:50-62`, `src/dataop_logger/review.py:118-160`, `src/dataop_logger/review.py:225-273`, `src/dataop_logger/viewer.py:133-166`, `src/dataop_logger/viewer.py:204`

**Failure scenario:** Every mutation is a load-modify-atomic-replace transaction, but there is no filesystem lock or revision compare-and-swap. `_WRITE_LOCK` protects threads in one viewer process only; a second viewer process, `dataop-logger propose`, or another Python caller does not share it. Two writers can load the same version, append different edits, and each atomically replace `review.json`; the last replacement silently drops the other edit. I forced two callers past `load()` together and added two different notes: both calls succeeded, but the final file contained only one note.

Atomic replacement prevents a torn JSON file, but not a lost update—which is particularly damaging for a human correction.

**Concrete fix:** Put each run's review mutations behind a cross-process `flock` and perform load, validation, mutation, and save while holding it. Add a monotonically increasing review revision and require API clients to send the revision they rendered, returning `409 Conflict` on stale writes. Make `propose()` use the same transaction helper. Add a multiprocessing test that performs concurrent additions and asserts both survive.

### M2. `step_boundary` accepts nonadjacent steps and frames that create overlaps or gaps

**File:** `src/dataop_logger/review.py:65-100`, `src/dataop_logger/review.py:164-187`, `src/dataop_logger/viewer.js:227-258`, `src/dataop_logger/viewer.js:317-337`

**Failure scenario:** Payload validation checks only that both names occur somewhere and permits any frame in `[0, length]`. `effective()` then changes every subgoal with either matching name and clamps the two sides independently. With `a=[0,2)`, `b=[2,5)`, `c=[5,8)`, I submitted `{before:"a", after:"c", frame:7}`. It was accepted and produced `a=[0,7)`, `b=[2,5)`, `c=[7,8)`, so `a` overlaps `b`. Moving an adjacent boundary past the second step's end similarly makes the two sides disagree instead of sharing one boundary. Repeated step names are also ambiguous because all matching occurrences are changed.

**Concrete fix:** Give derived subgoal occurrences stable IDs (or use adjacent indices) and bind the edit to one exact pair. Require the pair to be adjacent with `before.end_frame == after.start_frame`, require the new frame to lie in `[before.start_frame, after.end_frame]`, and set both `before.end_frame` and `after.start_frame` to exactly that frame. Validate the final effective subgoals are ordered, nonoverlapping, and within the run. Test nonadjacent names, duplicate names, both extreme boundary positions, multiple moves, and undo.

### M3. A run can be marked reviewed while agent proposals remain unresolved, hiding them from the queue

**File:** `src/dataop_logger/review.py:118-160`, `src/dataop_logger/viewer.py:153-163`, `src/dataop_logger/viewer.js:17-20`, `src/dataop_logger/viewer.js:90-96`, `src/dataop_logger/viewer.js:201-209`

**Failure scenario:** `set_reviewed()` unconditionally sets `reviewed=True`, and adding a later agent proposal does not clear it. The UI's `needsReview()` requires `!reviewed`, and “Mark reviewed” does not require pending proposals to be accepted or dismissed. I reproduced a valid review file with `reviewed: true` and a `proposed` failure-start edit. The Review tab classifies that run as done, so the proposal can wait forever even though agent edits are supposed to require a human decision.

**Concrete fix:** Refuse `reviewed=True` while any proposal is pending, or define `needs_review = pending_edits or not reviewed` on both server and client. Adding an agent proposal to a reviewed run should clear `reviewed`/`reviewed_at`. Have the server return a conflict if “done” races with a new proposal. Add tests for marking done with a pending proposal and proposing after a run was reviewed.

### M4. Undoing a superseding edit skips the previous correction and falls back to the manifest

**File:** `src/dataop_logger/review.py:109-115`, `src/dataop_logger/review.py:139-154`, `src/dataop_logger/review.py:164-178`, `src/dataop_logger/viewer.js:176-178`

**Failure scenario:** Accepting a new single-valued edit permanently changes the previous accepted edit to `rejected` and records `superseded_by`. The UI labels rejection of the current edit “Undo”, but rejecting it does not reactivate its predecessor. I applied failure modes `first` and then `second`; undoing `second` made the effective failure mode jump to the original manifest value rather than `first`. The same problem affects outcome, failure start, and reject-run corrections.

**Concrete fix:** Preserve a distinct superseded state/chain. When the active edit is undone, reactivate the nearest still-valid predecessor, or compute the effective value as the latest accepted/non-undone edit without destructively rejecting older entries. Add A → B → undo B → A and multi-level undo tests, including an accepted adjusted agent proposal followed by a human correction.

### M5. Proposal and inspection logic ignores accepted review corrections and can propose against the wrong takeover

**File:** `src/dataop_logger/review.py:212-273`, `src/dataop_logger/validate.py:345-364`

**Failure scenario:** `_step_budgets()`, `propose()`, and `inspect()` use only manifest outcome/subgoals/failure mode. I corrected a manifest failure to effective success and then ran `propose()`; it still added a failure-start proposal. With two takeovers and a failed later step, the heuristic always used `takeovers[0]`, proposing frame 4 before an early intervention rather than the later failure/rescue. `inspect()` likewise continues to count the raw failure and raw failure mode while the viewer shows success. An entirely stationary run also receives no idle proposal because both idle branches are guarded by `moving.any()`.

These are plausible-looking proposals and summaries, so reviewers may accept a bad span without noticing the layer drift.

**Concrete fix:** Fold `review.effective()` before choosing successful runs, failed steps, outcomes, and failure modes. Choose the takeover associated with the failed step (at minimum the first takeover at/after that step, not globally the first), and explicitly handle an all-idle run with a reject/exclude proposal or a documented no-op. Make `inspect()` report effective counts by default or clearly expose separate raw/effective sections. Add corrected-to-success, multiple-takeover, corrected-step-boundary, and all-idle tests.

### M6. A tampered run/review identifier is inserted into `innerHTML` without escaping

**File:** `src/dataop_logger/viewer.py:56-85`, `src/dataop_logger/viewer.js:98-108`, `src/dataop_logger/viewer.js:160-178`, `src/dataop_logger/viewer.js:263-272`

**Failure scenario:** The ordinary user-authored fields are escaped, but `r.run_id` is interpolated raw into a `data-id` attribute and the Run ID `<dd>`. Review edit IDs are also interpolated raw into button attributes. Logger-created IDs are safe, but neither `viewer.collect()` nor `review.load()` validates them; manifest integrity does not authenticate `run_id`. A downloaded or locally altered dataset can therefore place an HTML event-handler payload in `run_id`. Once parsed into `DATA`, `render()` writes it through `innerHTML`. Script executing in this same origin can call the review POST API with the required Host, Origin, and custom header, defeating the otherwise sound cross-origin protections and changing reviews.

**Concrete fix:** Escape IDs at every HTML interpolation (including `data-id`, `data-acc`, and `data-rej`), validate manifest run IDs as UUIDs matching their directory name, and validate review edit IDs as the generated hex format. Add a restrictive CSP; preferably build DOM nodes with `textContent`/properties instead of concatenating run data into HTML. Add browser-level injection tests for every manifest/review string field, not only the embedded `</script>` case.

## Low severity

### L1. Missing in-memory saved metadata silently disables the post-save length check

**File:** `src/dataop_logger/adapter.py:33-45`, `src/dataop_logger/logger.py:415-448`

**Failure scenario:** `saved_episode()` explicitly returns `None` when `latest_episode` is unavailable or does not name the expected episode, but `_save_and_commit()` treats `None` as permission to commit. The supported LeRobot 0.6 implementation normally provides the record, so the current regression is fixed, but an unexpected writer state removes the exact post-save guarantee without an error. A malformed record can also raise outside the save exception handler, leaving ambiguous staged/logger state.

**Concrete fix:** Fail closed: absence, mismatch, or malformed saved metadata after the episode count advances should retain a staged run, record `save_error`, and break the logger just like a length mismatch. Add fault injection for missing, wrong-index, malformed, and wrong-length `latest_episode` records.

### L2. Recorded task-step/failure-mode states permit internally contradictory summaries

**File:** `src/dataop_logger/logger.py:251-259`, `src/dataop_logger/logger.py:287-320`, `src/dataop_logger/logger.py:464-483`, `src/dataop_logger/validate.py:174-184`

**Failure scenario:** `Run.end("success", failure_mode="dropped")` is accepted even though the review payload validator correctly forbids a failure mode on success. Subgoal events validate individual values but not lifecycle: starting the same name twice overwrites its earlier start, multiple steps can remain open/overlap, and terminal events can repeat without a start. `derive_subgoals()` makes a deterministic summary, so validation recomputes the same contradiction and reports it as valid. Ambiguous/repeated names then make boundary edits unsafe.

**Concrete fix:** Apply the same success/failure-mode invariant in the logger and disk validator. Define and validate the subgoal state machine (sequential versus parallel, repeated-name occurrence identity, duplicate start/terminal behavior, and reason type) before staging and independently on disk. Add malformed event-sequence tests.

### L3. The POST API coerces non-boolean decision values instead of validating them

**File:** `src/dataop_logger/viewer.py:153-161`

**Failure scenario:** `bool(req.get("accept"))` and `bool(req.get("reviewed", True))` make JSON strings such as `"false"` evaluate to `True`. A non-browser API client intending to reject a proposal or unreview a run can do the opposite while receiving HTTP 200.

**Concrete fix:** Require the fields to be actual JSON booleans and return 400 otherwise. Also require `accept` for `decide` and `reviewed` for the reviewed operation rather than supplying truthy defaults. Add string, number, null, and missing-field tests.

### L4. Range parsing is not compliant or robust for common malformed/suffix requests

**File:** `src/dataop_logger/viewer.py:168-201`

**Failure scenario:** `bytes=-500` is interpreted as bytes 0–500 instead of the final 500 bytes; multiple or nonnumeric ranges raise uncaught `ValueError`; and an end before start can produce a negative content length. Browser/video-backend variations can therefore break seeking or terminate a request handler instead of returning 416.

**Concrete fix:** Parse exactly one RFC 7233 byte range, implement suffix ranges, reject invalid/multiple/reversed ranges with `416` and `Content-Range: bytes */size`, and add request-level tests for open-ended, suffix, malformed, reversed, and out-of-bounds ranges.

## Important missing tests

1. Failure masks across `base → none → base → expert`, `base → blend`, failure starts on each mode, no later takeover, and effective success.
2. Multiprocess review additions/decisions and stale browser revisions.
3. Step boundaries for nonadjacent/duplicate names, out-of-span frames, repeated moves, and undo chains.
4. Pending proposals before “reviewed” and proposals added after review completion.
5. Effective outcome/step corrections through `propose()` and `inspect()`, multiple takeovers, and all-idle runs.
6. Browser-level XSS cases plus POST attempts from evil Origin, DNS-rebound Host, `Origin: null`, and a read-only server.
7. Missing/malformed `latest_episode`, strict API booleans, and complete HTTP Range behavior.
