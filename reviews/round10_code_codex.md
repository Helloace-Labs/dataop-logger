# Round 10 code review: dataop-logger

## Verdict

The suite passes at HEAD: **60 passed in 5.01s**. The core snapshot/restore logic correctly handles added edits, `replace_edit`, interpretations, idle-finder additions, and multi-decision steps; undo leaves `seq` monotonic and the restored edits pass `check_review()`.

There are **no high-severity findings**. Three medium issues remain around failed/stale browser undos, reload persistence, and validation of stored history. I tested an archive of HEAD because unrelated uncommitted viewer edits appeared in the live worktree during the review; I did not modify or include them.

## High severity

None.

## Medium severity

### M1. A failed undo is removed from the page stack before the server accepts it

**File:** `src/dataop_logger/viewer.js:183-191`

**Failure scenario:** `undoLast()` pops first and only then sends the request. If another window made even an unrelated edit on that run, the first attempt gets the intended 409 and refreshes `r.review`, but the undo entry is already gone. Retrying would have succeeded when the touched edit was unchanged; similarly, a network error or “undo the later change first” response permanently removes the only UI handle for the step.

**Fix:** Peek at the entry, send the request, and pop only after a successful undo. Retain it on 409/network/ordering failures; remove it only when success or an authoritative response proves the entry was already undone or pruned. Add a two-window test where an unrelated write causes one 409 and the retry succeeds.

### M2. Reloading the viewer discards all undo capability despite persistent history

**File:** `src/dataop_logger/viewer.js:17`; `src/dataop_logger/viewer.py:56-60`

**Failure scenario:** `undoStack` always starts empty, while `_review_state()` omits history. After a cut, label replacement, or Cut all, refreshing the page removes the Undo button and Cmd+Z says there is nothing to undo, even though the entry remains in `review.json`. This also makes the stored 200-step history unusable without manually extracting IDs.

**Fix:** Persist the page's ordered stack in dataset-scoped `sessionStorage` and validate it against the server after reload, or return a compact list of undoable `{entry, what, run_id, order}` records and hydrate the stack. If reconstructing across runs server-side, add an unambiguous global/high-resolution order; the current second-resolution `at` cannot order same-second steps on different runs.

### M3. History snapshots are not validated and `undo()` can restore corrupt edits

**File:** `src/dataop_logger/review.py:72-88`, `src/dataop_logger/review.py:137-156`, `src/dataop_logger/review.py:397-428`

**Failure scenario:** `load()` and `check_review()` validate current edits but ignore `history`; `undo()` trusts each entry's shape and copies `before` directly into `edits`. I changed the saved `before.payload.start` of an accepted cut to `-50`; `undo()` returned successfully, bumped `rev`, and persisted the invalid edit, after which `effective()` failed closed. Malformed history types can also raise uncaught exceptions in the API rather than a review error.

**Fix:** Validate history length, entry IDs/fields, unique touched IDs, and both snapshots using the same edit validator. Before committing an undo, apply it to the in-memory review and run `check_review(manifest, review)`; pass the manifest into `undo()`. Convert malformed-history failures to `ReviewError` and add corruption fixtures for invalid snapshots, duplicate changes, oversized history, and missing fields.

## Low severity

### L1. One `recording()` block can silently produce several history steps

**File:** `src/dataop_logger/review.py:95-130`

**Failure scenario:** Each nested transaction appends independently and overwrites `box["entry"]`. I wrapped two `add_edit()` calls in one `recording("two cuts")`; it stored two history entries, returned only the second ID, and undoing that ID left the first cut active. This contradicts the context manager's “one undoable step” contract, though current viewer operations each use one transaction.

**Fix:** Reject a second transaction inside one recording context, or accumulate one before/after set and persist a single entry at context exit. Add a two-transaction recording test.

### L2. Cross-run undo bypasses a saved draft on the target run

**File:** `src/dataop_logger/viewer.js:16`, `src/dataop_logger/viewer.js:183-191`

**Failure scenario:** `busy()` checks only the current run. If the target run has a dirty draft stored in `drafts`, cross-run undo calls `render()` directly, hiding rather than restoring or reconciling that draft. A later normal switch can resurrect it against an edit the undo removed, producing a stale-label save failure.

**Fix:** Route the switch through draft-aware selection logic. If the target has a dirty draft, restore it and require Save/Cancel before undo, or explicitly reconcile/discard it after confirmation.

### L3. The 200-entry limit does not bound history bytes or retained text

**File:** `src/dataop_logger/review.py:117-130`; `src/dataop_logger/review.py:321-337`

**Failure scenario:** Every change stores full before and after edit snapshots; one `decide_many` step may contain 1,000 edits, and 200 such entries can make `review.json` very large and expensive to parse/rewrite. Snapshots also retain old label text, reasons, attribution, and interpretation history after a user changes or undoes them. History is correctly omitted from viewer responses, so this is not an HTML/API leak, but it is persistent dataset retention and potential local storage/latency bloat.

**Fix:** Store compact field deltas or deduplicated edit versions, cap total serialized history bytes/change count as well as entry count, and document that deleted/corrected review text remains in undo history until pruning.
