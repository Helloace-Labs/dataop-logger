# Round 11 code review: dataop-logger

## Verdict

The suite passes: **61 passed in 5.09s**. There are **no high-severity issues**. Four Round 10 findings are fixed; history validation and the byte cap are partly fixed. Two new medium issues remain: stale-but-not-undone history can permanently block Cmd+Z, and resizing a new label keeps defaults computed for its old span.

## Round 10 audit

| Round 10 finding | Status | Evidence |
| --- | --- | --- |
| M1: failed undo was removed from the page stack | **fixed** | The client no longer pops a local stack: `viewer.js:17-19,184-195` derives undo steps from server state, and a failed request leaves that state intact. A 409 refreshes the run at `viewer.js:140-144`, so an unrelated concurrent write can be retried. |
| M2: reload discarded undo capability | **fixed** | `_review_state()` returns persisted undo metadata (`viewer.py:56-62`); `undoList()` rebuilds and orders the cross-run list using nanosecond `t` values (`viewer.js:17-19`). Reload, successful removal, and cross-run switching now follow server history. |
| M3: corrupt history could restore invalid edits | **partly fixed** | `undo()` now validates the candidate review before commit (`review.py:148-170`), so I could not make an invalid snapshot corrupt current edits. However, stored snapshots receive only shallow id/kind/status checks (`review.py:438-454`): an accepted snapshot missing `payload` and `source` still passes `check_review()`, is advertised for undo, and then fails with `KeyError`/400. Reuse the complete edit/payload/reading validator for both snapshots and validate unique history IDs, `at`, positive non-bool `t`, and the byte limit. |
| L1: one recording could silently create several steps | **fixed** | `transaction()` rejects a second transaction after the first committed one (`review.py:117-127`), matching the accepted “one transaction per recording” design; the new test covers it. |
| L2: cross-run undo bypassed a target-run draft | **fixed** | `undoLast()` switches through `select()` and then calls `busy()` on the restored target draft before writing (`viewer.js:184-195`). |
| L3: history had no practical size bound | **partly fixed** | Count and normal multi-step growth are pruned at `review.py:92-93,131-135`, and SPEC documents retained text. But `len(hist) > 1` deliberately leaves one oversized step: I produced a valid **2,549,579-byte** history against the 2,000,000-byte limit. Existing oversized history also passes `check_review()`. If one new step exceeds the cap, store a compact delta or omit that undo entry; enforce the serialized-byte limit during validation too. |

## New findings by severity

### High

None.

### Medium

#### M1. A non-recorded edit can leave Cmd+Z permanently stuck on an impossible step

**File:** `src/dataop_logger/review.py:143-145`; `src/dataop_logger/viewer.js:184-195`

**Failure scenario:** `undoable()` means only “not marked undone”; it does not verify that every current edit still equals the step's `after` snapshot. I accepted a label through a recorded viewer-style step, then interpreted it through the CLI (which is intentionally not a viewer history step). The old step remained newest in `undoable`, but every Cmd+Z failed with “changed again since.” Because the UI always retries that same entry, older unrelated steps become unreachable too.

**Fix:** Return only currently applicable entries—every touched edit must equal `after`. This naturally hides an older step while a later step is active and reveals it again after that later step is undone. Add viewer-step → CLI interpretation/replacement tests and verify older independent undos remain reachable.

#### M2. Resizing a new selection preserves driver/segment defaults from the previous span

**File:** `src/dataop_logger/viewer.js:229-236`, `src/dataop_logger/viewer.js:445-446`

**Failure scenario:** A first drag over policy/grasp correctly defaults to `policy` and `grasp`. Dragging again over rescue/lift before saving takes the `sel ? {...sel}` branch, so the panel still says policy/grasp even though `defaultsFor()` would now return rescue/lift. Those stale notes can mislead the later interpreting agent.

**Fix:** Track whether driver and segment were auto-filled versus explicitly changed. Recompute untouched defaults whenever a new-label span is resized; preserve user choices and values on an existing saved label. Add boundary, tie, blend/none, resize-before-save, and manual-override tests.
