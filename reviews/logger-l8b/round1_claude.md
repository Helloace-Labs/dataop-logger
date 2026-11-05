## Review by Claude (fallback, Codex at limit)

Reviewed commit 69e928660f88a6d97d8d1f10202ed375cfe2770c (round 1, full round) for Aftershock issue #89. Read only: no tests or code were run.

**Verdict: approve.** No High or Medium findings. Three Lows below.

### What I checked

- **Item 1, `retract`.** Keeps `decided_at` and `decided_by`, sets `retracted_at` and `retracted_by`, default step `what` is `take back`, refuses anything not `accepted`, and refuses `unknown` as `by` or `now`. The checks run in the same order as the reference, so the error class matches. Accept, reject, replace and retract all refuse a taken back edit through the status check.
- **Item 2, keys and checks.** `EDIT_KEYS`, `_new_edit` and `_f1_edit` carry both keys. `check_edit` has the set together, unknown only together and only on a take back rules. `check_review` refuses a take back that names nobody outside a migrated review, in edits and history snapshots. `_edit_to_format1` refuses `retracted_by`.
- **Item 3, lone surrogates.** `_str` refuses them in every field. `dump_review` refuses them. The format 1 read refuses them through `check_review` of the migrated result. Every operation checks its inputs before the no-change paths, including `propose_many` and `decide_many`.
- **Item 4, `unknown`.** It is allowed only when both values are `unknown`, only on agent edits, and only in a migrated review (snapshots included). `_check_proposal` refuses a caller-passed `unknown`, so `propose`, `propose_many` and `rv.add_edit` all refuse it.
- **N1.** `check_review` refuses `retracted_at` `unknown` when `legacy_subtask_map` is set. The test covers the edit case, the snapshot-only case and the non-migrated case, and also compares them with the reference live.
- **Item 6, L8 files.** `_before_take_back_fields` matches the reference line for line. `load_review` and `read` both apply it, so `rv.load` does too. `review_digest` is computed from the upgraded text. Every logger path that computes a batch lock digest (`batches`, `export`) reads through `rv.load`. `effective()` uses only active edits, so the recorded batch digests still verify. The test checks this against digests the 4f4395e logger itself recorded. The L8 corpus is real output from 4f4395e, taken from git history, and the expected upgrades are right (`unknown` only in the non-migrated take back file).
- **Same bytes.** Id draws follow the reference: in `propose`, the id is drawn before the reading and corrects checks. In `replace`, it is now drawn after the payload check. The step id is drawn only when something changed. The `take_back` and `migrated_take_back` cases interleave refused calls with later id draws, and the random sequences now include `retract`.
- **The extra change is in scope.** The shared contract has `propose` and `replace` apply the segment naming rule themselves. Without the move, `fmt.propose` and `fmt.replace` on a migrated review with a new segment name would refuse where the reference writes the file. The `review.py` behavior is the same rule in one place, and `propose_many` now applies it per item, which is consistent. The payload-before-id order is needed so that ids match after a refused replace.

### Low

**L1. Viewer take backs are not stored with the step `what` `take back`.**
Files: `src/aftershock_logger/review.py` (`decide`, which passes `what=box["what"]` to `fmt.retract`), `src/aftershock_logger/viewer.js` (`decide(r, id, false, "delete label")`, `"undo cut ..."`, `"keep ..."`, `"undo ..."`), `SPEC.md` (Operations).
The shared contract says a take back's step `what` is `take back`. SPEC.md now says the same for taking an edit back "in the viewer". But every take back made in the viewer is stored under the viewer's label, such as `delete label`. Only CLI and direct calls write `take back`. The old suffix used to name the person; it is gone now that `retracted_by` does that, so nothing is lost, but the doc and the stored file disagree.
Fix: either make `fmt.retract` always write `take back` and keep the viewer's label for display only, or keep the viewer labels and correct the SPEC.md sentence. With the second option, ask the lead to note in the contract that a logger step's `what` may carry the viewer's label.

**L2. Two lone surrogate cases pass without the check they test.**
File: `tests/test_review_format.py` (`_surrogate_calls`).
`("accept", dict(edit_id=cut, by=S))` and `("reject", dict(edit_id=cut, by=S))` target `cut`, which is accepted. Both calls are refused for status whether or not `_who` checks the text. The live comparison checks only the error class, so it cannot tell the two causes apart. The `by` check in accept and reject is therefore untested.
Fix: have `_with_open_proposal` return the open proposal's id and use it for the accept and reject cases.

**L3. Two of the new byte rules are pinned only in tests that need the reference checkout.**
Files: `tests/review_cases.py` (cases), `tests/test_review_format.py` (`test_random_changes_are_refused_or_accepted_like_the_reference`).
(a) No committed case has a replace refused for its payload followed by a call that draws an id. Only the random sequences cover the payload-before-id fix, and they skip without the reference. Today's CI would not catch a regression.
(b) The random file change corpus (`every_kind`, `lock_and_unlock` and the migrated all kinds text) holds no edit with `retracted_*` set, so the random changes never reach the new field rules on a real take back.
Fix: add a step to a case, for example a replace of an accepted `failure_start` with a label payload followed by a propose, so `expected.json` pins it. Add the `take_back` case's final text to the mutation texts. Optionally, add an L8 file where a take back was later undone, so that a snapshot is upgraded to `unknown` while the edit itself reads null.
