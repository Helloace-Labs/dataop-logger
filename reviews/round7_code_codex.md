# Round 7 code review: dataop-logger

## Verdict

The requested test suite passes:

```text
/Users/chadster/lerobot/.venv/bin/python -m pytest -q tests
50 passed in 5.48s
```

The new Browse/Edit flow is substantially simpler than the old Review tab, and the main happy path is sound: labels use half-open frame ranges, writes remain atomic and cross-process locked, the browser sends a rendered revision, `replace_edit()` gives a rewritten label a new ID without its old interpretation, and an interpreted `exclude` gates every reader target through `dataop.train_mask`. Recorded frames, manifests, and integrity digests are not changed.

There are **no high-severity findings**. I would still address the medium findings before treating this as the final review workflow. In particular, the legacy `decide()` path can bypass interpretation invalidation, two agents can silently replace one another's interpretation, and old structured corrections/proposals are still active but no longer manageable in the viewer.

## Product

The owner-requested core is present: there is one Browse view, an Edit button beside the timeline, drag/click span selection, plain-language labels with optional driver and segment, and click-to-change/delete. The CLI loop is discoverable in the README and `labels --uninterpreted` includes useful recorded context.

The product contract still needs tightening:

- “What happens here?” looks required, but the backend permits a driver-only or segment-only label. That weakens the plain-language handoff to the agent.
- Editing an interpreted label correctly creates a new uninterpreted label, but the page does not warn that saving immediately removes the old training exclusion until the agent interprets the replacement. The label does show “not read by an agent yet” after saving, which is useful but late.
- “Who drove” and “Task segment” read like corrections to training metadata. They are currently annotation-only: neither changes nor adds any per-frame field returned by `DataopDataset`. This should be made explicit, or the reader needs a defined review overlay.
- Existing labels cannot have their span resized; changing the span requires deleting and recreating the label. Segment choices are collected globally from every task/run, rather than from the current run/task, so unrelated step names will accumulate in the form.
- The old proposal/correction model remains in the CLI and backend, while the only UI capable of resolving or undoing it was removed. The README also still calls the default viewer “read-only” even though it now writes `review.json` unless `--read-only` is supplied.

## High severity

None.

## Medium severity

### M1. `decide()` can rewrite an interpreted label in place and apply the old decision to different frames

**File:** `src/dataop_logger/review.py:206-219`, `src/dataop_logger/review.py:225-253`, `src/dataop_logger/review.py:310-324`; `src/dataop_logger/viewer.py:170-173`

**Failure scenario:** The new UI correctly uses `replace_edit()`, but the still-public `decide()` operation accepts an already accepted label, replaces its payload, and calls `_accept()` without clearing `interpretation`. I reproduced an accepted label `[1,3)` interpreted as `exclude`, then called `decide(..., accept=True, payload={start:6,end:9,...})`. The same label ID retained the “camera blocked” interpretation and `train_mask()` silently excluded frames 6–8, even though those frames and text described a different event. The POST API still exposes `op="decide"`, so this is not only an internal helper edge case. It contradicts SPEC.md's promise that rewriting a label creates a new edit without the old interpretation.

**Suggested fix:** Restrict accepting/adjusting with `decide()` to edits whose current status is `proposed`; accepted edits should only be rejectable for undo. Route every accepted-label payload change through `replace_edit()`, or have `decide()` create a replacement label with a new ID and no interpretation. Add an API-level regression test that attempts to adjust an interpreted accepted label.

### M2. Re-interpreting a label silently overwrites the previous agent decision

**File:** `src/dataop_logger/review.py:241-253`; `src/dataop_logger/cli.py:77-83`

**Failure scenario:** `interpret()` writes a single `edit["interpretation"]` object unconditionally. Two agents can both read the same `labels --uninterpreted` output; the first records `exclude`, then the second records `keep`. Both commands succeed under the lock, but the second silently erases the first decision and changes the training mask. The lock prevents a torn/lost JSON transaction, but it does not detect this semantic stale write, and the CLI supplies neither an expected review revision nor an expected “still uninterpreted” condition. Even a repeated manual command can silently reverse a prior decision with no audit trail.

**Suggested fix:** Make interpretation creation compare-and-set: by default reject a label that already has an interpretation. If reinterpretation is required, expose an explicit `--replace`/`--force` operation and append an interpretation history (or create a new revisioned interpretation edit) rather than deleting the previous decision. Include the review revision in `list_labels()` and optionally require it on `interpret`. Test two interpreters racing and keep-to-exclude/exclude-to-keep replacement.

### M3. Legacy corrections still affect training, and `propose` still creates work, but the viewer no longer shows or resolves either

**File:** `src/dataop_logger/review.py:276-329`, `src/dataop_logger/review.py:346-407`; `src/dataop_logger/cli.py:41-42`, `src/dataop_logger/cli.py:86-89`; `src/dataop_logger/viewer.js:165-217`

**Failure scenario:** A dataset reviewed before 61b25e6 can contain accepted `exclude`, `failure_start`, or `reject_run` edits. Those edits continue changing `train_mask`, but the new page no longer draws excluded spans/failure starts or offers undo, so a user cannot see why apparently ordinary frames are missing from training. Separately, `dataop-logger propose` still creates pending agent edits and `inspect` counts them, but the only UI that could accept or dismiss those proposals has been removed. A user following the still-documented command can create a permanently open, invisible queue.

**Suggested fix:** Choose one coherent model. If plain-language labels replace the old review workflow, retire `propose` and the viewer mutation API for legacy kinds, migrate accepted legacy corrections into visible labels/interpretations where possible, and clearly report any unmigrated corrections. If structured corrections remain supported, add a compact read-only “training effects” summary plus explicit resolve/undo controls in Browse/Edit. At minimum, never let the CLI create proposals that no supported UI or CLI command can resolve.

### M4. Driver and segment selections look authoritative but have no effect on reader metadata or target semantics

**File:** `src/dataop_logger/review.py:149-164`, `src/dataop_logger/review.py:282-307`, `src/dataop_logger/review.py:410-437`; `src/dataop_logger/reader.py:45-46`, `src/dataop_logger/reader.py:98-107`

**Failure scenario:** A reviewer labels a span as “rescue” because the recorded timeline incorrectly says policy, or assigns a new task segment because recorded subgoals are missing. `effective()` retains those values only inside `labels`; `DataopDataset` continues returning the recorded `control_mode`, validity fields, and target masks, and exposes no effective driver/segment overlay. Thus an executed-residual target can still be treated as policy-driven even though the UI says rescue, while a new segment is unavailable to training altogether. The fields are useful agent context, but the UI wording does not distinguish context from correction.

**Suggested fix:** Decide and document the semantics. If these are context only, label them “Mentioned driver” / “Segment context” and state explicitly that only an agent's keep/exclude interpretation affects training. If they are corrections, expose immutable-recorded and effective-reviewed fields separately (for example `dataop.recorded_control_mode`, `dataop.review_driver`, and `dataop.review_segment`) and define how each target mask responds to policy/rescue labels, including overlaps and `blend`/`none`. Do not overwrite the recorded tick table.

### M5. The local viewer can be framed by another website, enabling clickjacking writes

**File:** `src/dataop_logger/viewer.html:8`; `src/dataop_logger/viewer.py:123-132`, `src/dataop_logger/viewer.py:143-151`

**Failure scenario:** Host/Origin/custom-header checks block cross-origin scripts from directly POSTing, but another site can embed the localhost viewer in an iframe and visually overlay instructions. Clicks performed inside that same-origin viewer legitimately send the required header and localhost Origin; a user can be tricked through Edit/drag/driver/Save or label/Delete interactions. The CSP exists only as a meta tag and has no `frame-ancestors` directive; in any case, `frame-ancestors` must be delivered as an HTTP response header to prevent embedding.

**Suggested fix:** Send the full CSP as an HTTP header and include `frame-ancestors 'none'`; also send `X-Frame-Options: DENY` as a compatibility defense. Keep the meta CSP for `--build-only`, but do not rely on it for framing protection. Add a browser test that attempts to load the served viewer in a cross-origin iframe and verifies it is blocked.

### M6. A revision conflict refreshes only the run list, leaving the editor on stale label state

**File:** `src/dataop_logger/viewer.js:102-119`, `src/dataop_logger/viewer.js:122-131`

**Failure scenario:** On HTTP 409, `send()` replaces `r.review` and redraws the list, but does not rerender the current detail panel or reconcile `sel`. If another window replaced/deleted the label being edited, the page tells the user to “check its labels” while continuing to display the stale label and draft. Clicking Save again uses the new revision with the old rejected ID and returns 400; the user must cancel/reopen or refresh to discover the actual state. The Round 6 implementation rerendered the current run on conflict, so this is a regression.

**Suggested fix:** On 409, rerender the current run from `out.state`. If the selected label still exists, refresh its saved fields while preserving the user's draft separately; if it was replaced/deleted, clear the selection and show a persistent conflict message with an explicit “copy draft to new label” choice. Add a two-window browser test for same-label replace/delete and unrelated-label changes.

## Low severity

### L1. Plain-language text is not actually required

**File:** `src/dataop_logger/review.py:149-164`

**Failure scenario:** The requested flow says the user supplies plain-language text and may optionally add driver/segment. `check_payload()` instead accepts an empty `text` whenever either optional field is present. A user can select “policy” and save without answering “What happens here?”, leaving the agent an uninterpreted label with no human explanation. This also makes the UI's required-looking question misleading.

**Suggested fix:** Require `text.strip()` to be non-empty for every label; keep driver and segment independently optional. Add backend and POST tests for driver-only and segment-only labels, plus client-side validation that leaves the draft visible with a clear error.

### L2. Form rerenders can discard an unsaved custom segment, and Enter in the textarea saves instead of adding a line

**File:** `src/dataop_logger/viewer.js:303-318`, `src/dataop_logger/viewer.js:324-329`

**Failure scenario:** After typing a new segment name, clicking a driver chip calls `render()` after copying only the textarea. The uncommitted segment input disappears, so the user must type it again. In the textarea, ordinary Enter immediately clicks Save; a first-time user trying to add a newline can prematurely save an incomplete label. Escape, switching runs, and Done also discard the draft without warning.

**Suggested fix:** Keep the entire form in `sel` on every input event (text, driver, segment) instead of scraping selected fields before rerenders. Preserve standard textarea behavior and use Cmd/Ctrl+Enter for Save. Warn before discarding a dirty draft. Cover these interactions with a browser test.

### L3. Segment choices are global and an existing label's span cannot be corrected

**File:** `src/dataop_logger/viewer.js:19-20`, `src/dataop_logger/viewer.js:133-147`, `src/dataop_logger/viewer.js:221-225`

**Failure scenario:** `segmentNames()` combines steps and labels from every run, so a multi-task dataset offers irrelevant step names from other tasks and grows an unwieldy chip list. After opening a label, its start/end are fixed in `sel`; dragging selects a separate new label rather than moving/resizing the current one. A slightly wrong range therefore requires delete-and-recreate, which is not obvious from “click a label to change it.”

**Suggested fix:** Offer the current run's effective subgoals first (optionally same-task names behind a chooser), and let an opened label reselect or resize its span while still saving through `replace_edit()` so interpretation invalidation remains correct.

### L4. Stored label payloads and interpretations are not validated when loaded

**File:** `src/dataop_logger/review.py:70-86`, `src/dataop_logger/review.py:282-324`; `src/dataop_logger/validate.py:188-190`

**Failure scenario:** `load()` validates only edit ID/kind/status/source. It does not validate a label's frame range, driver, segment, or interpretation shape/train value. Because `validate` only calls `load`, a copied, manually edited, or externally generated `review.json` can pass validation while a misspelled `train` silently keeps frames, a negative/out-of-range span masks the wrong frames through NumPy slicing, or a non-object interpretation crashes the reader/viewer. Atomic writes prevent torn files but do not make externally supplied content valid.

**Suggested fix:** Add manifest-aware review validation and call it from validation, the reader, and viewer collection. Check every payload with the same rules used at creation, validate interpretation keys/types/enums, revision/sequence types, uniqueness of edit IDs, and replacement references. Fail closed before constructing a training mask. Add malformed stored-review fixtures.

### L5. Documentation still describes the default viewer as read-only

**File:** `README.md:145-168`

**Failure scenario:** The README says the page is “read-only,” then immediately documents Edit and saved labels. A user may start the default server believing no local state can change, even though it writes `review.json`; the actual read-only mode is not mentioned in this section.

**Suggested fix:** Say that recorded data/manifests are read-only while the separate review layer is editable, and document `dataop-logger view --read-only` for a viewer that rejects all POSTs.

## Important missing tests

1. Adjusting an interpreted accepted label through `decide()`/the POST API must not retain the old interpretation.
2. Two concurrent interpreters, repeated interpretation, and an explicit audited reinterpretation path.
3. Upgrade fixtures containing accepted legacy exclusions/failure starts/rejections and pending proposals in the new viewer.
4. Browser-level drag in both directions, click selection at both ends, label span replacement, textarea/new-segment state, keyboard behavior, delete, and two-window conflicts.
5. Cross-origin iframe embedding, in addition to the existing Origin/Host/custom-header POST checks.
6. Reader and validator behavior for malformed stored label payloads and interpretations.
