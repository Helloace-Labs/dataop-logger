# Round 8 code review: dataop-logger

## Verdict

The requested test suite passes:

```text
/Users/chadster/lerobot/.venv/bin/python -m pytest -q tests
52 passed in 5.27s
```

The frame-cut implementation is correct on its main path: a person's cut and an accepted suggestion are `exclude` edits, use half-open ranges, gate every target through `dataop.train_mask`, leave the recording/manifest/digests untouched, and can be undone by rejecting the accepted edit. The built-in and external-agent paths also converge on the same proposed `exclude` shape with `reason` and `by`, and both are documented in the README and agent skill.

There are **no high-severity findings**. One medium correctness issue should be fixed before relying on “Cut all”: `idle_spans()` can merge two stationary plateaus across a large joint movement and propose the entire run as idle. The viewer API's revision check for the finder is also not atomic with its writes. Most Round 7 findings are fixed; the legacy-proposal migration and stored-review validation findings are only partly fixed.

## Round 7 audit

| Round 7 item | Status | Evidence |
| --- | --- | --- |
| Product / L1: plain-language text was optional | **fixed** | `review.py:149-164` now requires non-empty text, and `viewer.js:148-159` validates it before saving. |
| Product: editing an interpreted label cleared the reading without warning | **fixed** | `viewer.js:175-189` explains before save that a changed label loses its reading until the agent reads it again. |
| Product / M4: driver and segment looked like training corrections | **fixed** | `viewer.js:182-189` explicitly calls them notes for the agent and says only the interpretation changes training; README describes them as optional notes. Keeping the reader unchanged is now an explicit product choice. |
| Product / L3: no span resize and global segment choices | **fixed** | `viewer.js:35-39` limits choices to current-run steps plus same-task labels; `viewer.js:367-390` resizes an open label while preserving its draft, and save still uses `replace_edit()`. |
| Product / M3: legacy corrections/proposals were invisible and `propose` created unresolvable work | **partly fixed** | The `propose` command and heuristic are removed; accepted cuts/corrections are listed with Undo at `viewer.js:265-269`, and new proposals are cut suggestions the UI can resolve. However, an existing pre-upgrade proposed `failure_start` (or another non-`exclude` proposal) is still counted by `inspect` but is neither rendered nor resolvable. See L1. |
| M1: `decide()` could re-aim an interpreted accepted label | **fixed** | `review.py:202-222` permits acceptance only from `proposed`; accepted labels must go through `replace_edit()`, which gives them a new ID and no interpretation. |
| M2: reinterpretation silently overwrote the prior decision | **fixed** | `review.py:241-261` rejects reinterpretation by default; explicit replacement appends the prior value to `interpretation_history`. CLI exposes `--replace` and `--by` at `cli.py:44-51`. |
| M3: legacy corrections/proposals | **partly fixed** | Same evidence as the Product/M3 row above. The normal current-version flow is fixed, but upgrade state is incomplete. |
| M4: driver/segment semantics | **fixed** | Same evidence as the Product/M4 row above. |
| M5: clickjacking | **fixed** | Every response now carries `Content-Security-Policy: frame-ancestors 'none'` and `X-Frame-Options: DENY` at `viewer.py:125-130`; the existing nonce-based meta CSP remains. |
| M6: stale editor after HTTP 409 | **fixed** | `viewer.js:124-145` installs the returned state, reconciles deleted labels/cuts, and rerenders the current run while retaining the notice/draft. |
| L2: form state and textarea Enter behavior | **fixed** | `viewer.js:397-403` keeps text/new-segment state on input; `viewer.js:414-419` preserves ordinary Enter and uses Cmd/Ctrl+Enter to save. A separate cross-run draft lifecycle issue remains as L2 below. |
| L4: stored review contents were not validated | **partly fixed** | `review.py:274-301` revalidates stored payloads and current interpretations, and `validate.py:186-190`, the reader, and viewer fail closed through `effective()`. It still accepts malformed interpretation history and inconsistent sequence metadata; see L4. |
| L5: README called the default viewer read-only | **fixed** | `README.md:145-155` distinguishes immutable recorded data from mutable labels/cuts and documents `--read-only`. |

## Product

The basic flow is understandable for a first-time user: Edit, drag a span, then either Cut or describe it. Cuts are visibly hatched, their panel says the recording is unchanged, and Undo is available both from the cut and the “Left out of training” card. Suggested cuts are visually distinct, attributed, described, and individually actionable; Cut all / Keep all provide the requested batch flow.

The AI-feature contract is mostly symmetric. The built-in button and `find-idle` use `suggest_cut()`, while a user's agent uses `suggest-cut`; both produce proposed `exclude` edits with the same payload, reason, attribution, lock, viewer, and human-decision path. Both commands are discoverable in `README.md:157-181` and `skills/dataop-logger/SKILL.md:121-135`.

Remaining product rough edges:

- “Keep” permanently rejects that exact proposed span because duplicate suppression includes rejected edits, but the UI does not explain that it means “dismiss and do not suggest again.” Re-running the finder can then appear to do nothing.
- The “Left out of training” card also contains outcome and step-boundary corrections, which do not necessarily leave frames out. A separate “Training corrections” heading would be more accurate.
- Draft behavior is not consistent: a new textual label can survive a run switch, but edits to an existing label do not, and Done can leave a saved draft that unexpectedly reappears later.
- “Cut all” is appropriately powerful, but the idle finder currently over-selects across a movement boundary. The operation is undoable and never deletes recorded data, but its suggestions should not be presented as reliable until M1 is fixed.

## High severity

None.

## Medium severity

### M1. `idle_spans()` takes the union of still windows and can bridge a large movement

**File:** `src/dataop_logger/review.py:368-391`

**Failure scenario:** The docstring says a frame is idle only when every one-second window containing it is still, but the implementation starts with all frames false and ORs in every still window (`idle[i:i+w] = True`). I reproduced 15 frames at joint position 0 followed by 15 frames at position 10, at 10 Hz with a 1.5 threshold. Despite the instantaneous 10-unit movement, `idle_spans()` returned `[(0, 30)]`: quiet windows wholly inside the first plateau mark frames 0–14, quiet windows wholly inside the second mark 15–29, and the boolean merge joins them. Through the command fallback, `find_idle()` then proposed cutting the whole run with the misleading reason “arm still for 3.0 s.” A person using Cut all would exclude the actual motion along with both pauses.

**Suggested fix:** Implement the documented conservative rule: a frame is idle only if all full windows containing it pass, or explicitly detect movement boundaries and prevent quiet intervals on opposite sides from touching/merging. An efficient rolling min/max or per-window failure count can do this without quadratic work. Add table-driven tests for two plateaus separated by a jump, a jump at each edge, multiple jumps less than one second apart, a single spike, exactly one window, and runs shorter than a window. Assert both spans and reasons through `find_idle()` on measured state and command fallback.

### M2. The `find_idle` API revision check is a time-of-check/time-of-use guard, not an atomic compare-and-set

**File:** `src/dataop_logger/viewer.py:179-183`; `src/dataop_logger/review.py:394-426`

**Failure scenario:** The handler loads and compares `rev`, releases all protection, reads the data, computes spans, and then `find_idle()` calls `suggest_cut()`/`add_edit()` once per span without `expect_rev`. If another window changes the review after line 182, the stale request still adds suggestions and returns 200 instead of 409. With several spans, each suggestion is a separate transaction, so another writer can interleave between them and a later failure can leave a partial batch. No JSON update is torn or lost, but the viewer's advertised revision contract is not actually enforced for this operation.

**Suggested fix:** Compute candidate spans without mutating, then open one `transaction(run_path, expect_rev=rev)` and validate/deduplicate/append the entire proposal batch inside it, bumping the revision once. Expose a batch helper shared by the API and CLI. Add a barrier-based concurrency test that changes the review after candidate computation but before commit and requires a 409 with no idle proposals written.

## Low severity

### L1. Pre-upgrade non-cut proposals remain invisible and unresolvable

**File:** `src/dataop_logger/viewer.js:24-34`, `src/dataop_logger/viewer.js:199-211`, `src/dataop_logger/validate.py:361-388`

**Failure scenario:** The new UI treats only proposed `exclude` edits as suggestions. A dataset carrying a pending Round 6/7 `failure_start`, `reject_run`, or step-boundary proposal still has `status="proposed"`; `inspect` counts it in `open_suggestions`, but neither the suggested-cuts card nor corrections card displays it. There is no longer a CLI command to accept or dismiss it. The proposal does not affect training, but it produces a permanent unexplained queue count and prevents a clean migration.

**Suggested fix:** Add a one-time migration/report for unsupported pending kinds. Either reject them with an explicit migration record, convert an eligible proposed exclusion to the new suggestion schema, or render a “Legacy proposals” card with Keep/Accept controls. Make `inspect.open_suggestions` count only actionable cut suggestions and report unsupported legacy proposals separately.

### L2. Cross-run draft lifecycle can lose existing-label edits or resurrect a dismissed new draft

**File:** `src/dataop_logger/viewer.js:14-16`, `src/dataop_logger/viewer.js:214-220`, `src/dataop_logger/viewer.js:350-351`, `src/dataop_logger/viewer.js:428-429`

**Failure scenario:** `select()` saves a draft only when it is a new label with non-empty text (`!sel.id && !sel.cutId`). Switching runs while editing an existing label silently discards those changes. Conversely, after a new-label draft is saved in `drafts`, returning to that run and clicking Done/E does `sel = null` but does not delete `drafts[run_id]`; switching away and back resurrects the draft the user appeared to dismiss. Opening another label can also replace an unsaved selection without warning.

**Suggested fix:** Give every draft a dirty flag and store it by run regardless of whether it is new or replacing a label. Done, Cancel, switching labels, and switching runs should each have explicit preserve/discard semantics; deleting the draft must remove the map entry. Add browser tests for new labels, existing-label edits, cuts, Done, Escape, row switching, and opening another label.

### L3. Overlapping cuts are counted more than once in viewer and inspect totals

**File:** `src/dataop_logger/viewer.js:50-60`; `src/dataop_logger/validate.py:361-388`; `src/dataop_logger/review.py:328-347`

**Failure scenario:** Training correctly applies overlapping excludes as a union because it writes false into the same mask. The displayed `s cut` and `inspect.after_review.cut_frames`, however, sum each edit's length. If a person cuts `[0,10)` and then accepts a suggestion `[5,15)`, only 15 unique frames are excluded but both reports say 20; enough overlaps can report more cut frames than the run contains.

**Suggested fix:** Compute cut counts from the union of active exclude intervals (or directly from `~train_mask` if the intended statistic includes all exclusion mechanisms). Keep a separate raw edit count if useful. Add nested, touching, and partially overlapping cuts plus undo tests.

### L4. Stored-review validation does not cover interpretation history or sequence invariants

**File:** `src/dataop_logger/review.py:274-297`

**Failure scenario:** The new validation correctly catches malformed current payloads and interpretations, but it accepts `interpretation_history` as an arbitrary string and accepts `seq=0` with an accepted edit whose `accepted_seq=1`; I reproduced both with direct `check_review()` calls. A later explicit reinterpretation calls `.append()` on the malformed history and raises an uncaught `AttributeError`, and inconsistent/duplicate accepted sequences can make the “latest accepted edit wins” ordering ambiguous. The current training effect fails closed for the fields it reads, but the mutable audit trail is not fully validated as SPEC claims.

**Suggested fix:** Validate every history entry with the same interpretation schema including `at`; require history to be a list, validate `reason`/`by` types, require positive unique `accepted_seq` values no greater than `review.seq`, and ensure `seq` is at least the maximum ever accepted. Add malformed history and duplicate/out-of-range sequence fixtures through `validate`, the reader, and an attempted `interpret --replace`.

### L5. Clicking to select about one second can create fractional frame indices at non-integer rates

**File:** `src/dataop_logger/viewer.js:353-385`

**Failure scenario:** Pointer coordinates are rounded to integer frame boundaries, but the click-expansion branch subtracts/adds raw `r.fps`. With a valid non-integer action rate such as 29.97 Hz, `lo` or `hi` can become fractional. The panel displays a fractional range, and Save/Cut is then rejected by the backend's integer-frame validation. Dragging more than two frames works, so the failure appears specific to a simple click.

**Suggested fix:** Use an integer click width such as `Math.max(1, Math.round(r.fps))` and clamp integer boundaries. Add browser/unit cases for 29.97 Hz near the beginning, middle, and end of a run.

### L6. Duplicate suppression makes resolved or differently attributed suggestions look newly created

**File:** `src/dataop_logger/review.py:181-199`, `src/dataop_logger/review.py:394-426`; `src/dataop_logger/cli.py:105-119`

**Failure scenario:** `add_edit()` treats any agent edit with the same kind/payload as a duplicate, regardless of status, reason, or `by`, and returns it. After a person chooses Keep, running `find-idle` again returns the rejected edit as though it were a found suggestion; the CLI prints “suggest cut,” but nothing appears in the viewer. If a user's agent suggests the same span with a new rationale after the built-in finder, `suggest-cut` prints success while preserving the old finder attribution/reason. This makes the two AI entry points look asymmetric and makes the button appear ineffective without explaining that the span was previously dismissed.

**Suggested fix:** Return an explicit result such as `created`, `already_pending`, `already_cut`, or `previously_kept`, and make the CLI/UI report it. Define whether a new agent/reason may reopen a dismissed span; if not, preserve the dismissal deliberately and say so. Include provenance in duplicate policy or keep a proposal history so attribution is not silently discarded.

## Important missing tests

1. Idle-window boundaries: stationary plateaus separated by jumps, spikes, multiple movements, exact/short windows, and non-integer rates, through both measured-state and command-fallback paths.
2. Atomic API revision conflict during a multi-span idle-finder batch and failure after the first candidate.
3. Upgrade fixtures with pending legacy non-`exclude` proposals.
4. Browser tests for existing-label drafts, Done/E draft disposal, opening another label, click selection at fractional rates, and overlapping cuts.
5. Re-running find-idle/suggest-cut after Cut and Keep, including a second agent with a different reason/by.
6. Malformed interpretation history and accepted-sequence invariants.
