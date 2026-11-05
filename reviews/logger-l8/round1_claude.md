## Review by Claude (fallback, Codex at limit)

Reviewed SHA: `722e94e66c884a59a50500e1ff0472cbd5d85336` (read only; no tests or code were run).

**Verdict: request changes.** There is one High and four Medium findings.

**What holds up.** The new `review_format.py` follows the reference implementation of the shared review format closely, operation by operation. I compared the strict reader, the canonical writer, `propose`/`accept`/`reject`/`replace`/`undo`/`set_reviewed`/`lock`/`unlock`, the history cap, `migrate_format1`/`to_format1`/`normalize_format1` and the manifest and task checks, and found no difference in what they write. The migration refuses rather than drops anything format 1 never wrote. Without a task, every name stays `unresolved`: `transaction`, `load` and `migrate-reviews` all migrate with no task, so nothing is mapped by guess. `effective()` of a migrated review keeps the format 1 shape (label `interpretation` with `at`, the `segment_plan` key in `set_by`, `failure_type` and `name` only when set), so existing batch lock digests should still match. A no-op leaves a format 1 file on disk untouched.

### High

**H1. A segment drawn in format 2 shows up in `effective()` under its derived id, not its name, and loses the recorded step's status.** `effective()` names a segment by `legacy_name` when it has one and otherwise by `subtask_id`. A segment the viewer or `annotate` writes has `legacy_name: null` and `subtask_id = subtask_id_for(name)`, so "Pick tube" becomes `pick-tube`. `match()` then compares that id with the recorded step name ("Pick tube"), finds nothing, and every drawn segment gets `status: "completed"` and `reason: null`. A recorded `failed` step therefore no longer carries through when a person redraws the segments. That changes what `events.major_events` estimates as the failure moment, what `runs` returns to agents, the event keys (`segment:<name>@<frame>`), and the segment status in export snapshots. `lists._seed` (on a dataset with no lists file yet) would also add `pick-tube` as a list entry next to "Pick tube". The tests don't catch this because the fixture names (`reach`, `grasp`, `lift`) are already slugs.
*Fix:* in `effective()`, turn the id back into the display name before naming and matching. Either compare against `subtask_id_for(recorded name)`, or resolve through the lists, as the viewer does. Add a test with a mixed-case, multi-word segment name over a recorded `failed` step that checks the name, status and reason.

### Medium

**M1. Derived subtask ids can collide, and the viewer derives them differently from Python.**
- "Pick tube", "Pick  tube", "pick-tube" and "Pick.tube " all map to one id, and the 64-character cut adds more collisions. `lists._find` then returns whichever entry comes first.
- For a name with no ASCII letters or digits (any non-Latin script), Python gives `segment-<hash>`, but `subtaskId` in `viewer.js` gives `""`. The viewer then can't find the list entry and shows the raw hash.
- Saving a segment plan from the viewer sends every segment by name. One edit to one segment therefore gives an id to every migrated `unresolved` segment from its name, even though the person never mapped those segments.

*Fix:* refuse to add or rename a list entry whose id collides with another entry's id. Have the server send each segment's id and display name so the browser doesn't re-derive them, or mirror the hash fallback in JS. Keep a migrated segment's `unresolved` id and `legacy_name` unless the person changed that segment.

**M2. `retract` is a valid file but loses who accepted the edit.** The result passes the strict reader: a rejected edit may keep `accepted_seq`, as format 1 did. But `retract` overwrites `decided_at`/`decided_by` from the acceptance with the person who took it back, so after the history cap drops the step, nothing shows who accepted it or when. The shared operations can't produce this state; they take an accepted edit back only through `undo`, which returns it to its earlier state. A consumer that scores agent proposals will see an accepted-then-deleted proposal as a plain rejection.
*Fix:* either get `retract` added to the shared format with defined fields (for example, keep the acceptance and record the retraction separately), or implement delete in the viewer as `undo` of the step that accepted the edit when that step is still undoable. Until then, check retracted files with the reference reader in the live tests (see M3).

**M3. Test gaps on the claims that matter most:**
- `test_real_review_files_migrate_losslessly_on_copies` calls `rv.effective(...)` but asserts nothing about it. Nothing on real data shows that recorded batch lock digests still match. Assert that `batches` validation reports no "changed after batch was locked" errors on the copied datasets, or compare `review_digest(m, migrated)` with each lock's recorded `review_digest`.
- `propose_many`, `decide_many` and `retract` results are never loaded by the reference implementation. In the live reference tests, load a review after each of them with `impl.load` and `impl.check_manifest` and compare the dumped bytes.
- The random sequences never call the logger-only operations, so they can't show that the reference operations still apply cleanly afterwards (for example `undo` across a `retract`).
- Add the H1 case.

**M4. Possible reader difference on strings with lone surrogates.** The plain-Python checks accept a lone surrogate (for example a note text `"\ud800"`, which a browser's `JSON.stringify` can send). The reference reader validates constrained strings through pydantic-core, which I expect refuses them as invalid unicode. The logger could then write a file the other side refuses as a whole. Please confirm, and add `"\ud800"` to the values in `_mutations`. If they differ, refuse lone surrogates in `_str`, or reject them in `_input`.

### Low

- **L1.** New agent proposals from `annotate`, `interpret`, `note` and `suggest-cut` record `agent_version: "unknown"`, which the format reserves for edits migrated from format 1; only find-idle sets a real version. They also get a fresh `job_id` per call, so running the same `annotate` twice adds duplicate open proposals instead of returning the existing ones. Add `--job-id` and `--agent-version` to these commands, and default the version to the logger's own when the caller gives none.
- **L2.** `migrate-reviews` counts runs in a locked batch as "refused" and exits 1, so a dataset with any locked batch never exits cleanly. Report them as skipped (locked) and keep exit 1 for real migration failures.
- **L3.** The "Proposed by agents" list shows a segments proposal by raw `subtask_id`. Use `canon("task_segments", id)` so it shows the list name.

