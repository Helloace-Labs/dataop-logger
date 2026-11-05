## Review by Claude (fallback, Codex at limit), confirmation round

Reviewed SHA: `41766bb5e380f8bd0ff0333fb8957a346dc6be67` (read only; no tests or code were run). Scope: `git diff 722e94e..41766bb` and the code it touches.

**Verdict: request changes.** One new Medium (introduced by the L1 fix) and two Lows. Every round 1 finding is resolved.

**Resolved.**
- **H1/M1.** Named segments are now stored as `unresolved` with the trimmed name in `legacy_name`; `subtask_id_for`, the id lookup in `lists.py` and `subtaskId` in `viewer.js` are gone, so there are no derived ids, no collisions and no Python/JS split. `effective()` names by `legacy_name`, so recorded step status and reason carry through; the new test covers mixed case, multiple words, a non-Latin name and a recorded `failed` step. In a migrated review `_segment_names` keeps a mapped id, adds new names as `unresolved` entries (exact name lookup, matching the reader's agreement check), and copies the review before changing the map, so a deduplicated proposal leaves the map untouched. `replace_edit` and correcting a proposal go through the same path.
- **M2.** `retract` now only flips `status` to `rejected`, keeping `accepted_seq`, `decided_at` and `decided_by`; the take back is its own history step whose `at` and `what` record when and who.
- **M3.** Batch lock digests are compared with the format 1 logger read from history (`a051fa0` is the right commit), on fixtures and on real files, and again after the rewrite. `propose_many`, `decide_many` and `retract` results (including a label with a reading) load in the reference reader with identical bytes, and undo across take backs, accept, replace and sign off continue identically on both sides.
- **M4.** `_str` refuses lone surrogates only in fields with a limit, and the new test sets every string field of three reviews to three surrogate values and requires the reference's answer.
- **L1/L2/L3.** Agent proposals carry a real version and accept `--job-id`/`--agent-version`; locked batches are skipped in `migrate-reviews`; the proposals list shows list names.

### Medium

**N1. The rerun dedup ignores `job_id` and `agent_version`, unlike the shared `propose`.** `_open_from` (used by `apply_agent_edit` and `add_event_note`) returns any open proposal of the same value from the same source and `by`, "whatever the job id". The shared `propose` (and `fmt.propose`, via `_same_open`) only dedups when source, `by`, `job_id` and `agent_version` all match, so that a new job or a new agent version proposing the same value gets its own edit and can be scored. With this change a second job, or a newer model passing `--agent-version m-2`, gets back the earlier job's edit and its own version is silently dropped; `test_agent_values_name_a_version_and_a_rerun_adds_nothing` asserts exactly that with `job_id="another-job"`.
*Fix:* drop `_open_from` and let `fmt.propose` dedup as the shared operation does. Since the commands now take `--job-id`, a rerun that wants no duplicates passes the same job id (and version). Change the test to expect the same edit for the same job and version, and a new edit for a new job or version.

### Low

- **N2.** In `migrate-reviews`, any `BatchError` from `batches.check_editable` is reported as "skipped, its batch is locked" and the command exits 0. `batch_of` also raises `BatchError` for an unreadable or malformed batch file, which would then be hidden as a skip. Count only the locked case as skipped (for example check `batch_of(...)["status"] == "locked"` directly) and let other batch errors count as refused.
- **N3.** (Follows the M2 choice; already on the list for the shared format question.) After a take back, the edit is `rejected` with `decided_by`/`decided_at` of the person who accepted it. Once the history cap drops the take back step, the file reads as that person having rejected it. Fine as an interim state, but worth stating in the format question so consumers that score proposals know how to read a rejected edit that has an `accepted_seq`.

