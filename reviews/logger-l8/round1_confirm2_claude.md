## Review by Claude (fallback, Codex at limit), confirmation round 2

Reviewed SHA: `d6b6eead8b2851f7e03f3012bbb660012d30fe58` (read only; no tests or code were run). Scope: `git diff 41766bb..d6b6eea` and the code it touches.

**Verdict: approve.** N1 and N2 are resolved. No new findings. N3 is tracked separately and not re-reviewed here.

**Resolved.**
- **N1.** `_open_from` is removed, and `apply_agent_edit` and `add_event_note` no longer return an open proposal before proposing. Repeats now go only through `_propose` to `fmt.propose`, whose `_same_open` matches source, `by`, `job_id`, `agent_version`, kind and payload. A new job or a new agent version therefore gets its own edit. A rerun with the same job id and version returns the open edit, and `_propose` leaves the review untouched in that case (including any new segment-name entries), so nothing is written. `annotate` still uses one job id per call and passes the caller's `--job-id` and `--agent-version`, so a rerun that wants no repeats can pass the same values. The `--job-id` help text says so. The test now checks all three cases: the same job adds nothing, and a new version and a new job each add their own edit.
- **N2.** `migrate-reviews` now calls `batches.batch_of` directly. It counts a run as skipped only when the batch status is `locked`, which is the same condition `check_editable` uses. A `BatchError` from an unreadable or malformed registry becomes a `ReviewError`, so the run counts as refused and the command exits 1. The test covers both the locked skip (exit 0, file unchanged) and the broken registry (exit 1, reason printed, file unchanged).

Nit (not counted): removing `_open_from` leaves three blank lines before `add_event_note` in `review.py`.
