request changes

1. **Medium — Migration still silently skips broken review links.**  
   Files: `src/aftershock_logger/review.py:133`, `file_format`; `src/aftershock_logger/cli.py:285`, `migrate-reviews`; `tests/test_review_format.py`, migration tests.

   If `review.json` is a symlink whose target is missing, `p.exists()` returns false. The command skips it and exits successfully with “0 refused,” including during `--dry-run`. This still mistakes an unreadable review entry for a missing review. The existence check also sits outside the error handler.

   Fix: use `lstat()` within error handling to distinguish a missing entry from a broken link. Convert lookup and read failures to `ReviewError`. Add command tests for a dangling link and lookup errors, checking the refusal count, diagnostic and nonzero exit in both modes.

`tests/data/review_format2/expected.json` changes only by adding the two new cases. No further obsolete-reading acceptance path was found by inspection.

No tests or code were run. `RULES.md`, `docs/WORKFLOW.md`, `docs/CODEX.md` and `contracts/review.md` are absent.