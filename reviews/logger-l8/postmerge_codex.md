# Codex post-merge confirmation, pull request #6

Reviewed squash commit `4f4395e`, judged against main. Posted on the pull request as "Review by Codex (post-merge)".

request changes

1. **High — An obsolete reading can be accepted without affecting training.**  
   Files on main: `src/aftershock_logger/review.py`, `decide` and `replace_edit`; `src/aftershock_logger/review_format.py`, `accept`, `decide_many` and `reading_in_effect`; `src/aftershock_logger/viewer.js`, `proposalText` and `aiBar`.

   Create a label, propose an `exclude` reading, then replace the label. The old reading remains offered for acceptance. Accepting it succeeds, but it affects no frames because its target label is rejected. The proposal shows neither the target label nor its span, so this can look like a successful exclusion while the corrected span remains eligible for training.

   Fix: check that a reading still targets an accepted label before accepting it, including bulk decisions. Refuse obsolete targets and show the target text and span. Add regression coverage for replacement and deletion before acceptance.

2. **Medium — Migration silently skips unreadable review files.**  
   Files on main: `src/aftershock_logger/cli.py:284`, `migrate-reviews`; `src/aftershock_logger/review.py:128`, `file_format`.

   `file_format` returns `None` for malformed JSON, unreadable files and unknown formats. The command skips these before entering its error handler. A truncated review therefore produces exit status 0 and “0 refused,” including during `--dry-run`.

   Fix: distinguish a missing review from an invalid existing file. Count unreadable or unrecognized files as refused, print the reason and exit nonzero. Add command tests for malformed JSON and unknown formats.

No tests or scripts were run. `RULES.md`, `docs/WORKFLOW.md`, `docs/CODEX.md` and `contracts/review.md` were absent from this checkout.