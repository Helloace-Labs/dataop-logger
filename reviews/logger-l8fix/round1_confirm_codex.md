approve

No High, Medium or Low findings. The fix in `6a3603b` is correct and complete for Medium 1.

In `src/aftershock_logger/review.py`, missing entries retain their existing behavior. Dangling links and lookup or read failures raise `ReviewError`. The new tests in `tests/test_review_format.py` check refusal counts, diagnostics and exit status in both migration modes.

No adjacent regression was found by inspection. `inspect` reports the error through the CLI handler. The viewer shows the error and disables editing for that run.

No tests or code were run. `RULES.md` is absent from the checkout.