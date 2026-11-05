# Round 1 decisions: logger-l5fix (pull request #8)

Codex round 1 reviewed 9d21997 and requested changes with one Medium finding. It is accepted and fixed.

| Finding | Decision | Change | Files |
| --- | --- | --- | --- |
| Medium 1: the existing stalled-writer test can close the logger while its writer is still inside `add_frame`, so the new `close()` may raise depending on scheduling | Accepted | After opening the gate the test joins the queue's writer thread (5 s bound) and asserts it exited before leaving the `RunLogger` block; the gate is also released in `finally`. The test passes `stop_timeout=0.2` so it does not wait the default 10 s. It passed 5 runs out of 5 | `tests/test_frame_queue.py` (`test_memory_stays_bounded_under_a_stalled_writer`) |

The race could not be made to fail on demand on this machine (the old test passed 3 runs out of 3), so this is a hardening of the test, not a fix with a failing test first.

## Confirmation round

Codex confirmed the fix at a9b0a01 (`round1_confirm_codex.md`): approve, no new findings. The change is done under the review rules: every round 1 finding has a decision and its fix passed a confirmation round with no new High or Medium.
