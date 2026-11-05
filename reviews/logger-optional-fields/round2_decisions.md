# Round 2 decisions: logger-optional-fields

Round 2 is the confirmation round of round 1 (`round1_confirm_codex.md`). It found a High in fix 4, so under the
review rules it counts as a full round. Codex confirmed fixes 1, 2, 3, 5, 6 and 7 as correct and complete, and accepted
the rejection of finding 8 for this public repository.

| # | Severity | Finding | Decision | Files |
| --- | --- | --- | --- | --- |
| 4 | High | A mapping with `"policy": null` was checked against the control modes after normalization: a policy frame with no mapped base action had already become `unknown`, and an expert frame with a valid shadow base action passed, so both imported with a false "no policy ran" | Accepted. The check now uses the modes as mapped, before normalization, and also refuses any valid base action. Such an episode is skipped with the reason. Tests cover both cases and an accepted all-expert episode with no policy evidence | `src/aftershock_logger/importer.py`, `tests/test_planned_fields.py` |

Tests: 200 passed.
