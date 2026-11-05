# Round 3 decisions: logger-optional-fields

Codex verdict: request changes, 2 High.

| # | Severity | Finding | Decision | Files |
| --- | --- | --- | --- | --- |
| 1 | High | Auto-detection picks the `demos` preset whenever no column says a policy drove, and the preset declared a null policy and planner, so an automatic import could claim none ran without evidence | Accepted. No preset declares a null policy or planner any more; the `demos` preset keeps only its control mode and outcome. A null needs an explicit `"policy": null` or `"planner": null` in the mapping. Test: an auto import of a plain recording keeps both `unknown`, and an explicit declaration writes null | `src/aftershock_logger/importer.py`, `SPEC.md`, `tests/test_planned_fields.py` |
| 2 | High | `validate` accepted a null policy beside ticks with a base action or policy control | Accepted. `validate` reads the policy through `manifest_fields` (so a legacy null still reads as unknown) and reports a null policy when any tick carries a `base_action` or is `base` or `blend`. Tests cover a policy frame, an expert frame with a shadow base action, and the same manifest without `task_rev` (legacy, accepted) | `src/aftershock_logger/validate.py`, `SPEC.md`, `tests/test_planned_fields.py` |

Tests: 202 passed. The 40-run recorded fixture still validates (its null policies have no `task_rev` key, so they read as unknown).
