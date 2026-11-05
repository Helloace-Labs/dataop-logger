request changes.

1. **Policy default: correct and complete.** [logger.py](src/aftershock_logger/logger.py) defaults to `"unknown"` and rejects base actions or base/blend control with `policy=None` before adding a frame. No nearby regression found.

2. **Unknown counters: correct and complete under the revised decision.** [reader.py](src/aftershock_logger/reader.py) converts `-1` to `"unknown"` in `tick_records`, including legacy defaults. Disk and tensor forms remain unchanged. No nearby regression found.

3. **Omitted subtask command: correct and complete.** [logger.py](src/aftershock_logger/logger.py) always defaults to `"unknown"` and preserves explicit null. No nearby regression found.

4. **High — Null-policy imports still accept contradictory evidence.** Files: [importer.py](src/aftershock_logger/importer.py), [test_planned_fields.py](tests/test_planned_fields.py). The fix is incomplete. The new check only examines control modes after normalization. A base-driven frame with no mapped base action becomes `"unknown"` before this check, so `policy: null` is accepted. An expert-controlled frame with a valid shadow base action also passes. Both falsely claim no policy ran.

   Fix: check the original mapped modes before normalization and reject null policy whenever any base action is valid. Skip the episode with a reason. Add regression coverage for both cases. The explicit declarations, planner mapping, and unknown subtask defaults otherwise look correct.

5. **Changing prompts without a planner: correct and complete.** [schema.py](src/aftershock_logger/schema.py) retains the last known prompt across unknown ticks. Recording rejects changes and validation reports them. No nearby regression found.

6. **Viewer policy handling: correct and complete.** [viewer.py](src/aftershock_logger/viewer.py) safely handles null and `"unknown"` while preserving record access. No nearby regression found.

7. **Empty prompts: correct and complete.** [schema.py](src/aftershock_logger/schema.py) accepts any string. [validate.py](src/aftershock_logger/validate.py) accepts empty strings and reports null prompts. No nearby regression found.

8. **Acceptance checks: rejection accepted for this public repository.** [round1_decisions.md](reviews/logger-optional-fields/round1_decisions.md) records the private checks and their results. Those results cannot be independently confirmed here. No code change or nearby regression applies.

Static review only. No files were edited. No tests or application code were run.