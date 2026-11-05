request changes.

1. **High — Policy-driven runs can claim no policy ran.**  
   `src/aftershock_logger/logger.py`, `run()` and `_manifest()` (lines 100, 447–453). Omitting `policy` still allows base-controlled frames and writes `policy: null`. The new `task_rev` key makes readers preserve that false meaning. Default omitted policy to `"unknown"`. Reject explicit null when a policy drove. Add regression tests.

2. **High — Unknown counters violate the contract.**  
   `src/aftershock_logger/schema.py`, `tick_values()`; `src/aftershock_logger/reader.py`, `planned_columns()`. Unknown `prompt_rev` and `control_gen` become `-1`, including values exposed by the reader. The contract accepts nonnegative integers or literal `"unknown"`. Preserve that representation in records and reader output. Correct the tests that require `-1`.

3. **High — Watch mode guesses omitted subtask commands.**  
   `src/aftershock_logger/logger.py`, `add_frame()` (lines 185–186); `tests/test_planned_fields.py`, `test_no_planner_is_null_and_kept_apart_from_unknown()`. With `planner=None`, an omitted `subtask_cmd` becomes null. Watch mode requires omitted fields to remain `"unknown"`. Default it to `"unknown"` and preserve null only when explicitly passed. Update the test.

4. **High — Expert imports infer unsupported absence.**  
   `src/aftershock_logger/importer.py`, `_build_run()` (lines 380–417). Any constant expert mapping sets unmapped policy, planner and subtask commands to null. An expert may be a scripted skill, with a planner or shadow policy running. Require an explicit declaration that none ran. Otherwise retain `"unknown"`. Test an all-expert skill import.

5. **High — Changing prompts can contradict a null planner.**  
   `src/aftershock_logger/schema.py`, `TickSequence.problem()` (lines 248–258). With `planner=None`, different prompts at increasing revisions pass recording and validation. The schema says changing prompts means a planner is present. Reject this contradiction before adding the frame. Add writer and validator tests.

6. **Medium — Unknown policy crashes the viewer.**  
   `src/aftershock_logger/importer.py`, `_build_run()`; `src/aftershock_logger/viewer.py`, `collect()` (lines 95–107). Imports now write `policy: "unknown"`. The viewer calls `.get()` on that string and raises `AttributeError`. Handle record, unknown and null separately. Add viewer coverage for an unmapped-policy import.

7. **Medium — Valid empty prompts are rejected.**  
   `src/aftershock_logger/schema.py`, `tick_values()`; `src/aftershock_logger/validate.py`, `_check_planned_ticks()`. Both reject `prompt=""`, although the contract accepts any string. Allow empty prompts and preserve them on round trip. Replace the tests that expect rejection.

8. **Medium — Two acceptance checks are missing.**  
   `tests/test_planned_fields.py`, manifest round-trip and legacy-dataset tests. These use local validators, without the Aftershock strict contract models. The legacy fixture has two episodes, as recorded in `tests/data/dataop_v0_1_0/meta/info.json`. Add contract-model validation of serialized manifests and validation of the required recorded 40-run fixture.

Static review only. No files were edited. No tests or application code were run.