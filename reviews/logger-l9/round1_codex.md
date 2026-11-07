request changes

1. **High — Failed-run coverage does not prove frames survived.**  
   `src/aftershock_logger/session.py:411` (`write_pending`) and `:629` (`session_problems`); `tests/test_session.py:222`. A save failure moves only the staged sidecar into quarantine. It does not preserve observations and cameras if saving failed before writing them. Validation still counts the entire range as covered. It also accepts a failed quarantine whose ticks were deleted. Preserve full frames in recoverable storage and verify their counts and checksums. Extend the failure test to read every retained frame.

2. **High — Closing silently discards the active gap.**  
   `src/aftershock_logger/session.py:202` (`__exit__`, `close`). Closing calls the gap logger’s `close`, which discards its active run. Gap frames, including a short stop awaiting finalization, disappear. The session record and evidence are not flushed. Validation can report success with only an open-session warning. Save or quarantine the active gap before closing. Flush its frame count and evidence, and report any preservation failure.

3. **High — Gap finalization failures are hidden.**  
   `src/aftershock_logger/session.py:206` (`close`) and `:517` (`finalize`). Every exception from `gap_dataset.finalize()` is suppressed after the final session record has been written. A metadata or data flush can fail while the caller receives a successful finalized session. Complete storage finalization before publishing success. Record `persistence_error`, propagate the failure, and retain enough state to retry.

4. **High — Short-stop replay changes recorded ticks.**  
   `src/aftershock_logger/session.py:237` (`add_frame`) and `:374` (`seal_episode`). Replay uses the original call arguments. Rewards subsequently added through `run.complete_transition()` disappear. Caller-supplied terminal flags survive replay and can prevent further gap frames or gap sealing. The discarded run’s events also disappear. Move the completed tick records and events, remap their indices, and clear terminal flags for gap storage. Add tests for delayed rewards and a truncated stopping tick.

5. **High — Validation accepts records the cloud rejects.**  
   `src/aftershock_logger/session.py:80` (`_check_go`) and `:580` (`session_problems`). A string `go.scope` and an invalid timestamp are accepted and written, although the cloud session models require a scope record and UTC timestamps ending in `Z`. Validation also omits nested field and type checks. Implement the supplied contracts at construction and validation boundaries. Add negative cases for malformed timestamps, scopes, run entries, and reset checks.

6. **High — Gap annotations can disagree silently.**  
   `src/aftershock_logger/session.py:639` (`session_problems`). The session-to-gap comparison omits `short_stop`, `moved_by`, and `reset_checks`. Changing a valid `short_stop` range in `session.json` alone still passes validation, allowing recorded reset frames to be reported as a stopped episode. Compare all gap metadata with its sidecar. Add mutation tests for each omitted field.

7. **Medium — An index failure mislabels a committed run.**  
   `src/aftershock_logger/session.py:423` (`write_pending`); `src/aftershock_logger/logger.py:810` (`_save_and_commit`). If index rebuilding raises after the manifest commits, the session marks the run failed. The committed run remains outside quarantine, and the writer remains usable. The session then contradicts its manifest. Make write status follow the durable commit boundary and report a derived-index failure separately. Test failure immediately after manifest commit.

8. **Medium — The required cloud round trip is optional.**  
   `tests/test_session.py:416` (`test_cloud_models_load_the_records`); `.github/workflows/linux-sandbox.yml:52`. The test skips without externally supplied models. CI never supplies them, so acceptance test 2 is not enforced. Add a required integration job with a pinned package of the cloud session models. Fail that job when the models are unavailable.

Read-only review. No tests or code were run.