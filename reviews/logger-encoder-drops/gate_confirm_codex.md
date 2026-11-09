request changes

1. **High — Finding 1 is incomplete.** The normal staging failure now reaches quarantine. But `src/aftershock_logger/logger.py:809` deletes the partial staging directory before the protected write. If staging leaves files and cleanup raises, such as after a filesystem becomes read-only, the run remains `quarantined` without entering `_unsaved_runs`. Closing the logger then skips retry and spill. The recorded prefix can still be lost when the process exits. Protect cleanup failures with the same recovery path. The new test raises before creating staging files, so it misses this case.

2. **High — Finding 2 is resolved.** Expected cameras now come from dataset features. Missing camera metadata produces an unknown count unless batch deferral is explicitly established. Save refuses the commit, validation reports the problem, and repair cannot commit it. The two-camera tests cover removed and null metadata. No adjacent regression found.

3. **High — Finding 3 is resolved.** `src/aftershock_logger/session.py:448` checks the stored stop before the short-stop branch. Other termination reasons leave the episode open. Sealing as `system_error` preserves the run and backlog label. The short-episode test covers this sequence. No adjacent regression found.

Read-only review. No tests or code were run.