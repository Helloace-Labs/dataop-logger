request changes

**High — Hooks can recreate writable storage through nested namespaces.**

Files and sections: `src/aftershock_logger/hook_sandbox.py`, `_bwrap_cmd` and `seccomp_program`; `src/aftershock_logger/_hook_runner.py`, `_probe`; `tests/test_hook_sandbox.py`, storage and weakened-sandbox tests.

The final remounts block ordinary file creation, but the filter permits `unshare` and mounting syscalls. Where nested user namespaces are allowed, a hook can call `unshare(CLONE_NEWUSER | CLONE_NEWNS)`, establish its ID mapping through the writable proc filesystem, and mount fresh tmpfs over an existing directory. Creating that namespace grants namespace-local capabilities despite the earlier capability drop. This follows the documented [Linux user namespace behavior](https://man7.org/linux/man-pages/man7/user_namespaces.7.html).

The hook can then fill many individually capped files. Their combined storage bypasses `RLIMIT_AS` and the per-file size limit. The new probe never attempts this, so it can approve this sandbox.

Concrete fix: deny namespace creation and mounting syscalls through seccomp on both supported architectures. Add a real Linux probe and bounded hostile-hook test for this route. Verify that weakening those controls makes the probe refuse hooks.

| Item | Assessment |
| --- | --- |
| 1. Linux workflow | Correct for the stated CI boundary. Read-only permissions, disabled persisted credentials, pinned actions, and no secret references introduce no identified privileged fork access. The required switch prevents unavailable-sandbox skips from producing a passing suite. |
| 2. Remove `PWD` | Correct and complete for the reported environment failure. It runs before the probe and hook. No adjacent regression found. |
| 3. Storage mounts | Correct mount order, incomplete protection as described above. Result file binds remain writable. The shm directory inherits the device mount’s read-only state; no mqueue filesystem is mounted. Proc and devpts remain separate writable mounts, though neither permits arbitrary regular-file creation. |
| 4. Signal reporting | Correct for the specified Linux exit-code range. No adjacent security regression found. |
| 5. Tests | The weakened-command test detects its listed mutations. The runtime test requires both loader spellings and successful hook execution. The required-sandbox test fails when proof is unavailable. The storage test covers direct writes, but misses namespace-created storage. Its `not_run` assertion also accepts refusal by the startup probe; it does not independently prove hook-level denial. |
| 6. README note | Correct for the reported AppArmor restriction. Export still fails closed without working isolation. No adjacent documentation regression found. |

Reviewed only `8ce8eff..bd6bf85` and supporting context. `RULES.md` and the named workflow, agent-process, plan, and queue docs were absent. No tests or code were run. No files were edited.