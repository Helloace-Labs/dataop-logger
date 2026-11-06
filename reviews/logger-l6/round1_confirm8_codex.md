approve

No High, Medium or Low findings in `bd6bf85..4e4ccb3`.

- `src/aftershock_logger/hook_sandbox.py`, `_SECCOMP_ARCH` and `seccomp_program`: the added syscall numbers match the [x86_64](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/arch/x86/entry/syscalls/syscall_64.tbl?h=v6.12) and [aarch64](https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git/tree/include/uapi/asm-generic/unistd.h?h=v6.12) definitions. Namespace creation through clone flags remains blocked unconditionally. Architecture and x32 guards remain intact. No remaining mount or namespace escape route was identified within scope.
- `src/aftershock_logger/_hook_runner.py`, `_probe`, and `hook_sandbox.py`, `_judge`: Linux probes both unshare and mount. Either success or a missing denial refuses hooks.
- `tests/test_hook_sandbox.py`: the hostile hook requires an actual unshare refusal. The weakened-filter case checks probe rejection. The storage test now rejects startup-probe refusal as evidence of hook-level protection.

No adjacent file exposure or write-surface regression was found by inspection.

Read `AGENTS.md`, `SPEC.md` and the prior review decisions. `RULES.md` and the named files under `docs/` were absent. No tests or code were run. No files were edited.