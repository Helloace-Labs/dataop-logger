# Logger L6, round 1 seventh confirmation decisions

This round confirmed the Linux workflow, the PWD fix, the remount order, the signal report and the README note. It found one new High, so it counts as a full round under the cap rule; the change stays open until a confirmation round approves the fix.

| # | Severity | Finding | Decision | Change |
| --- | --- | --- | --- | --- |
| 1 | High | A hook could unshare a new user namespace, gain namespace-local capabilities and mount a fresh tmpfs, which no limit bounds | Changed | `hook_sandbox.py` `_SECCOMP_ARCH`: the filter also refuses `unshare`, `setns`, `mount`, `umount2`, `pivot_root`, `chroot` and the new mount API (`open_tree` to `mount_setattr`) on x86_64 and aarch64 (`clone` and `clone3` were already refused). `_hook_runner.py` probe tries `unshare(CLONE_NEWUSER\|CLONE_NEWNS)` and a tmpfs mount; on Linux `_judge` refuses hooks unless both are denied. Tests: `test_hook_cannot_make_a_namespace_or_mount_storage` (a hostile hook, run on Linux CI), the weakened case `no-namespace-filter` in `test_the_probe_catches_a_weakened_bubblewrap` (on the CI runner, removing these calls from the filter lets unshare work, and the real probe refuses hooks), the filter table test, and the judge test. |
| 2 | Note | The storage test accepted a refusal by the probe | Changed | It now requires that the hook itself ran and its write failed (`Error` in the reason, no `self test`). |
