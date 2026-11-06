# Logger L6, round 1 sixth confirmation decisions, and the Linux CI run

The sixth confirmation round approved with no findings.

The Linux bubblewrap mode then ran for real for the first time, in a new GitHub workflow (`.github/workflows/linux-sandbox.yml`, ubuntu-latest, bubblewrap, `pull_request`, `contents: read`, no secrets, sandbox required so no sandboxed test can skip). It found these problems, which macOS could not show. Each is fixed in this change.

| # | Found by | Problem | Change |
| --- | --- | --- | --- |
| 1 | CI | Ubuntu 24.04 AppArmor blocks unprivileged user namespaces, so bubblewrap failed (`loopback: Failed RTM_NEWADDR`) and export refused every hook | The workflow loads an AppArmor profile that allows user namespaces for `/usr/bin/bwrap` only. The README tells hosts to do the same. Export still fails closed without it. |
| 2 | CI | Bubblewrap sets `PWD` after `--clearenv`, so the environment was not empty | `_hook_runner.py` drops `PWD` before anything else runs. |
| 3 | CI | The private `/tmp` was a writable tmpfs, and bubblewrap's root and `/dev` are writable tmpfs too. Nothing bounds storage there (not RLIMIT_AS, not RLIMIT_FSIZE across files) | No `/tmp` tmpfs. `_bwrap_cmd` ends with `--remount-ro /dev --remount-ro /`, so only the three result file binds are writable. The probe tries to store files beside the job and in `/`, `/tmp`, `/dev` and `/dev/shm`. Test: `test_hook_cannot_store_anything_in_the_sandbox_itself`. |
| 4 | CI | A hook killed by a signal showed only "exit code 139" (bubblewrap exits with 128 plus the signal) | The reason names the signal in Linux mode. |
| 5 | Lead | No Linux test proved the real probe catches a weakened sandbox | `test_the_probe_catches_a_weakened_bubblewrap` removes the syscall filter, the read-only remounts or the namespaces, exposes the host root, or plants a variable, and the real probe must refuse hooks. |
| 6 | Lead | The `/lib64` symlink and `..` library cases had only command tests | `test_real_linux_runtime_has_the_loader_spellings_and_hooks_run_through_them` requires that the runner's real runtime has both spellings and runs a hook that uses numpy's bundled BLAS. Every sandboxed hook test on the runner goes through them. |
