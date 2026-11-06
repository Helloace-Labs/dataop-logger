# Logger L6, round 1 confirmation decisions

The confirmation round found new High findings, so it counts as full round 2. Fixes 2, 5 and 6 of round 1 were confirmed.

| # | Severity | Finding | Decision | Change |
| --- | --- | --- | --- | --- |
| 1 | High | Linux still mounts whole library folders and masks with a cached list | Changed | `hook_sandbox.py` `_mirror`, `_shared_libs`, `_bwrap_cmd`: each runtime folder is mounted from a private read-only copy holding only its code files (hard links or copies, built once per process, so files added later are simply absent). Shared libraries are mounted one file each, from `ldd` on the interpreter and the runtime's extension modules; no `/lib` or `/usr/lib` folder is mounted. On macOS the profile no longer grants `/usr/lib` or `/System` (the libraries come from the dyld shared cache); only `/` itself and three devices. Test: `test_bwrap_command_mounts_only_code_and_the_result_files`. |
| 2 | High | The writable tmpfs used for hiding is unbounded | Changed | No tmpfs over runtime folders any more; hidden folders are simply not in the copy. The test asserts no tmpfs besides the private `/tmp` and that the only writable binds are the three result files. |
| 3 | High | Linux unix socket probe proves nothing | Changed | The probe now creates unix, inet and socketpair sockets directly, and on Linux any one created refuses hooks (`_judge`). Test: `test_linux_probe_requires_that_no_socket_can_be_created`. |
| 4 | Medium | Rule and mount order hide a numpy inside site-packages | Changed | The profile grants the standard library, then denies its site-packages, then grants numpy (the last match wins). The Linux copy leaves an empty mount point for a nested numpy and mounts outer folders first. Both layouts are tested for both modes. |
| 5 | Medium | `/dev/null` masks read as allowed in the probe | Changed | Masks are gone; a non-code file is absent from the copy, so opening it fails. |
| 6 | Medium | Symlink and fifo test expectations cannot hold on Linux | Changed | Those hook tests only expect the parent message on macOS; on Linux they only require not_run. New parent-level tests prepare symlink, fifo, directory and hard link result files and extra output files directly (`test_parent_reads_only_plain_result_files`, `test_parent_refuses_extra_output_files`). |
