request changes

The six fixes assess as follows:

| Fix | Assessment |
| --- | --- |
| 1. Runtime reads | Incomplete. Broad library mounts remain. Rule and mount order can also hide NumPy. |
| 2. Process creation | Correct by inspection. BPF jump offsets reach the intended returns. Architecture checks and process-creation blocks are present. |
| 3. Output files | The read-only directory followed by writable file binds is correct. Nearby writable mounts and test expectations still need fixes. |
| 4. Sockets | The syscall blocks are correct. The Linux probe does not prove they work. |
| 5. Result schema | Correct and complete for the reported cases. No adjacent regression found. |
| 6. Nested JSON | Correct. `RecursionError` produces `not_run`. No adjacent regression found. |

1. **High — Linux still exposes unapproved runtime files.**  
   Files and sections: `src/aftershock_logger/hook_sandbox.py`, `_non_code_files` and `_bwrap_cmd`, lines 216–274.  
   Entire system library trees remain readable. Masking only covers `_runtime()["dirs"]`, so non-code files elsewhere in those trees remain exposed. Cached masks also miss files added later. Build a read-only runtime from approved code files and required shared libraries. Avoid broad host directory binds and cached exclusion lists.

2. **High — Runtime hiding creates another unbounded write surface.**  
   File and section: `src/aftershock_logger/hook_sandbox.py`, `_bwrap_cmd`, lines 272–274.  
   The new hiding mount is a writable tmpfs. A hook can create many capped files there. Their combined storage is not bounded by `RLIMIT_FSIZE` or the process address-space limit. Use an empty read-only mount for hiding. Verify that the hook cannot create files there.

3. **High — The Linux Unix-socket probe can pass without socket blocking.**  
   Files and sections: `src/aftershock_logger/hook_sandbox.py`, `_probe`; `src/aftershock_logger/_hook_runner.py`, `_probe`; `tests/test_hook_sandbox.py`, `test_hook_cannot_reach_a_unix_socket`.  
   The listener is outside the child’s mounted filesystem. Connection failure therefore does not prove seccomp blocked sockets. Removing the socket blocks can still leave this probe and connectivity test passing. Probe `socket` and `socketpair` creation directly and require denial. Add a weakened-filter regression case.

4. **Medium — Rule and mount order hide permitted NumPy installs.**  
   File and sections: `src/aftershock_logger/hook_sandbox.py`, `_seatbelt_profile`, lines 187–198; `_bwrap_cmd`, lines 268–274.  
   When NumPy lives beneath the standard library’s `site-packages`, the later Seatbelt deny overrides its read grant. The later bubblewrap tmpfs also hides its earlier bind. The runner then cannot import NumPy. Hide other packages first, then grant or mount NumPy explicitly. Cover both nested and separate installation layouts.

5. **Medium — The Linux runtime probe rejects successful masking.**  
   Files and sections: `src/aftershock_logger/hook_sandbox.py`, `_bwrap_cmd`; `src/aftershock_logger/_hook_runner.py`, `_probe`, lines 97–106.  
   A file masked with `/dev/null` still opens successfully and returns empty bytes. The probe labels that as allowed, so a populated runtime sample causes all hooks to be refused. Make the probe distinguish hidden contents from exposed contents, or use isolation that denies opening the file. Keep a positive startup check.

6. **Medium — Result-file hardening breaks two Linux test expectations.**  
   File and section: `tests/test_hook_sandbox.py`, `test_a_failing_or_wrong_hook_is_not_run_with_the_reason`, lines 237–240.  
   The symlink and FIFO cases now unlink precreated bind-mounted files. Linux rejects that operation before parent result validation. The expected parent-validation messages cannot occur. Assert sandbox denial on Linux, and test parent file-type rejection separately with prepared result fixtures.

This was a read-only review. No tests or code were run.