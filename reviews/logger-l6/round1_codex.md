request changes

1. **High — Runtime mounts expose files outside the inputs.**  
   `src/aftershock_logger/hook_sandbox.py`, `_readable`, `_seatbelt_profile`, and `_bwrap_cmd` (lines 148–193).  
   Whole installation prefixes are readable. A hook can read any secret or dataset placed beneath those prefixes and return its contents. The probe only checks a canary elsewhere. Expose a minimal runtime containing approved files. Add a planted-secret test inside a runtime prefix.

2. **High — Linux permits process creation and aggregate memory exhaustion.**  
   `src/aftershock_logger/hook_sandbox.py`, `_bwrap_cmd` and `_spawn` (lines 185–197, 293–296).  
   Namespaces hide host processes but do not block `fork`, `clone`, or execution. Each descendant gets a separate memory allowance. A hook can create many processes and exhaust host resources before the timeout. Block process creation with a syscall policy and enforce aggregate resource limits. Refuse hooks when those controls are unavailable. Probe process creation explicitly.

3. **High — Linux output checks allow unbounded disk use.**  
   `src/aftershock_logger/hook_sandbox.py`, output bind and watcher (lines 194–196, 341–344).  
   The entire output directory is writable. A hook can create a directory named `keep.npy` and fill it with files. The watcher accepts that name without inspecting its type or contents. Per-file limits do not bound total storage, and cleanup runs in the parent. Bind only precreated result files as writable. Reject directories and undeclared files after exit. Add a bounded regression test for this case.

4. **High — Linux can reach host Unix sockets.**  
   `src/aftershock_logger/hook_sandbox.py`, `_bwrap_cmd`; `src/aftershock_logger/_hook_runner.py`, network probe (lines 85–91).  
   Network namespaces do not isolate pathname Unix sockets exposed through mounted directories. A hook can connect to a host listener beneath a readable runtime prefix. The TCP-only probe misses this. Block socket operations with a syscall policy and test a planted Unix listener. Cover local service access in the platform probes.

5. **High — Invalid result metadata can become a pass.**  
   `src/aftershock_logger/hook_sandbox.py`, `run_hook` result validation (lines 451–466).  
   A hook can write `{"keys":["kepe"]}` to `result.json` and exit successfully. The parent ignores the unknown key, reports `ok`, and keeps the original selection. Validation inside the runner cannot be trusted after hook code executes. Require the complete result schema in the parent. Reject missing fields, unknown keys, wrong types, and undeclared result files. Test forged metadata directly.

6. **Medium — Nested JSON crashes the parent’s error path.**  
   `src/aftershock_logger/hook_sandbox.py`, result parsing (lines 452–464).  
   Deeply nested JSON within the size cap raises `RecursionError`. That exception escapes instead of producing a failure with the hook’s name and reason. Limit nesting and handle parser recursion failures. Add a malicious result-file test.

This was a read-only review. No tests or code were run.