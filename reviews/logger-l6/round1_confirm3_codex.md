request changes

- **Fix 1 — Correct and complete by inspection.** `_parse_ldd` preserves spaces and rejects directories. The regression test covers the reported directory-prefix case. No adjacent regression found.
- **Fix 2 — Incomplete.** Folder-before-file ordering and direct symlink placeholders are correct. Paths containing `..` can still leave required library aliases without mount points.

**Medium — Library placeholders can land outside the intended mirror.**

Files and sections: `src/aftershock_logger/hook_sandbox.py`, `_mirror` lines 266–272 and `_bwrap_cmd` lines 350–361; `tests/test_hook_sandbox.py`, `test_bwrap_command_mounts_only_code_and_the_result_files`.

An accepted library path can contain `numpy/_core/../../numpy.libs/libalias.so`. `_mirror` checks containment against that raw spelling, but `os.path.relpath` normalizes it. The placeholder therefore lands beside the NumPy mirror, while the separate `numpy.libs` mirror gets none. If `libalias.so` is a symlink, the mirror omits it and the later file bind cannot create it inside the read-only mount. A valid runtime then fails startup.

Canonicalize each library’s parent directory while preserving its final alias name. Use the same destination for containment checks, placeholders and binds. Add a fixture combining `..`, a separate bundled-library mirror and a symlinked library.

Read-only review. No tests or code were run. No files were edited.