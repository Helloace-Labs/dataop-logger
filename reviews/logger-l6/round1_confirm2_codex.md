request changes

| Fix | Assessment |
| --- | --- |
| 1. Minimal runtime | Incomplete. The mirror excludes non-code files, but library parsing can reintroduce directory mounts. See findings 1 and 2. The broad macOS system-directory grants are removed; startup remains unverified. |
| 2. Writable hiding mounts | Correct and complete by inspection. Removed. No adjacent regression found. |
| 3. Linux socket probe | Correct by inspection. Socket and socketpair creation are attempted, and any success rejects the sandbox. The new test checks report judgment; it does not exercise a weakened filter. |
| 4. Nested NumPy | The reported directory and profile ordering is corrected for both layouts. Adjacent library mount ordering remains incomplete. See finding 2. |
| 5. Readable masks | Correct and complete by inspection. Masks are gone, so omitted non-code files cannot open successfully. |
| 6. Result-file tests | Correct and complete by inspection. Expectations account for Linux mount denial. Parent fixtures cover unsafe file types and extra files. No adjacent regression found. |

1. **High — Library parsing can expose whole directories.**  
   Files and sections: `src/aftershock_logger/hook_sandbox.py`, `_shared_libs` lines 278–282 and `_bwrap_cmd` lines 329–330; `tests/test_hook_sandbox.py`, command construction test.

   `line.split()` truncates library paths containing spaces. If the truncated prefix names an existing directory, `exists()` accepts it. The command then mounts that whole host directory, exposing non-code files. Other runtime mounts can supply the actual library, so startup need not fail. The runtime-file probe does not check the unintended directory.

   Parse the full resolved path while preserving spaces. Require a regular library file before mounting it. Add parser cases where the truncated prefix names an existing directory or unrelated file. The current command test mocks this parser.

2. **Medium — Later mirrors hide required library aliases.**  
   Files and sections: `src/aftershock_logger/hook_sandbox.py`, `_mirror` lines 241–246 and `_bwrap_cmd` lines 329–334; `tests/test_hook_sandbox.py`, runtime fixtures.

   Required library files are mounted before runtime directories. A later directory mount hides those earlier mounts beneath it. The mirror omits symlinks, so a required library alias inside a runtime folder disappears even when its versioned target is copied. This breaks valid bundled-library layouts and prevents hook startup.

   Create mount points for required leaf files in the mirrors. Mount runtime directories first, then bind the required files. Add a fixture with a required library symlink inside a mirrored runtime folder.

This review covered only the six fixes and adjacent behavior. No tests or code were run, and no files were edited.