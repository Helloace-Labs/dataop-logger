request changes

The `..` placeholder fix is correct by inspection, but it introduces a loader-path regression.

**Medium — Resolving parent folders removes required loader aliases.**

Files and sections: `src/aftershock_logger/hook_sandbox.py`, `_canon_lib` lines 281–284 and `_bwrap_cmd` lines 357–368; `tests/test_hook_sandbox.py`, regression test starting at line 438.

When a library or ELF loader path crosses a symlinked folder, canonicalization discards that folder’s alias. The sandbox mounts only the resolved destination and never recreates the alias. The kernel still requests the interpreter’s original `PT_INTERP` path, so a valid runtime can fail before Python starts. Library search paths can fail similarly. The new fixture covers a symlinked filename, but no symlinked parent folder.

Keep canonical paths for mirror placement, and preserve loader-requested aliases through explicit file mounts or sandbox-created symlinks. Add coverage for a symlinked parent folder, including the ELF interpreter path.

No new file exposure or write surface was found in this fix. Resolving the folder still binds only the accepted dependency file read-only, not its surrounding directory.

Read-only review. No tests or code were run. No files were edited.