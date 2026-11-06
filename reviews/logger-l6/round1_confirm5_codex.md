request changes

The fix handles the reported aliases, but remains incomplete.

**Medium — Symlink targets can require omitted directories.**

Files and sections: `src/aftershock_logger/hook_sandbox.py`, `_folder_links` lines 312–325 and `_mirror` lines 275–280; `tests/test_hook_sandbox.py`, `test_loader_spellings_through_symlinked_folders_keep_working`.

Consider `numpy/_libs -> ../scratch/../numpy.libs`, where `scratch` is an ordinary host directory outside the runtime folders. The dependency exists and passes parsing. The sandbox recreates the symlink and mounts the canonical library, but omits `scratch`. The kernel must traverse `scratch` before processing `..`, so the loader fails.

Preserve required traversal directories as empty read-only placeholders, without mounting their host contents. Add this fixture and verify the loader spelling resolves; the current test only checks the simple targets and canonical mount entries.

No new host-file exposure, host write surface, or mount-hiding regression was found by inspection. Symlinks resolve within the sandbox, and dependency files remain mounted read-only after runtime folders.

Read-only review. No tests or code were run. No files were edited.