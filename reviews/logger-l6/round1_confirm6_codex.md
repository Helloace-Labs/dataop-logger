approve

The fix in `74d790e` is correct and complete by inspection. No High, Medium or Low findings.

In `src/aftershock_logger/hook_sandbox.py`, `_folder_links` preserves the plain traversal folders. `_mirror` creates those inside runtime copies, which are mounted read-only. `_bwrap_cmd` creates the others empty, without binding host contents. Folder creation precedes runtime and library mounts, so it cannot hide them. No meaningful new write surface or adjacent regression was found.

In `tests/test_hook_sandbox.py`, `test_loader_spellings_through_symlinked_folders_keep_working` covers the reported target and checks placeholder creation, mount order and absence of a host-folder bind.

Read-only review. No tests or code were run. No files were edited.