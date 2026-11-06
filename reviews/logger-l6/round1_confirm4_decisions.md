# Logger L6, round 1 fourth confirmation decisions

This confirmation round found no new High. The `..` placement fix of the previous round was confirmed; it found that resolving folders loses the loader's own spellings.

| # | Severity | Finding | Decision | Change |
| --- | --- | --- | --- | --- |
| 1 | Medium | Canonical library paths drop symlinked folder aliases the loader asks for (the ELF interpreter's `/lib64` on a merged `/usr`) | Changed | `hook_sandbox.py`: `_parse_ldd` keeps ldd's spelling. `_bwrap_cmd` mounts each library at `_canon_lib` (as before) and recreates every symlinked folder the kernel passes through on ldd's spelling (`_folder_links`, which follows `..` and the targets' own symlinked folders): inside a runtime folder as the same symlink in its code copy, elsewhere with `--symlink`. A symlink exposes nothing that is not mounted. Test: `test_loader_spellings_through_symlinked_folders_keep_working`, with a `lib64 -> usr/lib64` interpreter path and a symlinked folder inside numpy; it fails without the links. |
