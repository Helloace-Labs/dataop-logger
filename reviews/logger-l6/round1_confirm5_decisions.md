# Logger L6, round 1 fifth confirmation decisions

This confirmation round found no new High and no new exposure, write surface or mount hiding. It found one more path shape the loader could not follow.

| # | Severity | Finding | Decision | Change |
| --- | --- | --- | --- | --- |
| 1 | Medium | A symlink target that walks through a plain folder before `..` (`../scratch/../numpy.libs`) needs that folder to exist in the sandbox | Changed | `hook_sandbox.py` `_folder_links` also returns every plain folder the kernel walks through. Those outside the runtime folders are created empty with `--dir` before any mount; those inside a runtime folder exist in its code copy. No host folder's contents are mounted for them. Test: `test_loader_spellings_through_symlinked_folders_keep_working` adds this target with a file inside `scratch`, checks `scratch` is created empty and never mounted, and fails without the `--dir` step. |
