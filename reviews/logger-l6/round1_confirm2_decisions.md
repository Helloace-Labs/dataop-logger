# Logger L6, round 1 second confirmation decisions

This confirmation round found a new High, so it counts as full round 3. Fixes 2, 3, 5 and 6 of the previous round were confirmed.

| # | Severity | Finding | Decision | Change |
| --- | --- | --- | --- | --- |
| 1 | High | ldd parsing splits on spaces and can mount a folder | Changed | `hook_sandbox.py` `_parse_ldd` matches the whole path up to ` (0x...)`, keeps spaces, and keeps only existing regular files, so a folder can never be mounted as a library. Test: `test_ldd_parsing_keeps_spaces_and_only_regular_files`, where the truncated prefix is an existing folder. |
| 2 | Medium | Later folder mounts hide library mounts; symlinked bundled libraries vanish | Changed | `_bwrap_cmd` mounts runtime folders first, then the interpreter files, then each library. `_mirror` leaves an empty mount point for every needed library (and interpreter file) inside a runtime folder, so a library reached by a symlink name is mounted there from its real file. The command test adds a bundled library behind a symlink and checks the mount point, the source and the order. |
