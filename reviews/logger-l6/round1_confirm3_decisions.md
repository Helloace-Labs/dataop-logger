# Logger L6, round 1 third confirmation decisions

This confirmation round found no new High. Fix 1 of the previous round (ldd parsing) was confirmed.

| # | Severity | Finding | Decision | Change |
| --- | --- | --- | --- | --- |
| 1 | Medium | A library path spelled with `..` gets its mount point beside the numpy copy, not in its own folder's copy | Changed | `hook_sandbox.py` `_canon_lib` resolves a library's folder (no `..`, no symlinked folders) and keeps its own name, so an alias stays an alias. `_parse_ldd` returns canonical paths, and `_bwrap_cmd` canonicalizes again before building the mirror, so containment checks, mount points and binds use one path. Test: `test_library_reached_through_dotdot_gets_its_mount_point_in_its_own_folder`, with a separate `numpy.libs` folder, a symlinked library and a `..` spelling; it fails without the fix. |
