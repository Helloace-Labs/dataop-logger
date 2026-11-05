# Round 1 decisions: logger-rename

Codex verdict: request changes, 1 High and 1 Medium.

| # | Severity | Finding | Decision | Files |
| --- | --- | --- | --- | --- |
| 1 | High | LeRobot version lost: `_pkg_version` became local to `_own_version`, so `_lerobot_version` hit a NameError that was swallowed and new manifests and export records stored a null LeRobot version | Accepted. `_pkg_version` and `PackageNotFoundError` are imported at module level again. Tests now assert the manifest's `env.software.lerobot` and `_lerobot_version()` (used by export records) equal the installed LeRobot version | `src/aftershock_logger/logger.py`, `tests/test_rename.py` |
| 2 | Medium | `runpy.run_module("dataop_logger.cli")` fails after the alias was imported, because runpy takes the real spec from sys.modules and its loader rejects the old name | Accepted. When a module is first reached through the old name, its real spec's loader is wrapped in a loader that maps the old name to the new one and passes everything else through. New test runs `runpy.run_module` under both names after importing either name first, and checks `importlib.reload` through the old name | `src/dataop_logger/__init__.py`, `tests/test_rename.py` |

Both new tests fail on the round 1 code and pass after the fixes. Tests: 136 passed (101 existing, 35 new).
