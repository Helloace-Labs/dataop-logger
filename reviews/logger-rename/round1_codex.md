request changes

1. **High: LeRobot version is silently lost.**  
   Files: `src/aftershock_logger/logger.py`, `_own_version` and `_lerobot_version` (lines 23–51); `tests/test_rename.py`, lines 224–226.

   `_pkg_version` now exists only inside `_own_version`. Calling it from `_lerobot_version` raises `NameError`, which is swallowed. New manifests and export records therefore always store a null LeRobot version. The new test passes because it checks the key but never checks its value.

   Fix: import the version function at module scope or inside both functions. Assert that recorded and exported LeRobot versions match the installed distribution.

2. **Medium: Running an imported alias through runpy fails.**  
   Files: `src/dataop_logger/__init__.py`, `exec_module` and `get_code` (lines 41–50); `tests/test_rename.py`, `test_python_dash_m_works_with_both_names`.

   After `import dataop_logger.cli`, the shared module has the canonical spec and source loader. `runpy.run_module("dataop_logger.cli", run_name="__main__")` then asks that loader for code under the old name. The loader rejects the name. The alias loader’s `get_code` is bypassed. Existing tests cover only fresh processes.

   Fix: make code loading accept both names while preserving shared module identity and the canonical spec. Add a regression test that imports the alias before executing it through runpy.

Tests were read, not executed.