request changes

1. **Original High fix: correct and complete.**  
   In `src/aftershock_logger/logger.py`, the module-level import restores `_pkg_version` for `_lerobot_version`. Manifests and exports both use that function. The new assertions detect the original failure. No nearby regression found.

2. **Medium: Reload removes alias compatibility.**  
   Files: `src/dataop_logger/__init__.py`, `_AliasLoader.find_spec` and `exec_module`; `tests/test_rename.py`, `test_runpy_and_reload_after_the_alias_was_imported`.

   The direct import case is fixed, but the fix is incomplete. After importing `dataop_logger.cli`, reloading that module uses its canonical name. The alias finder skips that name, so reload installs a fresh source loader without `_BothNamesLoader`. A later `runpy.run_module("dataop_logger.cli", run_name="__main__")` fails again.

   The regression test reloads `store` but executes `cli`, so it misses this sequence.

   Fix: preserve the compatible loader when canonical modules are reloaded. Add coverage that imports, reloads and executes the same module through the old name, with both import orders.

Tests were read, not executed.