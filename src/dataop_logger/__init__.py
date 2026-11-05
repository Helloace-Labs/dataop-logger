"""Old name of aftershock_logger, kept so existing code keeps working.

`import dataop_logger` and `import dataop_logger.<module>` return the very same module objects as
`aftershock_logger` and `aftershock_logger.<module>`, so classes, module state and monkeypatches are shared,
and `python -m dataop_logger.cli` still runs the CLI. New code should import `aftershock_logger`.
"""
import importlib
import importlib.abc
import importlib.util
import sys

_OLD, _NEW = "dataop_logger", "aftershock_logger"


class _AliasLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve `dataop_logger.X` to the already imported (or newly imported) `aftershock_logger.X`."""

    def __init__(self):
        self._real_specs = {}

    @staticmethod
    def _target(fullname: str) -> str:
        return _NEW + fullname[len(_OLD):]

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_OLD + "."):
            return None
        real = importlib.util.find_spec(self._target(fullname))
        if real is None:
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=real.submodule_search_locations is not None)

    def create_module(self, spec):
        # The old name becomes only another key in sys.modules for the same object. The import system keeps
        # the module's __name__, __package__ and __loader__ but overwrites __spec__ with the alias spec, so
        # remember the real one and put it back in exec_module.
        module = importlib.import_module(self._target(spec.name))
        self._real_specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module):
        # Already executed under its new name; only undo the __spec__ overwrite.
        real = self._real_specs.pop(module.__spec__.name, None)
        if real is not None:
            module.__spec__ = real

    # For runpy (`python -m dataop_logger.cli`): run the new module's code under the old name.
    def get_code(self, fullname):
        target = self._target(fullname)
        return importlib.util.find_spec(target).loader.get_code(target)

    def is_package(self, fullname):
        real = importlib.util.find_spec(self._target(fullname))
        return real is not None and real.submodule_search_locations is not None


if not any(isinstance(f, _AliasLoader) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasLoader())

_pkg = importlib.import_module(_NEW)
sys.modules[_OLD] = _pkg                                 # `dataop_logger` is `aftershock_logger`
