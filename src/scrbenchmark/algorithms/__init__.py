"""Algorithms module with resilient auto-registration imports.

This package automatically discovers and imports all algorithm modules so decorators
can register them into ``AlgorithmRegistry``. A missing optional dependency in one
module must not prevent the CLI from loading other available algorithms.
"""

from importlib import import_module
import inspect
import logging
import pkgutil
# pylint: disable=import-error
from core.algorithm_registry import BaseAlgorithm

logger = logging.getLogger(__name__)

__all__ = []

# Deprecated benchmark algorithms kept on disk for old imports, but no longer
# exposed through the registry, CLI, or UI.
DISABLED_MODULES = {
    "simple_autoencoder",
    "scvi_dct",
}

# Automatically discover and import all modules in this package
for _, modname, _ in pkgutil.iter_modules(__path__):
    if modname.startswith('_') or modname.startswith('template'):
        continue
    if modname in DISABLED_MODULES:
        logger.info("Skipping disabled algorithm module '%s'", modname)
        continue

    try:
        module = import_module(f".{modname}", __name__)
        # Expose subclasses of BaseAlgorithm in the package namespace
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseAlgorithm) and obj is not BaseAlgorithm:
                globals()[name] = obj
                if name not in __all__:
                    __all__.append(name)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Skipping algorithm module '%s' due to import error: %s",
            modname,
            exc,
        )

