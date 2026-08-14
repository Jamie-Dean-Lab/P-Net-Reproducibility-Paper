"""
Shared pytest setup for the architecture test suite.

The test modules use two import styles: some load the module under test directly
(`from pipeline import ...`, which needs architecture/ on sys.path) while the
modules they load import through the package (`from architecture.pnet_model
import TFModel`, which needs the project root). Both entries are added here so
the imports resolve the same way no matter which file pytest collects first, or
whether a single file is run on its own.
"""

import os
import sys

_ARCH_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_ARCH_DIR)
_ROOT_DIR = os.path.dirname(_PACKAGE_DIR)

for _p in (_ARCH_DIR, _PACKAGE_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
