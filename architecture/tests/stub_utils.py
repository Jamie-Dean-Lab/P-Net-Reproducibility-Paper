"""
Helpers for test modules that must stand in fake modules while importing the
code under test.

Both test_coef_weights_utils.py and test_evaluation.py load their target module
by file path so it can be exercised without TensorFlow, Keras or shap actually
being importable. That requires the stubs to be present in sys.modules at import
time, but leaving them there afterwards breaks every later test module that
wants the real packages - pytest shares one interpreter across the whole run.

stubbed_modules() installs the stubs for the duration of the import and then puts
sys.modules back exactly as it found it.
"""

import contextlib
import sys

_MISSING = object()


@contextlib.contextmanager
def stubbed_modules(stubs: dict):
    """
    Context manager which installs stub modules into sys.modules and restores the
    previous state on exit.

    Only the named keys are touched, so modules imported inside the block (for
    instance matplotlib.pyplot, pulled in by the module under test) are left
    alone. Names absent beforehand are removed again; names that were already
    present are put back unchanged.

    A module imported inside the block keeps its own references to the stubs it
    bound at import time, so the code under test carries on seeing the fakes
    after sys.modules has been restored.

    args:
        stubs (dict[str, ModuleType]) : mapping of module name to stub module

    yields:
        None
    """
    saved = {name: sys.modules.get(name, _MISSING) for name in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
