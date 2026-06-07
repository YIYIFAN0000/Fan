from __future__ import annotations

import os
import sys


def configure_runtime() -> None:
    """Keep local Windows scientific Python imports quiet and usable."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PANDAS_USE_NUMEXPR", "False")
    os.environ.setdefault("PANDAS_USE_BOTTLENECK", "False")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # The current conda env has NumPy 2.x plus old optional pandas accelerators.
    # Blocking their optional import avoids noisy import-time tracebacks.
    for module_name in ("numexpr", "bottleneck"):
        if module_name not in sys.modules:
            sys.modules[module_name] = None
