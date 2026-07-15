"""
_logging.py - Library logging helper
=====================================
Provides :func:`get_logger` so modules can emit progress/warnings without
hardcoding ``print`` calls. All loggers live under the ``quant_research``
namespace, so callers can route them however they like:

    import logging
    logging.basicConfig(level=logging.INFO)        # show everything
    logging.getLogger("quant_research").setLevel(logging.WARNING)  # quiet

The ``QUANT_LOG_LEVEL`` environment variable provides a one-shot default;
if unset, the library behaves like any well-behaved Python library and
attaches a :class:`logging.NullHandler` so nothing is printed unless the
application configures logging.
"""

from __future__ import annotations

import logging
import os

_ROOT_NAME = "quant_research"
_configured = False


def _configure_root_once() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger(_ROOT_NAME)
    # Library convention: attach NullHandler so "no config" stays silent.
    if not any(isinstance(h, logging.NullHandler) for h in root.handlers):
        root.addHandler(logging.NullHandler())
    level = os.environ.get("QUANT_LOG_LEVEL")
    if level:
        root.setLevel(level.upper())
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Pass ``__name__`` from callers."""
    _configure_root_once()
    if not name.startswith(_ROOT_NAME):
        name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(name)
