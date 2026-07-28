"""One place where a missing optional dependency is explained.

The base install stays small on purpose: matching two CSVs with an API-hosted model
should not download torch. When a user reaches for something that does need it, the
error must say which extra to install, not raise a bare ImportError from a library
they have never heard of.
"""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingExtra(ImportError):
    """An optional dependency is required but not installed."""


def require(extra: str, module: str, *, purpose: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtra(
            f"{purpose} requires the optional dependency {module!r}, which is part of "
            f"xwalk[{extra}].\n\n    pip install 'xwalk[{extra}]'\n\n"
            f"underlying import error: {exc}"
        ) from exc
