"""Pytest bootstrap: make the ``zpds`` package importable from the repo root.

The shipped package lives under ``python/zpds`` (so the repo root stays free of
the package tree), while the tests live in ``tests/`` and the benchmarks in
``benchmarks/``. Adding ``python/`` to ``sys.path`` lets ``pytest`` run from the
root without an editable install; the repo root itself is on ``sys.path`` (this
conftest's directory) so ``import benchmarks`` resolves too.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.join(_ROOT, "python")
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)
