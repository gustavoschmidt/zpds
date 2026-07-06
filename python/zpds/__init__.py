"""zpds — fast probabilistic data structures backed by a native Zig core.

Exposes four structures over a simple, ergonomic Python API:

- :class:`BloomFilter`   — membership, tunable false-positive rate.
- :class:`CuckooFilter`  — membership with deletion support.
- :class:`HyperLogLog`   — cardinality estimation.
- :class:`CountMinSketch` — frequency estimation.
"""

from __future__ import annotations

from ._native import as_bytes as _as_bytes, lib as _lib
from .bloom import BloomFilter
from .countmin import CountMinSketch
from .cuckoo import CuckooFilter, Full
from .hll import HyperLogLog

__all__ = [
    "BloomFilter",
    "CuckooFilter",
    "Full",
    "HyperLogLog",
    "CountMinSketch",
    "hash64",
    "native_version",
    "__version__",
]

__version__ = "0.0.1"


def native_version() -> tuple[int, int, int]:
    """The native core's version as a ``(major, minor, patch)`` tuple."""
    v = _lib.zpds_version()
    return (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF


def hash64(item, seed: int = 0) -> int:
    """64-bit wyhash of ``item`` (str/bytes) under ``seed``."""
    data = _as_bytes(item)
    return _lib.zpds_hash64(data, len(data), seed)
