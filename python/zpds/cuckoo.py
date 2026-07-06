"""Cuckoo filter — membership with deletion support."""

from __future__ import annotations

from ._native import as_bytes, ffi, lib


class Full(Exception):
    """Raised when a cuckoo filter cannot accommodate another insertion."""


class CuckooFilter:
    """A probabilistic set that, unlike a Bloom filter, supports deletion.

    False positives are possible (~0.01%); false negatives cannot occur for
    items that were inserted and not removed. Only delete items you actually
    inserted — deleting an absent item may remove a colliding fingerprint.

    >>> cf = CuckooFilter(capacity=1000)
    >>> cf.add("alice")
    >>> "alice" in cf
    True
    >>> cf.remove("alice")
    True
    >>> "alice" in cf
    False
    """

    __slots__ = ("_c",)

    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._c = lib.zpds_cuckoo_new(capacity, seed)
        if self._c == ffi.NULL:
            raise MemoryError("failed to allocate cuckoo filter")

    def add(self, item) -> None:
        """Insert an item. Raises :class:`Full` if the filter is full."""
        data = as_bytes(item)
        if not lib.zpds_cuckoo_add(self._c, data, len(data)):
            raise Full("cuckoo filter is full")

    def try_add(self, item) -> bool:
        """Insert an item, returning False instead of raising when full."""
        data = as_bytes(item)
        return bool(lib.zpds_cuckoo_add(self._c, data, len(data)))

    def contains(self, item) -> bool:
        data = as_bytes(item)
        return bool(lib.zpds_cuckoo_contains(self._c, data, len(data)))

    __contains__ = contains

    def remove(self, item) -> bool:
        """Remove one occurrence. Returns True if a match was removed."""
        data = as_bytes(item)
        return bool(lib.zpds_cuckoo_remove(self._c, data, len(data)))

    def clear(self) -> None:
        lib.zpds_cuckoo_clear(self._c)

    @property
    def capacity(self) -> int:
        """Total slot capacity."""
        return lib.zpds_cuckoo_capacity(self._c)

    def __len__(self) -> int:
        """Number of live fingerprints."""
        return lib.zpds_cuckoo_count(self._c)

    def __repr__(self) -> str:
        return f"CuckooFilter(size={len(self)}, capacity={self.capacity})"

    def __del__(self):
        c = getattr(self, "_c", None)
        if c is not None and c != ffi.NULL:
            lib.zpds_cuckoo_free(c)
            self._c = ffi.NULL
