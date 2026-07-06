"""Cuckoo filter — membership with deletion support."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from ._batch import (
    DEFAULT_BATCH_SIZE,
    BoolResults,
    Item,
    Items,
    check_batch_size,
    for_each_batch,
    iter_chunks,
    numpy_fast_view,
    pack_chunk,
    query_batches,
)
from ._native import as_bytes, ffi, lib

if TYPE_CHECKING:
    from types import TracebackType


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

    def add(self, item: Item) -> None:
        """Insert an item. Raises :class:`Full` if the filter is full."""
        data = as_bytes(item)
        if not lib.zpds_cuckoo_add(self._c, data, len(data)):
            raise Full("cuckoo filter is full")

    def try_add(self, item: Item) -> bool:
        """Insert an item, returning False instead of raising when full."""
        data = as_bytes(item)
        return bool(lib.zpds_cuckoo_add(self._c, data, len(data)))

    def add_many(
        self,
        items: Items,
        batch_size: int = DEFAULT_BATCH_SIZE,
        return_failed: bool = False,
    ):
        """Insert many items with one FFI crossing per batch.

        Unlike :meth:`add`, this never raises on a full filter. By default it
        returns the number of items actually inserted (which is < the input size
        once the table fills up).

        With ``return_failed=True`` it instead returns a ``list[int]`` of the
        positions (indices into ``items``) that could **not** be inserted, so a
        caller can recover the rejected keys when the filter fills mid-batch.

        ``items`` may be a numpy array or any iterable; ``batch_size`` sets the
        generic/streaming chunk size (ignored on the numpy fast path).
        """
        check_batch_size(batch_size)
        if not return_failed:
            return for_each_batch(
                items,
                batch_size,
                lambda b, o, w, n: lib.zpds_cuckoo_add_many(self._c, b, o, w, n, ffi.NULL),
            )

        failed: List[int] = []
        base = 0

        def go(blob, offsets, width, n):
            nonlocal base
            out = ffi.new("uint8_t[]", max(n, 1))
            lib.zpds_cuckoo_add_many(self._c, blob, offsets, width, n, out)
            for i in range(n):
                if not out[i]:
                    failed.append(base + i)
            base += n

        view = numpy_fast_view(items)
        if view is not None:
            buf, width, n = view
            go(buf, ffi.NULL, width, n)
        else:
            for chunk in iter_chunks(items, batch_size):
                blob, offsets, n = pack_chunk(chunk)
                go(blob, offsets, 0, n)
        return failed

    def contains(self, item: Item) -> bool:
        data = as_bytes(item)
        return bool(lib.zpds_cuckoo_contains(self._c, data, len(data)))

    __contains__ = contains

    def contains_many(
        self, items: Items, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> BoolResults:
        """Query many items at once. Returns a numpy bool array for numpy input,
        otherwise a ``list[bool]`` aligned with ``items``."""
        return query_batches(
            items,
            check_batch_size(batch_size),
            "uint8_t",
            lambda b, o, w, n, out: lib.zpds_cuckoo_contains_many(self._c, b, o, w, n, out),
        )

    def remove(self, item: Item) -> bool:
        """Remove one occurrence. Returns True if a match was removed."""
        data = as_bytes(item)
        return bool(lib.zpds_cuckoo_remove(self._c, data, len(data)))

    def remove_many(self, items: Items, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
        """Remove many items with one FFI crossing per batch. Returns the number
        of items actually removed."""
        return for_each_batch(
            items,
            check_batch_size(batch_size),
            lambda b, o, w, n: lib.zpds_cuckoo_remove_many(self._c, b, o, w, n, ffi.NULL),
        )

    def copy(self) -> "CuckooFilter":
        """Return an independent copy of this filter."""
        obj = CuckooFilter.__new__(CuckooFilter)
        obj._c = lib.zpds_cuckoo_clone(self._c)
        if obj._c == ffi.NULL:
            raise MemoryError("failed to copy cuckoo filter")
        return obj

    def clear(self) -> None:
        lib.zpds_cuckoo_clear(self._c)

    @property
    def capacity(self) -> int:
        """Total slot capacity."""
        return lib.zpds_cuckoo_capacity(self._c)

    @property
    def count(self) -> int:
        """Number of live fingerprints (same as ``len(self)``)."""
        return lib.zpds_cuckoo_count(self._c)

    def __len__(self) -> int:
        """Number of live fingerprints."""
        return lib.zpds_cuckoo_count(self._c)

    def close(self) -> None:
        """Free the native filter. Safe to call more than once."""
        c = getattr(self, "_c", None)
        if c is not None and c != ffi.NULL:
            lib.zpds_cuckoo_free(c)
            self._c = ffi.NULL

    def __enter__(self) -> "CuckooFilter":
        return self

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"CuckooFilter(size={self.count}, capacity={self.capacity})"

    def __del__(self):
        self.close()
