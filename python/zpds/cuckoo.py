"""Cuckoo filter — membership with deletion support."""

from __future__ import annotations

from ._batch import BatchSizeMixin, numpy_fast_view, pack_chunk, iter_chunks, query_batches
from ._native import as_bytes, ffi, lib


class Full(Exception):
    """Raised when a cuckoo filter cannot accommodate another insertion."""


class CuckooFilter(BatchSizeMixin):
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

    __slots__ = ("_c", "_batch_size")

    def __init__(self, capacity: int, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.init_batch_size()
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

    def add_many(self, items) -> int:
        """Insert many items with one FFI crossing per batch.

        Unlike :meth:`add`, this never raises on a full filter — it returns the
        number of items actually inserted (which is < the input size once the
        table fills up). ``items`` may be a numpy array or any iterable.
        """
        inserted = 0

        def go(blob, offsets, width, n):
            nonlocal inserted
            inserted += lib.zpds_cuckoo_add_many(self._c, blob, offsets, width, n, ffi.NULL)

        view = numpy_fast_view(items)
        if view is not None:
            buf, width, n = view
            go(buf, ffi.NULL, width, n)
        else:
            for chunk in iter_chunks(items, self._batch_size):
                blob, offsets, n = pack_chunk(chunk)
                go(blob, offsets, 0, n)
        return inserted

    def contains(self, item) -> bool:
        data = as_bytes(item)
        return bool(lib.zpds_cuckoo_contains(self._c, data, len(data)))

    __contains__ = contains

    def contains_many(self, items):
        """Query many items at once. Returns a numpy bool array for numpy input,
        otherwise a ``list[bool]`` aligned with ``items``."""
        return query_batches(
            items,
            self._batch_size,
            "uint8_t",
            lambda b, o, w, n, out: lib.zpds_cuckoo_contains_many(self._c, b, o, w, n, out),
        )

    def remove(self, item) -> bool:
        """Remove one occurrence. Returns True if a match was removed."""
        data = as_bytes(item)
        return bool(lib.zpds_cuckoo_remove(self._c, data, len(data)))

    def remove_many(self, items) -> int:
        """Remove many items with one FFI crossing per batch. Returns the number
        of items actually removed."""
        removed = 0

        def go(blob, offsets, width, n):
            nonlocal removed
            removed += lib.zpds_cuckoo_remove_many(self._c, blob, offsets, width, n, ffi.NULL)

        view = numpy_fast_view(items)
        if view is not None:
            buf, width, n = view
            go(buf, ffi.NULL, width, n)
        else:
            for chunk in iter_chunks(items, self._batch_size):
                blob, offsets, n = pack_chunk(chunk)
                go(blob, offsets, 0, n)
        return removed

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
