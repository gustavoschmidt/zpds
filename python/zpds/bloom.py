"""Bloom filter — membership with a tunable false-positive rate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._batch import (
    DEFAULT_BATCH_SIZE,
    BoolResults,
    Item,
    Items,
    check_batch_size,
    for_each_batch,
    query_batches,
)
from ._native import as_bytes, ffi, lib

if TYPE_CHECKING:
    from types import TracebackType


class BloomFilter:
    """A space-efficient probabilistic set.

    False positives are possible at approximately ``error_rate``; false
    negatives never occur.

    >>> bf = BloomFilter(capacity=1000, error_rate=0.01)
    >>> bf.add("alice")
    >>> "alice" in bf
    True
    """

    __slots__ = ("_c",)

    def __init__(self, capacity: int, error_rate: float = 0.01, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 < error_rate < 1.0:
            raise ValueError("error_rate must be in (0, 1)")
        self._c = lib.zpds_bloom_new(capacity, error_rate, seed)
        if self._c == ffi.NULL:
            raise MemoryError("failed to allocate Bloom filter")

    @classmethod
    def from_bit_count(cls, n_bits: int, k: int, seed: int = 0) -> "BloomFilter":
        """Construct with an explicit bit count ``n_bits`` and hash-function
        count ``k`` instead of a capacity / error-rate pair."""
        obj = cls.__new__(cls)
        obj._c = lib.zpds_bloom_new_with_params(n_bits, k, seed)
        if obj._c == ffi.NULL:
            raise MemoryError("failed to allocate Bloom filter")
        return obj

    def add(self, item: Item) -> None:
        data = as_bytes(item)
        lib.zpds_bloom_add(self._c, data, len(data))

    def add_many(self, items: Items, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Add many items with a single FFI crossing per batch.

        ``items`` may be a numpy array (passed zero-copy when its dtype is
        fixed-width), a list, or any iterable/generator. ``batch_size`` sets how
        many items are packed per crossing for the generic/streaming path (so
        streams stay memory-bounded); it is ignored on the numpy fast path.
        """
        for_each_batch(
            items,
            check_batch_size(batch_size),
            lambda b, o, w, n: lib.zpds_bloom_add_many(self._c, b, o, w, n),
        )

    def contains(self, item: Item) -> bool:
        data = as_bytes(item)
        return bool(lib.zpds_bloom_contains(self._c, data, len(data)))

    __contains__ = contains

    def contains_many(
        self, items: Items, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> BoolResults:
        """Query many items at once.

        Returns a numpy bool array for numpy-array input, otherwise a
        ``list[bool]`` aligned with ``items``.
        """
        return query_batches(
            items,
            check_batch_size(batch_size),
            "uint8_t",
            lambda b, o, w, n, out: lib.zpds_bloom_contains_many(self._c, b, o, w, n, out),
        )

    def copy(self) -> "BloomFilter":
        """Return an independent copy of this filter."""
        obj = BloomFilter.__new__(BloomFilter)
        obj._c = lib.zpds_bloom_clone(self._c)
        if obj._c == ffi.NULL:
            raise MemoryError("failed to copy Bloom filter")
        return obj

    def clear(self) -> None:
        lib.zpds_bloom_clear(self._c)

    @property
    def bits(self) -> int:
        """Number of bits in the filter (m)."""
        return lib.zpds_bloom_bits(self._c)

    @property
    def num_hashes(self) -> int:
        """Number of hash functions (k)."""
        return lib.zpds_bloom_k(self._c)

    @property
    def items_added(self) -> int:
        """Number of ``add`` calls so far (counts duplicates).

        A Bloom filter cannot report distinct membership, so this is *not* a set
        size — hence a named property rather than ``__len__``.
        """
        return lib.zpds_bloom_count(self._c)

    def close(self) -> None:
        """Free the native filter. Safe to call more than once."""
        c = getattr(self, "_c", None)
        if c is not None and c != ffi.NULL:
            lib.zpds_bloom_free(c)
            self._c = ffi.NULL

    def __enter__(self) -> "BloomFilter":
        return self

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"BloomFilter(bits={self.bits}, k={self.num_hashes}, added={self.items_added})"

    def __del__(self):
        self.close()
