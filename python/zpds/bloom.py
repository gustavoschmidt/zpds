"""Bloom filter — membership with a tunable false-positive rate."""

from __future__ import annotations

from ._batch import BatchSizeMixin, for_each_batch, query_batches
from ._native import as_bytes, ffi, lib


class BloomFilter(BatchSizeMixin):
    """A space-efficient probabilistic set.

    False positives are possible at approximately ``error_rate``; false
    negatives never occur.

    >>> bf = BloomFilter(capacity=1000, error_rate=0.01)
    >>> bf.add("alice")
    >>> "alice" in bf
    True
    """

    __slots__ = ("_c", "_batch_size")

    def __init__(self, capacity: int, error_rate: float = 0.01, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0.0 < error_rate < 1.0:
            raise ValueError("error_rate must be in (0, 1)")
        self.init_batch_size()
        self._c = lib.zpds_bloom_new(capacity, error_rate, seed)
        if self._c == ffi.NULL:
            raise MemoryError("failed to allocate Bloom filter")

    @classmethod
    def from_params(cls, n_bits: int, k: int, seed: int = 0) -> "BloomFilter":
        """Construct with an explicit bit count and hash-function count."""
        obj = cls.__new__(cls)
        obj.init_batch_size()
        obj._c = lib.zpds_bloom_new_with_params(n_bits, k, seed)
        if obj._c == ffi.NULL:
            raise MemoryError("failed to allocate Bloom filter")
        return obj

    def add(self, item) -> None:
        data = as_bytes(item)
        lib.zpds_bloom_add(self._c, data, len(data))

    def add_many(self, items) -> None:
        """Add many items with a single FFI crossing per batch.

        ``items`` may be a numpy array (passed zero-copy when its dtype is
        fixed-width), a list, or any iterable/generator (consumed in
        ``batch_size`` chunks, so streams stay memory-bounded).
        """
        for_each_batch(
            items,
            self._batch_size,
            lambda b, o, w, n: lib.zpds_bloom_add_many(self._c, b, o, w, n),
        )

    # Back-compat alias for the pre-batch API.
    update = add_many

    def contains(self, item) -> bool:
        data = as_bytes(item)
        return bool(lib.zpds_bloom_contains(self._c, data, len(data)))

    __contains__ = contains

    def contains_many(self, items):
        """Query many items at once.

        Returns a numpy bool array for numpy-array input, otherwise a
        ``list[bool]`` aligned with ``items``.
        """
        return query_batches(
            items,
            self._batch_size,
            "uint8_t",
            lambda b, o, w, n, out: lib.zpds_bloom_contains_many(self._c, b, o, w, n, out),
        )

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

    def __len__(self) -> int:
        """Number of ``add`` calls (counts duplicates)."""
        return lib.zpds_bloom_count(self._c)

    def __repr__(self) -> str:
        return f"BloomFilter(bits={self.bits}, k={self.num_hashes}, added={len(self)})"

    def __del__(self):
        c = getattr(self, "_c", None)
        if c is not None and c != ffi.NULL:
            lib.zpds_bloom_free(c)
            self._c = ffi.NULL
