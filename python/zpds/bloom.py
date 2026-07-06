"""Bloom filter — membership with a tunable false-positive rate."""

from __future__ import annotations

from ._native import as_bytes, ffi, lib


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
    def from_params(cls, n_bits: int, k: int, seed: int = 0) -> "BloomFilter":
        """Construct with an explicit bit count and hash-function count."""
        obj = cls.__new__(cls)
        obj._c = lib.zpds_bloom_new_with_params(n_bits, k, seed)
        if obj._c == ffi.NULL:
            raise MemoryError("failed to allocate Bloom filter")
        return obj

    def add(self, item) -> None:
        data = as_bytes(item)
        lib.zpds_bloom_add(self._c, data, len(data))

    def update(self, items) -> None:
        """Add every element of an iterable."""
        add = self.add
        for item in items:
            add(item)

    def contains(self, item) -> bool:
        data = as_bytes(item)
        return bool(lib.zpds_bloom_contains(self._c, data, len(data)))

    __contains__ = contains

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
