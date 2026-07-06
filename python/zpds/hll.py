"""HyperLogLog — distinct-count (cardinality) estimation."""

from __future__ import annotations

from ._batch import BatchSizeMixin, for_each_batch
from ._native import as_bytes, ffi, lib


class HyperLogLog(BatchSizeMixin):
    """Estimate the number of distinct items in a stream in fixed memory.

    ``precision`` (4..18) trades memory for accuracy: memory is ``2**precision``
    bytes and the standard relative error is ~``1.04 / sqrt(2**precision)``
    (about 0.8% at the default precision 14).

    >>> hll = HyperLogLog(precision=14)
    >>> for i in range(100_000):
    ...     hll.add(f"user-{i}")
    >>> abs(len(hll) - 100_000) / 100_000 < 0.02
    True
    """

    __slots__ = ("_c", "_batch_size")

    def __init__(self, precision: int = 14, seed: int = 0):
        self.init_batch_size()
        self._c = lib.zpds_hll_new(precision, seed)
        if self._c == ffi.NULL:
            raise MemoryError("failed to allocate HyperLogLog")

    def add(self, item) -> None:
        data = as_bytes(item)
        lib.zpds_hll_add(self._c, data, len(data))

    def add_many(self, items) -> None:
        """Add many items with a single FFI crossing per batch.

        ``items`` may be a numpy array (zero-copy for fixed-width dtypes), a
        list, or any iterable/generator (consumed in ``batch_size`` chunks).
        """
        for_each_batch(
            items,
            self._batch_size,
            lambda b, o, w, n: lib.zpds_hll_add_many(self._c, b, o, w, n),
        )

    # Back-compat alias for the pre-batch API.
    update = add_many

    def estimate(self) -> float:
        """Raw (floating-point) cardinality estimate."""
        return lib.zpds_hll_estimate(self._c)

    def __len__(self) -> int:
        """Rounded cardinality estimate."""
        return lib.zpds_hll_count(self._c)

    @property
    def num_registers(self) -> int:
        return lib.zpds_hll_size(self._c)

    @property
    def relative_error(self) -> float:
        """Expected standard relative error, ~1.04/sqrt(m)."""
        return lib.zpds_hll_error(self._c)

    def merge(self, other: "HyperLogLog") -> None:
        """Union another sketch into this one (requires equal precision)."""
        if not isinstance(other, HyperLogLog):
            raise TypeError("can only merge with another HyperLogLog")
        if not lib.zpds_hll_merge(self._c, other._c):
            raise ValueError("cannot merge HyperLogLogs of different precision")

    def clear(self) -> None:
        lib.zpds_hll_clear(self._c)

    def __repr__(self) -> str:
        return f"HyperLogLog(registers={self.num_registers}, estimate={len(self)})"

    def __del__(self):
        c = getattr(self, "_c", None)
        if c is not None and c != ffi.NULL:
            lib.zpds_hll_free(c)
            self._c = ffi.NULL
