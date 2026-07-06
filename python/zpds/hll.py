"""HyperLogLog — distinct-count (cardinality) estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._batch import DEFAULT_BATCH_SIZE, Item, Items, check_batch_size, for_each_batch
from ._native import as_bytes, ffi, lib

if TYPE_CHECKING:
    from types import TracebackType


class HyperLogLog:
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

    __slots__ = ("_c",)

    def __init__(self, precision: int = 14, seed: int = 0):
        self._c = lib.zpds_hll_new(precision, seed)
        if self._c == ffi.NULL:
            raise MemoryError("failed to allocate HyperLogLog")

    def add(self, item: Item) -> None:
        data = as_bytes(item)
        lib.zpds_hll_add(self._c, data, len(data))

    def add_many(self, items: Items, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        """Add many items with a single FFI crossing per batch.

        ``items`` may be a numpy array (zero-copy for fixed-width dtypes), a
        list, or any iterable/generator. ``batch_size`` sets how many items are
        packed per crossing for the generic/streaming path; it is ignored on the
        numpy fast path.
        """
        for_each_batch(
            items,
            check_batch_size(batch_size),
            lambda b, o, w, n: lib.zpds_hll_add_many(self._c, b, o, w, n),
        )

    def estimate(self) -> float:
        """Raw (floating-point) cardinality estimate."""
        return lib.zpds_hll_estimate(self._c)

    @property
    def cardinality(self) -> int:
        """Rounded distinct-count estimate (the integer form of ``estimate``)."""
        return lib.zpds_hll_count(self._c)

    def __len__(self) -> int:
        """Rounded cardinality estimate (same as :attr:`cardinality`)."""
        return lib.zpds_hll_count(self._c)

    @property
    def num_registers(self) -> int:
        return lib.zpds_hll_size(self._c)

    @property
    def relative_error(self) -> float:
        """Expected standard relative error, ~1.04/sqrt(m)."""
        return lib.zpds_hll_error(self._c)

    def merge(self, other: "HyperLogLog") -> None:
        """Union another sketch into this one in place (requires equal precision)."""
        if not isinstance(other, HyperLogLog):
            raise TypeError("can only merge with another HyperLogLog")
        if not lib.zpds_hll_merge(self._c, other._c):
            raise ValueError("cannot merge HyperLogLogs of different precision")

    def copy(self) -> "HyperLogLog":
        """Return an independent copy of this sketch."""
        obj = HyperLogLog.__new__(HyperLogLog)
        obj._c = lib.zpds_hll_clone(self._c)
        if obj._c == ffi.NULL:
            raise MemoryError("failed to copy HyperLogLog")
        return obj

    def __ior__(self, other: "HyperLogLog") -> "HyperLogLog":
        """``a |= b`` — union ``b`` into ``a`` in place."""
        self.merge(other)
        return self

    def __or__(self, other: "HyperLogLog") -> "HyperLogLog":
        """``a | b`` — a new sketch that is the union of ``a`` and ``b``."""
        result = self.copy()
        result.merge(other)
        return result

    def clear(self) -> None:
        lib.zpds_hll_clear(self._c)

    def close(self) -> None:
        """Free the native sketch. Safe to call more than once."""
        c = getattr(self, "_c", None)
        if c is not None and c != ffi.NULL:
            lib.zpds_hll_free(c)
            self._c = ffi.NULL

    def __enter__(self) -> "HyperLogLog":
        return self

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"HyperLogLog(registers={self.num_registers}, estimate={self.cardinality})"

    def __del__(self):
        self.close()
