"""Count-Min Sketch — frequency estimation over a stream."""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, Iterable, Optional

from ._batch import (
    DEFAULT_BATCH_SIZE,
    IntResults,
    Item,
    Items,
    _numpy,
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


class CountMinSketch:
    """Estimate per-item frequencies in sublinear space.

    Estimates never *under*count. With the default sizing the overestimate is
    at most ``epsilon * total`` with probability ``1 - delta``. For an explicit
    counter grid use :meth:`from_shape`.

    >>> cms = CountMinSketch(epsilon=0.001, delta=0.001)
    >>> cms.add("apple", 3)
    >>> cms.estimate("apple")
    3
    """

    __slots__ = ("_c",)

    def __init__(self, epsilon: float = 0.001, delta: float = 0.001, seed: int = 0):
        if not 0.0 < epsilon < 1.0 or not 0.0 < delta < 1.0:
            raise ValueError("epsilon and delta must be in (0, 1)")
        self._c = lib.zpds_countmin_new(epsilon, delta, seed)
        if self._c == ffi.NULL:
            raise MemoryError("failed to allocate Count-Min sketch")

    @classmethod
    def from_shape(cls, width: int, depth: int, seed: int = 0) -> "CountMinSketch":
        """Construct with an explicit counter grid (``width`` columns, ``depth``
        rows) instead of an ``epsilon``/``delta`` pair."""
        if width <= 0 or depth <= 0:
            raise ValueError("width and depth must be positive")
        obj = cls.__new__(cls)
        obj._c = lib.zpds_countmin_new_with_params(width, depth, seed)
        if obj._c == ffi.NULL:
            raise MemoryError("failed to allocate Count-Min sketch")
        return obj

    def add(self, item: Item, count: int = 1) -> None:
        """Add ``count`` occurrences of ``item``."""
        data = as_bytes(item)
        lib.zpds_countmin_add(self._c, data, len(data), count)

    def add_many(
        self,
        items: Items,
        counts: Optional[Iterable[int]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Add many items with a single FFI crossing per batch.

        ``counts`` is optional: ``None`` adds each item once; otherwise it is a
        sequence (or numpy uint64 array) of per-item increments aligned with
        ``items``. ``items`` may be a numpy array (zero-copy for fixed-width
        dtypes when ``counts`` is None or a numpy array) or any iterable.
        ``batch_size`` sets the generic/streaming chunk size.
        """
        check_batch_size(batch_size)
        if counts is None:
            for_each_batch(
                items,
                batch_size,
                lambda b, o, w, n: lib.zpds_countmin_add_many(self._c, b, o, w, n, ffi.NULL),
            )
            return

        # Per-item counts. Fast path: numpy items + numpy counts, zero-copy both.
        np = _numpy()
        view = numpy_fast_view(items)
        if view is not None and np is not None and isinstance(counts, np.ndarray):
            buf, width, n = view
            c = np.ascontiguousarray(counts, dtype=np.uint64)
            if c.shape[0] != n:
                raise ValueError("counts must be the same length as items")
            cbuf = ffi.from_buffer("uint64_t[]", c) if c.size else ffi.NULL
            lib.zpds_countmin_add_many(self._c, buf, ffi.NULL, width, n, cbuf)
            return

        # Generic path: walk items and counts together in chunks.
        it_items = iter(items)
        it_counts = iter(counts)
        while True:
            chunk = list(islice(it_items, batch_size))
            if not chunk:
                break
            cchunk = list(islice(it_counts, len(chunk)))
            if len(cchunk) != len(chunk):
                raise ValueError("counts must be the same length as items")
            blob, offsets, n = pack_chunk(chunk)
            counts_arr = ffi.new("uint64_t[]", cchunk)
            lib.zpds_countmin_add_many(self._c, blob, offsets, 0, n, counts_arr)

    def estimate(self, item: Item) -> int:
        """Estimated frequency of ``item`` (never an underestimate)."""
        data = as_bytes(item)
        return lib.zpds_countmin_estimate(self._c, data, len(data))

    def estimate_many(
        self, items: Items, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> IntResults:
        """Estimate the frequency of many items at once. Returns a numpy uint64
        array for numpy input, otherwise a ``list[int]`` aligned with ``items``."""
        return query_batches(
            items,
            check_batch_size(batch_size),
            "uint64_t",
            lambda b, o, w, n, out: lib.zpds_countmin_estimate_many(self._c, b, o, w, n, out),
        )

    def merge(self, other: "CountMinSketch") -> None:
        """Add another sketch counter-wise in place (requires equal shape and seed)."""
        if not isinstance(other, CountMinSketch):
            raise TypeError("can only merge with another CountMinSketch")
        if not lib.zpds_countmin_merge(self._c, other._c):
            raise ValueError("cannot merge sketches of different width/depth/seed")

    def copy(self) -> "CountMinSketch":
        """Return an independent copy of this sketch."""
        obj = CountMinSketch.__new__(CountMinSketch)
        obj._c = lib.zpds_countmin_clone(self._c)
        if obj._c == ffi.NULL:
            raise MemoryError("failed to copy Count-Min sketch")
        return obj

    def __iadd__(self, other: "CountMinSketch") -> "CountMinSketch":
        """``a += b`` — add ``b`` into ``a`` counter-wise, in place."""
        self.merge(other)
        return self

    def __add__(self, other: "CountMinSketch") -> "CountMinSketch":
        """``a + b`` — a new sketch that is the counter-wise sum of ``a`` and ``b``."""
        result = self.copy()
        result.merge(other)
        return result

    def clear(self) -> None:
        lib.zpds_countmin_clear(self._c)

    @property
    def width(self) -> int:
        return lib.zpds_countmin_width(self._c)

    @property
    def depth(self) -> int:
        return lib.zpds_countmin_depth(self._c)

    @property
    def total(self) -> int:
        """Sum of all increments added."""
        return lib.zpds_countmin_total(self._c)

    def close(self) -> None:
        """Free the native sketch. Safe to call more than once."""
        c = getattr(self, "_c", None)
        if c is not None and c != ffi.NULL:
            lib.zpds_countmin_free(c)
            self._c = ffi.NULL

    def __enter__(self) -> "CountMinSketch":
        return self

    def __exit__(
        self,
        exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        tb: "TracebackType | None",
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"CountMinSketch(width={self.width}, depth={self.depth}, total={self.total})"

    def __del__(self):
        self.close()
