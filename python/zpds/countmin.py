"""Count-Min Sketch — frequency estimation over a stream."""

from __future__ import annotations

from itertools import islice

from ._batch import (
    BatchSizeMixin,
    for_each_batch,
    iter_chunks,
    numpy_fast_view,
    pack_chunk,
    query_batches,
    _numpy,
)
from ._native import as_bytes, ffi, lib


class CountMinSketch(BatchSizeMixin):
    """Estimate per-item frequencies in sublinear space.

    Estimates never *under*count. With the default sizing the overestimate is
    at most ``epsilon * total_count`` with probability ``1 - delta``.

    Provide either ``epsilon``/``delta`` (the sketch is sized for you) or an
    explicit ``width``/``depth``.

    >>> cms = CountMinSketch(epsilon=0.001, delta=0.001)
    >>> cms.add("apple", 3)
    >>> cms.estimate("apple")
    3
    """

    __slots__ = ("_c", "_batch_size")

    def __init__(
        self,
        epsilon: float = 0.001,
        delta: float = 0.001,
        seed: int = 0,
        *,
        width: int | None = None,
        depth: int | None = None,
    ):
        self.init_batch_size()
        if (width is None) != (depth is None):
            raise ValueError("specify both width and depth, or neither")
        if width is not None:
            if width <= 0 or depth <= 0:
                raise ValueError("width and depth must be positive")
            self._c = lib.zpds_countmin_new_with_params(width, depth, seed)
        else:
            if not 0.0 < epsilon < 1.0 or not 0.0 < delta < 1.0:
                raise ValueError("epsilon and delta must be in (0, 1)")
            self._c = lib.zpds_countmin_new(epsilon, delta, seed)
        if self._c == ffi.NULL:
            raise MemoryError("failed to allocate Count-Min sketch")

    def add(self, item, count: int = 1) -> None:
        """Add ``count`` occurrences of ``item``."""
        data = as_bytes(item)
        lib.zpds_countmin_add(self._c, data, len(data), count)

    def add_many(self, items, counts=None) -> None:
        """Add many items with a single FFI crossing per batch.

        ``counts`` is optional: ``None`` adds each item once; otherwise it is a
        sequence (or numpy uint64 array) of per-item increments aligned with
        ``items``. ``items`` may be a numpy array (zero-copy for fixed-width
        dtypes when ``counts`` is None or a numpy array) or any iterable.
        """
        if counts is None:
            for_each_batch(
                items,
                self._batch_size,
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
            chunk = list(islice(it_items, self._batch_size))
            if not chunk:
                break
            cchunk = list(islice(it_counts, len(chunk)))
            if len(cchunk) != len(chunk):
                raise ValueError("counts must be the same length as items")
            blob, offsets, n = pack_chunk(chunk)
            counts_arr = ffi.new("uint64_t[]", cchunk)
            lib.zpds_countmin_add_many(self._c, blob, offsets, 0, n, counts_arr)

    def estimate(self, item) -> int:
        """Estimated frequency of ``item`` (never an underestimate)."""
        data = as_bytes(item)
        return lib.zpds_countmin_estimate(self._c, data, len(data))

    def estimate_many(self, items):
        """Estimate the frequency of many items at once. Returns a numpy uint64
        array for numpy input, otherwise a ``list[int]`` aligned with ``items``."""
        return query_batches(
            items,
            self._batch_size,
            "uint64_t",
            lambda b, o, w, n, out: lib.zpds_countmin_estimate_many(self._c, b, o, w, n, out),
        )

    def merge(self, other: "CountMinSketch") -> None:
        """Add another sketch counter-wise (requires equal shape and seed)."""
        if not isinstance(other, CountMinSketch):
            raise TypeError("can only merge with another CountMinSketch")
        if not lib.zpds_countmin_merge(self._c, other._c):
            raise ValueError("cannot merge sketches of different width/depth/seed")

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

    def __repr__(self) -> str:
        return f"CountMinSketch(width={self.width}, depth={self.depth}, total={self.total})"

    def __del__(self):
        c = getattr(self, "_c", None)
        if c is not None and c != ffi.NULL:
            lib.zpds_countmin_free(c)
            self._c = ffi.NULL
