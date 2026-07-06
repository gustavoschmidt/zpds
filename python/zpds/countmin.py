"""Count-Min Sketch — frequency estimation over a stream."""

from __future__ import annotations

from ._native import as_bytes, ffi, lib


class CountMinSketch:
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

    __slots__ = ("_c",)

    def __init__(
        self,
        epsilon: float = 0.001,
        delta: float = 0.001,
        seed: int = 0,
        *,
        width: int | None = None,
        depth: int | None = None,
    ):
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

    def estimate(self, item) -> int:
        """Estimated frequency of ``item`` (never an underestimate)."""
        data = as_bytes(item)
        return lib.zpds_countmin_estimate(self._c, data, len(data))

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
