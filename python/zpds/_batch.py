"""Batch plumbing shared by all four structures.

The whole point of the ``*_many`` API is to cross the FFI boundary once per
*batch* instead of once per *item* — the per-item loop then runs in native
code. This module turns arbitrary Python inputs into the ``(blob, offsets,
item_width, n)`` shape the C ABI expects, choosing the cheapest route:

- **numpy fast path** — a C-contiguous array with a fixed-width, byte-meaningful
  dtype (integers, floats, bool, raw bytes ``S``/``V``) is passed *zero-copy*:
  the array's own buffer becomes ``blob`` and ``item_width`` is its itemsize, so
  no per-item Python work happens at all.
- **generic / streaming path** — any other iterable (list, generator, ...) is
  consumed in chunks of ``batch_size`` and each chunk is packed into a
  concatenated blob + offsets array. Streaming keeps memory bounded: only one
  chunk is materialized at a time.

Note on fixed-width numpy strings: an ``S8`` field hashes all 8 bytes including
NUL padding, so ``b"abc"`` stored in ``S8`` is a different key than the Python
value ``b"abc"``. Add and query using the same representation.
"""

from __future__ import annotations

from itertools import islice

from ._native import as_bytes, ffi, lib

DEFAULT_BATCH_SIZE = 8192

# numpy dtype kinds whose raw bytes we can hash directly (fixed-width, every
# byte meaningful). 'U' (4-byte unicode) and 'O' (object) go the generic route.
_FAST_KINDS = frozenset("iufbSV")


def _numpy():
    try:
        import numpy as np
    except ImportError:
        return None
    return np


def numpy_fast_view(obj):
    """Return ``(cdata, item_width, n)`` for a zero-copy view of a numpy array,
    or ``None`` if ``obj`` is not a fast-pathable array."""
    np = _numpy()
    if np is None or not isinstance(obj, np.ndarray):
        return None
    if obj.dtype.kind not in _FAST_KINDS:
        return None
    if obj.ndim == 1:
        item_width = obj.dtype.itemsize
        n = obj.shape[0]
    elif obj.ndim == 2 and obj.dtype.itemsize == 1:
        item_width = obj.shape[1]
        n = obj.shape[0]
    else:
        return None
    arr = np.ascontiguousarray(obj)
    # from_buffer keeps `arr` alive for the lifetime of the returned cdata.
    buf = ffi.from_buffer("uint8_t[]", arr) if arr.nbytes else ffi.NULL
    return buf, item_width, n


def iter_chunks(iterable, size):
    """Yield successive lists of up to ``size`` items from ``iterable``."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def pack_chunk(items):
    """Pack a list of str/bytes-like items into ``(blob, offsets_cdata, n)``."""
    encoded = [as_bytes(x) for x in items]
    blob = b"".join(encoded)
    offsets = [0]
    total = 0
    for e in encoded:
        total += len(e)
        offsets.append(total)
    # A non-null blob pointer is required whenever n > 0 (even if every item is
    # empty and the join produced b"").
    blob_arg = blob if blob else b"\x00"
    return blob_arg, ffi.new("size_t[]", offsets), len(encoded)


def for_each_batch(items, batch_size, fn):
    """Invoke ``fn(blob, offsets, item_width, n)`` once for a numpy fast view,
    or once per packed chunk of a generic iterable."""
    view = numpy_fast_view(items)
    if view is not None:
        buf, width, n = view
        fn(buf, ffi.NULL, width, n)
        return
    for chunk in iter_chunks(items, batch_size):
        blob, offsets, n = pack_chunk(chunk)
        fn(blob, offsets, 0, n)


def _to_results(out, n, ctype, as_numpy):
    """Convert a filled output buffer to a numpy array (numpy input) or a list."""
    np = _numpy()
    if ctype == "uint8_t":
        if as_numpy and np is not None:
            return np.frombuffer(ffi.buffer(out, n), dtype=np.uint8).astype(bool) if n else np.empty(0, bool)
        return [out[i] != 0 for i in range(n)]
    # uint64_t
    if as_numpy and np is not None:
        return np.frombuffer(ffi.buffer(out, n * 8), dtype=np.uint64).copy() if n else np.empty(0, np.uint64)
    return [int(out[i]) for i in range(n)]


def query_batches(items, batch_size, ctype, fn):
    """Run a per-item query op across a batch, returning a numpy array for numpy
    inputs or a list otherwise. ``fn(blob, offsets, item_width, n, out)`` fills
    ``out[0:n]``."""
    view = numpy_fast_view(items)
    if view is not None:
        buf, width, n = view
        out = ffi.new(ctype + "[]", max(n, 1))
        fn(buf, ffi.NULL, width, n, out)
        return _to_results(out, n, ctype, as_numpy=True)

    results = []
    for chunk in iter_chunks(items, batch_size):
        blob, offsets, n = pack_chunk(chunk)
        out = ffi.new(ctype + "[]", max(n, 1))
        fn(blob, offsets, 0, n, out)
        results.extend(_to_results(out, n, ctype, as_numpy=False))
    return results


class BatchSizeMixin:
    """Adds a configurable ``batch_size`` used when chunking generic iterables.

    Concrete classes must include ``"_batch_size"`` in their ``__slots__`` and
    initialize it (see ``BatchSizeMixin.init_batch_size``).
    """

    __slots__ = ()

    def init_batch_size(self):
        self._batch_size = DEFAULT_BATCH_SIZE

    @property
    def batch_size(self) -> int:
        """Number of items packed per FFI crossing for generic iterables.

        Ignored on the numpy fast path, where the whole array crosses at once.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError("batch_size must be positive")
        self._batch_size = int(value)
