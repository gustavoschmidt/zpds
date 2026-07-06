# zpds

Fast probabilistic data structures for Python — Bloom filter, Cuckoo filter,
HyperLogLog, and Count-Min Sketch — powered by a native Zig core.

It targets high-throughput streaming workloads: deduplication, cardinality
counting, and frequency tracking over millions of items. On an ordinary laptop
it inserts **~90M items/s** into a Bloom filter from a NumPy array — ~100× a
pure-Python implementation ([benchmarks](benchmarks/README.md)).

- **Bloom filter** — membership, tunable false-positive rate.
- **Cuckoo filter** — membership with deletion support.
- **HyperLogLog** — cardinality estimation.
- **Count-Min Sketch** — frequency estimation.

## Install

```sh
pip install zpds          # prebuilt wheel, no Zig toolchain needed
pip install zpds[numpy]   # add the zero-copy NumPy batch fast path
```

Wheels for Linux, macOS, and Windows; any Python ≥ 3.9, no compiler needed.

## Quick start

```python
from zpds import BloomFilter, CuckooFilter, HyperLogLog, CountMinSketch

bf = BloomFilter(capacity=1_000_000, error_rate=0.01)
bf.add("alice")
"alice" in bf                       # True

cf = CuckooFilter(capacity=1_000_000)
cf.add("bob")
cf.remove("bob")                    # deletion, unlike a Bloom filter

hll = HyperLogLog(precision=14)
hll.add_many(f"user-{i}" for i in range(1_000_000))
len(hll)                            # ≈ 1_000_000, ~0.8% error

cms = CountMinSketch(epsilon=0.001, delta=0.001)
cms.add("apple", 3)
cms.estimate("apple")               # ≥ 3, never an underestimate
```

Each object owns a native buffer freed on garbage collection; use it as a
context manager (`with BloomFilter(...) as bf:`) or call `.close()` for
deterministic cleanup. The mergeable sketches copy and combine as values:

```python
merged = hll_a | hll_b              # union (HyperLogLog); hll_a |= hll_b in place
totals = cms_a + cms_b             # counter-wise sum (Count-Min); cms_a += cms_b
snapshot = bf.copy()               # independent copy of any structure
```

## Batch & NumPy API

Every structure has `*_many` methods that cross the Python↔native boundary
once per batch instead of once per item. A contiguous fixed-width NumPy array
is passed zero-copy — this is where the ~100× speedups come from.

```python
import numpy as np

bf = BloomFilter(capacity=1_000_000)
bf.add_many(np.arange(1_000_000, dtype=np.uint64))              # zero-copy
mask = bf.contains_many(np.array([1, 2, 3], dtype=np.uint64))   # bool array

bf.add_many(open("keys.txt"))       # any iterable streams in bounded memory
bf.add_many(stream, batch_size=50_000)   # tune the per-crossing chunk size

cf = CuckooFilter(capacity=1_000)
failed = cf.add_many(keys, return_failed=True)   # indices that didn't fit

cms = CountMinSketch(epsilon=1e-3, delta=1e-3)
cms.add_many(words, counts)         # optional per-item counts
freqs = cms.estimate_many(words)    # list[int] or np.uint64 array
```

For non-numeric NumPy dtypes (`S`/`V`), the full fixed-width field is hashed
(including NUL padding), so add and query with the same representation.

## Performance

Two levers compound: the native core, and the batch APIs that cross the FFI
boundary once per *workload* instead of once per *item*. A contiguous NumPy
array takes the zero-copy path — its buffer goes straight to Zig with no
per-item Python work.

| Workload          | pure-Python  | zpds single-item | zpds batch (NumPy) |
|-------------------|--------------|------------------|--------------------|
| Bloom — add       | 0.83 M ops/s | 4.5 M ops/s (5×) | 89 M ops/s (108×)  |
| Bloom — query     | 0.87 M ops/s | 4.0 M ops/s (5×) | 101 M ops/s (117×) |
| HyperLogLog — add | 1.5 M ops/s  | 4.5 M ops/s (3×) | 255 M ops/s (167×) |
| Count-Min — add   | 1.0 M ops/s  | 4.2 M ops/s (4×) | 95 M ops/s (92×)   |

Speedups in parentheses are relative to the pure-Python baseline. Apple
Silicon, CPython 3.12, `ReleaseSafe`, 200k items — see
[benchmarks](benchmarks/README.md) for the harness and full tables (including
the `add_many(list)` middle path).

## Building from source

Requires [Zig](https://ziglang.org/) `0.16.0` or newer.

```sh
zig build          # builds zig-out/lib/libzpds.{dylib,so,a}
zig build test     # runs the Zig unit + statistical test suite
```

To build a wheel from source (compiles the Zig core and bundles it):

```sh
pip install build ziglang==0.16.0
python -m build --wheel
```

## Development

The Zig core lives in `zig/` (C ABI in `zig/ffi.zig`, header in
`zig/include/zpds.h`), the Python package in `python/zpds/`, tests in `tests/`,
and benchmarks in `benchmarks/`. The bindings load a prebuilt `libzpds` shared
library via `cffi` in ABI (`dlopen`) mode — one `py3-none-<platform>` wheel per
platform serves every supported Python 3.

```sh
zig build          # produces zig-out/lib/libzpds.{so,dylib}
pytest             # loads the library from zig-out/lib (dev fallback)
```

Set `ZPDS_LIBRARY_PATH` to point the bindings at a specific shared library.

## License

MIT — see [LICENSE](LICENSE).
