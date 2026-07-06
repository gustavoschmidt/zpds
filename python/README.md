# zpds (Python)

Python bindings for [zpds](../README.md) — fast probabilistic data structures
backed by a native Zig core.

```python
from zpds import BloomFilter, CuckooFilter, HyperLogLog, CountMinSketch

bf = BloomFilter(capacity=1_000_000, error_rate=0.01)
bf.add("alice")
assert "alice" in bf

cf = CuckooFilter(capacity=1_000_000)
cf.add("bob")
cf.remove("bob")           # deletion, unlike a Bloom filter

hll = HyperLogLog(precision=14)
hll.update(f"user-{i}" for i in range(1_000_000))
len(hll)                   # ≈ distinct count, ~0.8% error

cms = CountMinSketch(epsilon=0.001, delta=0.001)
cms.add("apple", 3)
cms.estimate("apple")      # ≥ 3, never an underestimate
```

## Batch & numpy API

Every structure has `*_many` methods that cross the FFI boundary once per
*batch* rather than once per *item*. They accept a list, any iterable/generator
(streamed in `batch_size` chunks, so memory stays bounded), or a numpy array —
and a fixed-width numpy array is passed **zero-copy** for the biggest speedup
(~20× over single-item `add`; see `benchmarks/`).

```python
import numpy as np

bf = BloomFilter(capacity=1_000_000)
bf.add_many(np.arange(1_000_000, dtype=np.uint64))   # zero-copy, one crossing
mask = bf.contains_many(np.array([1, 2, 3], dtype=np.uint64))  # -> np.bool_ array

bf.batch_size = 50_000                     # tune the chunk size for streams
bf.add_many(open("keys.txt"))              # stream a file, bounded memory

cf = CuckooFilter(capacity=1_000_000)
inserted = cf.add_many(keys)               # returns how many actually fit
removed = cf.remove_many(keys[:100])

cms = CountMinSketch(epsilon=1e-3, delta=1e-3)
cms.add_many(words, counts)                # optional per-item counts
freqs = cms.estimate_many(words)           # list[int] or np.uint64 array
```

For non-numeric numpy dtypes (`S`/`V`), the full fixed-width field is hashed
(including NUL padding), so add and query with the same representation.

## Development

The bindings load a prebuilt `libzpds` shared library via `cffi` in ABI mode.
Build the core first, then point the tests at it (or rely on the `zig-out/lib`
dev fallback):

```sh
cd ..            # repo root
zig build        # produces zig-out/lib/libzpds.{so,dylib}

cd python
pip install cffi pytest
pytest           # loads the library from ../zig-out/lib
```

Set `ZPDS_LIBRARY_PATH` to override which shared library is loaded.
