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
