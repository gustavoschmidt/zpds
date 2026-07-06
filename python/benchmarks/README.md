# Benchmarks

Compares the native Zig core against dependency-free pure-Python implementations
of the same algorithms (`pure_python.py`). We measure **per-item** `add` and
query throughput, since that streaming / single-item regime is zpds's niche.

```sh
cd python
python -m benchmarks.bench --n 200000
```

Representative run (Apple M-series, `ReleaseSafe`, n = 200k):

| structure         | op    | zpds ops/sec | speedup vs pure-Python |
| ----------------- | ----- | -----------: | ---------------------: |
| Bloom filter      | add   |        ~4.6M |                  ~5.5x |
| Bloom filter      | query |        ~4.4M |                  ~4.9x |
| HyperLogLog       | add   |        ~4.4M |                  ~2.9x |
| Count-Min Sketch  | add   |        ~4.2M |                  ~4.1x |
| Count-Min Sketch  | query |        ~4.5M |                  ~5.2x |

## Reading these numbers

The per-call `cffi` FFI crossing (~100–200ns) dominates each of these tiny
operations, so the wrapper — not the algorithm — is the current ceiling: zpds
sits around 4–5M ops/sec regardless of structure. The pure-Python baselines are
not naïve either; they hash with BLAKE2b (native C), which is why the ratio is
"only" a few ×.

Two follow-ups (see `PLAN.md`) widen the gap substantially:

- a **batch API** (`add_many(iterable)`) amortizes one FFI call over many items;
- a **CPython extension** replaces the `cffi` ABI shim with a direct C call.

Third-party libraries (`pybloom_live`, `datasketch`) are folded into the
comparison automatically if they are importable.
