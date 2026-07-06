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

## Batch API — amortizing the crossing

`add_many` / `contains_many` / `estimate_many` make one FFI crossing per *batch*
instead of per *item*, so the per-item loop runs in native code. Representative
run (`--n 500000`):

| strategy            | ops/sec | speedup vs single-item |
| ------------------- | ------: | ---------------------: |
| single-item `add`   |   ~4.4M |                   1.0x |
| `add_many(list)`    |   ~4.9M |                  ~1.1x |
| `add_many(numpy)`   |  ~87M   |                 ~20x   |

The gap between the two batch rows *is* the FFI-overhead story:

- `add_many(list)` still does per-item Python work (encode each `str`, `join`,
  build the offsets array), so it roughly trades FFI cost for packing cost — a
  modest win.
- `add_many(numpy)` on a fixed-width array is **zero-copy**: the array's buffer
  is handed straight to Zig, no per-item Python work at all. Now the crossing is
  fully amortized and you see the native core's real throughput (~20x).

Takeaway: reach for the numpy path (or a `bytes`/2-D `uint8` buffer) when you
have contiguous fixed-width keys; use `add_many` on an iterable/generator to
stream large inputs in `batch_size` chunks with bounded memory.

## Remaining follow-up

A **CPython extension** would replace the `cffi` ABI shim with a direct C call,
lifting the ~4–5M ops/sec single-item ceiling itself — the one regime batching
can't help, where events genuinely arrive one at a time.

Third-party libraries (`pybloom_live`, `datasketch`) are folded into the
comparison automatically if they are importable.
