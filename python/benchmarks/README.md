# Benchmarks

`zpds` vs dependency-free pure-Python implementations of the same structures
(and optional `pybloom_live` / `datasketch`).

## Running

```sh
zig build -Doptimize=ReleaseSafe   # optimized native core (important!)
python benchmarks/bench.py         # needs cffi; numpy/pybloom_live/datasketch optional
```

The always-available baseline is the pure-Python code in `pure_python.py` —
there is no stdlib Bloom/HLL/Count-Min to compare against. It is not a straw
man: it hashes with BLAKE2b (native C), so the speedups below come from the
architecture, not a slow baseline hash.

## Sample results

Apple Silicon (macOS), CPython 3.12, Zig 0.16 `ReleaseSafe`, zpds 0.1.0,
200k items. Numbers vary by machine; the relative speedups are the point.

### Bloom filter — add

| Implementation        | Rate          | vs pure-Python |
|-----------------------|---------------|----------------|
| zpds (batch, numpy)   | 89 M ops/s    | **~108×**      |
| zpds (batch, list)    | 6.7 M ops/s   | ~8×            |
| zpds (scalar)         | 4.5 M ops/s   | ~5×            |
| pure-python           | 0.83 M ops/s  | 1.0×           |

### Bloom filter — query

| Implementation        | Rate          | vs pure-Python |
|-----------------------|---------------|----------------|
| zpds (batch, numpy)   | 101 M ops/s   | **~117×**      |
| zpds (batch, list)    | 5.8 M ops/s   | ~7×            |
| zpds (scalar)         | 4.0 M ops/s   | ~5×            |
| pure-python           | 0.87 M ops/s  | 1.0×           |

### HyperLogLog — add

| Implementation        | Rate          | vs pure-Python |
|-----------------------|---------------|----------------|
| zpds (batch, numpy)   | 255 M ops/s   | **~167×**      |
| zpds (batch, list)    | 7.0 M ops/s   | ~4.6×          |
| zpds (scalar)         | 4.5 M ops/s   | ~2.9×          |
| pure-python           | 1.5 M ops/s   | 1.0×           |

### Count-Min Sketch — add

| Implementation        | Rate          | vs pure-Python |
|-----------------------|---------------|----------------|
| zpds (batch, numpy)   | 95 M ops/s    | **~92×**       |
| zpds (batch, list)    | 6.8 M ops/s   | ~6.6×          |
| zpds (scalar)         | 4.2 M ops/s   | ~4.1×          |
| pure-python           | 1.0 M ops/s   | 1.0×           |

(`estimate_many` mirrors these numbers for the query side.)

## Takeaway

Two levers compound:

1. **The native core** does the hashing and bit/register work in Zig instead of
   pure Python.
2. **Batch APIs** cross the Python↔C boundary once per *workload* instead of
   once per *item*.

Scalar single-item calls are dominated by the per-call FFI crossing (~100–200
ns), so they plateau around 4–5 M ops/s regardless of structure and buy a solid
3–5×. `add_many(list)` still pays per-item Python packing (encode each `str`,
`join`, build offsets), so it roughly trades FFI cost for packing cost — a
modest extra win. The design pays off on the **numpy zero-copy path**: a
contiguous fixed-width array's buffer is handed straight to Zig with *no*
per-item Python work, and throughput jumps ~2 orders of magnitude (tens to
hundreds of millions of ops/s) — the native core's real speed, finally
unmasked.

The one regime batching can't help — events arriving genuinely one at a time —
is what the future CPython extension targets, by removing the `cffi` trampoline
from the scalar path itself.
