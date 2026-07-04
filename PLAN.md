# zpds — Zig Probabilistic Data Structures

A fast, reliable library of **probabilistic data structures** for Python, backed by a
native **Zig** core.

## Motivation

Existing Python options (`pybloom`, much of `datasketch`) are pure-Python and slow.
The algorithms have real depth, but the API boundary is trivial
(`add(bytes)`, `contains(bytes) -> bool`, `count() -> int`) — an ideal Zig fit.
Correctness is statistically testable against theoretical error bounds.

## Structures

- **Bloom filter** — membership, tunable false-positive rate.
- **Cuckoo filter** — membership with deletion support.
- **HyperLogLog** — cardinality estimation.
- **Count-Min Sketch** — frequency estimation.

## Target niche

**Streaming / single-item / low-latency CPU** use: dedup in a stream, membership
checks in a request path. This is the regime where a tight CPU core wins and a GPU
is the wrong tool — a PCIe round-trip (~µs) dwarfs a CPU probe (~tens of ns), and the
random-access, low-compute profile suits the CPU.

### GPU is not a threat here

Verified against datasketch docs (2026-07): datasketch's GPU support is **experimental,
MinHash-only**, and only accelerates the `update_batch()` permutation-apply +
min-reduction step via CuPy/CUDA. It is off by default (`gpu_mode='disable'`), and their
own docs note transfer overhead makes small batches CPU-favorable. There is no GPU path
for Bloom / HLL / Count-Min. GPU only wins the batch regime, not this library's niche.

## Architecture

- **Core:** Zig (`src/`), exposed over a **C ABI** (`export fn ...`).
  - Keep the boundary simple: bytes / ints / bools across the FFI line.
  - Needs a good **non-cryptographic hash** (xxHash / wyHash).
- **Python wrapper:** start with **`cffi`/`ctypes`**; graduate to a CPython extension later.
- **Testing:** measured false-positive / error rates checked against theoretical bounds;
  benchmark vs pure-Python implementations.

## Milestones

1. `build.zig` + Zig core skeleton with a C-ABI entry point.
2. Non-crypto hash (xxHash / wyHash) + test vectors.
3. Bloom filter + statistical FP-rate tests.
4. HyperLogLog + cardinality-accuracy tests.
5. Cuckoo filter and Count-Min Sketch.
6. Python `cffi` wrapper + pytest suite.
7. Benchmarks vs pure-Python libraries.
8. **Wheel packaging** (`cibuildwheel`: manylinux / macOS / Windows).

## Notes

- Make-or-break is **wheels**: prebuilt binaries so `pip install` just works.
