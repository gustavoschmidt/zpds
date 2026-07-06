# zpds — Zig Probabilistic Data Structures

A fast, reliable library of **probabilistic data structures** for Python, backed
by a native **Zig** core.

- **Bloom filter** — membership, tunable false-positive rate.
- **Cuckoo filter** — membership with deletion support.
- **HyperLogLog** — cardinality estimation.
- **Count-Min Sketch** — frequency estimation.

The Zig core is exposed over a small C ABI (`include/zpds.h`); a Python wrapper
sits on top. The target niche is streaming / single-item / low-latency CPU work
where a tight CPU core wins. See [`PLAN.md`](PLAN.md) for the full design.

## Building the native core

Requires [Zig](https://ziglang.org/) `0.16.0` or newer.

```sh
zig build          # builds zig-out/lib/libzpds.{dylib,so,a}
zig build test     # runs the Zig unit + statistical test suite
```

## Layout

| Path              | What                                             |
| ----------------- | ------------------------------------------------ |
| `src/`            | Zig core: hashes and data structures             |
| `src/ffi.zig`     | C-ABI entry points (`export fn ...`)             |
| `include/zpds.h`  | C header for the ABI                             |
| `python/`         | Python (`cffi`) wrapper and tests                |
