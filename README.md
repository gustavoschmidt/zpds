# zpds — Zig Probabilistic Data Structures

A fast, reliable library of **probabilistic data structures** for Python, backed
by a native **Zig** core.

- **Bloom filter** — membership, tunable false-positive rate.
- **Cuckoo filter** — membership with deletion support.
- **HyperLogLog** — cardinality estimation.
- **Count-Min Sketch** — frequency estimation.

The Zig core is exposed over a small C ABI (`include/zpds.h`); a Python wrapper
sits on top. The target niche is streaming / single-item / low-latency CPU work
where a tight CPU core wins. On an ordinary laptop it inserts **~90M items/s**
into a Bloom filter from a NumPy array — ~100× a pure-Python implementation
([benchmarks](python/benchmarks/README.md)). See [`PLAN.md`](PLAN.md) for the
full design.

## Install (Python)

Prebuilt wheels bundle the native library, so `pip install` just works — no Zig
toolchain required at install time:

```sh
pip install zpds
```

Wheels are produced for manylinux, macOS (arm64 + x86_64) and Windows by
`cibuildwheel` (see `.github/workflows/wheels.yml`). Because the bindings use
cffi's `dlopen` mode rather than the CPython ABI, one `py3-none-<platform>`
wheel per platform serves every supported Python 3.

## Performance

See [`python/benchmarks/`](python/benchmarks/README.md) for the harness and full
tables. Representative (Apple Silicon, `ReleaseSafe`, 200k items, vs a
dependency-free pure-Python baseline):

| Workload                     | zpds (batch, numpy) | zpds (scalar) | vs pure-Python |
|------------------------------|---------------------|---------------|----------------|
| Bloom — add                  | 89 M ops/s          | 4.5 M ops/s   | ~108× / ~5×    |
| Bloom — query                | 101 M ops/s         | 4.0 M ops/s   | ~117× / ~5×    |
| HyperLogLog — add            | 255 M ops/s         | 4.5 M ops/s   | ~167× / ~3×    |
| Count-Min — add              | 95 M ops/s          | 4.2 M ops/s   | ~92× / ~4×     |

Two levers compound: the native core, and **batch APIs** (`add_many` /
`contains_many` / `estimate_many`) that cross the Python↔C boundary once per
*workload* instead of once per *item*. A contiguous NumPy array takes the
zero-copy path — its buffer goes straight to Zig with no per-item Python work.

## Building the native core

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

## Layout

| Path              | What                                             |
| ----------------- | ------------------------------------------------ |
| `src/`            | Zig core: hashes and data structures             |
| `src/ffi.zig`     | C-ABI entry points (`export fn ...`)             |
| `include/zpds.h`  | C header for the ABI                             |
| `python/`         | Python (`cffi`) wrapper and tests                |
