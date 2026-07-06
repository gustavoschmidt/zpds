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
