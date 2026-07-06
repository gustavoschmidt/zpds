"""Benchmark zpds against pure-Python implementations of the same structures.

Run with an optimized native core for a fair comparison::

    zig build -Doptimize=ReleaseSafe
    python benchmarks/bench.py            # needs cffi; numpy/pybloom_live/datasketch optional

The always-available baseline is the dependency-free pure-Python code in
``pure_python.py`` (there is no stdlib Bloom/HLL/Count-Min to compare against).
``pybloom_live`` (Bloom) and ``datasketch`` (HyperLogLog) are folded in when
importable; ``numpy`` unlocks the zero-copy batch row.
"""

from __future__ import annotations

import os
import sys
from time import perf_counter

# Allow running as a plain script (`python benchmarks/bench.py`) or via importlib.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # for `import pure_python`
sys.path.insert(0, os.path.dirname(_HERE))  # python/ dir, for `import zpds`

import zpds  # noqa: E402
from pure_python import PyBloomFilter, PyCountMinSketch, PyHyperLogLog  # noqa: E402

try:
    import numpy as np

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_NUMPY = False

try:
    from pybloom_live import BloomFilter as PyBloomLive

    HAVE_PYBLOOM = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_PYBLOOM = False

try:
    from datasketch import HyperLogLog as DsHyperLogLog

    HAVE_DATASKETCH = True
except ImportError:  # pragma: no cover - optional dependency
    HAVE_DATASKETCH = False


def _rate(fn, workload, n=None) -> float:
    """Return operations/second for ``fn`` applied over ``workload``."""
    n = len(workload) if n is None else n
    t0 = perf_counter()
    fn(workload)
    elapsed = perf_counter() - t0
    return n / elapsed if elapsed > 0 else float("inf")


# --- benchmarks ------------------------------------------------------------


def bench_bloom_add(n=200_000):
    keys = [f"item-{i}".encode() for i in range(n)]

    z1 = zpds.BloomFilter(capacity=n, error_rate=0.01)
    z2 = zpds.BloomFilter(capacity=n, error_rate=0.01)
    results = {
        "zpds (batch, list)": _rate(z2.add_many, keys),
        "zpds (scalar)": _rate(lambda w: [z1.add(k) for k in w], keys),
    }
    if HAVE_NUMPY:
        npk = np.arange(n, dtype=np.uint64)
        z3 = zpds.BloomFilter(capacity=n, error_rate=0.01)
        results["zpds (batch, numpy)"] = _rate(z3.add_many, npk, n=n)

    p = PyBloomFilter(capacity=n, error_rate=0.01)
    results["pure-python"] = _rate(lambda w: [p.add(k) for k in w], keys)
    if HAVE_PYBLOOM:
        try:
            pbl = PyBloomLive(capacity=n, error_rate=0.01)
            results["pybloom_live"] = _rate(lambda w: [pbl.add(k) for k in w], keys)
        except Exception:  # pragma: no cover - defensive
            pass
    return results


def bench_bloom_query(n=200_000):
    keys = [f"item-{i}".encode() for i in range(n)]
    z = zpds.BloomFilter(capacity=n, error_rate=0.01)
    z.add_many(keys)
    results = {
        "zpds (batch, list)": _rate(z.contains_many, keys),
        "zpds (scalar)": _rate(lambda w: [k in z for k in w], keys),
    }
    if HAVE_NUMPY:
        npk = np.arange(n, dtype=np.uint64)
        zn = zpds.BloomFilter(capacity=n, error_rate=0.01)
        zn.add_many(npk)
        results["zpds (batch, numpy)"] = _rate(zn.contains_many, npk, n=n)

    p = PyBloomFilter(capacity=n, error_rate=0.01)
    for k in keys:
        p.add(k)
    results["pure-python"] = _rate(lambda w: [k in p for k in w], keys)
    return results


def bench_hll_add(n=200_000):
    keys = [f"item-{i}".encode() for i in range(n)]

    z1 = zpds.HyperLogLog(precision=14)
    z2 = zpds.HyperLogLog(precision=14)
    results = {
        "zpds (batch, list)": _rate(z2.add_many, keys),
        "zpds (scalar)": _rate(lambda w: [z1.add(k) for k in w], keys),
    }
    if HAVE_NUMPY:
        npk = np.arange(n, dtype=np.uint64)
        z3 = zpds.HyperLogLog(precision=14)
        results["zpds (batch, numpy)"] = _rate(z3.add_many, npk, n=n)

    p = PyHyperLogLog(precision=14)
    results["pure-python"] = _rate(lambda w: [p.add(k) for k in w], keys)
    if HAVE_DATASKETCH:
        try:
            ds = DsHyperLogLog(p=14)
            results["datasketch"] = _rate(lambda w: [ds.update(k) for k in w], keys)
        except Exception:  # pragma: no cover - defensive
            pass
    return results


def bench_countmin_add(n=200_000):
    keys = [f"item-{i}".encode() for i in range(n)]

    z1 = zpds.CountMinSketch(epsilon=0.001, delta=0.001)
    z2 = zpds.CountMinSketch(epsilon=0.001, delta=0.001)
    results = {
        "zpds (batch, list)": _rate(z2.add_many, keys),
        "zpds (scalar)": _rate(lambda w: [z1.add(k) for k in w], keys),
    }
    if HAVE_NUMPY:
        npk = np.arange(n, dtype=np.uint64)
        z3 = zpds.CountMinSketch(epsilon=0.001, delta=0.001)
        results["zpds (batch, numpy)"] = _rate(z3.add_many, npk, n=n)

    p = PyCountMinSketch(epsilon=0.001, delta=0.001)
    results["pure-python"] = _rate(lambda w: [p.add(k) for k in w], keys)
    return results


def bench_countmin_query(n=200_000):
    keys = [f"item-{i}".encode() for i in range(n)]
    z = zpds.CountMinSketch(epsilon=0.001, delta=0.001)
    z.add_many(keys)
    results = {
        "zpds (batch, list)": _rate(z.estimate_many, keys),
        "zpds (scalar)": _rate(lambda w: [z.estimate(k) for k in w], keys),
    }
    p = PyCountMinSketch(epsilon=0.001, delta=0.001)
    for k in keys:
        p.add(k)
    results["pure-python"] = _rate(lambda w: [p.estimate(k) for k in w], keys)
    return results


def _print_table(title, results):
    baseline = results.get("pure-python")
    print(f"\n{title}")
    print("-" * len(title))
    for impl, rate in sorted(results.items(), key=lambda kv: -kv[1]):
        speedup = f"{rate / baseline:9.1f}x" if baseline else "    -    "
        print(f"  {impl:22} {rate:14,.0f} ops/s  {speedup}")


def main():
    print(f"zpds {zpds.__version__}  (native core {'.'.join(map(str, zpds.native_version()))})")
    print(f"numpy: {HAVE_NUMPY}  |  pybloom_live: {HAVE_PYBLOOM}  |  datasketch: {HAVE_DATASKETCH}")
    _print_table("Bloom filter — add (200k items)", bench_bloom_add())
    _print_table("Bloom filter — query (200k items)", bench_bloom_query())
    _print_table("HyperLogLog — add (200k items)", bench_hll_add())
    _print_table("Count-Min Sketch — add (200k items)", bench_countmin_add())
    _print_table("Count-Min Sketch — query (200k items)", bench_countmin_query())


if __name__ == "__main__":
    main()
