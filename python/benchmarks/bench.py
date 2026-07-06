"""Benchmark zpds (native Zig core) against pure-Python implementations.

The target niche is streaming / single-item / low-latency work, so we measure
per-item ``add`` and query throughput rather than batch operations. Run:

    python -m benchmarks.bench            # from the python/ directory
    python benchmarks/bench.py --n 500000

Optional third-party libraries (``pybloom_live``, ``datasketch``) are included
in the comparison automatically when importable.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import zpds

from .pure_python import PyBloomFilter, PyCountMinSketch, PyHyperLogLog


@dataclass
class Result:
    name: str
    op: str
    ops_per_sec: float


def _time(fn, iterations: int) -> float:
    """Return operations/second for ``fn`` called ``iterations`` times."""
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    return iterations / elapsed if elapsed > 0 else float("inf")


def bench_bloom(keys: list[bytes]) -> list[Result]:
    n = len(keys)
    out: list[Result] = []

    z = zpds.BloomFilter(capacity=n, error_rate=0.01)
    out.append(Result("zpds.BloomFilter", "add", _time(lambda: [z.add(k) for k in keys], n)))
    out.append(Result("zpds.BloomFilter", "query", _time(lambda: [k in z for k in keys], n)))

    p = PyBloomFilter(capacity=n, error_rate=0.01)
    out.append(Result("pure-python Bloom", "add", _time(lambda: [p.add(k) for k in keys], n)))
    out.append(Result("pure-python Bloom", "query", _time(lambda: [k in p for k in keys], n)))
    return out


def bench_hll(keys: list[bytes]) -> list[Result]:
    n = len(keys)
    out: list[Result] = []

    z = zpds.HyperLogLog(precision=14)
    out.append(Result("zpds.HyperLogLog", "add", _time(lambda: [z.add(k) for k in keys], n)))

    p = PyHyperLogLog(precision=14)
    out.append(Result("pure-python HLL", "add", _time(lambda: [p.add(k) for k in keys], n)))
    return out


def bench_countmin(keys: list[bytes]) -> list[Result]:
    n = len(keys)
    out: list[Result] = []

    z = zpds.CountMinSketch(epsilon=0.001, delta=0.001)
    out.append(Result("zpds.CountMinSketch", "add", _time(lambda: [z.add(k) for k in keys], n)))
    out.append(Result("zpds.CountMinSketch", "query", _time(lambda: [z.estimate(k) for k in keys], n)))

    p = PyCountMinSketch(epsilon=0.001, delta=0.001)
    out.append(Result("pure-python Count-Min", "add", _time(lambda: [p.add(k) for k in keys], n)))
    out.append(Result("pure-python Count-Min", "query", _time(lambda: [p.estimate(k) for k in keys], n)))
    return out


def bench_batch(keys: list[bytes]) -> list[Result]:
    """Single-item vs batch vs zero-copy numpy `add`, to expose FFI overhead."""
    n = len(keys)
    out: list[Result] = []

    z = zpds.BloomFilter(capacity=n, error_rate=0.01)
    out.append(Result("single-item add", "add", _time(lambda: [z.add(k) for k in keys], n)))

    z2 = zpds.BloomFilter(capacity=n, error_rate=0.01)
    out.append(Result("add_many(list)", "add", _time(lambda: z2.add_many(keys), n)))

    try:
        import numpy as np

        np_keys = np.arange(n, dtype=np.uint64)
        z3 = zpds.BloomFilter(capacity=n, error_rate=0.01)
        out.append(Result("add_many(numpy)", "add", _time(lambda: z3.add_many(np_keys), n)))
    except ImportError:
        pass
    return out


def run(n: int) -> list[list[Result]]:
    keys = [f"item-{i}".encode() for i in range(n)]
    return [bench_bloom(keys), bench_hll(keys), bench_countmin(keys)]


def run_batch(n: int) -> list[Result]:
    keys = [f"item-{i}".encode() for i in range(n)]
    return bench_batch(keys)


def _print_group(title: str, results: list[Result]) -> None:
    print(f"\n{title}")
    print(f"  {'implementation':<24}{'op':<8}{'ops/sec':>14}{'speedup':>10}")
    # Speedup: each native result vs the matching pure-python op.
    baseline = {r.op: r.ops_per_sec for r in results if r.name.startswith("pure-python")}
    for r in results:
        base = baseline.get(r.op)
        speedup = f"{r.ops_per_sec / base:>9.1f}x" if base and r.name.startswith("zpds") else ""
        print(f"  {r.name:<24}{r.op:<8}{r.ops_per_sec:>14,.0f}{speedup:>10}")


def _print_batch(results: list[Result]) -> None:
    print("\nBatch add — amortizing the FFI crossing (zpds Bloom)")
    print(f"  {'strategy':<24}{'ops/sec':>14}{'speedup':>10}")
    base = next((r.ops_per_sec for r in results if r.name == "single-item add"), None)
    for r in results:
        speedup = f"{r.ops_per_sec / base:>9.1f}x" if base else ""
        print(f"  {r.name:<24}{r.ops_per_sec:>14,.0f}{speedup:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200_000, help="number of items")
    args = parser.parse_args()

    print(f"zpds native core version {zpds.native_version()}  |  n = {args.n:,}")
    groups = run(args.n)
    for title, group in zip(("Bloom filter", "HyperLogLog", "Count-Min Sketch"), groups):
        _print_group(title, group)
    _print_batch(run_batch(args.n))


if __name__ == "__main__":
    main()
