"""Tests for the benchmark harness and its pure-Python reference impls.

These validate that the baselines used for comparison are themselves correct
(they are independent implementations of the same algorithms) and that the
benchmark runs end-to-end and shows the native core outrunning pure Python.
"""

import zpds
from benchmarks.bench import run
from benchmarks.pure_python import PyBloomFilter, PyCountMinSketch, PyHyperLogLog


def test_pure_bloom_no_false_negatives_and_reasonable_fp():
    n = 5_000
    bf = PyBloomFilter(capacity=n, error_rate=0.01)
    keys = [f"k-{i}".encode() for i in range(n)]
    for k in keys:
        bf.add(k)
    assert all(k in bf for k in keys)  # no false negatives
    fp = sum(1 for i in range(20_000) if f"absent-{i}".encode() in bf)
    assert fp / 20_000 < 0.03


def test_pure_hll_accuracy():
    n = 100_000
    hll = PyHyperLogLog(precision=14)
    for i in range(n):
        hll.add(f"e-{i}".encode())
    assert abs(len(hll) - n) / n < 0.03


def test_pure_countmin_never_underestimates():
    cms = PyCountMinSketch(epsilon=0.001, delta=0.001)
    for i in range(1, 501):
        cms.add(f"w-{i}".encode(), i)
    for i in range(1, 501):
        assert cms.estimate(f"w-{i}".encode()) >= i


def test_native_and_reference_agree_on_hll_cardinality():
    # Independent implementations should land within a few % of each other on
    # the same stream (both target ~0.8% error at precision 14).
    n = 50_000
    keys = [f"e-{i}".encode() for i in range(n)]
    z = zpds.HyperLogLog(precision=14, seed=0)
    p = PyHyperLogLog(precision=14, seed=0)
    for k in keys:
        z.add(k)
        p.add(k)
    assert abs(z.estimate() - p.estimate()) / n < 0.05


def test_benchmark_runs_and_native_is_faster():
    groups = run(20_000)
    flat = [r for g in groups for r in g]
    # Every measurement completed with a positive rate.
    assert all(r.ops_per_sec > 0 for r in flat)

    # For the 'add' op in each group, the native core beats pure Python.
    for group in groups:
        by = {(r.name.startswith("zpds"), r.op): r.ops_per_sec for r in group}
        assert by[(True, "add")] > by[(False, "add")]
