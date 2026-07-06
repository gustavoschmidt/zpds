"""Benchmark harness smoke tests + pure-Python reference cross-validation.

The harness is loaded from its file (it is a runnable script, not a package
submodule). The pure-Python implementations double as an honest baseline and as
independent implementations we check the native core against.
"""

import importlib.util
import os

import pytest

import zpds
from benchmarks.pure_python import PyBloomFilter, PyCountMinSketch, PyHyperLogLog

_BENCH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "bench.py")


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench", _BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- harness runs end-to-end -----------------------------------------------

def test_bloom_add_bench_runs(bench):
    r = bench.bench_bloom_add(n=2000)
    assert r["zpds (scalar)"] > 0
    assert r["zpds (batch, list)"] > 0
    assert "pure-python" in r


def test_bloom_query_bench_runs(bench):
    r = bench.bench_bloom_query(n=2000)
    assert r["zpds (batch, list)"] > 0


def test_hll_add_bench_runs(bench):
    assert bench.bench_hll_add(n=2000)["zpds (scalar)"] > 0


def test_countmin_bench_runs(bench):
    assert bench.bench_countmin_add(n=2000)["zpds (scalar)"] > 0
    assert bench.bench_countmin_query(n=2000)["zpds (scalar)"] > 0


def test_batch_beats_pure_python(bench):
    r = bench.bench_bloom_add(n=20_000)
    assert r["zpds (batch, list)"] > r["pure-python"]


# --- pure-Python references satisfy their own guarantees -------------------

def test_pure_bloom_no_false_negatives_and_reasonable_fp():
    n = 5_000
    bf = PyBloomFilter(capacity=n, error_rate=0.01)
    keys = [f"k-{i}".encode() for i in range(n)]
    for k in keys:
        bf.add(k)
    assert all(k in bf for k in keys)
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
    # Independent implementations should land within a few % on the same stream.
    n = 50_000
    keys = [f"e-{i}".encode() for i in range(n)]
    z = zpds.HyperLogLog(precision=14, seed=0)
    p = PyHyperLogLog(precision=14, seed=0)
    for k in keys:
        z.add(k)
        p.add(k)
    assert abs(z.estimate() - p.estimate()) / n < 0.05
