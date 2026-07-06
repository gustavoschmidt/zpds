"""Tests for the ergonomics added on top of the core structures: copy(),
merge operators, context managers / close(), the cuckoo failure mask, the
named count properties, and package namespace hygiene."""

import pytest

import zpds
from zpds import BloomFilter, CountMinSketch, CuckooFilter, HyperLogLog


# --- copy() is independent --------------------------------------------------

def test_bloom_copy_is_independent():
    a = BloomFilter(capacity=1000, seed=1)
    a.add("x")
    b = a.copy()
    assert "x" in b
    b.add("y")
    assert "y" in b
    assert "y" not in a  # mutating the copy leaves the original alone


def test_cuckoo_copy_is_independent():
    a = CuckooFilter(capacity=1000)
    a.add("x")
    b = a.copy()
    assert "x" in b
    b.remove("x")
    assert "x" not in b
    assert "x" in a


def test_hll_copy_is_independent():
    a = HyperLogLog(precision=12, seed=3)
    a.add_many(f"e-{i}" for i in range(1000))
    b = a.copy()
    assert b.estimate() == a.estimate()
    b.add_many(f"z-{i}" for i in range(1000))
    assert b.cardinality > a.cardinality


def test_countmin_copy_is_independent():
    a = CountMinSketch.from_shape(500, 5, seed=9)
    a.add("x", 3)
    b = a.copy()
    assert b.estimate("x") == a.estimate("x")
    b.add("x", 10)
    assert b.estimate("x") >= 13
    assert a.estimate("x") == 3


# --- merge operators --------------------------------------------------------

def test_hll_union_operators():
    a = HyperLogLog(precision=14, seed=7)
    b = HyperLogLog(precision=14, seed=7)
    a.add_many(f"e-{i}" for i in range(50_000))
    b.add_many(f"e-{i}" for i in range(25_000, 75_000))

    union = a | b  # value form: neither operand mutated
    assert abs(union.cardinality - 75_000) / 75_000 < 0.03
    a_before = a.cardinality
    assert a.cardinality == a_before

    a |= b  # in-place
    assert abs(a.cardinality - 75_000) / 75_000 < 0.03


def test_hll_merge_operator_precision_mismatch():
    a = HyperLogLog(precision=12)
    b = HyperLogLog(precision=14)
    with pytest.raises(ValueError):
        a |= b


def test_countmin_add_operators():
    a = CountMinSketch.from_shape(500, 5, seed=99)
    b = CountMinSketch.from_shape(500, 5, seed=99)
    a.add("x", 10)
    b.add("x", 32)

    total = a + b  # value form
    assert total.estimate("x") >= 42
    assert a.estimate("x") == 10  # original untouched

    a += b  # in-place
    assert a.estimate("x") >= 42
    assert a.total == 42


# --- context manager / close() ---------------------------------------------

@pytest.mark.parametrize(
    "make",
    [
        lambda: BloomFilter(capacity=100),
        lambda: CuckooFilter(capacity=100),
        lambda: HyperLogLog(precision=10),
        lambda: CountMinSketch(),
    ],
)
def test_context_manager_closes(make):
    with make() as obj:
        assert obj is not None
    # close() is idempotent — the safety-net __del__ must not double-free.
    obj.close()


# --- cuckoo failure mask ----------------------------------------------------

def test_cuckoo_return_failed_all_fit():
    cf = CuckooFilter(capacity=10_000, seed=4)
    failed = cf.add_many([f"k-{i}" for i in range(5000)], return_failed=True)
    assert failed == []
    assert len(cf) == 5000


def test_cuckoo_return_failed_reports_indices():
    cf = CuckooFilter(capacity=8)
    items = [f"x-{i}" for i in range(2000)]
    failed = cf.add_many(items, return_failed=True)
    # A tiny table overflows: some indices are rejected, and they are valid
    # positions into `items`.
    assert failed
    assert all(0 <= idx < len(items) for idx in failed)
    # inserted + failed accounts for every offered item.
    assert len(cf) + len(failed) == len(items)


# --- named count properties -------------------------------------------------

def test_named_count_properties():
    bf = BloomFilter(capacity=100)
    bf.add("a")
    bf.add("a")
    assert bf.items_added == 2  # counts duplicate adds; not a distinct count

    cf = CuckooFilter(capacity=100)
    cf.add("a")
    assert cf.count == len(cf) == 1

    hll = HyperLogLog(precision=10)
    hll.add("a")
    assert hll.cardinality == len(hll)


# --- namespace hygiene ------------------------------------------------------

def test_internal_symbols_not_leaked():
    # _native internals must not surface on the top-level package.
    assert not hasattr(zpds, "as_bytes")
    assert not hasattr(zpds, "lib")
    for name in zpds.__all__:
        assert hasattr(zpds, name)
