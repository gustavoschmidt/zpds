"""Batch API: numpy fast path, streaming iterables, and equivalence to the
single-item API."""

import pytest

from zpds import BloomFilter, CountMinSketch, CuckooFilter, HyperLogLog

np = pytest.importorskip("numpy")


# --- Bloom -----------------------------------------------------------------

def test_bloom_add_many_matches_individual_adds():
    keys = [f"item-{i}" for i in range(5000)]
    a = BloomFilter(capacity=10_000, error_rate=0.01, seed=1)
    b = BloomFilter(capacity=10_000, error_rate=0.01, seed=1)
    for k in keys:
        a.add(k)
    b.add_many(keys)
    # Same params/seed/inserts -> identical membership over a probe set.
    probe = keys + [f"absent-{i}" for i in range(5000)]
    assert [k in a for k in probe] == [k in b for k in probe]
    assert a.items_added == b.items_added == 5000


def test_bloom_add_many_from_generator_stream():
    bf = BloomFilter(capacity=10_000, seed=2)
    # small batch_size exercises multi-chunk streaming
    bf.add_many((f"s-{i}" for i in range(5000)), batch_size=128)
    assert bf.items_added == 5000
    assert all(f"s-{i}" in bf for i in range(5000))


def test_bloom_contains_many_list_returns_list():
    bf = BloomFilter(capacity=1000)
    bf.add_many(["a", "b", "c"])
    res = bf.contains_many(["a", "x", "b", "y"])
    assert isinstance(res, list)
    assert res[0] and res[2]
    assert not res[1] and not res[3]


def test_bloom_numpy_uint64_zero_copy():
    keys = np.arange(2000, dtype=np.uint64)
    bf = BloomFilter(capacity=20_000)
    bf.add_many(keys)
    present = bf.contains_many(keys)
    assert isinstance(present, np.ndarray) and present.dtype == bool
    assert present.all()
    absent = np.arange(10_000_000, 10_002_000, dtype=np.uint64)
    assert bf.contains_many(absent).mean() < 0.05


def test_bloom_numpy_2d_uint8_fixed_width():
    # 500 keys, each an 8-byte row.
    keys = np.random.randint(0, 256, size=(500, 8), dtype=np.uint8)
    bf = BloomFilter(capacity=5000)
    bf.add_many(keys)
    assert bf.contains_many(keys).all()


def test_bloom_numpy_object_array_uses_generic_path():
    keys = np.array([f"o-{i}" for i in range(300)], dtype=object)
    bf = BloomFilter(capacity=3000)
    bf.add_many(keys)
    # object dtype -> generic path -> str encoding, so single-item calls agree.
    assert all(f"o-{i}" in bf for i in range(300))


def test_batch_size_keyword():
    bf = BloomFilter(capacity=100)
    # A tiny batch_size still streams the whole iterable correctly.
    bf.add_many((f"i-{i}" for i in range(50)), batch_size=1)
    assert bf.items_added == 50
    with pytest.raises(ValueError):
        bf.add_many(["a"], batch_size=0)


def test_empty_batch_is_noop():
    bf = BloomFilter(capacity=100)
    bf.add_many([])
    bf.add_many(np.empty(0, dtype=np.uint64))
    assert bf.items_added == 0
    assert bf.contains_many([]) == []
    assert list(bf.contains_many(np.empty(0, dtype=np.uint64))) == []


# --- HyperLogLog -----------------------------------------------------------

def test_hll_add_many_matches_individual():
    keys = [f"e-{i}" for i in range(20_000)]
    a = HyperLogLog(precision=14, seed=3)
    b = HyperLogLog(precision=14, seed=3)
    for k in keys:
        a.add(k)
    b.add_many(keys)
    # Identical register state -> identical estimate.
    assert a.estimate() == b.estimate()


def test_hll_add_many_numpy():
    keys = np.arange(100_000, dtype=np.uint64)
    hll = HyperLogLog(precision=14)
    hll.add_many(keys)
    assert abs(len(hll) - 100_000) / 100_000 < 0.02


# --- Cuckoo ----------------------------------------------------------------

def test_cuckoo_add_many_returns_inserted_and_membership():
    keys = [f"k-{i}" for i in range(5000)]
    cf = CuckooFilter(capacity=10_000, seed=4)
    inserted = cf.add_many(keys)
    assert inserted == 5000
    assert len(cf) == 5000
    assert all(cf.contains_many(keys))


def test_cuckoo_remove_many():
    keys = [f"k-{i}" for i in range(1000)]
    cf = CuckooFilter(capacity=5000, seed=5)
    cf.add_many(keys)
    removed = cf.remove_many(keys[:400])
    assert removed == 400
    assert len(cf) == 600
    assert all(cf.contains_many(keys[400:]))


def test_cuckoo_add_many_reports_partial_on_full():
    cf = CuckooFilter(capacity=8)
    inserted = cf.add_many(f"x-{i}" for i in range(100_000))
    # Small table fills up: fewer inserted than offered.
    assert 0 < inserted < 100_000


def test_cuckoo_numpy():
    keys = np.arange(3000, dtype=np.uint64)
    cf = CuckooFilter(capacity=10_000)
    assert cf.add_many(keys) == 3000
    assert cf.contains_many(keys).all()


# --- Count-Min -------------------------------------------------------------

def test_countmin_add_many_default_counts():
    a = CountMinSketch.from_shape(1000, 5, seed=6)
    b = CountMinSketch.from_shape(1000, 5, seed=6)
    keys = [f"w-{i}" for i in range(2000)]
    for k in keys:
        a.add(k)
    b.add_many(keys)
    est_a = a.estimate_many(keys)
    est_b = b.estimate_many(keys)
    assert est_a == est_b


def test_countmin_add_many_with_counts_list():
    cms = CountMinSketch(epsilon=0.001, delta=0.001, seed=7)
    keys = [f"w-{i}" for i in range(1, 501)]
    counts = list(range(1, 501))
    cms.add_many(keys, counts)
    est = cms.estimate_many(keys)
    assert all(e >= c for e, c in zip(est, counts))
    assert cms.total == sum(counts)


def test_countmin_counts_length_mismatch():
    cms = CountMinSketch.from_shape(100, 4)
    with pytest.raises(ValueError):
        cms.add_many(["a", "b", "c"], [1, 2])


def test_countmin_numpy_items_and_counts():
    items = np.arange(1000, dtype=np.uint64)
    counts = np.arange(1, 1001, dtype=np.uint64)
    cms = CountMinSketch(epsilon=0.0005, delta=0.001)
    cms.add_many(items, counts)
    est = cms.estimate_many(items)
    assert isinstance(est, np.ndarray) and est.dtype == np.uint64
    assert (est >= counts).all()
    assert cms.total == int(counts.sum())


def test_countmin_estimate_many_list():
    cms = CountMinSketch.from_shape(1000, 5)
    cms.add("apple", 3)
    cms.add("banana", 5)
    est = cms.estimate_many(["apple", "banana", "cherry"])
    assert isinstance(est, list)
    assert est[0] >= 3 and est[1] >= 5
