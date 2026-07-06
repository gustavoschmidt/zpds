import pytest

from zpds import BloomFilter


def test_add_and_contains():
    bf = BloomFilter(capacity=1000, error_rate=0.01)
    bf.add("alice")
    assert "alice" in bf
    assert "bob" not in bf


def test_accepts_str_and_bytes():
    bf = BloomFilter(capacity=1000)
    bf.add(b"\x00\x01\x02")
    assert b"\x00\x01\x02" in bf


def test_no_false_negatives():
    bf = BloomFilter(capacity=10_000, error_rate=0.01)
    items = [f"item-{i}" for i in range(10_000)]
    bf.add_many(items)
    assert all(x in bf for x in items)
    assert bf.items_added == 10_000


def test_false_positive_rate_tracks_target():
    n, target = 20_000, 0.01
    bf = BloomFilter(capacity=n, error_rate=target)
    for i in range(n):
        bf.add(f"present-{i}")

    trials = 50_000
    fp = sum(1 for i in range(trials) if f"absent-{i}" in bf)
    observed = fp / trials
    # Empirical FP rate should be near the target, comfortably below 2x.
    assert observed < target * 2


def test_clear():
    bf = BloomFilter(capacity=100)
    bf.add("x")
    assert "x" in bf
    bf.clear()
    assert "x" not in bf
    assert bf.items_added == 0


def test_params_and_repr():
    bf = BloomFilter(capacity=1_000_000, error_rate=0.01)
    assert bf.bits % 64 == 0
    assert bf.num_hashes == 7
    assert "BloomFilter" in repr(bf)


def test_from_bit_count():
    bf = BloomFilter.from_bit_count(n_bits=1024, k=4)
    bf.add("x")
    assert "x" in bf


def test_invalid_args():
    with pytest.raises(ValueError):
        BloomFilter(capacity=0)
    with pytest.raises(ValueError):
        BloomFilter(capacity=10, error_rate=1.5)


def test_type_error_on_bad_item():
    bf = BloomFilter(capacity=10)
    with pytest.raises(TypeError):
        bf.add(123)
