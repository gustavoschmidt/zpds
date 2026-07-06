import pytest

from zpds import CuckooFilter, Full


def test_add_contains_remove():
    cf = CuckooFilter(capacity=1000)
    cf.add("alice")
    assert "alice" in cf
    assert cf.remove("alice") is True
    assert "alice" not in cf


def test_no_false_negatives():
    cf = CuckooFilter(capacity=10_000, seed=0x9999)
    items = [f"item-{i}" for i in range(10_000)]
    for x in items:
        cf.add(x)
    assert len(cf) == 10_000
    assert all(x in cf for x in items)


def test_remove_updates_count():
    cf = CuckooFilter(capacity=1000, seed=7)
    for i in range(200):
        cf.add(f"k-{i}")
    assert len(cf) == 200
    for i in range(100):
        assert cf.remove(f"k-{i}")
    assert len(cf) == 100
    # Remaining half still present.
    assert all(f"k-{i}" in cf for i in range(100, 200))


def test_remove_absent_returns_false():
    cf = CuckooFilter(capacity=100)
    assert cf.remove("never-added") is False


def test_low_false_positive_rate():
    n = 20_000
    cf = CuckooFilter(capacity=n, seed=0xABCD)
    for i in range(n):
        cf.try_add(f"present-{i}")
    trials = 50_000
    fp = sum(1 for i in range(trials) if f"absent-{i}" in cf)
    assert fp / trials < 0.001


def test_full_raises():
    # A tiny filter overflows quickly; try_add eventually returns False and
    # add() raises Full.
    cf = CuckooFilter(capacity=8)
    overflowed = False
    for i in range(100_000):
        if not cf.try_add(f"x-{i}"):
            overflowed = True
            break
    assert overflowed
    with pytest.raises(Full):
        for i in range(100_000):
            cf.add(f"y-{i}")


def test_capacity_and_repr():
    cf = CuckooFilter(capacity=1000)
    assert cf.capacity >= 1000
    assert "CuckooFilter" in repr(cf)
