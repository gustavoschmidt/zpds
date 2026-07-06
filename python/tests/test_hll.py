import pytest

from zpds import HyperLogLog


def test_empty_is_zero():
    hll = HyperLogLog()
    assert len(hll) == 0


def test_duplicates_do_not_inflate():
    hll = HyperLogLog()
    for _ in range(1000):
        hll.add("same")
    assert len(hll) <= 2


@pytest.mark.parametrize("true_n", [100, 1_000, 10_000, 100_000])
def test_accuracy(true_n):
    hll = HyperLogLog(precision=14, seed=0xC0FFEE)
    hll.update(f"elem-{i}" for i in range(true_n))
    rel_err = abs(len(hll) - true_n) / true_n
    assert rel_err < 0.03


def test_relative_error_property():
    hll = HyperLogLog(precision=14)
    # ~1.04 / sqrt(16384) ≈ 0.0081
    assert 0.005 < hll.relative_error < 0.012
    assert hll.num_registers == 16384


def test_merge_union():
    a = HyperLogLog(precision=14, seed=7)
    b = HyperLogLog(precision=14, seed=7)
    a.update(f"e-{i}" for i in range(50_000))
    b.update(f"e-{i}" for i in range(25_000, 75_000))
    a.merge(b)
    assert abs(len(a) - 75_000) / 75_000 < 0.03


def test_merge_precision_mismatch():
    a = HyperLogLog(precision=12)
    b = HyperLogLog(precision=14)
    with pytest.raises(ValueError):
        a.merge(b)


def test_clear():
    hll = HyperLogLog()
    hll.add("x")
    hll.clear()
    assert len(hll) == 0
