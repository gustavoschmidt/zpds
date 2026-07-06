import pytest

from zpds import CountMinSketch


def test_empty_is_zero():
    cms = CountMinSketch()
    assert cms.estimate("anything") == 0
    assert cms.total == 0


def test_basic_counts():
    cms = CountMinSketch(epsilon=0.001, delta=0.001)
    cms.add("apple", 3)
    cms.add("apple")
    cms.add("banana", 5)
    assert cms.estimate("apple") >= 4
    assert cms.estimate("banana") >= 5
    assert cms.total == 9


def test_never_underestimates():
    cms = CountMinSketch(epsilon=0.001, delta=0.001, seed=42)
    for i in range(1, 501):
        cms.add(f"w-{i}", i)
    for i in range(1, 501):
        assert cms.estimate(f"w-{i}") >= i


def test_error_bound():
    eps = 0.001
    cms = CountMinSketch(epsilon=eps, delta=0.001, seed=7)
    for i in range(1, 1001):
        cms.add(f"w-{i}", i)
    bound = eps * cms.total  # actual guarantee uses e/width; this is looser
    # Verify the tighter e/width * total bound holds for every probed item.
    import math

    tight = math.e / cms.width * cms.total
    for i in range(1, 1001):
        assert cms.estimate(f"w-{i}") - i <= tight


def test_explicit_params():
    cms = CountMinSketch(width=500, depth=5)
    assert cms.width == 500
    assert cms.depth == 5


def test_sizing_from_epsilon_delta():
    cms = CountMinSketch(epsilon=0.01, delta=0.01)
    assert cms.width == 272
    assert cms.depth == 5


def test_merge():
    a = CountMinSketch(width=500, depth=5, seed=99)
    b = CountMinSketch(width=500, depth=5, seed=99)
    a.add("x", 10)
    b.add("x", 32)
    b.add("y", 5)
    a.merge(b)
    assert a.estimate("x") >= 42
    assert a.estimate("y") >= 5
    assert a.total == 47


def test_merge_mismatch():
    a = CountMinSketch(width=500, depth=5, seed=1)
    b = CountMinSketch(width=500, depth=5, seed=2)
    with pytest.raises(ValueError):
        a.merge(b)


def test_invalid_args():
    with pytest.raises(ValueError):
        CountMinSketch(width=10)  # depth missing
    with pytest.raises(ValueError):
        CountMinSketch(epsilon=2.0)
