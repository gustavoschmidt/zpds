import zpds


def test_native_version():
    assert zpds.native_version() == (0, 0, 1)


def test_hash_is_deterministic():
    assert zpds.hash64("abc", 0) == zpds.hash64("abc", 0)
    assert zpds.hash64(b"abc", 0) == zpds.hash64("abc", 0)


def test_hash_golden_vectors():
    # Same vectors locked by the Zig test suite.
    assert zpds.hash64("", 0) == 0x0409638EE2BDE459
    assert zpds.hash64("abc", 0) == 0x02A4F1D7CB516C72
    assert zpds.hash64("abc", 1) == 0xDBE5B1E5823255B7


def test_hash_seed_sensitivity():
    assert zpds.hash64("abc", 0) != zpds.hash64("abc", 1)
