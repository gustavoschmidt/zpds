"""Load the native ``libzpds`` shared library and expose its C ABI via cffi.

We use cffi in ABI (``dlopen``) mode: the C declarations below mirror
``include/zpds.h`` exactly, and we open a prebuilt shared library at import
time. The library is located, in order, from:

1. ``$ZPDS_LIBRARY_PATH`` — a full path to the shared library, or a directory
   containing it.
2. The package directory (how a built wheel bundles the library).
3. ``zig-out/lib`` relative to the repository (the developer build tree).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cffi import FFI

# Mirrors include/zpds.h (declarations only, no preprocessor directives).
_CDEF = """
uint32_t zpds_version(void);
uint64_t zpds_hash64(const uint8_t *data, size_t len, uint64_t seed);

typedef struct zpds_bloom zpds_bloom;
zpds_bloom *zpds_bloom_new(uint64_t expected_items, double fp_rate, uint64_t seed);
zpds_bloom *zpds_bloom_new_with_params(uint64_t n_bits, uint32_t k, uint64_t seed);
void zpds_bloom_free(zpds_bloom *b);
void zpds_bloom_add(zpds_bloom *b, const uint8_t *data, size_t len);
bool zpds_bloom_contains(const zpds_bloom *b, const uint8_t *data, size_t len);
uint64_t zpds_bloom_count(const zpds_bloom *b);
uint64_t zpds_bloom_bits(const zpds_bloom *b);
uint32_t zpds_bloom_k(const zpds_bloom *b);
void zpds_bloom_clear(zpds_bloom *b);

typedef struct zpds_hll zpds_hll;
zpds_hll *zpds_hll_new(uint32_t precision, uint64_t seed);
void zpds_hll_free(zpds_hll *h);
void zpds_hll_add(zpds_hll *h, const uint8_t *data, size_t len);
uint64_t zpds_hll_count(const zpds_hll *h);
double zpds_hll_estimate(const zpds_hll *h);
uint64_t zpds_hll_size(const zpds_hll *h);
double zpds_hll_error(const zpds_hll *h);
void zpds_hll_clear(zpds_hll *h);
bool zpds_hll_merge(zpds_hll *dst, const zpds_hll *src);

typedef struct zpds_cuckoo zpds_cuckoo;
zpds_cuckoo *zpds_cuckoo_new(uint64_t capacity, uint64_t seed);
void zpds_cuckoo_free(zpds_cuckoo *c);
bool zpds_cuckoo_add(zpds_cuckoo *c, const uint8_t *data, size_t len);
bool zpds_cuckoo_contains(zpds_cuckoo *c, const uint8_t *data, size_t len);
bool zpds_cuckoo_remove(zpds_cuckoo *c, const uint8_t *data, size_t len);
uint64_t zpds_cuckoo_count(const zpds_cuckoo *c);
uint64_t zpds_cuckoo_capacity(const zpds_cuckoo *c);
void zpds_cuckoo_clear(zpds_cuckoo *c);

typedef struct zpds_countmin zpds_countmin;
zpds_countmin *zpds_countmin_new(double epsilon, double delta, uint64_t seed);
zpds_countmin *zpds_countmin_new_with_params(uint64_t width, uint64_t depth, uint64_t seed);
void zpds_countmin_free(zpds_countmin *cm);
void zpds_countmin_add(zpds_countmin *cm, const uint8_t *data, size_t len, uint64_t count);
uint64_t zpds_countmin_estimate(const zpds_countmin *cm, const uint8_t *data, size_t len);
uint64_t zpds_countmin_total(const zpds_countmin *cm);
uint64_t zpds_countmin_width(const zpds_countmin *cm);
uint64_t zpds_countmin_depth(const zpds_countmin *cm);
void zpds_countmin_clear(zpds_countmin *cm);
bool zpds_countmin_merge(zpds_countmin *dst, const zpds_countmin *src);
"""


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libzpds.dylib"
    if sys.platform == "win32":
        return "zpds.dll"
    return "libzpds.so"


def _candidate_paths() -> list[Path]:
    name = _library_name()
    candidates: list[Path] = []

    env = os.environ.get("ZPDS_LIBRARY_PATH")
    if env:
        p = Path(env)
        candidates.append(p if p.is_file() else p / name)

    here = Path(__file__).resolve().parent
    candidates.append(here / name)  # bundled inside the wheel
    # Developer build tree: python/zpds/_native.py -> repo root -> zig-out/lib.
    candidates.append(here.parent.parent / "zig-out" / "lib" / name)
    return candidates


def _load() -> tuple[FFI, object]:
    ffi = FFI()
    ffi.cdef(_CDEF)
    tried: list[str] = []
    for path in _candidate_paths():
        tried.append(str(path))
        if path.is_file():
            return ffi, ffi.dlopen(str(path))
    raise OSError(
        "Could not locate the zpds native library. Build it with `zig build` "
        "or set $ZPDS_LIBRARY_PATH. Looked in:\n  " + "\n  ".join(tried)
    )


ffi, lib = _load()


def as_bytes(item: object) -> bytes:
    """Coerce ``item`` (str/bytes/bytes-like) to ``bytes`` for hashing."""
    if isinstance(item, str):
        return item.encode("utf-8")
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    raise TypeError(f"expected str or bytes-like, got {type(item).__name__}")
