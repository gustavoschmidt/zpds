/*
 * zpds — Zig Probabilistic Data Structures.
 *
 * C-ABI surface for the native core. The Python wrapper (and any other C
 * consumer) links against libzpds and drives everything through these
 * declarations. The boundary deliberately stays simple: bytes, ints, bools.
 */
#ifndef ZPDS_H
#define ZPDS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Library version, packed as (major << 16) | (minor << 8) | patch. */
uint32_t zpds_version(void);

/* One-shot 64-bit non-cryptographic (wyhash) of `len` bytes at `data` under
 * `seed`. A NULL/empty buffer (len == 0) hashes the empty string. */
uint64_t zpds_hash64(const uint8_t *data, size_t len, uint64_t seed);

/* --- Bloom filter -------------------------------------------------------- */

typedef struct zpds_bloom zpds_bloom;

/* Allocate a Bloom filter sized for `expected_items` at target false-positive
 * `fp_rate` (0,1). Returns NULL on allocation failure. Free with
 * zpds_bloom_free. */
zpds_bloom *zpds_bloom_new(uint64_t expected_items, double fp_rate, uint64_t seed);

/* Allocate a Bloom filter with explicit bit count `n_bits` and `k` hash
 * functions. */
zpds_bloom *zpds_bloom_new_with_params(uint64_t n_bits, uint32_t k, uint64_t seed);

void zpds_bloom_free(zpds_bloom *b);
void zpds_bloom_add(zpds_bloom *b, const uint8_t *data, size_t len);
bool zpds_bloom_contains(const zpds_bloom *b, const uint8_t *data, size_t len);
uint64_t zpds_bloom_count(const zpds_bloom *b);   /* number of add() calls */
uint64_t zpds_bloom_bits(const zpds_bloom *b);    /* bit count m */
uint32_t zpds_bloom_k(const zpds_bloom *b);       /* hash-function count k */
void zpds_bloom_clear(zpds_bloom *b);

#ifdef __cplusplus
}
#endif

#endif /* ZPDS_H */
