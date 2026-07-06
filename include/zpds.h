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

/* --- HyperLogLog --------------------------------------------------------- */

typedef struct zpds_hll zpds_hll;

/* Allocate a HyperLogLog with `precision` register-index bits (clamped to
 * [4, 18]; higher precision -> more memory, less error). Free with
 * zpds_hll_free. */
zpds_hll *zpds_hll_new(uint32_t precision, uint64_t seed);

void zpds_hll_free(zpds_hll *h);
void zpds_hll_add(zpds_hll *h, const uint8_t *data, size_t len);
uint64_t zpds_hll_count(const zpds_hll *h);     /* rounded cardinality estimate */
double zpds_hll_estimate(const zpds_hll *h);    /* raw cardinality estimate */
uint64_t zpds_hll_size(const zpds_hll *h);      /* register count m = 2^precision */
double zpds_hll_error(const zpds_hll *h);       /* expected relative error */
void zpds_hll_clear(zpds_hll *h);

/* Merge `src` into `dst` (register-wise max). Returns false on precision
 * mismatch, leaving `dst` unchanged. */
bool zpds_hll_merge(zpds_hll *dst, const zpds_hll *src);

/* --- Cuckoo filter ------------------------------------------------------- */

typedef struct zpds_cuckoo zpds_cuckoo;

/* Allocate a cuckoo filter that can hold roughly `capacity` items. Free with
 * zpds_cuckoo_free. */
zpds_cuckoo *zpds_cuckoo_new(uint64_t capacity, uint64_t seed);

void zpds_cuckoo_free(zpds_cuckoo *c);
/* Insert an item. Returns false if the filter is full. */
bool zpds_cuckoo_add(zpds_cuckoo *c, const uint8_t *data, size_t len);
bool zpds_cuckoo_contains(zpds_cuckoo *c, const uint8_t *data, size_t len);
/* Remove one occurrence. Returns true if a match was removed. */
bool zpds_cuckoo_remove(zpds_cuckoo *c, const uint8_t *data, size_t len);
uint64_t zpds_cuckoo_count(const zpds_cuckoo *c);     /* live fingerprints */
uint64_t zpds_cuckoo_capacity(const zpds_cuckoo *c);  /* total slot capacity */
void zpds_cuckoo_clear(zpds_cuckoo *c);

/* --- Count-Min Sketch ---------------------------------------------------- */

typedef struct zpds_countmin zpds_countmin;

/* Allocate a sketch sized for additive error `epsilon * total` with failure
 * probability `delta`. Free with zpds_countmin_free. */
zpds_countmin *zpds_countmin_new(double epsilon, double delta, uint64_t seed);

/* Allocate a sketch with explicit width and depth. */
zpds_countmin *zpds_countmin_new_with_params(uint64_t width, uint64_t depth, uint64_t seed);

void zpds_countmin_free(zpds_countmin *cm);
void zpds_countmin_add(zpds_countmin *cm, const uint8_t *data, size_t len, uint64_t count);
uint64_t zpds_countmin_estimate(const zpds_countmin *cm, const uint8_t *data, size_t len);
uint64_t zpds_countmin_total(const zpds_countmin *cm);
uint64_t zpds_countmin_width(const zpds_countmin *cm);
uint64_t zpds_countmin_depth(const zpds_countmin *cm);
void zpds_countmin_clear(zpds_countmin *cm);

/* Merge `src` into `dst` (counter-wise sum). Returns false on width/depth/seed
 * mismatch, leaving `dst` unchanged. */
bool zpds_countmin_merge(zpds_countmin *dst, const zpds_countmin *src);

#ifdef __cplusplus
}
#endif

#endif /* ZPDS_H */
