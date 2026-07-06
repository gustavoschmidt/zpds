/*
 * zpds — Zig Probabilistic Data Structures.
 *
 * C-ABI surface for the native core. The Python wrapper (and any other C
 * consumer) links against libzpds and drives everything through these
 * declarations. The boundary deliberately stays simple: bytes, ints, bools.
 */
#ifndef ZPDS_H
#define ZPDS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Library version, packed as (major << 16) | (minor << 8) | patch. */
uint32_t zpds_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ZPDS_H */
