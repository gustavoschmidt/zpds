//! C-ABI surface for zpds.
//!
//! Everything crossing the FFI boundary stays simple: raw byte pointers,
//! integers and booleans. Higher-level ergonomics live in the Python wrapper.

const std = @import("std");
const root = @import("root.zig");
const hash = root.hash;
const Bloom = root.Bloom;
const HyperLogLog = root.HyperLogLog;

/// Allocator backing every heap object handed across the C ABI. `page_allocator`
/// is stateless and libc-free, keeping the shared library self-contained.
const gpa = std.heap.page_allocator;

/// Materialize a byte slice from a (possibly null) C pointer + length.
inline fn slice(ptr: ?[*]const u8, len: usize) []const u8 {
    return if (len == 0) &.{} else ptr.?[0..len];
}

/// Return the library version packed as (major << 16) | (minor << 8) | patch.
export fn zpds_version() u32 {
    return root.versionInt();
}

/// One-shot 64-bit wyhash of `len` bytes at `data` under `seed`. A null/empty
/// buffer (len == 0) hashes the empty string.
export fn zpds_hash64(data: ?[*]const u8, len: usize, seed: u64) u64 {
    return hash.hash64(slice(data, len), seed);
}

// --- Bloom filter -----------------------------------------------------------

/// Allocate a Bloom filter sized for `expected_items` at `fp_rate`. Returns
/// null on allocation failure.
export fn zpds_bloom_new(expected_items: u64, fp_rate: f64, seed: u64) ?*Bloom {
    const b = gpa.create(Bloom) catch return null;
    b.* = Bloom.init(gpa, expected_items, fp_rate, seed) catch {
        gpa.destroy(b);
        return null;
    };
    return b;
}

/// Allocate a Bloom filter with explicit bit count and hash-function count.
export fn zpds_bloom_new_with_params(n_bits: u64, k: u32, seed: u64) ?*Bloom {
    const b = gpa.create(Bloom) catch return null;
    b.* = Bloom.initWithParams(gpa, n_bits, k, seed) catch {
        gpa.destroy(b);
        return null;
    };
    return b;
}

export fn zpds_bloom_free(b: ?*Bloom) void {
    if (b) |ptr| {
        ptr.deinit(gpa);
        gpa.destroy(ptr);
    }
}

export fn zpds_bloom_add(b: *Bloom, data: ?[*]const u8, len: usize) void {
    b.add(slice(data, len));
}

export fn zpds_bloom_contains(b: *const Bloom, data: ?[*]const u8, len: usize) bool {
    return b.contains(slice(data, len));
}

export fn zpds_bloom_count(b: *const Bloom) u64 {
    return b.len();
}

export fn zpds_bloom_bits(b: *const Bloom) u64 {
    return b.n_bits;
}

export fn zpds_bloom_k(b: *const Bloom) u32 {
    return b.k;
}

export fn zpds_bloom_clear(b: *Bloom) void {
    b.clear();
}

// --- HyperLogLog ------------------------------------------------------------

/// Allocate a HyperLogLog with `precision` register-index bits (clamped to
/// [4, 18]). Returns null on allocation failure.
export fn zpds_hll_new(precision: u32, seed: u64) ?*HyperLogLog {
    const p: u5 = @intCast(std.math.clamp(precision, 4, 18));
    const h = gpa.create(HyperLogLog) catch return null;
    h.* = HyperLogLog.init(gpa, p, seed) catch {
        gpa.destroy(h);
        return null;
    };
    return h;
}

export fn zpds_hll_free(h: ?*HyperLogLog) void {
    if (h) |ptr| {
        ptr.deinit(gpa);
        gpa.destroy(ptr);
    }
}

export fn zpds_hll_add(h: *HyperLogLog, data: ?[*]const u8, len: usize) void {
    h.add(slice(data, len));
}

export fn zpds_hll_count(h: *const HyperLogLog) u64 {
    return h.count();
}

export fn zpds_hll_estimate(h: *const HyperLogLog) f64 {
    return h.estimate();
}

export fn zpds_hll_size(h: *const HyperLogLog) u64 {
    return h.size();
}

export fn zpds_hll_error(h: *const HyperLogLog) f64 {
    return h.relativeError();
}

export fn zpds_hll_clear(h: *HyperLogLog) void {
    h.clear();
}

/// Merge `src` into `dst` (register-wise max). Returns false on precision
/// mismatch, leaving `dst` unchanged.
export fn zpds_hll_merge(dst: *HyperLogLog, src: *const HyperLogLog) bool {
    dst.merge(src) catch return false;
    return true;
}
