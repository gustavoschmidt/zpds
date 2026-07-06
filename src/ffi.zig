//! C-ABI surface for zpds.
//!
//! Everything crossing the FFI boundary stays simple: raw byte pointers,
//! integers and booleans. Higher-level ergonomics live in the Python wrapper.

const std = @import("std");
const root = @import("root.zig");
const hash = root.hash;
const Bloom = root.Bloom;
const HyperLogLog = root.HyperLogLog;
const CuckooFilter = root.CuckooFilter;
const CountMin = root.CountMin;

/// Allocator backing every heap object handed across the C ABI. `page_allocator`
/// is stateless and libc-free, keeping the shared library self-contained.
const gpa = std.heap.page_allocator;

/// Materialize a byte slice from a (possibly null) C pointer + length.
inline fn slice(ptr: ?[*]const u8, len: usize) []const u8 {
    return if (len == 0) &.{} else ptr.?[0..len];
}

/// Address item `i` of a batch. Two layouts share every `*_many` entry point,
/// selected by `item_width`:
///   - fixed-width (`item_width != 0`): item i is `blob[i*w .. (i+1)*w]`. This
///     is the zero-copy path for a contiguous numpy buffer; `offsets` is unused.
///   - variable-length (`item_width == 0`): item i is
///     `blob[offsets[i] .. offsets[i+1]]`, so `offsets` has `n + 1` entries.
inline fn itemAt(blob: [*]const u8, offsets: ?[*]const usize, item_width: usize, i: usize) []const u8 {
    if (item_width != 0) {
        const start = i * item_width;
        return blob[start .. start + item_width];
    }
    const off = offsets.?;
    return blob[off[i]..off[i + 1]];
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

// --- Cuckoo filter ----------------------------------------------------------

/// Allocate a cuckoo filter that can hold roughly `capacity` items. Returns
/// null on allocation failure.
export fn zpds_cuckoo_new(capacity: u64, seed: u64) ?*CuckooFilter {
    const c = gpa.create(CuckooFilter) catch return null;
    c.* = CuckooFilter.init(gpa, capacity, seed) catch {
        gpa.destroy(c);
        return null;
    };
    return c;
}

export fn zpds_cuckoo_free(c: ?*CuckooFilter) void {
    if (c) |ptr| {
        ptr.deinit(gpa);
        gpa.destroy(ptr);
    }
}

/// Insert an item. Returns false if the filter is full (insertion failed).
export fn zpds_cuckoo_add(c: *CuckooFilter, data: ?[*]const u8, len: usize) bool {
    c.add(slice(data, len)) catch return false;
    return true;
}

export fn zpds_cuckoo_contains(c: *CuckooFilter, data: ?[*]const u8, len: usize) bool {
    return c.contains(slice(data, len));
}

/// Remove one occurrence of an item. Returns true if a match was removed.
export fn zpds_cuckoo_remove(c: *CuckooFilter, data: ?[*]const u8, len: usize) bool {
    return c.remove(slice(data, len));
}

export fn zpds_cuckoo_count(c: *const CuckooFilter) u64 {
    return c.len();
}

export fn zpds_cuckoo_capacity(c: *const CuckooFilter) u64 {
    return c.capacity();
}

export fn zpds_cuckoo_clear(c: *CuckooFilter) void {
    c.clear();
}

// --- Count-Min Sketch -------------------------------------------------------

/// Allocate a sketch sized for additive error `epsilon * total` with failure
/// probability `delta`. Returns null on allocation failure.
export fn zpds_countmin_new(epsilon: f64, delta: f64, seed: u64) ?*CountMin {
    const cm = gpa.create(CountMin) catch return null;
    cm.* = CountMin.init(gpa, epsilon, delta, seed) catch {
        gpa.destroy(cm);
        return null;
    };
    return cm;
}

/// Allocate a sketch with explicit `width` and `depth`.
export fn zpds_countmin_new_with_params(width: u64, depth: u64, seed: u64) ?*CountMin {
    const cm = gpa.create(CountMin) catch return null;
    cm.* = CountMin.initWithParams(gpa, width, depth, seed) catch {
        gpa.destroy(cm);
        return null;
    };
    return cm;
}

export fn zpds_countmin_free(cm: ?*CountMin) void {
    if (cm) |ptr| {
        ptr.deinit(gpa);
        gpa.destroy(ptr);
    }
}

export fn zpds_countmin_add(cm: *CountMin, data: ?[*]const u8, len: usize, count: u64) void {
    cm.add(slice(data, len), count);
}

export fn zpds_countmin_estimate(cm: *const CountMin, data: ?[*]const u8, len: usize) u64 {
    return cm.estimate(slice(data, len));
}

export fn zpds_countmin_total(cm: *const CountMin) u64 {
    return cm.totalCount();
}

export fn zpds_countmin_width(cm: *const CountMin) u64 {
    return cm.width;
}

export fn zpds_countmin_depth(cm: *const CountMin) u64 {
    return cm.depth;
}

export fn zpds_countmin_clear(cm: *CountMin) void {
    cm.clear();
}

/// Merge `src` into `dst` (counter-wise sum). Returns false if the sketches
/// differ in width, depth or seed, leaving `dst` unchanged.
export fn zpds_countmin_merge(dst: *CountMin, src: *const CountMin) bool {
    dst.merge(src) catch return false;
    return true;
}

// --- Batch operations -------------------------------------------------------
//
// Each of these amortizes a single FFI crossing over `n` items: the per-item
// loop runs entirely in native code. See `itemAt` for the two supported memory
// layouts (fixed-width / offsets).

export fn zpds_bloom_add_many(
    b: *Bloom,
    blob: ?[*]const u8,
    offsets: ?[*]const usize,
    item_width: usize,
    n: usize,
) void {
    if (n == 0) return;
    const data = blob.?;
    var i: usize = 0;
    while (i < n) : (i += 1) b.add(itemAt(data, offsets, item_width, i));
}

/// Query `n` items; writes `1`/`0` into `out[i]` for present/absent.
export fn zpds_bloom_contains_many(
    b: *const Bloom,
    blob: ?[*]const u8,
    offsets: ?[*]const usize,
    item_width: usize,
    n: usize,
    out: ?[*]u8,
) void {
    if (n == 0) return;
    const data = blob.?;
    const o = out.?;
    var i: usize = 0;
    while (i < n) : (i += 1) o[i] = if (b.contains(itemAt(data, offsets, item_width, i))) 1 else 0;
}

/// Insert `n` items. Returns the number successfully inserted. If `out_ok` is
/// non-null, `out_ok[i]` is set to 1/0 per item.
export fn zpds_cuckoo_add_many(
    c: *CuckooFilter,
    blob: ?[*]const u8,
    offsets: ?[*]const usize,
    item_width: usize,
    n: usize,
    out_ok: ?[*]u8,
) usize {
    if (n == 0) return 0;
    const data = blob.?;
    var inserted: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (c.add(itemAt(data, offsets, item_width, i))) |_| {
            inserted += 1;
            if (out_ok) |o| o[i] = 1;
        } else |_| {
            if (out_ok) |o| o[i] = 0;
        }
    }
    return inserted;
}

export fn zpds_cuckoo_contains_many(
    c: *CuckooFilter,
    blob: ?[*]const u8,
    offsets: ?[*]const usize,
    item_width: usize,
    n: usize,
    out: ?[*]u8,
) void {
    if (n == 0) return;
    const data = blob.?;
    const o = out.?;
    var i: usize = 0;
    while (i < n) : (i += 1) o[i] = if (c.contains(itemAt(data, offsets, item_width, i))) 1 else 0;
}

/// Remove `n` items. Returns the number removed. If `out_ok` is non-null,
/// `out_ok[i]` is set to 1/0 per item.
export fn zpds_cuckoo_remove_many(
    c: *CuckooFilter,
    blob: ?[*]const u8,
    offsets: ?[*]const usize,
    item_width: usize,
    n: usize,
    out_ok: ?[*]u8,
) usize {
    if (n == 0) return 0;
    const data = blob.?;
    var removed: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const ok = c.remove(itemAt(data, offsets, item_width, i));
        if (out_ok) |o| o[i] = if (ok) 1 else 0;
        if (ok) removed += 1;
    }
    return removed;
}

export fn zpds_hll_add_many(
    h: *HyperLogLog,
    blob: ?[*]const u8,
    offsets: ?[*]const usize,
    item_width: usize,
    n: usize,
) void {
    if (n == 0) return;
    const data = blob.?;
    var i: usize = 0;
    while (i < n) : (i += 1) h.add(itemAt(data, offsets, item_width, i));
}

/// Add `n` items. `counts` (if non-null) gives the per-item increment;
/// otherwise each item is added once.
export fn zpds_countmin_add_many(
    cm: *CountMin,
    blob: ?[*]const u8,
    offsets: ?[*]const usize,
    item_width: usize,
    n: usize,
    counts: ?[*]const u64,
) void {
    if (n == 0) return;
    const data = blob.?;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const count: u64 = if (counts) |cc| cc[i] else 1;
        cm.add(itemAt(data, offsets, item_width, i), count);
    }
}

/// Estimate the frequency of `n` items into `out[i]`.
export fn zpds_countmin_estimate_many(
    cm: *const CountMin,
    blob: ?[*]const u8,
    offsets: ?[*]const usize,
    item_width: usize,
    n: usize,
    out: ?[*]u64,
) void {
    if (n == 0) return;
    const data = blob.?;
    const o = out.?;
    var i: usize = 0;
    while (i < n) : (i += 1) o[i] = cm.estimate(itemAt(data, offsets, item_width, i));
}

// --- Batch tests ------------------------------------------------------------

test "batch add/contains via offsets layout" {
    const b = zpds_bloom_new(1000, 0.01, 0).?;
    defer zpds_bloom_free(b);

    // Items: "alpha" (5), "beta" (4), "gamma" (5).
    const blob = "alphabetagamma";
    const offsets = [_]usize{ 0, 5, 9, 14 };
    zpds_bloom_add_many(b, blob, &offsets, 0, 3);
    try std.testing.expectEqual(@as(u64, 3), zpds_bloom_count(b));

    var out = [_]u8{ 9, 9, 9 };
    zpds_bloom_contains_many(b, blob, &offsets, 0, 3, &out);
    try std.testing.expectEqual([_]u8{ 1, 1, 1 }, out);

    // A single-item query agrees with the batch insert.
    try std.testing.expect(zpds_bloom_contains(b, "beta", 4));
    try std.testing.expect(!zpds_bloom_contains(b, "delta", 5));
}

test "batch fixed-width layout with per-item counts" {
    const cm = zpds_countmin_new_with_params(500, 5, 0).?;
    defer zpds_countmin_free(cm);

    // Three 4-byte items, contiguous — the zero-copy numpy layout.
    const blob = "aaaabbbbcccc";
    const counts = [_]u64{ 3, 5, 7 };
    zpds_countmin_add_many(cm, blob, null, 4, 3, &counts);
    try std.testing.expectEqual(@as(u64, 15), zpds_countmin_total(cm));

    var out = [_]u64{ 0, 0, 0 };
    zpds_countmin_estimate_many(cm, blob, null, 4, 3, &out);
    try std.testing.expect(out[0] >= 3);
    try std.testing.expect(out[1] >= 5);
    try std.testing.expect(out[2] >= 7);
}

test "cuckoo batch add/remove report counts" {
    const c = zpds_cuckoo_new(1000, 0).?;
    defer zpds_cuckoo_free(c);

    const blob = "aaaabbbbcccc"; // 3 fixed-width 4-byte items
    var ok = [_]u8{ 0, 0, 0 };
    try std.testing.expectEqual(@as(usize, 3), zpds_cuckoo_add_many(c, blob, null, 4, 3, &ok));
    try std.testing.expectEqual([_]u8{ 1, 1, 1 }, ok);

    var present = [_]u8{ 0, 0, 0 };
    zpds_cuckoo_contains_many(c, blob, null, 4, 3, &present);
    try std.testing.expectEqual([_]u8{ 1, 1, 1 }, present);

    try std.testing.expectEqual(@as(usize, 3), zpds_cuckoo_remove_many(c, blob, null, 4, 3, null));
    try std.testing.expectEqual(@as(u64, 0), zpds_cuckoo_count(c));
}
