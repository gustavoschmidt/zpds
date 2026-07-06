//! Cuckoo filter — set membership with deletion support (Fan, Andersen,
//! Kaminsky & Mitzenmacher, 2014).
//!
//! Each item is reduced to a non-zero 16-bit fingerprint stored in one of two
//! candidate buckets. The two buckets are related by partial-key cuckoo
//! hashing: i2 = i1 XOR h(fingerprint), an involution so either bucket can be
//! derived from the other using only the stored fingerprint. Insertion evicts
//! and relocates fingerprints (cuckoo-style) when both buckets are full.
//!
//! False positives are possible (~2b / 2^f); false negatives cannot occur for
//! items that were inserted and not deleted. Deleting an item never inserted
//! may remove a colliding fingerprint — only delete items you added.

const std = @import("std");
const hash = @import("hash.zig");

pub const CuckooFilter = struct {
    /// Fingerprint slots, laid out as num_buckets * SLOTS_PER_BUCKET. 0 = empty.
    slots: []u16,
    num_buckets: u64,
    /// num_buckets - 1 (num_buckets is a power of two).
    bucket_mask: u64,
    /// Live fingerprint count.
    occupancy: u64,
    seed: u64,
    prng: std.Random.DefaultPrng,

    pub const SLOTS_PER_BUCKET: usize = 4;
    pub const MAX_KICKS: usize = 500;

    /// Target load factor used to size the table from a capacity hint.
    const LOAD_FACTOR: f64 = 0.94;
    /// Decorrelates the fingerprint's alternate-bucket hash from the item hash.
    const FP_SALT: u64 = 0x51_7C_C1_B7_27_22_0A_95;

    pub const Error = error{Full};

    /// Allocate a filter that can hold roughly `capacity` items before
    /// insertions begin to fail.
    pub fn init(allocator: std.mem.Allocator, cap: u64, seed: u64) !CuckooFilter {
        const need: f64 = @as(f64, @floatFromInt(@max(cap, 1))) /
            (@as(f64, @floatFromInt(SLOTS_PER_BUCKET)) * LOAD_FACTOR);
        var nb: u64 = 1;
        while (@as(f64, @floatFromInt(nb)) < need) nb <<= 1;
        return initWithBuckets(allocator, nb, seed);
    }

    /// Allocate a filter with an explicit bucket count (rounded up to a power
    /// of two, minimum 1).
    pub fn initWithBuckets(allocator: std.mem.Allocator, num_buckets: u64, seed: u64) !CuckooFilter {
        var nb: u64 = 1;
        while (nb < num_buckets) nb <<= 1;
        const slots = try allocator.alloc(u16, nb * SLOTS_PER_BUCKET);
        @memset(slots, 0);
        return .{
            .slots = slots,
            .num_buckets = nb,
            .bucket_mask = nb - 1,
            .occupancy = 0,
            .seed = seed,
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn deinit(self: *CuckooFilter, allocator: std.mem.Allocator) void {
        allocator.free(self.slots);
        self.* = undefined;
    }

    pub fn clear(self: *CuckooFilter) void {
        @memset(self.slots, 0);
        self.occupancy = 0;
    }

    const Loc = struct { fp: u16, idx1: u64 };

    fn locate(self: *const CuckooFilter, item: []const u8) Loc {
        const h = hash.hash64(item, self.seed);
        var fp: u16 = @truncate(h);
        if (fp == 0) fp = 1; // 0 is reserved for empty slots
        const idx1 = (h >> 32) & self.bucket_mask;
        return .{ .fp = fp, .idx1 = idx1 };
    }

    fn altIndex(self: *const CuckooFilter, index: u64, fp: u16) u64 {
        const hf = hash.hash64(std.mem.asBytes(&fp), self.seed ^ FP_SALT);
        return (index ^ (hf & self.bucket_mask)) & self.bucket_mask;
    }

    fn bucket(self: *CuckooFilter, index: u64) []u16 {
        const start = index * SLOTS_PER_BUCKET;
        return self.slots[start .. start + SLOTS_PER_BUCKET];
    }

    fn bucketInsert(self: *CuckooFilter, index: u64, fp: u16) bool {
        for (self.bucket(index)) |*slot| {
            if (slot.* == 0) {
                slot.* = fp;
                return true;
            }
        }
        return false;
    }

    fn bucketContains(self: *CuckooFilter, index: u64, fp: u16) bool {
        for (self.bucket(index)) |slot| {
            if (slot == fp) return true;
        }
        return false;
    }

    fn bucketDelete(self: *CuckooFilter, index: u64, fp: u16) bool {
        for (self.bucket(index)) |*slot| {
            if (slot.* == fp) {
                slot.* = 0;
                return true;
            }
        }
        return false;
    }

    /// Insert `item`. Returns error.Full if the table could not accommodate it
    /// after MAX_KICKS relocations (treat the filter as full).
    pub fn add(self: *CuckooFilter, item: []const u8) Error!void {
        const loc = self.locate(item);
        const idx2 = self.altIndex(loc.idx1, loc.fp);
        if (self.bucketInsert(loc.idx1, loc.fp) or self.bucketInsert(idx2, loc.fp)) {
            self.occupancy += 1;
            return;
        }

        // Both candidate buckets are full: evict and relocate.
        var index = if (self.prng.random().boolean()) loc.idx1 else idx2;
        var fp = loc.fp;
        var kicks: usize = 0;
        while (kicks < MAX_KICKS) : (kicks += 1) {
            const slot = self.prng.random().intRangeLessThan(usize, 0, SLOTS_PER_BUCKET);
            const victim = self.bucket(index)[slot];
            self.bucket(index)[slot] = fp;
            fp = victim;
            index = self.altIndex(index, fp);
            if (self.bucketInsert(index, fp)) {
                self.occupancy += 1;
                return;
            }
        }
        // Re-home the last displaced fingerprint so the table stays consistent,
        // then report the insertion as failed.
        _ = self.bucketInsert(index, fp);
        return error.Full;
    }

    /// Report whether `item` may be present.
    pub fn contains(self: *CuckooFilter, item: []const u8) bool {
        const loc = self.locate(item);
        if (self.bucketContains(loc.idx1, loc.fp)) return true;
        return self.bucketContains(self.altIndex(loc.idx1, loc.fp), loc.fp);
    }

    /// Remove one occurrence of `item`. Returns true if a matching fingerprint
    /// was found and removed.
    pub fn remove(self: *CuckooFilter, item: []const u8) bool {
        const loc = self.locate(item);
        if (self.bucketDelete(loc.idx1, loc.fp) or
            self.bucketDelete(self.altIndex(loc.idx1, loc.fp), loc.fp))
        {
            self.occupancy -= 1;
            return true;
        }
        return false;
    }

    /// Number of live fingerprints.
    pub fn len(self: *const CuckooFilter) u64 {
        return self.occupancy;
    }

    /// Total slot capacity (num_buckets * SLOTS_PER_BUCKET).
    pub fn capacity(self: *const CuckooFilter) u64 {
        return self.num_buckets * SLOTS_PER_BUCKET;
    }

    /// Fraction of slots currently occupied.
    pub fn loadFactor(self: *const CuckooFilter) f64 {
        return @as(f64, @floatFromInt(self.occupancy)) / @as(f64, @floatFromInt(self.capacity()));
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "empty filter contains nothing" {
    var c = try CuckooFilter.init(testing.allocator, 1000, 0);
    defer c.deinit(testing.allocator);
    try testing.expect(!c.contains("absent"));
    try testing.expectEqual(@as(u64, 0), c.len());
}

test "bucket count rounds up to a power of two" {
    var c = try CuckooFilter.initWithBuckets(testing.allocator, 100, 0);
    defer c.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 128), c.num_buckets);
    try testing.expectEqual(@as(u64, 0), c.num_buckets & (c.num_buckets - 1));
}

test "no false negatives up to load factor" {
    const n = 10_000;
    var c = try CuckooFilter.init(testing.allocator, n, 0x9999);
    defer c.deinit(testing.allocator);

    var buf: [24]u8 = undefined;
    var i: u32 = 0;
    var inserted: u32 = 0;
    while (i < n) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "item-{d}", .{i});
        c.add(key) catch break;
        inserted += 1;
    }
    // We should comfortably fit the requested capacity.
    try testing.expect(inserted >= n);
    try testing.expectEqual(@as(u64, inserted), c.len());

    i = 0;
    while (i < inserted) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "item-{d}", .{i});
        try testing.expect(c.contains(key));
    }
}

test "deletion removes membership and updates count" {
    var c = try CuckooFilter.init(testing.allocator, 1000, 7);
    defer c.deinit(testing.allocator);

    var buf: [24]u8 = undefined;
    var i: u32 = 0;
    while (i < 200) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "k-{d}", .{i});
        try c.add(key);
    }
    try testing.expectEqual(@as(u64, 200), c.len());

    // Delete half; each delete must succeed and decrement the count.
    i = 0;
    while (i < 100) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "k-{d}", .{i});
        try testing.expect(c.remove(key));
    }
    try testing.expectEqual(@as(u64, 100), c.len());

    // Deleting something never inserted fails.
    try testing.expect(!c.remove("never-added-xyz"));

    // The remaining half is still present.
    i = 100;
    while (i < 200) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "k-{d}", .{i});
        try testing.expect(c.contains(key));
    }
}

test "low false-positive rate on absent keys" {
    const n = 20_000;
    var c = try CuckooFilter.init(testing.allocator, n, 0xABCD);
    defer c.deinit(testing.allocator);

    var buf: [32]u8 = undefined;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "present-{d}", .{i});
        c.add(key) catch break;
    }

    const trials = 100_000;
    var fp: u32 = 0;
    i = 0;
    while (i < trials) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "absent-{d}", .{i});
        if (c.contains(key)) fp += 1;
    }
    const rate = @as(f64, @floatFromInt(fp)) / trials;
    // Theory is ~2*4/2^16 ≈ 1.2e-4; allow generous slack.
    try testing.expect(rate < 0.001);
}

test "clear empties the filter" {
    var c = try CuckooFilter.init(testing.allocator, 100, 0);
    defer c.deinit(testing.allocator);
    try c.add("x");
    try testing.expect(c.contains("x"));
    c.clear();
    try testing.expect(!c.contains("x"));
    try testing.expectEqual(@as(u64, 0), c.len());
}
