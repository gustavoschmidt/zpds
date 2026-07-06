//! Bloom filter — probabilistic set membership with a tunable false-positive
//! rate and zero false negatives.
//!
//! Sizing follows the standard optimum: for `n` expected items at target
//! false-positive rate `p`,
//!
//!     m = ceil(-n ln p / (ln 2)^2)      bits
//!     k = round((m / n) ln 2)           hash functions
//!
//! The `k` probe positions come from a single hash pair via double hashing
//! (`hash.Pair.nth`), so each `add`/`contains` hashes the input just once.

const std = @import("std");
const hash = @import("hash.zig");

pub const Bloom = struct {
    /// Backing bit array, packed into 64-bit words.
    bits: []u64,
    /// Number of bits (m). Always a multiple of 64.
    n_bits: u64,
    /// Number of hash functions (k).
    k: u32,
    /// Number of `add` calls (counts duplicates; not a distinct-item count).
    inserted: u64,
    /// Seed mixed into the base hash.
    seed: u64,

    pub const Params = struct {
        n_bits: u64,
        k: u32,
    };

    /// Compute the optimal (m, k) for `expected_items` at `fp_rate`.
    /// `expected_items` is floored to 1 and `fp_rate` clamped to (0, 1).
    pub fn optimalParams(expected_items: u64, fp_rate: f64) Params {
        const n: f64 = @floatFromInt(@max(expected_items, 1));
        const p = std.math.clamp(fp_rate, 1e-12, 1.0 - 1e-12);

        const ln2 = std.math.ln2;
        const m_ideal = -(n * @log(p)) / (ln2 * ln2);
        // Round up to a whole number of 64-bit words.
        var m: u64 = @intFromFloat(@ceil(m_ideal));
        m = @max(m, 64);
        m = (m + 63) & ~@as(u64, 63);

        const k_ideal = (@as(f64, @floatFromInt(m)) / n) * ln2;
        const k: u32 = @intFromFloat(@max(1.0, @round(k_ideal)));
        return .{ .n_bits = m, .k = @min(k, 64) };
    }

    /// Allocate a filter sized for `expected_items` at `fp_rate`.
    pub fn init(allocator: std.mem.Allocator, expected_items: u64, fp_rate: f64, seed: u64) !Bloom {
        const p = optimalParams(expected_items, fp_rate);
        return initWithParams(allocator, p.n_bits, p.k, seed);
    }

    /// Allocate a filter with explicit parameters. `n_bits` is rounded up to a
    /// multiple of 64; `k` is floored to 1.
    pub fn initWithParams(allocator: std.mem.Allocator, n_bits: u64, k: u32, seed: u64) !Bloom {
        const m = (@max(n_bits, 64) + 63) & ~@as(u64, 63);
        const words = try allocator.alloc(u64, m / 64);
        @memset(words, 0);
        return .{
            .bits = words,
            .n_bits = m,
            .k = @max(k, 1),
            .inserted = 0,
            .seed = seed,
        };
    }

    pub fn deinit(self: *Bloom, allocator: std.mem.Allocator) void {
        allocator.free(self.bits);
        self.* = undefined;
    }

    /// Reset the filter to empty without reallocating.
    pub fn clear(self: *Bloom) void {
        @memset(self.bits, 0);
        self.inserted = 0;
    }

    inline fn setBit(self: *Bloom, pos: u64) void {
        self.bits[pos >> 6] |= @as(u64, 1) << @truncate(pos);
    }

    inline fn testBit(self: *const Bloom, pos: u64) bool {
        return (self.bits[pos >> 6] & (@as(u64, 1) << @truncate(pos))) != 0;
    }

    /// Insert `item`.
    pub fn add(self: *Bloom, item: []const u8) void {
        const pair = hash.hashPair(item, self.seed);
        var i: u32 = 0;
        while (i < self.k) : (i += 1) {
            self.setBit(pair.nth(i) % self.n_bits);
        }
        self.inserted += 1;
    }

    /// Report whether `item` may be present. False positives are possible; false
    /// negatives are not.
    pub fn contains(self: *const Bloom, item: []const u8) bool {
        const pair = hash.hashPair(item, self.seed);
        var i: u32 = 0;
        while (i < self.k) : (i += 1) {
            if (!self.testBit(pair.nth(i) % self.n_bits)) return false;
        }
        return true;
    }

    /// Number of `add` calls so far.
    pub fn len(self: *const Bloom) u64 {
        return self.inserted;
    }

    /// Theoretical false-positive probability at the current fill:
    /// (1 - e^{-k*inserted/m})^k.
    pub fn estimatedFpRate(self: *const Bloom) f64 {
        const kf: f64 = @floatFromInt(self.k);
        const nf: f64 = @floatFromInt(self.inserted);
        const mf: f64 = @floatFromInt(self.n_bits);
        return std.math.pow(f64, 1.0 - @exp(-kf * nf / mf), kf);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "optimalParams roughly matches the textbook sizing" {
    // 1M items at 1% -> ~9.585 bits/item, k = 7.
    const p = Bloom.optimalParams(1_000_000, 0.01);
    const bits_per_item = @as(f64, @floatFromInt(p.n_bits)) / 1_000_000.0;
    try testing.expect(bits_per_item > 9.0 and bits_per_item < 10.5);
    try testing.expectEqual(@as(u32, 7), p.k);
    try testing.expectEqual(@as(u64, 0), p.n_bits % 64);
}

test "no false negatives: every inserted item is found" {
    var b = try Bloom.init(testing.allocator, 10_000, 0.01, 0xABCD);
    defer b.deinit(testing.allocator);

    var buf: [24]u8 = undefined;
    var i: u32 = 0;
    while (i < 10_000) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "item-{d}", .{i});
        b.add(key);
    }
    try testing.expectEqual(@as(u64, 10_000), b.len());

    i = 0;
    while (i < 10_000) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "item-{d}", .{i});
        try testing.expect(b.contains(key));
    }
}

test "empty filter contains nothing" {
    var b = try Bloom.init(testing.allocator, 1000, 0.01, 0);
    defer b.deinit(testing.allocator);
    try testing.expect(!b.contains("nope"));
    try testing.expect(!b.contains(""));
}

test "clear empties the filter" {
    var b = try Bloom.init(testing.allocator, 1000, 0.01, 0);
    defer b.deinit(testing.allocator);
    b.add("x");
    try testing.expect(b.contains("x"));
    b.clear();
    try testing.expect(!b.contains("x"));
    try testing.expectEqual(@as(u64, 0), b.len());
}

test "statistical: observed FP rate tracks the target" {
    // Fill to capacity, then probe a large disjoint key set and measure the
    // empirical false-positive rate. It must land near the theoretical rate,
    // not merely below the nominal target.
    const n = 50_000;
    const target = 0.01;
    var b = try Bloom.init(testing.allocator, n, target, 0x1234_5678);
    defer b.deinit(testing.allocator);

    var buf: [32]u8 = undefined;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "present-{d}", .{i});
        b.add(key);
    }

    const trials = 200_000;
    var fp: u32 = 0;
    i = 0;
    while (i < trials) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "absent-{d}", .{i});
        if (b.contains(key)) fp += 1;
    }

    const observed = @as(f64, @floatFromInt(fp)) / trials;
    const theoretical = b.estimatedFpRate();
    // Theoretical rate for these params is a hair under the 1% target.
    try testing.expect(theoretical < target * 1.1);
    // Empirical rate should sit within 40% of theory given 200k trials — tight
    // enough to catch a broken probe scheme, loose enough to never flake.
    try testing.expect(observed > theoretical * 0.6);
    try testing.expect(observed < theoretical * 1.4);
}

test "distinct seeds give distinct filters" {
    var a = try Bloom.initWithParams(testing.allocator, 1024, 4, 1);
    defer a.deinit(testing.allocator);
    var b = try Bloom.initWithParams(testing.allocator, 1024, 4, 2);
    defer b.deinit(testing.allocator);
    a.add("collision-test");
    b.add("collision-test");
    // Same key, different seeds -> almost surely different bit patterns.
    try testing.expect(!std.mem.eql(u64, a.bits, b.bits));
}
