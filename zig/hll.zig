//! HyperLogLog — cardinality (distinct-count) estimation in fixed memory.
//!
//! Classic Flajolet et al. estimator over `m = 2^precision` single-byte
//! registers. Each item's 64-bit hash splits into a `precision`-bit register
//! index and a suffix; the register keeps the largest observed "rank" (leftmost
//! 1-bit position) for its bucket. The harmonic mean of 2^-rank across
//! registers estimates the cardinality, with linear counting substituted in the
//! small-cardinality regime.
//!
//! Standard relative error is ~1.04/sqrt(m) (e.g. ~0.8% at precision 14). We
//! hash to 64 bits, so the 32-bit large-range correction is unnecessary.

const std = @import("std");
const hash = @import("hash.zig");

pub const MIN_PRECISION: u5 = 4;
pub const MAX_PRECISION: u5 = 18;

pub const HyperLogLog = struct {
    /// One rank per bucket; length is 2^precision.
    registers: []u8,
    precision: u5,
    seed: u64,

    /// Allocate an HLL with the given `precision` (clamped to [4, 18]).
    pub fn init(allocator: std.mem.Allocator, precision: u5, seed: u64) !HyperLogLog {
        const p = std.math.clamp(precision, MIN_PRECISION, MAX_PRECISION);
        const m = @as(usize, 1) << p;
        const registers = try allocator.alloc(u8, m);
        @memset(registers, 0);
        return .{ .registers = registers, .precision = p, .seed = seed };
    }

    pub fn deinit(self: *HyperLogLog, allocator: std.mem.Allocator) void {
        allocator.free(self.registers);
        self.* = undefined;
    }

    /// Return an independent copy of the sketch (its own register array).
    pub fn clone(self: *const HyperLogLog, allocator: std.mem.Allocator) !HyperLogLog {
        return .{
            .registers = try allocator.dupe(u8, self.registers),
            .precision = self.precision,
            .seed = self.seed,
        };
    }

    pub fn clear(self: *HyperLogLog) void {
        @memset(self.registers, 0);
    }

    /// Number of registers (m).
    pub fn size(self: *const HyperLogLog) usize {
        return self.registers.len;
    }

    /// Add an item to the multiset.
    pub fn add(self: *HyperLogLog, item: []const u8) void {
        const x = hash.hash64(item, self.seed);
        const p = self.precision;
        // Top `p` bits select the register.
        const idx: usize = @intCast(x >> @intCast(64 - @as(u32, p)));
        // Rank = leftmost 1-bit position within the remaining (64 - p) bits, +1.
        const w = x << p;
        const rank: u8 = @intCast(@min(@clz(w) + 1, 64 - @as(u32, p) + 1));
        if (rank > self.registers[idx]) self.registers[idx] = rank;
    }

    /// Bias-correction constant alpha_m for the harmonic-mean estimator.
    fn alpha(m: usize) f64 {
        return switch (m) {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            else => 0.7213 / (1.0 + 1.079 / @as(f64, @floatFromInt(m))),
        };
    }

    /// Estimated distinct-item count.
    pub fn estimate(self: *const HyperLogLog) f64 {
        const m = self.registers.len;
        const mf: f64 = @floatFromInt(m);

        var sum: f64 = 0;
        var zeros: usize = 0;
        for (self.registers) |r| {
            sum += std.math.ldexp(@as(f64, 1.0), -@as(i32, r)); // 2^-r
            if (r == 0) zeros += 1;
        }

        const raw = alpha(m) * mf * mf / sum;

        // Small-range correction: linear counting when many registers are empty.
        if (raw <= 2.5 * mf and zeros != 0) {
            return mf * @log(mf / @as(f64, @floatFromInt(zeros)));
        }
        return raw;
    }

    /// Estimated distinct-item count, rounded to an integer.
    pub fn count(self: *const HyperLogLog) u64 {
        return @intFromFloat(@round(self.estimate()));
    }

    /// Expected standard relative error, ~1.04/sqrt(m).
    pub fn relativeError(self: *const HyperLogLog) f64 {
        return 1.04 / @sqrt(@as(f64, @floatFromInt(self.registers.len)));
    }

    /// Merge `other` into `self` (register-wise max). Requires equal precision.
    pub fn merge(self: *HyperLogLog, other: *const HyperLogLog) error{PrecisionMismatch}!void {
        if (self.precision != other.precision) return error.PrecisionMismatch;
        for (self.registers, other.registers) |*a, b| {
            if (b > a.*) a.* = b;
        }
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn addRange(h: *HyperLogLog, start: u32, end: u32) !void {
    var buf: [32]u8 = undefined;
    var i = start;
    while (i < end) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "elem-{d}", .{i});
        h.add(key);
    }
}

test "empty HLL estimates zero" {
    var h = try HyperLogLog.init(testing.allocator, 14, 0);
    defer h.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 0), h.count());
}

test "precision is clamped to the valid range" {
    var lo = try HyperLogLog.init(testing.allocator, 1, 0);
    defer lo.deinit(testing.allocator);
    try testing.expectEqual(MIN_PRECISION, lo.precision);

    var hi = try HyperLogLog.init(testing.allocator, 25, 0);
    defer hi.deinit(testing.allocator);
    try testing.expectEqual(MAX_PRECISION, hi.precision);
}

test "duplicates do not inflate the estimate" {
    var h = try HyperLogLog.init(testing.allocator, 14, 0);
    defer h.deinit(testing.allocator);
    var i: u32 = 0;
    while (i < 1000) : (i += 1) h.add("same-key");
    // A single distinct element -> estimate near 1 (linear counting regime).
    try testing.expect(h.count() <= 2);
}

test "accuracy across cardinalities is within a few standard errors" {
    const cards = [_]u32{ 100, 1_000, 10_000, 100_000, 1_000_000 };
    for (cards) |true_n| {
        var h = try HyperLogLog.init(testing.allocator, 14, 0xC0FFEE);
        defer h.deinit(testing.allocator);
        try addRange(&h, 0, true_n);

        const est = h.estimate();
        const rel_err = @abs(est - @as(f64, @floatFromInt(true_n))) / @as(f64, @floatFromInt(true_n));
        // 3x the ~0.8% standard error at precision 14, with headroom. The keys
        // and hash are deterministic, so this never flakes.
        try testing.expect(rel_err < 0.03);
    }
}

test "small cardinalities use linear counting accurately" {
    var h = try HyperLogLog.init(testing.allocator, 14, 1);
    defer h.deinit(testing.allocator);
    try addRange(&h, 0, 50);
    const est = h.estimate();
    try testing.expect(est > 45 and est < 55);
}

test "merge unions two sketches" {
    var a = try HyperLogLog.init(testing.allocator, 14, 7);
    defer a.deinit(testing.allocator);
    var b = try HyperLogLog.init(testing.allocator, 14, 7);
    defer b.deinit(testing.allocator);

    try addRange(&a, 0, 50_000); // [0, 50k)
    try addRange(&b, 25_000, 75_000); // overlaps [25k, 50k)
    try a.merge(&b);

    // Union cardinality is 75k distinct.
    const est = a.estimate();
    const rel_err = @abs(est - 75_000.0) / 75_000.0;
    try testing.expect(rel_err < 0.03);
}

test "merge rejects precision mismatch" {
    var a = try HyperLogLog.init(testing.allocator, 12, 0);
    defer a.deinit(testing.allocator);
    var b = try HyperLogLog.init(testing.allocator, 14, 0);
    defer b.deinit(testing.allocator);
    try testing.expectError(error.PrecisionMismatch, a.merge(&b));
}
