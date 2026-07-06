//! Count-Min Sketch — sublinear frequency estimation for a stream (Cormode &
//! Muthukrishnan, 2005).
//!
//! A `depth × width` grid of counters. Each item increments one counter per
//! row (row r at column g_r(item) mod width, via double hashing). A frequency
//! query takes the minimum across rows. Estimates never *under*count; the
//! overestimate is at most `epsilon * total_count` with probability
//! `1 - delta`, given width = ceil(e/epsilon), depth = ceil(ln(1/delta)).

const std = @import("std");
const hash = @import("hash.zig");

pub const CountMin = struct {
    /// Row-major counters, depth * width entries.
    counters: []u64,
    width: u64,
    depth: u64,
    /// Sum of all added increments.
    total: u64,
    seed: u64,

    /// Size a sketch for additive error `epsilon * total` with failure
    /// probability `delta`. Both are clamped to (0, 1).
    pub fn init(allocator: std.mem.Allocator, epsilon: f64, delta: f64, seed: u64) !CountMin {
        const eps = std.math.clamp(epsilon, 1e-9, 1.0 - 1e-9);
        const del = std.math.clamp(delta, 1e-9, 1.0 - 1e-9);
        const width: u64 = @intFromFloat(@ceil(std.math.e / eps));
        const depth: u64 = @intFromFloat(@ceil(@log(1.0 / del)));
        return initWithParams(allocator, @max(width, 1), @max(depth, 1), seed);
    }

    /// Allocate a sketch with explicit `width` and `depth` (each floored to 1).
    pub fn initWithParams(allocator: std.mem.Allocator, width: u64, depth: u64, seed: u64) !CountMin {
        const w = @max(width, 1);
        const d = @max(depth, 1);
        const counters = try allocator.alloc(u64, w * d);
        @memset(counters, 0);
        return .{ .counters = counters, .width = w, .depth = d, .total = 0, .seed = seed };
    }

    pub fn deinit(self: *CountMin, allocator: std.mem.Allocator) void {
        allocator.free(self.counters);
        self.* = undefined;
    }

    /// Return an independent copy of the sketch (its own counter grid).
    pub fn clone(self: *const CountMin, allocator: std.mem.Allocator) !CountMin {
        return .{
            .counters = try allocator.dupe(u64, self.counters),
            .width = self.width,
            .depth = self.depth,
            .total = self.total,
            .seed = self.seed,
        };
    }

    pub fn clear(self: *CountMin) void {
        @memset(self.counters, 0);
        self.total = 0;
    }

    /// Add `count` occurrences of `item`.
    pub fn add(self: *CountMin, item: []const u8, count: u64) void {
        const pair = hash.hashPair(item, self.seed);
        var r: u64 = 0;
        while (r < self.depth) : (r += 1) {
            const col = pair.nth(r) % self.width;
            self.counters[r * self.width + col] += count;
        }
        self.total += count;
    }

    /// Add a single occurrence of `item`.
    pub fn increment(self: *CountMin, item: []const u8) void {
        self.add(item, 1);
    }

    /// Estimated frequency of `item` (never an underestimate).
    pub fn estimate(self: *const CountMin, item: []const u8) u64 {
        const pair = hash.hashPair(item, self.seed);
        var min: u64 = std.math.maxInt(u64);
        var r: u64 = 0;
        while (r < self.depth) : (r += 1) {
            const col = pair.nth(r) % self.width;
            min = @min(min, self.counters[r * self.width + col]);
        }
        return min;
    }

    /// Sum of all increments added so far.
    pub fn totalCount(self: *const CountMin) u64 {
        return self.total;
    }

    /// The additive error bound: estimate <= true + errorBound, w.h.p.
    pub fn errorBound(self: *const CountMin) f64 {
        return std.math.e / @as(f64, @floatFromInt(self.width)) * @as(f64, @floatFromInt(self.total));
    }

    /// Merge `other` into `self` (counter-wise sum). Requires equal dimensions
    /// and seed so the two sketches address counters identically.
    pub fn merge(self: *CountMin, other: *const CountMin) error{ShapeMismatch}!void {
        if (self.width != other.width or self.depth != other.depth or self.seed != other.seed) {
            return error.ShapeMismatch;
        }
        for (self.counters, other.counters) |*a, b| a.* += b;
        self.total += other.total;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "empty sketch estimates zero" {
    var cm = try CountMin.init(testing.allocator, 0.001, 0.001, 0);
    defer cm.deinit(testing.allocator);
    try testing.expectEqual(@as(u64, 0), cm.estimate("anything"));
    try testing.expectEqual(@as(u64, 0), cm.totalCount());
}

test "sizing matches epsilon/delta formulas" {
    var cm = try CountMin.init(testing.allocator, 0.01, 0.01, 0);
    defer cm.deinit(testing.allocator);
    // width = ceil(e/0.01) = 272, depth = ceil(ln(100)) = 5.
    try testing.expectEqual(@as(u64, 272), cm.width);
    try testing.expectEqual(@as(u64, 5), cm.depth);
}

test "estimate never underestimates the true frequency" {
    var cm = try CountMin.init(testing.allocator, 0.001, 0.001, 42);
    defer cm.deinit(testing.allocator);

    // Insert item i exactly i times.
    var buf: [24]u8 = undefined;
    var i: u32 = 1;
    while (i <= 500) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "w-{d}", .{i});
        cm.add(key, i);
    }

    i = 1;
    while (i <= 500) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "w-{d}", .{i});
        try testing.expect(cm.estimate(key) >= i);
    }
}

test "overestimate stays within the epsilon*total bound" {
    const eps = 0.001;
    var cm = try CountMin.init(testing.allocator, eps, 0.001, 7);
    defer cm.deinit(testing.allocator);

    var buf: [24]u8 = undefined;
    var i: u32 = 1;
    while (i <= 1000) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "w-{d}", .{i});
        cm.add(key, i);
    }

    const bound = cm.errorBound();
    i = 1;
    while (i <= 1000) : (i += 1) {
        const key = try std.fmt.bufPrint(&buf, "w-{d}", .{i});
        const over = @as(f64, @floatFromInt(cm.estimate(key) - i));
        try testing.expect(over <= bound);
    }
}

test "merge sums two sketches" {
    var a = try CountMin.initWithParams(testing.allocator, 500, 5, 99);
    defer a.deinit(testing.allocator);
    var b = try CountMin.initWithParams(testing.allocator, 500, 5, 99);
    defer b.deinit(testing.allocator);

    a.add("x", 10);
    b.add("x", 32);
    b.add("y", 5);
    try a.merge(&b);

    try testing.expect(a.estimate("x") >= 42);
    try testing.expect(a.estimate("y") >= 5);
    try testing.expectEqual(@as(u64, 47), a.totalCount());
}

test "merge rejects mismatched shape or seed" {
    var a = try CountMin.initWithParams(testing.allocator, 500, 5, 1);
    defer a.deinit(testing.allocator);
    var b = try CountMin.initWithParams(testing.allocator, 500, 5, 2); // different seed
    defer b.deinit(testing.allocator);
    try testing.expectError(error.ShapeMismatch, a.merge(&b));
}

test "clear resets counters and total" {
    var cm = try CountMin.init(testing.allocator, 0.01, 0.01, 0);
    defer cm.deinit(testing.allocator);
    cm.add("x", 7);
    try testing.expect(cm.estimate("x") >= 7);
    cm.clear();
    try testing.expectEqual(@as(u64, 0), cm.estimate("x"));
    try testing.expectEqual(@as(u64, 0), cm.totalCount());
}
