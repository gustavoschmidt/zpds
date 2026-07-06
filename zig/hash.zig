//! Non-cryptographic hashing for the probabilistic data structures.
//!
//! We use wyhash (`std.hash.Wyhash`): fast, good avalanche, and already
//! reference-verified inside the Zig standard library. On top of the raw 64-bit
//! hash we derive an independent hash *pair* used for double hashing
//! (Kirsch–Mitzenmacher) so a single pass over the input yields the `k` probe
//! positions a Bloom filter or Count-Min sketch needs.

const std = @import("std");
const Wyhash = std.hash.Wyhash;

/// One-shot 64-bit wyhash of `data` under `seed`.
pub fn hash64(data: []const u8, seed: u64) u64 {
    return Wyhash.hash(seed, data);
}

/// Two independent 64-bit hashes of the same input.
pub const Pair = struct {
    h1: u64,
    h2: u64,

    /// The i-th derived hash, via g_i = h1 + i*h2 (Kirsch–Mitzenmacher). This
    /// generates `k` well-distributed values from just two base hashes.
    pub fn nth(self: Pair, i: u64) u64 {
        return self.h1 +% (i *% self.h2);
    }
};

/// Golden-ratio constant used to decorrelate the second hash's seed.
const PHI64: u64 = 0x9E3779B97F4A7C15;

/// Derive a hash pair for `data`. `h2` is forced odd so the double-hashing
/// probe sequence g_i = h1 + i*h2 is a full-period walk when reduced modulo a
/// power of two, avoiding short cycles that would cluster probes.
pub fn hashPair(data: []const u8, seed: u64) Pair {
    const h1 = Wyhash.hash(seed, data);
    const h2 = Wyhash.hash(seed ^ PHI64, data) | 1;
    return .{ .h1 = h1, .h2 = h2 };
}

test "golden wyhash vectors" {
    // Regression vectors: these lock the exact hash our data structures depend
    // on. A change here means the on-wire/behavioral hash changed.
    try std.testing.expectEqual(@as(u64, 0x0409638ee2bde459), hash64("", 0));
    try std.testing.expectEqual(@as(u64, 0x28d2053309d28531), hash64("a", 0));
    try std.testing.expectEqual(@as(u64, 0x02a4f1d7cb516c72), hash64("abc", 0));
    try std.testing.expectEqual(@as(u64, 0x41d032e1df79b67e), hash64("message digest", 0));
    try std.testing.expectEqual(@as(u64, 0xdbe5b1e5823255b7), hash64("abc", 1));
    try std.testing.expectEqual(
        @as(u64, 0x4f0e75ed5d33843d),
        hash64("The quick brown fox jumps over the lazy dog", 42),
    );
}

test "determinism: same input+seed hashes identically" {
    const data = "streaming dedup key";
    try std.testing.expectEqual(hash64(data, 7), hash64(data, 7));
    const p = hashPair(data, 7);
    const q = hashPair(data, 7);
    try std.testing.expectEqual(p.h1, q.h1);
    try std.testing.expectEqual(p.h2, q.h2);
}

test "seed sensitivity: different seeds diverge" {
    const data = "abc";
    try std.testing.expect(hash64(data, 0) != hash64(data, 1));
}

test "hashPair: h2 is odd and independent of h1" {
    const p = hashPair("hello world", 123);
    try std.testing.expectEqual(@as(u64, 1), p.h2 & 1);
    try std.testing.expect(p.h1 != p.h2);
}

test "hashPair.nth is a linear probe sequence" {
    const p = hashPair("hello world", 123);
    try std.testing.expectEqual(p.h1, p.nth(0));
    try std.testing.expectEqual(p.h1 +% p.h2, p.nth(1));
    try std.testing.expectEqual(p.h1 +% (2 *% p.h2), p.nth(2));
}

test "uniformity: bucket distribution is roughly even" {
    // Hash many distinct keys into 256 buckets; the fill should be close to
    // uniform. A chi-square-style deviation check catches a badly broken hash.
    const n_buckets = 256;
    const n_keys = 100_000;
    var counts = [_]u32{0} ** n_buckets;
    var buf: [32]u8 = undefined;
    var i: u32 = 0;
    while (i < n_keys) : (i += 1) {
        const key = std.fmt.bufPrint(&buf, "key-{d}", .{i}) catch unreachable;
        counts[hash64(key, 0) % n_buckets] += 1;
    }

    const expected: f64 = @as(f64, n_keys) / n_buckets;
    var chi2: f64 = 0;
    for (counts) |c| {
        const d = @as(f64, @floatFromInt(c)) - expected;
        chi2 += (d * d) / expected;
    }
    // 255 degrees of freedom: the chi-square statistic is ~255 on average and
    // effectively never exceeds ~360 for a good hash. A loose ceiling flags
    // only genuine breakage while staying non-flaky.
    try std.testing.expect(chi2 < 360.0);
}
