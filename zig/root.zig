//! zpds — Zig Probabilistic Data Structures.
//!
//! This is the root module. It re-exports the public Zig API and force-includes
//! the C-ABI layer (`ffi.zig`) so the exported entry points survive linking.

const std = @import("std");

pub const hash = @import("hash.zig");
pub const bloom = @import("bloom.zig");
pub const Bloom = bloom.Bloom;
pub const hll = @import("hll.zig");
pub const HyperLogLog = hll.HyperLogLog;
pub const cuckoo = @import("cuckoo.zig");
pub const CuckooFilter = cuckoo.CuckooFilter;
pub const count_min = @import("count_min.zig");
pub const CountMin = count_min.CountMin;

pub const version = struct {
    pub const major: u32 = 0;
    pub const minor: u32 = 0;
    pub const patch: u32 = 1;
};

/// The library version packed as (major << 16) | (minor << 8) | patch.
pub fn versionInt() u32 {
    return (version.major << 16) | (version.minor << 8) | version.patch;
}

// Force the C-ABI exports to be linked into the shared/static library.
comptime {
    _ = @import("ffi.zig");
}

test "version is 0.0.1" {
    try std.testing.expectEqual(@as(u32, 0x000001), versionInt());
}
