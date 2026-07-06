//! zpds — Zig Probabilistic Data Structures.
//!
//! This is the root module. It re-exports the public Zig API and force-includes
//! the C-ABI layer (`ffi.zig`) so the exported entry points survive linking.

const std = @import("std");

pub const hash = @import("hash.zig");

pub const version = struct {
    pub const major: u32 = 0;
    pub const minor: u32 = 1;
    pub const patch: u32 = 0;
};

/// The library version packed as (major << 16) | (minor << 8) | patch.
pub fn versionInt() u32 {
    return (version.major << 16) | (version.minor << 8) | version.patch;
}

// Force the C-ABI exports to be linked into the shared/static library.
comptime {
    _ = @import("ffi.zig");
}

test "version is 0.1.0" {
    try std.testing.expectEqual(@as(u32, 0x000100), versionInt());
}
