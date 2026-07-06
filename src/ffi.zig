//! C-ABI surface for zpds.
//!
//! Everything crossing the FFI boundary stays simple: raw byte pointers,
//! integers and booleans. Higher-level ergonomics live in the Python wrapper.

const root = @import("root.zig");
const hash = root.hash;

/// Return the library version packed as (major << 16) | (minor << 8) | patch.
export fn zpds_version() u32 {
    return root.versionInt();
}

/// One-shot 64-bit wyhash of `len` bytes at `data` under `seed`. A null/empty
/// buffer (len == 0) hashes the empty string.
export fn zpds_hash64(data: ?[*]const u8, len: usize, seed: u64) u64 {
    const bytes: []const u8 = if (len == 0) &.{} else data.?[0..len];
    return hash.hash64(bytes, seed);
}
