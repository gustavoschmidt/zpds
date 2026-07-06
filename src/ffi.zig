//! C-ABI surface for zpds.
//!
//! Everything crossing the FFI boundary stays simple: raw byte pointers,
//! integers and booleans. Higher-level ergonomics live in the Python wrapper.

const root = @import("root.zig");

/// Return the library version packed as (major << 16) | (minor << 8) | patch.
export fn zpds_version() u32 {
    return root.versionInt();
}
