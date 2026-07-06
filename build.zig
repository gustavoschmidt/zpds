const std = @import("std");

// zpds — Zig Probabilistic Data Structures.
//
// Builds a C-ABI shared library (consumed by the Python wrapper) and a static
// library, and wires up the `zig build test` step that runs every `test` block
// in the core module.
pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The core module: all of the Zig source, rooted at src/root.zig. root.zig
    // force-references src/ffi.zig so the `export fn` C-ABI entry points are
    // retained in the linked artifacts.
    const core = b.addModule("zpds", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const shared = b.addLibrary(.{
        .name = "zpds",
        .root_module = core,
        .linkage = .dynamic,
    });
    b.installArtifact(shared);

    const static = b.addLibrary(.{
        .name = "zpds",
        .root_module = core,
        .linkage = .static,
    });
    b.installArtifact(static);

    // Unit tests: every `test` block reachable from the core module.
    const core_tests = b.addTest(.{ .root_module = core });
    const run_core_tests = b.addRunArtifact(core_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_core_tests.step);
}
