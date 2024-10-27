const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});

    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "mlir-lang-cpp",
        .target = target,
        .optimize = optimize,
        .pic = true,
    });

    exe.addCSourceFiles(.{
        .files = &.{
            "main.cpp",
        },
        .flags = &.{
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
        },
    });

    exe.linkLibC();
    exe.linkLibCpp();

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the executable");
    run_step.dependOn(&run_cmd.step);

    // Unit Tests
    const exe_unit_tests = b.addExecutable(.{
        .target = target,
        .optimize = optimize,
        .name = "unit_tests",
        .pic = true,
    });
    exe_unit_tests.addCSourceFiles(.{
        .files = &.{
            "tests/test_lexer.cpp",
        },
        .flags = &.{
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
        },
    });

    const catch2 = b.addStaticLibrary(.{
        .name = "catch2",
        .optimize = optimize,
        .pic = true,
        .target = target,
    });
    catch2.addCSourceFiles(.{
        .files = &.{
            "third-party/catch2/catch_amalgamated.cpp",
        },
        .flags = &.{
            "-std=c++20",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
        },
    });
    catch2.linkLibC();
    catch2.linkLibCpp();

    exe_unit_tests.linkLibC();
    exe_unit_tests.linkLibCpp();
    exe_unit_tests.linkLibrary(catch2);
    b.installArtifact(exe_unit_tests);

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    if (b.args) |args| {
        run_exe_unit_tests.addArgs(args);
    }

    const test_step = b.step("test", "Run the unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
