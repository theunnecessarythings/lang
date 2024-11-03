const std = @import("std");

fn generateTablegen(b: *std.Build) !*std.Build.Step.Run {
    const llvm_path = "/mnt/ubuntu/home/sreeraj/Documents/llvm-project";
    const mlir_tblgen = llvm_path ++ "/build/bin/mlir-tblgen";

    const dialect_dir = try std.fs.cwd().openDir(
        "src/tablegen/dialect/",
        .{ .iterate = true },
    );
    var iter = dialect_dir.iterate();

    var gen_dialect_decls: ?*std.Build.Step.Run = null;
    var gen_dialect_defs: ?*std.Build.Step.Run = null;
    while (try iter.next()) |entry| {
        std.debug.print("{s}", .{entry.name});
        const output_file_defs = try std.fmt.allocPrint(
            b.allocator,
            "{s}/{s}.cpp.inc",
            .{ b.path("src/dialect/").getPath(b), entry.name[0 .. entry.name.len - 3] },
        );
        const output_file_decls = try std.fmt.allocPrint(
            b.allocator,
            "{s}/{s}.h.inc",
            .{ b.path("src/dialect/").getPath(b), entry.name[0 .. entry.name.len - 3] },
        );
        const input_file = try std.fmt.allocPrint(
            b.allocator,
            "{s}/{s}",
            .{ b.path("src/tablegen/dialect/").getPath(b), entry.name },
        );

        gen_dialect_decls = b.addSystemCommand(&.{
            mlir_tblgen,
            "-gen-dialect-decls",
            "-o",
            output_file_decls,
            input_file,
            "-I" ++ llvm_path ++ "/mlir/include",
        });
        if (gen_dialect_defs != null) {
            gen_dialect_decls.?.step.dependOn(&gen_dialect_defs.?.step);
        }

        gen_dialect_defs = b.addSystemCommand(&.{
            mlir_tblgen,
            "-gen-dialect-defs",
            "-o",
            output_file_defs,
            input_file,
            "-I" ++ llvm_path ++ "/mlir/include",
        });

        gen_dialect_defs.?.step.dependOn(&gen_dialect_decls.?.step);
    }

    const ops_dir = try std.fs.cwd().openDir(
        "src/tablegen/ops/",
        .{ .iterate = true },
    );
    iter = ops_dir.iterate();

    var gen_op_defs: ?*std.Build.Step.Run = null;
    var gen_op_decls: ?*std.Build.Step.Run = null;

    while (try iter.next()) |entry| {
        const output_file_defs = try std.fmt.allocPrint(
            b.allocator,
            "{s}/{s}.cpp.inc",
            .{ b.path("src/ops/").getPath(b), entry.name[0 .. entry.name.len - 3] },
        );
        const output_file_decls = try std.fmt.allocPrint(
            b.allocator,
            "{s}/{s}.h.inc",
            .{ b.path("src/ops/").getPath(b), entry.name[0 .. entry.name.len - 3] },
        );
        const input_file = try std.fmt.allocPrint(
            b.allocator,
            "{s}/{s}",
            .{ b.path("src/tablegen/ops/").getPath(b), entry.name },
        );

        gen_op_decls = b.addSystemCommand(&.{
            mlir_tblgen,
            "-gen-op-decls",
            "-o",
            output_file_decls,
            input_file,
            "-I" ++ llvm_path ++ "/mlir/include",
        });
        if (gen_op_defs != null) {
            gen_op_decls.?.step.dependOn(&gen_op_defs.?.step);
        }

        gen_op_defs = b.addSystemCommand(&.{
            mlir_tblgen,
            "-gen-op-defs",
            "-o",
            output_file_defs,
            input_file,
            "-I" ++ llvm_path ++ "/mlir/include",
        });

        gen_op_defs.?.step.dependOn(&gen_op_decls.?.step);
    }

    gen_op_defs.?.step.dependOn(&gen_dialect_defs.?.step);

    return gen_op_defs.?;
}

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
            "tests/test_parser.cpp",
        },
        .flags = &.{
            "-std=c++20",
            // "-Wall",
            // "-Wextra",
            // "-Wpedantic",
        },
    });

    var tablegen_step = generateTablegen(b) catch unreachable;
    exe_unit_tests.step.dependOn(&tablegen_step.step);

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
