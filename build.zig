const std = @import("std");

const LLVM_PATH = "/mnt/ubuntu/home/sreeraj/Documents/llvm-project";

fn generateTablegen(b: *std.Build) !void {
    const mlir_tblgen = LLVM_PATH ++ "/build/bin/mlir-tblgen";

    const dialect_dir = try std.fs.cwd().openDir(
        "src/tablegen/",
        .{ .iterate = true },
    );

    const mlir_types = &[_][]const u8{ "dialect", "op", "typedef" };

    var iter = dialect_dir.iterate();

    for (mlir_types) |mlir_type| {
        iter.reset();
        while (try iter.next()) |entry| {
            const output_file_defs = try std.fmt.allocPrint(
                b.allocator,
                "{s}/{s}/{s}.cpp.inc",
                .{ b.path("src/").getPath(b), mlir_type, entry.name[0 .. entry.name.len - 3] },
            );
            const output_file_decls = try std.fmt.allocPrint(
                b.allocator,
                "{s}/{s}/{s}.h.inc",
                .{ b.path("src/").getPath(b), mlir_type, entry.name[0 .. entry.name.len - 3] },
            );
            const input_file = try std.fmt.allocPrint(
                b.allocator,
                "{s}/{s}",
                .{ b.path("src/tablegen/").getPath(b), entry.name },
            );

            const gen_decls = b.addSystemCommand(&.{
                mlir_tblgen,
                try std.fmt.allocPrint(b.allocator, "-gen-{s}-decls", .{mlir_type}),
                "-o",
                output_file_decls,
                input_file,
                "-I" ++ LLVM_PATH ++ "/mlir/include",
            });
            b.getInstallStep().dependOn(&gen_decls.step);

            const gen_defs = b.addSystemCommand(&.{
                mlir_tblgen,
                try std.fmt.allocPrint(b.allocator, "-gen-{s}-defs", .{mlir_type}),
                "-o",
                output_file_defs,
                input_file,
                "-I" ++ LLVM_PATH ++ "/mlir/include",
            });

            b.getInstallStep().dependOn(&gen_defs.step);
        }
    }
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
            "src/Dialect.cpp",
        },
        .flags = &.{
            // "-std=c++20",
            // "-Wall",
            // "-Wextra",
            // "-Wpedantic",
            \\-I/mnt/ubuntu/home/sreeraj/Documents/llvm-project/llvm/include -I/mnt/ubuntu/home/sreeraj/Documents/llvm-project/build/include -std=c++17   -fno-exceptions -funwind-tables -fno-rtti -D_GNU_SOURCE -D_DEBUG -D_GLIBCXX_ASSERTIONS -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS
            \\-L/mnt/ubuntu/home/sreeraj/Documents/llvm-project/build/lib
            \\-lLLVMWindowsManifest -lLLVMXRay -lLLVMLibDriver -lLLVMDlltoolDriver -lLLVMTextAPIBinaryReader -lLLVMCoverage -lLLVMLineEditor -lLLVMX86TargetMCA -lLLVMX86Disassembler -lLLVMX86AsmParser -lLLVMX86CodeGen -lLLVMX86Desc -lLLVMX86Info -lLLVMNVPTXCodeGen -lLLVMNVPTXDesc -lLLVMNVPTXInfo -lLLVMOrcDebugging -lLLVMOrcJIT -lLLVMWindowsDriver -lLLVMMCJIT -lLLVMJITLink -lLLVMInterpreter -lLLVMExecutionEngine -lLLVMRuntimeDyld -lLLVMOrcTargetProcess -lLLVMOrcShared -lLLVMDWP -lLLVMDebugInfoLogicalView -lLLVMDebugInfoGSYM -lLLVMOption -lLLVMObjectYAML -lLLVMObjCopy -lLLVMMCA -lLLVMMCDisassembler -lLLVMLTO -lLLVMPasses -lLLVMHipStdPar -lLLVMCFGuard -lLLVMCoroutines -lLLVMipo -lLLVMVectorize -lLLVMSandboxIR -lLLVMLinker -lLLVMInstrumentation -lLLVMFrontendOpenMP -lLLVMFrontendOffloading -lLLVMFrontendOpenACC -lLLVMFrontendHLSL -lLLVMFrontendDriver -lLLVMFrontendAtomic -lLLVMExtensions -lLLVMDWARFLinkerParallel -lLLVMDWARFLinkerClassic -lLLVMDWARFLinker -lLLVMGlobalISel -lLLVMMIRParser -lLLVMAsmPrinter -lLLVMSelectionDAG -lLLVMCodeGen -lLLVMTarget -lLLVMObjCARCOpts -lLLVMCodeGenTypes -lLLVMCGData -lLLVMIRPrinter -lLLVMInterfaceStub -lLLVMFileCheck -lLLVMFuzzMutate -lLLVMScalarOpts -lLLVMInstCombine -lLLVMAggressiveInstCombine -lLLVMTransformUtils -lLLVMBitWriter -lLLVMAnalysis -lLLVMProfileData -lLLVMSymbolize -lLLVMDebugInfoBTF -lLLVMDebugInfoPDB -lLLVMDebugInfoMSF -lLLVMDebugInfoCodeView -lLLVMDebugInfoDWARF -lLLVMObject -lLLVMTextAPI -lLLVMMCParser -lLLVMIRReader -lLLVMAsmParser -lLLVMMC -lLLVMBitReader -lLLVMFuzzerCLI -lLLVMCore -lLLVMRemarks -lLLVMBitstreamReader -lLLVMBinaryFormat -lLLVMTargetParser -lLLVMTableGen -lLLVMSupport -lLLVMDemangle
        },
    });
    exe.addObjectFile(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" });
    exe_unit_tests.addIncludePath(b.path("include"));
    exe_unit_tests.addIncludePath(
        .{ .cwd_relative = LLVM_PATH ++ "/build/include" },
    );
    exe_unit_tests.addIncludePath(
        .{ .cwd_relative = LLVM_PATH ++ "/build/tools/mlir/include" },
    );

    exe_unit_tests.addIncludePath(
        .{ .cwd_relative = LLVM_PATH ++ "/mlir/include" },
    );
    exe_unit_tests.addIncludePath(
        .{ .cwd_relative = LLVM_PATH ++ "/llvm/include" },
    );

    exe_unit_tests.addLibraryPath(
        .{ .cwd_relative = LLVM_PATH ++ "/build/lib" },
    );
    // exe_unit_tests.linkSystemLibrary("LLVMSupport");
    exe_unit_tests.linkSystemLibrary("MLIRAnalysis");
    exe_unit_tests.linkSystemLibrary("MLIRFunctionInterfaces");
    exe_unit_tests.linkSystemLibrary("MLIRIR");
    exe_unit_tests.linkSystemLibrary("MLIRParser");
    exe_unit_tests.linkSystemLibrary("MLIRSideEffectInterfaces");
    exe_unit_tests.linkSystemLibrary("MLIRTransforms");

    generateTablegen(b) catch unreachable;

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

    // exe_unit_tests.linkLibC();
    // exe_unit_tests.linkLibCpp();
    exe_unit_tests.linkLibrary(catch2);
    b.installArtifact(exe_unit_tests);

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    if (b.args) |args| {
        run_exe_unit_tests.addArgs(args);
    }

    const test_step = b.step("test", "Run the unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
