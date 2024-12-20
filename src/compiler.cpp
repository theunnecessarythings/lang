/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
#include "LangGen.h"
#include "MLIRGen.h"
#include "analyzer.hpp"
#include "ast.hpp"
#include "dialect/LangDialect.h"
#include "dialect/LangOpsDialect.cpp.inc"
#include "json_dumper.hpp"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "parser.hpp"
#include "passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <system_error>
#include <utility>

#define DEBUG_TYPE "compiler"

std::unique_ptr<Program>
Compiler::parseInputFile(llvm::StringRef filename,
                         std::shared_ptr<Context> context) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file_or_err =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = file_or_err.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = file_or_err.get()->getBuffer();
  auto lexer = std::make_unique<Lexer>(buffer.str(), 1);
  auto parser = Parser(std::move(lexer), context);
  context->source_mgr.AddNewSourceBuffer(std::move(*file_or_err),
                                         llvm::SMLoc());
  auto tree = parser.parseProgram();
  auto analyzer = Analyzer(context);
  analyzer.analyze(tree.get());
  return tree;
}

int Compiler::dumpJSON(InputType input_type, llvm::StringRef input_filename) {
  if (input_type == InputType::MLIR) {
    llvm::errs() << "Can't dump a Lang AST JSON when the input is MLIR\n";
    return 5;
  }

  auto context = std::make_shared<Context>();
  auto module_ast = parseInputFile(input_filename, context);
  if (!module_ast)
    return 1;
  auto dumper = JsonDumper();
  dumper.dump(module_ast.get());
  auto str = dumper.toString();
  llvm::errs() << str << "\n";
  return 0;
}

int Compiler::dumpAST(InputType input_type, llvm::StringRef input_filename) {
  if (input_type == InputType::MLIR) {
    llvm::errs() << "Can't dump a Lang AST when the input is MLIR\n";
    return 5;
  }

  auto context = std::make_shared<Context>();
  auto module_ast = parseInputFile(input_filename, context);
  if (!module_ast)
    return 1;

  auto dumper = AstDumper();
  dumper.dump(module_ast.get());
  llvm::errs() << dumper.toString() << "\n";
  return 0;
}

int Compiler::loadMLIR(InputType input_type, llvm::StringRef input_filename,
                       llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module, bool lang) {
  if (input_type != InputType::MLIR &&
      !llvm::StringRef(input_filename).ends_with(".mlir")) {
    auto compiler_context = std::make_shared<Context>();
    auto module_ast = parseInputFile(input_filename, compiler_context);
    if (!module_ast)
      return 6;
    if (!lang)
      module = mlirGen(context, module_ast.get());
    else
      module = langGen(context, module_ast.get(), *compiler_context);
    return !module ? 1 : 0;
  }

  // Otherwise, the input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file_or_err =
      llvm::MemoryBuffer::getFileOrSTDIN(input_filename);
  if (std::error_code ec = file_or_err.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir.
  sourceMgr.AddNewSourceBuffer(std::move(*file_or_err), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << input_filename << "\n";
    return 3;
  }
  return 0;
}

int Compiler::runJit(mlir::ModuleOp module, bool enable_opt) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto opt_pipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enable_opt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engine_options;
  engine_options.transformer = opt_pipeline;
  auto maybe_engine = mlir::ExecutionEngine::create(module, engine_options);
  assert(maybe_engine && "failed to construct an execution engine");
  auto engine = std::move(maybe_engine.get());

  // Invoke the JIT-compiled function.
  auto invocation_result = engine->invokePacked("main", {});

  if (invocation_result) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int Compiler::dumpMLIRLang(InputType input_type, llvm::StringRef input_filename,
                           bool enable_opt) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);

  mlir::registerAllPasses();
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);

  mlir::MLIRContext context(registry);

  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);

  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::lang::LangDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  context.getOrLoadDialect<mlir::index::IndexDialect>();
  // context.getOrLoadDialect<mlir::arith::ArithDialect>();
  // context.getOrLoadDialect<mlir::scf::SCFDialect>();
  //
  // context.getOrLoadDialect<mlir::affine::AffineDialect>();
  // context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  // context.getOrLoadDialect<mlir::func::FuncDialect>();
  // context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  // context.getOrLoadDialect<mlir::BuiltinDialect>();
  // context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  // context.getOrLoadDialect<mlir::linalg::LinalgDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler source_mgr_handle(source_mgr, &context);
  if (int error = loadMLIR(input_type, input_filename, source_mgr, context,
                           module, true))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Custom passes
  // mlir::OpPassManager &cast_pm = pm.nest<mlir::lang::FuncOp>();
  // cast_pm.addPass(mlir::lang::createLiteralCastPass());
  // pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(mlir::lang::createComptimeEvalPass());
  pm.addPass(mlir::lang::createLowerToAffinePass());
  pm.addPass(mlir::lang::createUnrealizedConversionCastResolverPass());

  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Add a few cleanups post lowering.
  mlir::OpPassManager &opt_pm = pm.nest<mlir::func::FuncOp>();
  opt_pm.addPass(mlir::createCanonicalizerPass());
  opt_pm.addPass(mlir::createCSEPass());
  opt_pm.addPass(mlir::affine::createLoopFusionPass());
  opt_pm.addPass(mlir::affine::createAffineScalarReplacementPass());

  // Add passes to lower the module to LLVM dialect.
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createCanonicalizerPass());

  mlir::bufferization::OneShotBufferizationOptions bufferization_opts;
  bufferization_opts.bufferizeFunctionBoundaries = true;
  pm.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferization_opts));

  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(pm.run(*module))) {
    LLVM_DEBUG(module->dump());
    return 4;
  }
  LLVM_DEBUG(llvm::errs() << "MLIR after lowering to LLVM:\n");
  LLVM_DEBUG(module->dump());
  runJit(module.get(), enable_opt);
  return 0;
}

int Compiler::dumpMLIR(InputType input_type, llvm::StringRef input_filename,
                       bool enable_opt, Action emit_action) {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  //
  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  // context.getOrLoadDialect<mlir::lang::LangDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler source_mgr_handler(source_mgr, &context);
  if (int error =
          loadMLIR(input_type, input_filename, source_mgr, context, module))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // pm.addPass(ComptimeEvalPass::create());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Check to see what granularity of MLIR we are compiling to.
  bool is_lowering_to_affine = emit_action >= Action::DumpMLIRAffine;

  if (enable_opt || is_lowering_to_affine) {
    // Inline all functions into main and then delete them.
    // pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations.
    mlir::OpPassManager &opt_pm = pm.nest<mlir::func::FuncOp>();
    opt_pm.addPass(mlir::createCanonicalizerPass());
    opt_pm.addPass(mlir::createCSEPass());
  }

  if (is_lowering_to_affine) {
    // Partially lower the toy dialect.
    pm.addPass(mlir::createLowerAffinePass());

    // Add a few cleanups post lowering.
    mlir::OpPassManager &opt_pm = pm.nest<mlir::func::FuncOp>();
    opt_pm.addPass(mlir::createCanonicalizerPass());
    opt_pm.addPass(mlir::createCSEPass());

    // Add optimizations if enabled.
    if (enable_opt) {
      opt_pm.addPass(mlir::affine::createLoopFusionPass());
      opt_pm.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  // Add passes to lower the module to LLVM dialect.
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());

  if (mlir::failed(pm.run(*module)))
    return 4;

  // module->dump();

  // Compile the MLIR to LLVM IR.
  runJit(module.get(), enable_opt);
  return 0;
}
