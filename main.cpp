#include "LangGen.h"
#include "MLIRGen.h"
#include "ast.hpp"
#include "dialect/LangDialect.h"
#include "dialect/LangOpsDialect.cpp.inc"
#include "include/dialect/LangOps.h"
#include "json_dumper.hpp"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
// #include "comptime_eval_pass.hpp"
#include "analyzer.hpp"
#include "lexer.hpp"
#include "parser.hpp"

namespace cl = llvm::cl;

static cl::opt<std::string> input_filename(cl::Positional,
                                           cl::desc("<input lang file>"),
                                           cl::init("-"),
                                           cl::value_desc("filename"));

namespace {
enum InputType { Lang, MLIR };
} // namespace
static cl::opt<enum InputType>
    input_type("x", cl::init(Lang),
               cl::desc("Decided the kind of output desired"),
               cl::values(clEnumValN(Lang, "lang",
                                     "load the input file as a Toy source.")),
               cl::values(clEnumValN(MLIR, "mlir",
                                     "load the input file as an MLIR file")));

namespace {
enum Action { None, DumpAST, DumpJSON, DumpMLIR, DumpMLIRLang, DumpMLIRAffine };
} // namespace
static cl::opt<enum Action> emit_action(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpJSON, "json", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRLang, "lang",
                          "output the MLIR lang dialect dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")));

static cl::opt<bool> enable_opt("opt", cl::desc("Enable optimizations"));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<Program> parseInputFile(llvm::StringRef filename,
                                        std::shared_ptr<Context> context) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file_or_err =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = file_or_err.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = file_or_err.get()->getBuffer();
  auto lexer = std::make_unique<Lexer>(buffer.str(), 0);
  auto parser = Parser(std::move(lexer), context);
  context->source_mgr.AddNewSourceBuffer(std::move(*file_or_err),
                                         llvm::SMLoc());
  auto tree = parser.parseProgram();
  auto analyzer = Analyzer(context);
  analyzer.analyze(tree.get());
  return tree;
}

int dumpJSON() {
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

int dumpAST() {
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

int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module, bool lang = false) {
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

int runJit(mlir::ModuleOp module) {
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

int dumpMLIRLang() {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::lang::LangDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr source_mgr;
  mlir::SourceMgrDiagnosticHandler source_mgr_handle(source_mgr, &context);
  if (int error = loadMLIR(source_mgr, context, module, true))
    return error;

  mlir::PassManager pm(module.get()->getName());
  // Apply any generic pass manager command line options and run the pipeline.
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  // Custom passes
  mlir::OpPassManager &cast_pm = pm.nest<mlir::lang::FuncOp>();
  cast_pm.addPass(mlir::lang::createLiteralCastPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::lang::createLowerToAffinePass());
  pm.addPass(mlir::lang::createUnrealizedConversionCastResolverPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Add a few cleanups post lowering.
  mlir::OpPassManager &opt_pm = pm.nest<mlir::func::FuncOp>();
  opt_pm.addPass(mlir::createCanonicalizerPass());
  opt_pm.addPass(mlir::createCSEPass());
  opt_pm.addPass(mlir::affine::createLoopFusionPass());
  opt_pm.addPass(mlir::affine::createAffineScalarReplacementPass());

  // Add passes to lower the module to LLVM dialect.
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(pm.run(*module)))
    return 4;
  module->dump();
  runJit(module.get());
  return 0;
}

int dumpMLIR() {
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
  if (int error = loadMLIR(source_mgr, context, module))
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

  module->dump();

  // Compile the MLIR to LLVM IR.
  runJit(module.get());
  return 0;
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "lang compiler\n");

  switch (emit_action) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpJSON:
    return dumpJSON();
  case Action::DumpMLIR:
  case Action::DumpMLIRAffine:
    return dumpMLIR();
  case Action::DumpMLIRLang:
    return dumpMLIRLang();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
