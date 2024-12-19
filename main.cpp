#include "analyzer.hpp"
#include "compiler.hpp"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;

static cl::opt<std::string> input_filename(cl::Positional,
                                           cl::desc("<input lang file>"),
                                           cl::init("-"),
                                           cl::value_desc("filename"));

// namespace
static cl::opt<enum InputType>
    input_type("x", cl::init(Lang),
               cl::desc("Decided the kind of output desired"),
               cl::values(clEnumValN(Lang, "lang",
                                     "load the input file as a Toy source.")),
               cl::values(clEnumValN(MLIR, "mlir",
                                     "load the input file as an MLIR file")));

// namespace
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

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "lang compiler\n");

  switch (emit_action) {
  case Action::DumpAST:
    return Compiler::dumpAST(input_type, input_filename);
  case Action::DumpJSON:
    return Compiler::dumpJSON(input_type, input_filename);
  case Action::DumpMLIR:
  case Action::DumpMLIRAffine:
    return Compiler::dumpMLIR(input_type, input_filename, enable_opt,
                              emit_action);
  case Action::DumpMLIRLang:
    return Compiler::dumpMLIRLang(input_type, input_filename, enable_opt);
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
