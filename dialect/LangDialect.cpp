#include "dialect/LangDialect.h"
#include "dialect/LangOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
using namespace mlir;
using namespace mlir::lang;

//===----------------------------------------------------------------------===//
// Lang dialect.
//===----------------------------------------------------------------------===//

void LangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialect/LangOps.cpp.inc"
      >();

  addTypes<>();
  getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}
