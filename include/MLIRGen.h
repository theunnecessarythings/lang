#pragma once

namespace mlir {
class MLIRContext;
template <typename T> class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace Ast {
struct Program;
}

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          const Ast::Program &program);
