#pragma once
#include "analyzer.hpp"
#include "ast.hpp"
#include "compiler.hpp"

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // namespace mlir

struct Program;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> langGen(mlir::MLIRContext &context,
                                          Program *program, Context &cxt);
