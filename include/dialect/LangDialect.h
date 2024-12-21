#ifndef LANG_LANGDIALECT_H
#define LANG_LANGDIALECT_H

#include "LangOps.h"
#include "mlir/AsmParser/CodeComplete.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"

#include "dialect/LangEnumAttrDefs.h.inc"
#include "dialect/LangOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "dialect/LangOpsAttrDefs.h.inc"

namespace mlir {
namespace lang {
struct LangDialectInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All call operations can be inlined.
  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// All operations can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }

  /// All functions can be inlined.
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }
  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op, mlir::Block *newDest) const final {
    // Only return needs to be handled here.
    auto return_op = mlir::dyn_cast<mlir::lang::ReturnOp>(op);
    if (!return_op)
      return;

    // Replace the return with a branch to the dest.
    mlir::OpBuilder builder(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest,
                                       return_op.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToRepl) const final {
    // Only return needs to be handled here.
    auto return_op = mlir::cast<mlir::lang::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(return_op.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(return_op.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
} // namespace lang
} // namespace mlir

#endif // LANG_LANGDIALECT_H
