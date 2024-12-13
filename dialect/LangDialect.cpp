#include "dialect/LangDialect.h"
#include "dialect/LangOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#define GET_ATTRDEF_CLASSES
#include "dialect/LangOpsAttrDefs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "dialect/LangOpsTypes.cpp.inc"

#include "dialect/LangEnumAttrDefs.cpp.inc"
//===----------------------------------------------------------------------===//
// Lang dialect.
//===----------------------------------------------------------------------===//

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

void mlir::lang::LangDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialect/LangOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "dialect/LangOpsAttrDefs.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "dialect/LangOpsTypes.cpp.inc"
      >();

  addInterfaces<LangDialectInlinerInterface>();

  getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  getContext()->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
}

void mlir::lang::StructAccessOp::build(mlir::OpBuilder &b,
                                       mlir::OperationState &state,
                                       mlir::Value input, size_t index) {
  // Extract the result type from the input type.
  StructType struct_ty = llvm::cast<mlir::lang::StructType>(input.getType());
  assert(index < struct_ty.getNumElementTypes());
  mlir::Type result_type = struct_ty.getElementTypes()[index];

  // Call into the auto-generated build method.
  build(b, state, result_type, input, b.getI64IntegerAttr(index));
}

llvm::LogicalResult mlir::lang::StructAccessOp::verify() {
  StructType struct_ty =
      llvm::cast<mlir::lang::StructType>(getInput().getType());
  size_t index_value = getIndex();
  if (index_value >= struct_ty.getNumElementTypes())
    return emitOpError()
           << "index should be within the range of the input struct type";
  mlir::Type result_type = getResult().getType();
  if (result_type != struct_ty.getElementTypes()[index_value])
    return emitOpError() << "must have the same result type as the struct "
                            "element referred to by the index";
  return mlir::success();
}

mlir::OpFoldResult mlir::lang::TypeConstOp::fold(FoldAdaptor adaptor) {
  auto type_attr = getTypeAttr();
  if (!type_attr)
    return {};

  // Return the type attribute as the fold result
  auto type = TypeValueType::get(getContext(), type_attr.getValue());
  return TypeAttr::get(type);
}
