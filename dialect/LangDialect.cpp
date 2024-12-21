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
