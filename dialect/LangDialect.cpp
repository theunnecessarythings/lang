#include "dialect/LangDialect.h"
#include "dialect/LangOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#define GET_ATTRDEF_CLASSES
#include "dialect/LangOpsAttrDefs.cpp.inc"
//===----------------------------------------------------------------------===//
// Lang dialect.
//===----------------------------------------------------------------------===//
template <> struct mlir::FieldParser<llvm::APInt> {
  static FailureOr<llvm::APInt> parse(AsmParser &parser) {
    llvm::APInt value;
    if (parser.parseInteger(value))
      return failure();
    return value;
  }
};
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
    auto returnOp = mlir::dyn_cast<mlir::lang::ReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the return with a branch to the dest.
    mlir::OpBuilder builder(op);
    builder.create<mlir::cf::BranchOp>(op->getLoc(), newDest,
                                       returnOp.getOperands());
    op->erase();
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToRepl) const final {
    // Only return needs to be handled here.
    auto returnOp = mlir::cast<mlir::lang::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }
};
// void mlir::lang::registerInlinerExtension(DialectRegistry &registry) {
//   registry.addExtension(
//       +[](MLIRContext *ctx, mlir::lang::LangDialect *dialect) {
//         dialect->addInterfaces<FuncInlinerInterface>();
//
//         // The inliner extension relies on the ControlFlow dialect.
//         ctx->getOrLoadDialect<cf::ControlFlowDialect>();
//       });
// }

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
  addTypes<StructType, TypeValueType, StringType, PointerType,
           IntLiteralType>();

  addInterfaces<LangDialectInlinerInterface>();

  getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  getContext()->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
}

mlir::Type mlir::lang::LangDialect::parseType(DialectAsmParser &parser) const {
  // Attempt to parse 'typevalue' type
  mlir::StringRef type_name;
  if (parser.parseOptionalKeyword(&type_name, {"typevalue", "struct", "string",
                                               "int_literal", "ptr"})) {
    return Type();
  }
  if (type_name == "typevalue") {
    if (parser.parseLess())
      return Type();
    mlir::Type innerType;
    if (parser.parseType(innerType))
      return Type();
    if (parser.parseGreater())
      return Type();
    return TypeValueType::get(innerType);
  }

  // Attempt to parse 'int_literal' type
  else if (type_name == "int_literal") {
    return IntLiteralType::get(getContext());
  }

  else if (type_name == "ptr") {
    return PointerType::get(getContext());
  }

  // Attempt to parse 'struct' type, with a string attribute name
  // struct ::= `struct` `<` type (`,` type)* `>` string
  else if (type_name == "struct") {
    if (parser.parseLess())
      return Type();

    SmallVector<mlir::Type, 1> elementTypes;
    do {
      // Parse the current element type.
      mlir::Type elementType;
      if (parser.parseType(elementType))
        return Type();

      elementTypes.push_back(elementType);
    } while (succeeded(parser.parseOptionalComma()));

    // Parse: `>`
    if (parser.parseGreater())
      return Type();

    // Parse the struct name.
    llvm::StringRef name;
    if (parser.parseKeyword(&name))
      return Type();

    return StructType::get(elementTypes, name);
  }

  else if (type_name == "string") {
    return StringType::get(getContext());
  }

  // If neither 'typevalue' nor 'struct' was parsed, attempt to parse other
  // types
  mlir::Type type;
  if (succeeded(parser.parseType(type)))
    return type;

  // If parsing failed, report an error
  parser.emitError(parser.getNameLoc(), "unknown type");
  return Type();
}

void mlir::lang::LangDialect::printType(Type type,
                                        DialectAsmPrinter &printer) const {
  if (mlir::isa<TypeValueType>(type)) {
    TypeValueType typeValue = mlir::cast<TypeValueType>(type);
    printer << "typevalue<";
    printer.printType(typeValue.getAliasedType());
    printer << '>';
  } else if (mlir::isa<StructType>(type)) {
    StructType structType = mlir::cast<StructType>(type);
    printer << "struct<";
    llvm::interleaveComma(structType.getElementTypes(), printer);
    printer << '>' << structType.getName();
  } else if (mlir::isa<StringType>(type)) {
    printer << "string";
  } else if (mlir::isa<IntLiteralType>(type)) {
    printer << "int_literal";
  } else if (mlir::isa<PointerType>(type)) {
    printer << "ptr";
  } else
    llvm_unreachable("unknown type in LangDialect");
}

void mlir::lang::StructAccessOp::build(mlir::OpBuilder &b,
                                       mlir::OperationState &state,
                                       mlir::Value input, size_t index) {
  // Extract the result type from the input type.
  StructType structTy = llvm::cast<mlir::lang::StructType>(input.getType());
  assert(index < structTy.getNumElementTypes());
  mlir::Type resultType = structTy.getElementTypes()[index];

  // Call into the auto-generated build method.
  build(b, state, resultType, input, b.getI64IntegerAttr(index));
}

llvm::LogicalResult mlir::lang::StructAccessOp::verify() {
  StructType structTy =
      llvm::cast<mlir::lang::StructType>(getInput().getType());
  size_t indexValue = getIndex();
  if (indexValue >= structTy.getNumElementTypes())
    return emitOpError()
           << "index should be within the range of the input struct type";
  mlir::Type resultType = getResult().getType();
  if (resultType != structTy.getElementTypes()[indexValue])
    return emitOpError() << "must have the same result type as the struct "
                            "element referred to by the index";
  return mlir::success();
}

mlir::OpFoldResult mlir::lang::TypeConstOp::fold(FoldAdaptor adaptor) {
  auto typeAttr = getTypeAttr();
  if (!typeAttr)
    return {};

  // Return the type attribute as the fold result
  auto type = TypeValueType::get(typeAttr.getValue());
  return TypeAttr::get(type);
}

llvm::ArrayRef<mlir::Type> mlir::lang::StructType::getElementTypes() {
  return getImpl()->elementTypes;
}

llvm::StringRef mlir::lang::StructType::getName() { return getImpl()->name; }

size_t mlir::lang::StructType::getNumElementTypes() {
  return getElementTypes().size();
}

mlir::Type mlir::lang::TypeValueType::getAliasedType() {
  return getImpl()->aliasedType;
}
