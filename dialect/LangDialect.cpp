#include "dialect/LangDialect.h"
#include "dialect/LangOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// Lang dialect.
//===----------------------------------------------------------------------===//

void mlir::lang::LangDialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "dialect/LangOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "dialect/LangOpsTypes.cpp.inc"
      >();
  addTypes<StructType, TypeValueType, StringType>();
  getContext()->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}

Type mlir::lang::LangDialect::parseType(DialectAsmParser &parser) const {
  // Attempt to parse 'typevalue' type
  mlir::StringRef type_name;
  if (parser.parseOptionalKeyword(&type_name,
                                  {"typevalue", "struct", "string"})) {
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

  // Attempt to parse 'struct' type
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
    return StructType::get(elementTypes);
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
    printer << '>';
  } else if (mlir::isa<StringType>(type)) {
    printer << "string";
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

OpFoldResult mlir::lang::TypeConstOp::fold(FoldAdaptor adaptor) {
  auto typeAttr = getTypeAttr();
  if (!typeAttr)
    return {};

  // Return the type attribute as the fold result
  auto type = TypeValueType::get(typeAttr.getValue());
  return TypeAttr::get(type);
}

struct StructTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }
  llvm::ArrayRef<mlir::Type> elementTypes;
};

llvm::ArrayRef<mlir::Type> mlir::lang::StructType::getElementTypes() {
  return getImpl()->elementTypes;
}

size_t mlir::lang::StructType::getNumElementTypes() {
  return getElementTypes().size();
}

mlir::Type mlir::lang::TypeValueType::getAliasedType() {
  return getImpl()->aliasedType;
}
