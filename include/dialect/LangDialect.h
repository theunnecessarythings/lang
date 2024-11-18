#ifndef LANG_LANGDIALECT_H
#define LANG_LANGDIALECT_H

#include "mlir/AsmParser/CodeComplete.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"

#include "dialect/LangOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "dialect/LangOpsAttrDefs.h.inc"

namespace mlir {
namespace lang {

struct TypeValueTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;

  TypeValueTypeStorage(mlir::Type aliasedType) : aliasedType(aliasedType) {}

  bool operator==(const KeyTy &key) const { return key == aliasedType; }

  static KeyTy getKey(mlir::Type aliasedType) { return aliasedType; }

  static TypeValueTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    return new (allocator.allocate<TypeValueTypeStorage>())
        TypeValueTypeStorage(key);
  }

  mlir::Type aliasedType;
};

class TypeValueType : public mlir::Type::TypeBase<TypeValueType, mlir::Type,
                                                  TypeValueTypeStorage> {
public:
  using Base::Base;

  static TypeValueType get(mlir::Type type) {
    auto ctx = type.getContext();
    return Base::get(ctx, type);
  }

  mlir::Type getAliasedType();
  static constexpr mlir::StringLiteral name = "lang.typevalue";
};

struct StructTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<llvm::ArrayRef<mlir::Type>, llvm::StringRef>;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes,
                    llvm::StringRef name)
      : elementTypes(elementTypes), name(name) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(elementTypes, name);
  }

  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes,
                      llvm::StringRef name) {
    return KeyTy(elementTypes, name);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the element types and name into the allocator's memory
    auto elementTypes = allocator.copyInto(key.first);
    auto name = allocator.copyInto(key.second);
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes, name);
  }
  llvm::ArrayRef<mlir::Type> elementTypes;
  llvm::StringRef name;
};

class StructType
    : public mlir::Type::TypeBase<StructType, mlir::Type, StructTypeStorage> {
public:
  using Base::Base;

  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes,
                        llvm::StringRef name) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    auto *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes, name);
  }

  llvm::ArrayRef<mlir::Type> getElementTypes();
  llvm::StringRef getName();

  size_t getNumElementTypes();
  static constexpr mlir::StringLiteral name = "lang.struct";
};

class StringType
    : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;

  static StringType get(MLIRContext *ctx) { return Base::get(ctx); }
  static constexpr mlir::StringLiteral name = "lang.string";
};

class PointerType
    : public mlir::Type::TypeBase<PointerType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;
  static PointerType get(MLIRContext *ctx) { return Base::get(ctx); }
  static constexpr mlir::StringLiteral name = "lang.ptr";
};

class IntLiteralType : public mlir::Type::TypeBase<IntLiteralType, mlir::Type,
                                                   mlir::TypeStorage> {
public:
  using Base::Base;

  static IntLiteralType get(MLIRContext *ctx) { return Base::get(ctx); }

  static constexpr mlir::StringLiteral name = "lang.int_literal";
};

} // namespace lang
} // namespace mlir
#endif // LANG_LANGDIALECT_H
