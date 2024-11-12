#ifndef LANG_LANGDIALECT_H
#define LANG_LANGDIALECT_H

#include "mlir/IR/Dialect.h"

#include "dialect/LangOpsDialect.h.inc"
#include "dialect/LangOpsTypes.h.inc"

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
    mlir::MLIRContext *ctx = type.getContext();
    return Base::get(ctx, type);
  }

  mlir::Type getAliasedType();
  static constexpr mlir::StringLiteral name = "lang.typevalue";
};

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

class StructType
    : public mlir::Type::TypeBase<StructType, mlir::Type, StructTypeStorage> {
public:
  using Base::Base;

  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  llvm::ArrayRef<mlir::Type> getElementTypes();

  size_t getNumElementTypes();
  static constexpr mlir::StringLiteral name = "lang.struct";
};

class StringType
    : public mlir::Type::TypeBase<StringType, mlir::Type, mlir::TypeStorage> {
public:
  using Base::Base;

  static StringType get(mlir::MLIRContext *ctx) { return Base::get(ctx); }
  static constexpr mlir::StringLiteral name = "lang.string";
};

} // namespace lang
} // namespace mlir
#endif // LANG_LANGDIALECT_H
