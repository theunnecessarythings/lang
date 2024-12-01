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

  TypeValueTypeStorage(mlir::Type aliased_type) : aliased_type(aliased_type) {}

  bool operator==(const KeyTy &key) const { return key == aliased_type; }

  static KeyTy getKey(mlir::Type aliasedType) { return aliasedType; }

  static TypeValueTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    return new (allocator.allocate<TypeValueTypeStorage>())
        TypeValueTypeStorage(key);
  }

  mlir::Type aliased_type;
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

  StructTypeStorage(llvm::ArrayRef<mlir::Type> element_types,
                    llvm::StringRef name)
      : element_types(element_types), name(name) {}

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(element_types, name);
  }

  static KeyTy getKey(llvm::ArrayRef<mlir::Type> element_types,
                      llvm::StringRef name) {
    return KeyTy(element_types, name);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the element types and name into the allocator's memory
    auto element_types = allocator.copyInto(key.first);
    auto name = allocator.copyInto(key.second);
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(element_types, name);
  }
  llvm::ArrayRef<mlir::Type> element_types;
  llvm::StringRef name;
};

struct SliceTypeStorage : public mlir::TypeStorage {
  explicit SliceTypeStorage(mlir::Type element_type)
      : element_type(element_type) {}

  using KeyTy = mlir::Type;

  bool operator==(const KeyTy &key) const { return key == element_type; }

  static SliceTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<SliceTypeStorage>()) SliceTypeStorage(key);
  }

  mlir::Type element_type;
};

struct ArrayTypeStorage : public mlir::TypeStorage {
  explicit ArrayTypeStorage(mlir::Type element_type, int64_t size)
      : element_type(element_type), size(size) {}

  using KeyTy = std::pair<mlir::Type, int64_t>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(element_type, size);
  }

  static KeyTy getKey(mlir::Type element_type, int64_t size) {
    return KeyTy(element_type, size);
  }

  static ArrayTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<ArrayTypeStorage>())
        ArrayTypeStorage(key.first, key.second);
  }

  mlir::Type getElementType() const { return element_type; }

  int64_t getSize() const { return size; }

  mlir::Type element_type;
  int64_t size;
};

class StructType
    : public mlir::Type::TypeBase<StructType, mlir::Type, StructTypeStorage> {
public:
  using Base::Base;

  static StructType get(llvm::ArrayRef<mlir::Type> element_types,
                        llvm::StringRef name) {
    assert(!element_types.empty() && "expected at least 1 element type");

    auto *ctx = element_types.front().getContext();
    return Base::get(ctx, element_types, name);
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

class SliceType
    : public mlir::Type::TypeBase<SliceType, mlir::Type, SliceTypeStorage> {
public:
  using Base::Base;

  // Factory method to create a new SliceType
  static SliceType get(mlir::Type element_type) {
    assert(element_type && "Pointer type must be non-null");

    auto *ctx = element_type.getContext();
    return Base::get(ctx, element_type);
  }

  mlir::Type getElementType() const { return getImpl()->element_type; }

  static mlir::Type getLengthType(mlir::MLIRContext *context) {
    return mlir::IntegerType::get(context, 64);
  }

  static constexpr mlir::StringLiteral name = "lang.slice";
};

class ArrayType
    : public mlir::Type::TypeBase<ArrayType, mlir::Type, ArrayTypeStorage> {
public:
  using Base::Base;

  static ArrayType get(mlir::Type elementType, int64_t size) {
    assert(elementType && "Array type must be non-null");

    auto *ctx = elementType.getContext();
    return Base::get(ctx, elementType, size);
  }

  mlir::Type getElementType() const { return getImpl()->getElementType(); }

  int64_t getSize() const { return getImpl()->getSize(); }

  static constexpr mlir::StringLiteral name = "lang.array";

  static mlir::Type getLengthType(mlir::MLIRContext *context) {
    return mlir::IntegerType::get(context, 64);
  }
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
