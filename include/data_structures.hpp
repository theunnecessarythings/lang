#include "dialect/LangOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>

namespace mlir {

enum class ResultState { Failure, SuccessNoValue, SuccessWithValue };

template <typename T> class [[nodiscard]] Result {
public:
  Result(LogicalResult lr)
      : state(mlir::failed(lr) ? ResultState::Failure
                               : ResultState::SuccessNoValue) {}

  Result(T val) : state(ResultState::SuccessWithValue), value(std::move(val)) {}

  Result(InFlightDiagnostic &&diag) { state = ResultState::Failure; }

  static Result failure() { return Result(ResultState::Failure); }

  static Result successNoValue() { return Result(ResultState::SuccessNoValue); }

  static Result success(T val) { return Result(std::move(val)); }

  bool succeeded() const {
    return state == ResultState::SuccessNoValue ||
           state == ResultState::SuccessWithValue;
  }

  bool failed() const { return state == ResultState::Failure; }

  bool hasValue() const { return state == ResultState::SuccessWithValue; }

  const T &getValue() const {
    assert(hasValue() && "No value available");
    return *value;
  }

  T &getValue() {
    assert(hasValue() && "No value available");
    return *value;
  }

  // Convert to LogicalResult for seamless integration.
  operator LogicalResult() const {
    return succeeded() ? mlir::success() : mlir::failure();
  }

  const T *operator->() const {
    assert(hasValue() && "No value to access");
    return &*value;
  }

  T *operator->() {
    assert(hasValue() && "No value to access");
    return &*value;
  }

  const T &operator*() const {
    assert(hasValue() && "No value to access");
    return *value;
  }

  T &operator*() {
    assert(hasValue() && "No value to access");
    return *value;
  }

private:
  Result(ResultState s) : state(s) {}

  ResultState state;
  std::optional<T> value;
};

template <typename T> Result<T> toResult(FailureOr<T> fo) {
  return fo ? Result<T>::success(*fo) : Result<T>::failure();
}

} // namespace mlir

inline std::string mangle(llvm::StringRef base,
                          llvm::ArrayRef<mlir::Type> types) {
  std::string mangled_name = base.str();
  llvm::raw_string_ostream rso(mangled_name);
  if (!types.empty()) {
    rso << "_";
  }
  for (auto &type : types) {
    rso << "_";
    if (mlir::isa<mlir::lang::ArrayType>(type))
      rso << "!array";
    else if (mlir::isa<mlir::lang::SliceType>(type))
      rso << "!slice";
    else if (mlir::isa<mlir::TensorType>(type))
      rso << "tensor";
    else if (mlir::isa<mlir::TupleType>(type))
      rso << "tuple";
    // else if (mlir::isa<mlir::MemRefType>(type))
    //   rso << "memref";
    else if (mlir::isa<mlir::lang::StructType>(type))
      rso << mlir::cast<mlir::lang::StructType>(type).getName();
    else
      rso << type;
  }
  return rso.str();
}

inline std::string attrToStr(mlir::Attribute attr) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  if (mlir::isa<mlir::IntegerAttr>(attr)) {
    rso << mlir::cast<mlir::IntegerAttr>(attr).getInt();
  } else if (mlir::isa<mlir::FloatAttr>(attr)) {
    rso << mlir::cast<mlir::FloatAttr>(attr).getValueAsDouble();
  } else {
    llvm::errs() << "Unsupported attribute kind\n";
  }
  return rso.str();
}
