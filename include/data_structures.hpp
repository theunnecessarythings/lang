#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
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
