#include "dialect/LangOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

// Determines if an explicit cast is necessary between the literal type and
// target type.
bool needsExplicitCast(mlir::Type literalType, mlir::Type targetType) {
  if (auto literalIntType = mlir::dyn_cast<mlir::IntegerType>(literalType)) {
    if (auto targetIntType = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
      // Ensure explicit cast if types differ in signedness or bitwidth
      if (literalIntType.isSigned() != targetIntType.isSigned() ||
          literalIntType.getWidth() != targetIntType.getWidth()) {
        return true;
      }
      // If bitwidths differ (e.g., i64 to i32), also require an explicit cast
      if (literalIntType.getWidth() > targetIntType.getWidth()) {
        return true;
      }
    }
  }

  if (auto literalFloatType = mlir::dyn_cast<mlir::FloatType>(literalType)) {
    if (auto targetFloatType = mlir::dyn_cast<mlir::FloatType>(targetType)) {
      // Require explicit cast for narrowing or widening
      if (literalFloatType.getWidth() != targetFloatType.getWidth()) {
        return true;
      }
    }
  }

  // Check for integer-to-float or float-to-integer casting
  if ((mlir::isa<mlir::IntegerType>(literalType) &&
       mlir::isa<mlir::FloatType>(targetType)) ||
      (mlir::isa<mlir::FloatType>(literalType) &&
       mlir::isa<mlir::IntegerType>(targetType))) {
    return true; // Require explicit casts for int<->float conversions
  }

  return false;
}

// Checks if an implicit cast is allowed, based on if the literal can fit the
// target type without loss.
bool canImplicitlyCast(mlir::Type literalType, mlir::Type targetType) {
  if (auto literalIntType = mlir::dyn_cast<mlir::IntegerType>(literalType)) {
    if (auto targetIntType = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
      // Allow implicit widening, i.e., smaller integer fitting in a larger
      // integer
      return literalIntType.getWidth() <= targetIntType.getWidth() &&
             literalIntType.isSigned() == targetIntType.isSigned();
    }
  }

  if (auto literalFloatType = mlir::dyn_cast<mlir::FloatType>(literalType)) {
    if (auto targetFloatType = mlir::dyn_cast<mlir::FloatType>(targetType)) {
      // Allow implicit widening of floats (e.g., f32 to f64)
      return literalFloatType.getWidth() <= targetFloatType.getWidth();
    }
  }

  return true;
}

// Inserts an explicit cast operation as needed.
mlir::Value insertExplicitCast(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Type targetType, mlir::Value literal) {
  mlir::Type literalType = literal.getType();

  if (auto literalIntType = mlir::dyn_cast<mlir::IntegerType>(literalType)) {
    if (auto targetIntType = mlir::dyn_cast<mlir::IntegerType>(targetType)) {
      if (targetIntType.getWidth() > literalIntType.getWidth()) {
        // Widen integer: use sign or zero extension depending on signedness
        if (literalIntType.isSigned())
          return builder.create<mlir::arith::ExtSIOp>(loc, targetType, literal);
        else
          return builder.create<mlir::arith::ExtUIOp>(loc, targetType, literal);
      } else if (targetIntType.getWidth() < literalIntType.getWidth()) {
        // Truncate integer
        return builder.create<mlir::arith::TruncIOp>(loc, targetType, literal);
      }
    } else if (mlir::isa<mlir::FloatType>(targetType)) {
      // Convert integer to float
      if (literalIntType.isSigned())
        return builder.create<mlir::arith::SIToFPOp>(loc, targetType, literal);
      else
        return builder.create<mlir::arith::UIToFPOp>(loc, targetType, literal);
    }
  } else if (auto literalFloatType =
                 mlir::dyn_cast<mlir::FloatType>(literalType)) {
    if (auto targetFloatType = mlir::dyn_cast<mlir::FloatType>(targetType)) {
      if (targetFloatType.getWidth() > literalFloatType.getWidth()) {
        // Extend float
        return builder.create<mlir::arith::ExtFOp>(loc, targetType, literal);
      } else if (targetFloatType.getWidth() < literalFloatType.getWidth()) {
        // Truncate float
        return builder.create<mlir::arith::TruncFOp>(loc, targetType, literal);
      }
    } else if (mlir::isa<mlir::IntegerType>(targetType)) {
      // Convert float to integer
      return builder.create<mlir::arith::FPToSIOp>(loc, targetType, literal);
    }
  }

  // If no conversion needed, return the literal itself
  return literal;
}

void castIfRequired(mlir::Type literalType, mlir::Type targetType,
                    mlir::arith::ConstantOp constant_op, mlir::Value operand,
                    mlir::OpBuilder &builder, mlir::Operation *op) {

  // If explicit cast is needed, insert it
  if (needsExplicitCast(literalType, targetType)) {
    builder.setInsertionPointAfter(constant_op);
    mlir::Value castedValue =
        insertExplicitCast(builder, op->getLoc(), targetType, constant_op);
    operand.replaceAllUsesWith(castedValue);
  }
  // If an implicit cast is allowed, ensure it fits; otherwise, emit error
  else if (!canImplicitlyCast(literalType, targetType)) {
    emitError(op->getLoc())
        << "Type mismatch in literal expression requires explicit cast: "
        << "from " << literalType << " to " << targetType;
  }
}

struct HandleLiteralExpressionsPass
    : public mlir::PassWrapper<HandleLiteralExpressionsPass,
                               mlir::OperationPass<mlir::lang::FuncOp>> {
  void runOnOperation() override {
    mlir::lang::FuncOp function = getOperation();
    mlir::OpBuilder builder(function.getContext());

    function.walk([&](mlir::Operation *op) {
      // For each operation, process each operand
      for (auto operand : op->getOperands()) {
        auto literal = operand.getDefiningOp<mlir::arith::ConstantOp>();
        if (!literal) {
          continue; // Skip non-literal operands
        }
        mlir::Type targetType = operand.getType();
        mlir::Type literalType = literal.getType();

        castIfRequired(literalType, targetType, literal, operand, builder, op);
      }
    });
  }
};

namespace mlir {
namespace lang {
std::unique_ptr<mlir::Pass> createLiteralCastPass() {
  return std::make_unique<HandleLiteralExpressionsPass>();
}
} // namespace lang
} // namespace mlir
