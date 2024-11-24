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
bool needsExplicitCast(mlir::Type literal_type, mlir::Type target_type) {
  if (auto literal_int_type = mlir::dyn_cast<mlir::IntegerType>(literal_type)) {
    if (auto target_int_type = mlir::dyn_cast<mlir::IntegerType>(target_type)) {
      // Ensure explicit cast if types differ in signedness or bitwidth
      if (literal_int_type.isSigned() != target_int_type.isSigned() ||
          literal_int_type.getWidth() != target_int_type.getWidth()) {
        return true;
      }
      // If bitwidths differ (e.g., i64 to i32), also require an explicit cast
      if (literal_int_type.getWidth() > target_int_type.getWidth()) {
        return true;
      }
    }
  }

  if (auto literal_float_type = mlir::dyn_cast<mlir::FloatType>(literal_type)) {
    if (auto target_float_type = mlir::dyn_cast<mlir::FloatType>(target_type)) {
      // Require explicit cast for narrowing or widening
      if (literal_float_type.getWidth() != target_float_type.getWidth()) {
        return true;
      }
    }
  }

  // Check for integer-to-float or float-to-integer casting
  if ((mlir::isa<mlir::IntegerType>(literal_type) &&
       mlir::isa<mlir::FloatType>(target_type)) ||
      (mlir::isa<mlir::FloatType>(literal_type) &&
       mlir::isa<mlir::IntegerType>(target_type))) {
    return true; // Require explicit casts for int<->float conversions
  }

  return false;
}

// Checks if an implicit cast is allowed, based on if the literal can fit the
// target type without loss.
bool canImplicitlyCast(mlir::Type literal_type, mlir::Type target_type) {
  if (auto literal_int_type = mlir::dyn_cast<mlir::IntegerType>(literal_type)) {
    if (auto target_int_type = mlir::dyn_cast<mlir::IntegerType>(target_type)) {
      // Allow implicit widening, i.e., smaller integer fitting in a larger
      // integer
      return literal_int_type.getWidth() <= target_int_type.getWidth() &&
             literal_int_type.isSigned() == target_int_type.isSigned();
    }
  }

  if (auto literal_float_type = mlir::dyn_cast<mlir::FloatType>(literal_type)) {
    if (auto target_float_type = mlir::dyn_cast<mlir::FloatType>(target_type)) {
      // Allow implicit widening of floats (e.g., f32 to f64)
      return literal_float_type.getWidth() <= target_float_type.getWidth();
    }
  }

  return true;
}

// Inserts an explicit cast operation as needed.
mlir::Value insertExplicitCast(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Type target_type, mlir::Value literal) {
  mlir::Type literal_type = literal.getType();

  if (auto literal_int_type = mlir::dyn_cast<mlir::IntegerType>(literal_type)) {
    if (auto target_int_type = mlir::dyn_cast<mlir::IntegerType>(target_type)) {
      if (target_int_type.getWidth() > literal_int_type.getWidth()) {
        // Widen integer: use sign or zero extension depending on signedness
        if (literal_int_type.isSigned())
          return builder.create<mlir::arith::ExtSIOp>(loc, target_type,
                                                      literal);
        else
          return builder.create<mlir::arith::ExtUIOp>(loc, target_type,
                                                      literal);
      } else if (target_int_type.getWidth() < literal_int_type.getWidth()) {
        // Truncate integer
        return builder.create<mlir::arith::TruncIOp>(loc, target_type, literal);
      }
    } else if (mlir::isa<mlir::FloatType>(target_type)) {
      // Convert integer to float
      if (literal_int_type.isSigned())
        return builder.create<mlir::arith::SIToFPOp>(loc, target_type, literal);
      else
        return builder.create<mlir::arith::UIToFPOp>(loc, target_type, literal);
    }
  } else if (auto literal_float_type =
                 mlir::dyn_cast<mlir::FloatType>(literal_type)) {
    if (auto target_float_type = mlir::dyn_cast<mlir::FloatType>(target_type)) {
      if (target_float_type.getWidth() > literal_float_type.getWidth()) {
        // Extend float
        return builder.create<mlir::arith::ExtFOp>(loc, target_type, literal);
      } else if (target_float_type.getWidth() < literal_float_type.getWidth()) {
        // Truncate float
        return builder.create<mlir::arith::TruncFOp>(loc, target_type, literal);
      }
    } else if (mlir::isa<mlir::IntegerType>(target_type)) {
      // Convert float to integer
      return builder.create<mlir::arith::FPToSIOp>(loc, target_type, literal);
    }
  }

  // If no conversion needed, return the literal itself
  return literal;
}

void castIfRequired(mlir::Type literal_type, mlir::Type target_type,
                    mlir::arith::ConstantOp constant_op, mlir::Value operand,
                    mlir::OpBuilder &builder, mlir::Operation *op) {

  // If explicit cast is needed, insert it
  if (needsExplicitCast(literal_type, target_type)) {
    builder.setInsertionPointAfter(constant_op);
    mlir::Value casted_value =
        insertExplicitCast(builder, op->getLoc(), target_type, constant_op);
    operand.replaceAllUsesWith(casted_value);
  }
  // If an implicit cast is allowed, ensure it fits; otherwise, emit error
  else if (!canImplicitlyCast(literal_type, target_type)) {
    emitError(op->getLoc())
        << "Type mismatch in literal expression requires explicit cast: "
        << "from " << literal_type << " to " << target_type;
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
        mlir::Type target_type = operand.getType();
        mlir::Type literal_type = literal.getType();

        castIfRequired(literal_type, target_type, literal, operand, builder,
                       op);
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
