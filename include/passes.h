#pragma once

#include "dialect/LangOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>

namespace mlir {
class Pass;

namespace lang {

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createLiteralCastPass();
std::unique_ptr<mlir::Pass> createUnrealizedConversionCastResolverPass();
std::unique_ptr<mlir::Pass> createComptimeEvalPass();
} // namespace lang
} // namespace mlir
