#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace lang {

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
std::unique_ptr<mlir::Pass> createLowerToAffinePass();
std::unique_ptr<mlir::Pass> createLiteralCastPass();
std::unique_ptr<mlir::Pass> createUnrealizedConversionCastResolverPass();
} // namespace lang
} // namespace mlir
