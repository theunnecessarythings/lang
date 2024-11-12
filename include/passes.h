#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace lang {

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

} // namespace lang
} // namespace mlir
