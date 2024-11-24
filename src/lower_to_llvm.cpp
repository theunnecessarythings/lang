#include "dialect/LangDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <utility>
//===----------------------------------------------------------------------===//
// LangToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct LangToLLVMLoweringPass
    : public mlir::PassWrapper<LangToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LangToLLVMLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void LangToLLVMLoweringPass::runOnOperation() {
  // Add ConversionTarget
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LLVMTypeConverter type_converter(&getContext());

  mlir::RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(type_converter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(type_converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(type_converter,
                                                        patterns);
  populateFuncToLLVMConversionPatterns(type_converter, patterns);

  // patterns.add<PrintOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::lang::createLowerToLLVMPass() {
  return std::make_unique<LangToLLVMLoweringPass>();
}
