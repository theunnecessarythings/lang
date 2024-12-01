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
std::unique_ptr<mlir::Pass> createComptimeLoweringPass();

struct InlineComptimeOp
    : public mlir::OpConversionPattern<mlir::lang::ComptimeOp> {
  using mlir::OpConversionPattern<mlir::lang::ComptimeOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(mlir::lang::ComptimeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    llvm::errs() << "Inlining comptime op\n";
    // Get the parent block and the block to inline.
    mlir::Block *parentBlock = op->getBlock();
    mlir::Block &inlineBlock = op->getRegion(0).front();

    // Compute the insertion point: right after the op.
    auto insertionPoint = std::next(mlir::Block::iterator(op));
    rewriter.setInsertionPoint(parentBlock, insertionPoint);

    // Map block arguments to the operation's operands.
    mlir::IRMapping mapper;
    for (auto [arg, operand] :
         llvm::zip(inlineBlock.getArguments(), op->getOperands())) {
      mapper.map(arg, operand);
    }

    // Clone and inline the operations from the inlineBlock.
    for (auto &innerOp : inlineBlock.without_terminator()) {
      llvm::errs() << "Cloning op: " << innerOp << "\n";
      // rewriter.clone(innerOp, mapper);
    }

    // // Handle the terminator.
    // if (auto terminator =
    //         mlir::dyn_cast<mlir::lang::YieldOp>(inlineBlock.getTerminator()))
    //         {
    //   mlir::SmallVector<mlir::Value, 4> results;
    //   for (auto operand : terminator.getOperands()) {
    //     results.push_back(mapper.lookup(operand));
    //   }
    //   rewriter.replaceOp(op, results);
    //   rewriter.replaceAllOpUsesWith(op, results);
    // } else {
    //   return llvm::failure();
    // }
    rewriter.eraseOp(op);
    parentBlock->dump();
    return llvm::success();
  }
};

} // namespace lang
} // namespace mlir
