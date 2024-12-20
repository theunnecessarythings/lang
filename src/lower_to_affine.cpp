#include "dialect/LangDialect.h"
#include "dialect/LangOps.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>
#include <utility>

#include "mlir/AsmParser/AsmParser.h"

namespace mlir {
namespace lang {

class LangToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LangToLLVMTypeConverter(mlir::MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    addConversion(
        [&](mlir::lang::StructType struct_type) -> std::optional<mlir::Type> {
          mlir::MLIRContext *ctx = struct_type.getContext();
          mlir::SmallVector<mlir::Type, 4> element_types;
          for (mlir::Type field_type : struct_type.getElementTypes()) {
            element_types.push_back(convertType(field_type));
          }
          auto llvm_struct_type = mlir::LLVM::LLVMStructType::getIdentified(
              ctx, struct_type.getName());
          if (mlir::failed(llvm_struct_type.setBody(element_types,
                                                    /*isPacked=*/false))) {
            return std::nullopt;
          }
          return llvm_struct_type;
        });

    addConversion([](mlir::lang::IntLiteralType int_literal_type)
                      -> std::optional<mlir::Type> {
      return mlir::IntegerType::get(int_literal_type.getContext(), 64);
      // int_literal_type.getWidth());
    });

    addConversion([](mlir::lang::PointerType ptr_type) {
      return mlir::LLVM::LLVMPointerType::get(ptr_type.getContext());
    });

    addConversion(
        [](mlir::lang::StringType string_type) -> std::optional<mlir::Type> {
          return mlir::LLVM::LLVMPointerType::get(string_type.getContext());
        });

    addConversion([](mlir::lang::ArrayType array_type) {
      return mlir::MemRefType::get({array_type.getSize().getInt()},
                                   array_type.getElementType());
    });

    addConversion([](mlir::lang::SliceType slice_type) {
      return mlir::MemRefType::get({std::numeric_limits<int64_t>::min()},
                                   slice_type.getElementType());
    });

    addConversion([](mlir::TensorType tensor_type) { return tensor_type; });

    addConversion([](mlir::lang::TypeValueType type) { return type; });
  }
};

template <typename OpTy, typename... Args>
OpTy createOp(mlir::ConversionPatternRewriter &rewriter, mlir::Operation *op,
              Args &&...args) {
  auto new_op = rewriter.create<OpTy>(std::forward<Args>(args)...);
  new_op->setAttrs(op->getAttrs());
  return new_op;
}

struct FuncOpLowering : public mlir::OpConversionPattern<mlir::lang::FuncOp> {
  using OpConversionPattern<mlir::lang::FuncOp>::OpConversionPattern;

  FuncOpLowering(mlir::TypeConverter &type_converter,
                 mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::FuncOp>(type_converter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto func_type = op.getFunctionType();
    auto input_types = func_type.getInputs();
    auto result_types = func_type.getResults();
    mlir::TypeConverter::SignatureConversion signature_conversion(
        input_types.size());
    mlir::SmallVector<mlir::Type, 4> converted_result_types;

    for (const auto &type : llvm::enumerate(input_types)) {
      mlir::Type converted_type = getTypeConverter()->convertType(type.value());
      if (!converted_type) {
        return rewriter.notifyMatchFailure(op, "failed to convert input type");
      }
      signature_conversion.addInputs(type.index(), converted_type);
    }

    for (mlir::Type type : result_types) {
      mlir::Type converted_type = getTypeConverter()->convertType(type);
      if (!converted_type) {
        return rewriter.notifyMatchFailure(op, "failed to convert result type");
      }
      converted_result_types.push_back(converted_type);
    }

    auto converted_func_type = mlir::FunctionType::get(
        op.getContext(), signature_conversion.getConvertedTypes(),
        converted_result_types);

    // auto new_func_op = rewriter.create<mlir::func::FuncOp>(
    //     op.getLoc(), op.getName(), converted_func_type);
    auto new_func_op = createOp<mlir::func::FuncOp>(
        rewriter, op, op.getLoc(), op.getName(), converted_func_type);

    rewriter.inlineRegionBefore(op.getBody(), new_func_op.getBody(),
                                new_func_op.end());

    if (failed(rewriter.convertRegionTypes(&new_func_op.getRegion(),
                                           *this->getTypeConverter(),
                                           &signature_conversion))) {
      return rewriter.notifyMatchFailure(op, "failed to convert region types");
    }
    llvm::SmallVector<mlir::Location, 4> locations(result_types.size(),
                                                   rewriter.getUnknownLoc());
    auto return_block = rewriter.createBlock(
        &new_func_op.getBody(), new_func_op.getBody().end(),
        new_func_op.getResultTypes(), locations);
    rewriter.setInsertionPointToStart(return_block);
    rewriter.create<mlir::func::ReturnOp>(op.getLoc(),
                                          return_block->getArguments());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct CallOpLowering : public mlir::OpConversionPattern<mlir::lang::CallOp> {
  using OpConversionPattern<mlir::lang::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // auto call = rewriter.create<mlir::func::CallOp>(
    //     op.getLoc(), op.getCalleeAttr(), op.getResultTypes(),
    //     op.getOperands());
    auto call = createOp<mlir::func::CallOp>(
        rewriter, op, op.getLoc(), op.getCalleeAttr(), op.getResultTypes(),
        adaptor.getOperands());
    rewriter.replaceOp(op, call);
    return mlir::success();
  }
};

struct IfOpLowering : public mlir::OpConversionPattern<mlir::lang::IfOp> {
  using OpConversionPattern<mlir::lang::IfOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::IfOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto scf_if_attr = op->getAttrOfType<mlir::BoolAttr>("can_use_scf_if_op");
    auto terminated_by_return =
        op->getAttrOfType<mlir::BoolAttr>("terminated_by_return");

    if (!scf_if_attr || !scf_if_attr.getValue()) {
      return generate_unstructured_if(rewriter, op, adaptor.getCondition());
    }

    auto result_type =
        op.getResult() ? op.getResult().getType() : mlir::TypeRange();
    // auto if_op = rewriter.create<mlir::scf::IfOp>(op.getLoc(), result_type,
    //                                               adaptor.getCondition());
    auto if_op = createOp<mlir::scf::IfOp>(rewriter, op, op.getLoc(),
                                           result_type, adaptor.getCondition());

    rewriter.inlineRegionBefore(op.getThenRegion(), if_op.getThenRegion(),
                                if_op.getThenRegion().end());
    if (!op.getElseRegion().empty()) {
      rewriter.inlineRegionBefore(op.getElseRegion(), if_op.getElseRegion(),
                                  if_op.getElseRegion().end());
    }
    rewriter.replaceOp(op, if_op);
    if (terminated_by_return.getValue()) {
      rewriter.setInsertionPointAfter(op);
      auto branch_op = rewriter.create<mlir::cf::BranchOp>(
          op.getLoc(), if_op.getResults(),
          &op->getParentOfType<mlir::func::FuncOp>().getBlocks().back());

      // erase everything after the branch op in the parent block
      auto it = std::next(mlir::Block::iterator(branch_op));
      auto block = branch_op->getBlock();
      while (it != block->end()) {
        auto next = std::next(it);
        rewriter.eraseOp(&*it);
        it = next;
      }
    }

    return mlir::success();
  }

  llvm::LogicalResult
  generate_unstructured_if(mlir::ConversionPatternRewriter &rewriter,
                           mlir::lang::IfOp op, mlir::Value condition) const {
    auto loc = op.getLoc();

    // Get the parent block and function
    auto *current_block = rewriter.getInsertionBlock();
    auto *parent_region = current_block->getParent();

    // Create the blocks for the 'then', 'else', and continuation ('merge')
    // blocks
    auto *then_block = rewriter.createBlock(
        parent_region, std::next(current_block->getIterator()));
    mlir::Block *else_block = nullptr;
    if (!op.getElseRegion().empty()) {
      else_block = rewriter.createBlock(
          parent_region, std::next(current_block->getIterator()));
    }
    // Create the merge block by splitting the current block
    auto *merge_block = rewriter.splitBlock(current_block, op->getIterator());

    // Insert the conditional branch
    rewriter.setInsertionPointToEnd(current_block);
    if (else_block) {
      rewriter.create<mlir::cf::CondBranchOp>(loc, op.getCondition(),
                                              then_block, else_block);
    } else {
      rewriter.create<mlir::cf::CondBranchOp>(loc, op.getCondition(),
                                              then_block, merge_block);
    }

    // move the contents of the then region to the new block
    rewriter.mergeBlocks(&op.getThenRegion().front(), then_block);

    // If 'then' block does not end with a return, branch to the merge block
    if (then_block->empty() ||
        !mlir::isa<mlir::lang::ReturnOp, mlir::func::ReturnOp>(
            then_block->back())) {
      rewriter.setInsertionPointToEnd(then_block);
      rewriter.create<mlir::cf::BranchOp>(loc, merge_block);
    }
    // Build the 'else' block if it exists
    if (else_block) {
      // move the contents of the else region to the new block
      rewriter.mergeBlocks(&op.getElseRegion().front(), else_block);

      // If 'else' block does not end with a return, branch to the merge
      // block
      if (else_block->empty() ||
          !mlir::isa<mlir::lang::ReturnOp, mlir::func::ReturnOp>(
              else_block->back())) {
        rewriter.setInsertionPointToEnd(else_block);
        rewriter.create<mlir::cf::BranchOp>(loc, merge_block);
      }
    }

    // Continue building from the merge block
    rewriter.setInsertionPointToStart(merge_block);
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

struct YieldOpLowering : public mlir::OpConversionPattern<mlir::lang::YieldOp> {
  using OpConversionPattern<mlir::lang::YieldOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::YieldOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (!mlir::isa<mlir::scf::IfOp>(op->getParentOp())) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    auto yield_op =
        rewriter.create<mlir::scf::YieldOp>(op.getLoc(), adaptor.getOperands());
    rewriter.replaceOp(op, yield_op);
    return mlir::success();
  }
};

struct UndefOpLowering : public mlir::OpConversionPattern<mlir::lang::UndefOp> {
  using mlir::OpConversionPattern<mlir::lang::UndefOp>::OpConversionPattern;

  UndefOpLowering(mlir::TypeConverter &typeConverter,
                  mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::UndefOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::UndefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type type = op.getType();
    auto new_type = this->getTypeConverter()->convertType(type);
    if (!new_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert type");
    }
    // rewriter.replaceOpWithNewOp<mlir::LLVM::UndefOp>(op, new_type);
    // allocate an llvm struct
    auto llvm_struct_type =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(new_type);
    if (!llvm_struct_type) {
      return rewriter.notifyMatchFailure(
          op, "converted type is not an LLVM struct");
    }
    auto one_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    auto alloca_op = rewriter.create<mlir::LLVM::AllocaOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
        llvm_struct_type, one_val);
    rewriter.replaceOp(op, alloca_op);
    return mlir::success();
  }
};

struct ReturnOpLowering
    : public mlir::OpConversionPattern<mlir::lang::ReturnOp> {
  using mlir::OpConversionPattern<mlir::lang::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (mlir::isa<mlir::scf::IfOp>(op->getParentOp())) {
      auto yield_op = rewriter.create<mlir::scf::YieldOp>(
          op.getLoc(), adaptor.getOperands());
      rewriter.replaceOp(op, yield_op);
      return mlir::success();
    }
    // rewriter.setInsertionPointAfter(op);
    auto branch_op = rewriter.create<mlir::cf::BranchOp>(
        op.getLoc(),
        &op->getParentOfType<mlir::func::FuncOp>().getBlocks().back(),
        adaptor.getOperands());

    rewriter.replaceOp(op, branch_op);
    return mlir::success();
  }
};

struct IndexAccessOpLowering
    : public mlir::OpConversionPattern<mlir::lang::IndexAccessOp> {
  using mlir::OpConversionPattern<
      mlir::lang::IndexAccessOp>::OpConversionPattern;

  IndexAccessOpLowering(mlir::TypeConverter &typeConverter,
                        mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::IndexAccessOp>(typeConverter, context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::IndexAccessOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto container = adaptor.getContainer();
    auto index = adaptor.getIndex();

    if (index.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      index =
          index.getDefiningOp<mlir::UnrealizedConversionCastOp>().getOperand(0);
    }
    if (container.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      container = container.getDefiningOp<mlir::UnrealizedConversionCastOp>()
                      .getOperand(0);
    }

    if (index.getType() != rewriter.getIndexType()) {
      return op.emitError("index type must be index");
    }

    if (auto memref_type =
            mlir::dyn_cast<mlir::MemRefType>(container.getType())) {
      auto load_op = rewriter.create<mlir::memref::LoadOp>(
          op.getLoc(), memref_type.getElementType(), container, index);
      rewriter.replaceOp(op, load_op.getResult());
      return mlir::success();
    } else {
      return op.emitError("unsupported container type");
    }
    return mlir::success();
  }
};

struct ArrayOpLowering : public mlir::OpConversionPattern<mlir::lang::ArrayOp> {
  using mlir::OpConversionPattern<mlir::lang::ArrayOp>::OpConversionPattern;

  ArrayOpLowering(mlir::TypeConverter &typeConverter,
                  mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::ArrayOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::ArrayOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto converted_type = this->getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert type");
    }
    mlir::MemRefType memref_type =
        mlir::dyn_cast<mlir::MemRefType>(converted_type);
    if (!memref_type) {
      return rewriter.notifyMatchFailure(op, "converted type is not a memref");
    }
    // Use memref alloca to allocate memory for the array
    auto alloca_op =
        rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), memref_type);
    // Store the array
    rewriter.create<mlir::memref::StoreOp>(op.getLoc(), adaptor.getValues(),
                                           alloca_op.getResult());
    rewriter.replaceOp(op, alloca_op);
    return mlir::success();
  }
};

struct TypeConstOpLowering
    : public mlir::OpConversionPattern<mlir::lang::TypeConstOp> {
  using mlir::OpConversionPattern<mlir::lang::TypeConstOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::TypeConstOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ConstantOpLowering
    : public mlir::OpConversionPattern<mlir::lang::ConstantOp> {
  using mlir::OpConversionPattern<mlir::lang::ConstantOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::ConstantOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        op, adaptor.getValue().getType(), adaptor.getValue());
    return mlir::success();
  }
};

struct TupleOpLowering : public mlir::OpConversionPattern<mlir::lang::TupleOp> {

  TupleOpLowering(mlir::TypeConverter &typeConverter,
                  mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::TupleOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::TupleOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type converted_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");
    }
    auto one_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    auto alloca_op = rewriter.create<mlir::LLVM::AllocaOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
        converted_type, one_val);

    // Insert each field into the struct using LLVM::StoreOp
    for (auto it : llvm::enumerate(adaptor.getOperands())) {
      mlir::Value field = it.value();
      // if unrealized conversion cast, get the original value
      if (field.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        field =
            field.getDefiningOp<mlir::UnrealizedConversionCastOp>().getOperand(
                0);
      }

      // Optionally, convert the field type if necessary
      mlir::Type field_type = field.getType();
      mlir::Type converted_field_type = typeConverter->convertType(field_type);
      if (!converted_field_type)
        return rewriter.notifyMatchFailure(op, "Failed to convert field type");

      // If the field is a pointer (e.g., a nested tuple), load the value
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(field_type)) {
        auto field_alloca =
            mlir::dyn_cast<mlir::LLVM::AllocaOp>(field.getDefiningOp());
        if (field_alloca) {
          field = rewriter.create<mlir::LLVM::LoadOp>(
              op.getLoc(), field_alloca.getElemType(), field);
        }
      }

      auto index_val = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI64Type(),
          rewriter.getI64IntegerAttr(it.index()));
      auto zero_val = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
      // Insert the field into the struct
      auto gep_op = rewriter.create<mlir::LLVM::GEPOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
          converted_type, alloca_op, mlir::ValueRange{zero_val, index_val});
      rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), field, gep_op);
    }

    rewriter.replaceOp(op, alloca_op);
    return mlir::success();
  }
};

struct AssignOpLowering
    : public mlir::OpConversionPattern<mlir::lang::AssignOp> {
  using mlir::OpConversionPattern<mlir::lang::AssignOp>::OpConversionPattern;

  AssignOpLowering(mlir::TypeConverter &typeConverter,
                   mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::AssignOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::AssignOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Value lhs = adaptor.getTarget();
    mlir::Value rhs = adaptor.getValue();

    // Ensure types are convertible
    mlir::Type converted_lhs_type =
        this->getTypeConverter()->convertType(lhs.getType());
    mlir::Type converted_rhs_type =
        this->getTypeConverter()->convertType(rhs.getType());

    if (!converted_lhs_type || !converted_rhs_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert types");
    }
    if (lhs.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      auto original_lhs =
          lhs.getDefiningOp<mlir::UnrealizedConversionCastOp>().getOperand(0);
      if (mlir::isa<mlir::LLVM::LLVMPointerType, mlir::MemRefType>(
              original_lhs.getType())) {
        lhs = original_lhs;
      }
    }
    if (rhs.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      auto original_rhs =
          rhs.getDefiningOp<mlir::UnrealizedConversionCastOp>().getOperand(0);
      if (mlir::isa<mlir::LLVM::LLVMPointerType, mlir::MemRefType>(
              original_rhs.getType())) {
        rhs = original_rhs;
      }
    }
    if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(lhs.getType())) {
      rewriter.create<mlir::memref::StoreOp>(op.getLoc(), rhs, lhs);
      rewriter.replaceOp(op, rhs);
    } else {
      rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), rhs, lhs);
      rewriter.eraseOp(op);
    }
    return mlir::success();
  }
};

struct DerefOpLowering : public mlir::OpConversionPattern<mlir::lang::DerefOp> {
  using mlir::OpConversionPattern<mlir::lang::DerefOp>::OpConversionPattern;

  DerefOpLowering(mlir::TypeConverter &typeConverter,
                  mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::DerefOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::DerefOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Value addr = adaptor.getAddr();
    mlir::Type converted_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert type");
    }
    if (addr.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      addr =
          addr.getDefiningOp<mlir::UnrealizedConversionCastOp>().getOperand(0);
    }
    rewriter.replaceOp(op, rewriter.create<mlir::memref::LoadOp>(
                               op.getLoc(), converted_type, addr));
    return mlir::success();
  }
};

struct StructAccessOpLowering
    : public mlir::OpConversionPattern<mlir::lang::StructAccessOp> {
  using mlir::OpConversionPattern<
      mlir::lang::StructAccessOp>::OpConversionPattern;

  StructAccessOpLowering(mlir::TypeConverter &typeConverter,
                         mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::StructAccessOp>(typeConverter,
                                                        context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::StructAccessOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto input = adaptor.getInput();

    mlir::Type converted_struct_type =
        this->getTypeConverter()->convertType(input.getType());
    if (!converted_struct_type)
      return rewriter.notifyMatchFailure(op,
                                         "Failed to convert input struct type");

    auto llvm_struct_type =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(converted_struct_type);
    if (!llvm_struct_type)
      return rewriter.notifyMatchFailure(
          op, "Converted type is not an LLVM struct");

    int64_t field_index = op.getIndex();

    unsigned num_fields = llvm_struct_type.getBody().size();
    if (field_index < 0 || static_cast<unsigned>(field_index) >= num_fields)
      return rewriter.notifyMatchFailure(op, "Field index out of bounds");

    if (adaptor.getInput().getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      input = adaptor.getInput()
                  .getDefiningOp<mlir::UnrealizedConversionCastOp>()
                  .getInputs()[0];
    }

    auto zero_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    auto index_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(field_index));

    if (mlir::isa<mlir::LLVM::LLVMStructType>(input.getType())) {
      auto extracted_value = rewriter.create<mlir::LLVM::ExtractValueOp>(
          op.getLoc(), llvm_struct_type.getBody()[field_index], input,
          field_index);
      rewriter.replaceOp(op, extracted_value.getResult());
      return mlir::success();
    }

    auto gep = rewriter.create<mlir::LLVM::GEPOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
        llvm_struct_type, input, mlir::ValueRange{zero_val, index_val});
    auto extracted_value = rewriter.create<mlir::LLVM::LoadOp>(
        op.getLoc(), llvm_struct_type.getBody()[field_index], gep);

    rewriter.replaceOp(op, extracted_value);

    return mlir::success();
  }
};

struct ResolveCastPattern
    : public mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp> {
  using mlir::OpConversionPattern<
      mlir::UnrealizedConversionCastOp>::OpConversionPattern;

  ResolveCastPattern(mlir::TypeConverter &typeConverter,
                     mlir::MLIRContext *context)
      : OpConversionPattern<mlir::UnrealizedConversionCastOp>(typeConverter,
                                                              context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp castOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (castOp.getNumOperands() != 1) {
      return rewriter.notifyMatchFailure(castOp, "expected one operand");
    }
    mlir::Value original = castOp.getOperand(0);
    mlir::Type original_type = original.getType();
    mlir::Type converted_type =
        this->getTypeConverter()->convertType(original_type);
    mlir::Type target_type = castOp.getType(0);

    // llvm::errs() << "originalType: " << originalType << "\n";
    // llvm::errs() << "convertedType: " << convertedType << "\n";
    // llvm::errs() << "targetType: " << targetType << "\n";

    if (mlir::isa<mlir::LLVM::LLVMPointerType>(converted_type) &&
        mlir::isa<mlir::LLVM::LLVMStructType>(target_type)) {
      // Load the value from the pointer
      auto load_op = rewriter.create<mlir::LLVM::LoadOp>(castOp.getLoc(),
                                                         target_type, original);
      rewriter.replaceOp(castOp, load_op.getResult());
      return mlir::success();
    }
    // if targetType and convertedType are the same, we can erase the cast
    if (target_type == converted_type) {
      rewriter.replaceOp(castOp, original);
      return mlir::success();
    }
    if (!converted_type) {
      return rewriter.notifyMatchFailure(castOp,
                                         "failed to convert operand type");
    }

    // if either the original or converted type is a struct
    // then we can construct using the target struct's constructor function
    if (auto target_struct_type =
            mlir::dyn_cast<mlir::lang::StructType>(target_type)) {
      if (auto converted_struct_type =
              mlir::dyn_cast<mlir::lang::StructType>(converted_type)) {
        mlir::Operation *op = castOp.getOperation();
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto constructor_name = converted_struct_type.getName().str() + "_init";
        auto constructor =
            module.lookupSymbol<mlir::lang::FuncOp>(constructor_name);
        if (!constructor) {
          return rewriter.notifyMatchFailure(
              castOp, "constructor function not found for struct type");
        }

        // Call the constructor function
        mlir::ValueRange args(original);
        auto call_op = rewriter.create<mlir::lang::CallOp>(
            castOp.getLoc(), constructor.getFunctionType(), args);
        rewriter.replaceOp(castOp, call_op.getResult(0));
        return mlir::success();
      }
    }

    return mlir::success();
  }
};

struct CreateStructOpLowering
    : public mlir::OpConversionPattern<mlir::lang::CreateStructOp> {
  using mlir::OpConversionPattern<
      mlir::lang::CreateStructOp>::OpConversionPattern;

  CreateStructOpLowering(mlir::TypeConverter &typeConverter,
                         mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::CreateStructOp>(typeConverter,
                                                        context) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::lang::CreateStructOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Type converted_type =
        this->getTypeConverter()->convertType(op.getType());
    if (!converted_type) {
      llvm::errs() << "Failed to convert type: " << op.getType() << "\n";
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");
    }
    auto one_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    auto zero_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(op.getContext());
    auto alloca_op = rewriter.create<mlir::LLVM::AllocaOp>(
        op.getLoc(), ptr_type, converted_type, one_val);

    // Insert each field into the struct using LLVM::StoreOp
    for (auto it : llvm::enumerate(adaptor.getFields())) {
      mlir::Value field = it.value();

      // Optionally, convert the field type if necessary
      mlir::Type field_type = field.getType();
      mlir::Type converted_field_type = typeConverter->convertType(field_type);
      if (!converted_field_type)
        return rewriter.notifyMatchFailure(op, "Failed to convert field type");

      auto index_val = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI64Type(),
          rewriter.getI64IntegerAttr(it.index()));
      // Insert the field into the struct
      auto gep_op = rewriter.create<mlir::LLVM::GEPOp>(
          op.getLoc(), ptr_type, converted_type, alloca_op,
          mlir::ValueRange{zero_val, index_val});
      rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), field, gep_op);
    }
    rewriter.replaceOp(op, alloca_op);
    return mlir::success();
  }
};

struct VarDeclOpLowering
    : public mlir::OpConversionPattern<mlir::lang::VarDeclOp> {
  using mlir::OpConversionPattern<mlir::lang::VarDeclOp>::OpConversionPattern;

  VarDeclOpLowering(mlir::TypeConverter &typeConverter,
                    mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::VarDeclOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::VarDeclOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {

    mlir::Type var_type = op.getVarType().value_or(op.getInitValue().getType());

    if (mlir::isa<mlir::lang::StructType>(var_type)) {
      // NOTE: Check this
      rewriter.replaceOp(op, adaptor.getInitValue());
      return mlir::success();
    }

    if (mlir::isa<mlir::lang::StringType>(var_type)) {
      rewriter.replaceOp(op, adaptor.getInitValue());
      return mlir::success();
    }

    // If the variable type is a TypeValueType, get the aliased type
    if (mlir::isa<mlir::lang::TypeValueType>(var_type)) {
      // if the init value is also a TypeValueType, we can erase the VarDeclOp
      if (mlir::isa<mlir::lang::TypeValueType>(
              adaptor.getInitValue().getType())) {
        rewriter.eraseOp(op);
        return mlir::success();
      }
      auto type_value = mlir::cast<mlir::lang::TypeValueType>(var_type);
      var_type = type_value.getType();
    }

    // If the variable is mutable, we need to allocate memory for it
    if (adaptor.getIsMutable()) {
      var_type = mlir::MemRefType::get({}, var_type);
    }

    // Check if the variable type is a MemRefType
    if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(var_type)) {
      auto alloca_op =
          rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), memref_type);
      if (adaptor.getInitValue()) {
        rewriter.create<mlir::memref::StoreOp>(
            op.getLoc(), adaptor.getInitValue(), alloca_op);
      }
      rewriter.replaceOp(op, alloca_op.getResult());

      return mlir::success();
    }

    if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::VectorType,
                  mlir::TensorType, mlir::lang::IntLiteralType>(var_type)) {
      if (!adaptor.getInitValue()) {
        auto zero = rewriter.getZeroAttr(var_type);
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, var_type,
                                                             zero);
      } else {
        rewriter.replaceOp(op, adaptor.getInitValue());
      }
      return mlir::success();
    }

    if (mlir::isa<mlir::lang::ArrayType>(var_type)) {
      // if (adaptor.getInitValue().getType() != var_type) {
      //   return op.emitError("init value type does not match variable type");
      // }
      rewriter.replaceOp(op, adaptor.getInitValue());
      return mlir::success();
    }
    rewriter.replaceOp(op, adaptor.getInitValue());
    return mlir::success();
    // return mlir::emitError(op.getLoc(), "lowering of variable type ")
    //        << var_type << " not supported";
    return mlir::success();
  }
};

struct StringConstantOpLowering
    : public mlir::OpConversionPattern<mlir::lang::StringConstOp> {
  using OpConversionPattern<mlir::lang::StringConstOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::StringConstOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    mlir::StringRef value = op.getValue();

    // Get LLVM types
    auto *context = rewriter.getContext();
    auto llvm_i8_type = mlir::IntegerType::get(context, 8);
    auto llvm_i8_ptr_type = mlir::LLVM::LLVMPointerType::get(context);

    // Create a global string in the LLVM dialect
    size_t str_size = value.size() + 1; // +1 for null terminator
    auto array_type = mlir::LLVM::LLVMArrayType::get(llvm_i8_type, str_size);

    // Insert global at the module level
    auto module = op->getParentOfType<mlir::ModuleOp>();
    std::string global_name =
        "_str_constant_" +
        std::to_string(reinterpret_cast<uintptr_t>(op.getOperation()));
    if (!module.lookupSymbol<mlir::LLVM::GlobalOp>(global_name)) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      // Create initial value attribute
      auto initial_value = rewriter.getStringAttr(value.str() + '\0');
      rewriter.create<mlir::LLVM::GlobalOp>(
          loc, array_type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal,
          global_name, initial_value);
    }

    auto global_op = module.lookupSymbol<mlir::LLVM::GlobalOp>(global_name);
    // Get pointer to the first character
    auto global = rewriter.create<mlir::LLVM::AddressOfOp>(loc, global_op);
    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(32), rewriter.getI32IntegerAttr(0));
    auto ptr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, llvm_i8_ptr_type, array_type, global,
        mlir::ArrayRef<mlir::Value>{zero, zero});

    // Replace the original operation
    rewriter.replaceOp(op, ptr.getResult());

    return mlir::success();
  }
};

struct PrintOpLowering : public mlir::OpConversionPattern<mlir::lang::PrintOp> {
  using OpConversionPattern<mlir::lang::PrintOp>::OpConversionPattern;

  PrintOpLowering(mlir::TypeConverter &typeConverter,
                  mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::PrintOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {

    // Get the printf function
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto printf_symbol = getOrInsertPrintf(rewriter, module);

    // Get the format string
    auto format_str = op.getFormatAttr().getValue();

    // Create a global string in the LLVM dialect
    static int format_string_counter = 0;
    auto format_str_val = getOrCreateGlobalString(
        op.getLoc(), rewriter, "_fmt" + std::to_string(format_string_counter++),
        format_str, module);

    mlir::SmallVector<mlir::Value, 4> args;
    args.push_back(format_str_val);

    mlir::SmallVector<mlir::Type, 4> arg_types;
    arg_types.push_back(format_str_val.getType());

    for (mlir::Value operand : adaptor.getOperands()) {
      mlir::Type llvm_type =
          this->getTypeConverter()->convertType(operand.getType());
      if (!llvm_type)
        return op.emitError("failed to convert operand type");
      args.push_back(operand);
      arg_types.push_back(llvm_type);
    }

    auto i32_type = rewriter.getIntegerType(32);
    auto func_type = mlir::LLVM::LLVMFunctionType::get(i32_type, arg_types,
                                                       /*isVarArg=*/true);

    // Call the printf function
    rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), func_type, printf_symbol,
                                        mlir::ValueRange(args));

    // Replace the original operation
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  /// Create a function declaration for printf, the signature is:
  ///   * `i32 (i8*, ...)`
  static mlir::LLVM::LLVMFunctionType
  getPrintfType(mlir::MLIRContext *context) {
    auto llvm_i32_type = mlir::IntegerType::get(context, 32);
    auto llvm_ptr_type = mlir::LLVM::LLVMPointerType::get(context);
    auto llvm_fn_type =
        mlir::LLVM::LLVMFunctionType::get(llvm_i32_type, llvm_ptr_type,
                                          /*isVarArg=*/true);
    return llvm_fn_type;
  }

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static mlir::FlatSymbolRefAttr
  getOrInsertPrintf(mlir::PatternRewriter &rewriter, mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
      return mlir::SymbolRefAttr::get(context, "printf");

    // Insert the printf function into the body of the parent module.
    mlir::PatternRewriter::InsertionGuard insert_guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                            getPrintfType(context));
    return mlir::SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the
  /// given name, creating the string if necessary.
  static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                             mlir::OpBuilder &builder,
                                             mlir::StringRef name,
                                             mlir::StringRef value,
                                             mlir::ModuleOp module) {
    // Create the global at the entry of the module.
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insert_guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value),
          /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    mlir::Value global_ptr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getIndexAttr(0));
    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), global_ptr,
        mlir::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

} // namespace lang
} // namespace mlir

namespace {
struct LangToAffineLoweringPass
    : public mlir::PassWrapper<LangToAffineLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LangToAffineLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void LangToAffineLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  mlir::lang::LangToLLVMTypeConverter type_converter(&getContext());

  target.addLegalDialect<mlir::affine::AffineDialect, mlir::BuiltinDialect,
                         mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                         mlir::tensor::TensorDialect, mlir::LLVM::LLVMDialect,
                         mlir::memref::MemRefDialect,
                         mlir::index::IndexDialect>();

  // Mark all operations illegal.
  target.addIllegalDialect<mlir::lang::LangDialect>();
  target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<mlir::lang::IndexAccessOpLowering, mlir::lang::ArrayOpLowering,
               mlir::lang::CreateStructOpLowering, mlir::lang::DerefOpLowering,
               mlir::lang::FuncOpLowering, mlir::lang::AssignOpLowering,
               mlir::lang::ResolveCastPattern, mlir::lang::UndefOpLowering,
               mlir::lang::PrintOpLowering, mlir::lang::TupleOpLowering,
               mlir::lang::StructAccessOpLowering>(type_converter,
                                                   &getContext());

  patterns.add<mlir::lang::IfOpLowering, mlir::lang::CallOpLowering,
               mlir::lang::ReturnOpLowering, mlir::lang::VarDeclOpLowering,
               mlir::lang::TypeConstOpLowering,
               mlir::lang::StringConstantOpLowering,
               mlir::lang::ConstantOpLowering, mlir::lang::YieldOpLowering>(
      &getContext());

  // Apply partial conversion.
  if (failed(mlir::applyFullConversion(getOperation(), target,
                                       std::move(patterns)))) {

    llvm::errs() << "LangToAffineLoweringPass: Full conversion failed for\n";
    getOperation().dump();
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::lang::createLowerToAffinePass() {
  return std::make_unique<LangToAffineLoweringPass>();
}

namespace {
struct ResolveCastPatternPass
    : public mlir::PassWrapper<ResolveCastPatternPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResolveCastPatternPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ResolveCastPatternPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  mlir::lang::LangToLLVMTypeConverter type_converter(&getContext());

  target.addLegalDialect<mlir::affine::AffineDialect, mlir::BuiltinDialect,
                         mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                         mlir::tensor::TensorDialect>();

  // Mark all operations illegal.
  target.addIllegalDialect<mlir::lang::LangDialect>();
  target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<mlir::lang::ResolveCastPattern>(type_converter, &getContext());

  // Apply partial conversion.
  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass>
mlir::lang::createUnrealizedConversionCastResolverPass() {
  return std::make_unique<ResolveCastPatternPass>();
}
