#include "dialect/LangDialect.h"
#include "dialect/LangOps.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>
#include <utility>

#include "mlir/AsmParser/AsmParser.h"

class LangToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  LangToLLVMTypeConverter(mlir::MLIRContext *ctx) : LLVMTypeConverter(ctx) {
    addConversion(
        [&](mlir::lang::StructType struct_type) -> std::optional<mlir::Type> {
          mlir::MLIRContext *ctx = struct_type.getContext();
          mlir::SmallVector<mlir::Type, 4> elementTypes;
          for (mlir::Type fieldType : struct_type.getElementTypes()) {
            elementTypes.push_back(convertType(fieldType));
          }
          auto llvm_struct_type = mlir::LLVM::LLVMStructType::getIdentified(
              ctx, struct_type.getName());
          if (mlir::failed(
                  llvm_struct_type.setBody(elementTypes, /*isPacked=*/false))) {
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
  }
};

struct FuncOpLowering : public mlir::OpConversionPattern<mlir::lang::FuncOp> {
  using OpConversionPattern<mlir::lang::FuncOp>::OpConversionPattern;

  FuncOpLowering(mlir::TypeConverter &typeConverter, mlir::MLIRContext *context)
      : OpConversionPattern<mlir::lang::FuncOp>(typeConverter, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto func_type = op.getFunctionType();
    auto input_types = func_type.getInputs();
    auto result_types = func_type.getResults();
    mlir::TypeConverter::SignatureConversion signatureConversion(
        input_types.size());
    mlir::SmallVector<mlir::Type, 4> convertedResultTypes;

    for (const auto &type : llvm::enumerate(input_types)) {
      mlir::Type convertedType = getTypeConverter()->convertType(type.value());
      if (!convertedType) {
        return rewriter.notifyMatchFailure(op, "failed to convert input type");
      }
      signatureConversion.addInputs(type.index(), convertedType);
    }

    for (mlir::Type type : result_types) {
      mlir::Type convertedType = getTypeConverter()->convertType(type);
      if (!convertedType) {
        return rewriter.notifyMatchFailure(op, "failed to convert result type");
      }
      convertedResultTypes.push_back(convertedType);
    }

    auto convertedFuncType = mlir::FunctionType::get(
        op.getContext(), signatureConversion.getConvertedTypes(),
        convertedResultTypes);

    auto newFuncOp = rewriter.create<mlir::func::FuncOp>(
        op.getLoc(), op.getName(), convertedFuncType);

    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    if (failed(rewriter.convertRegionTypes(&newFuncOp.getRegion(),
                                           *this->getTypeConverter(),
                                           &signatureConversion))) {
      return rewriter.notifyMatchFailure(op, "failed to convert region types");
    }
    llvm::SmallVector<mlir::Location, 4> locations(result_types.size(),
                                                   rewriter.getUnknownLoc());
    auto return_block =
        rewriter.createBlock(&newFuncOp.getBody(), newFuncOp.getBody().end(),
                             newFuncOp.getResultTypes(), locations);
    rewriter.setInsertionPointToStart(return_block);
    rewriter.create<mlir::func::ReturnOp>(op.getLoc(),
                                          return_block->getArguments());
    rewriter.eraseOp(op);
    op.emitWarning() << "\nfunc op =>\n";
    return mlir::success();
  }
};

struct CallOpLowering : public mlir::OpConversionPattern<mlir::lang::CallOp> {
  using OpConversionPattern<mlir::lang::CallOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto call = rewriter.create<mlir::func::CallOp>(
        op.getLoc(), op.getCalleeAttr(), op.getResultTypes(), op.getOperands());
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
    auto if_op = rewriter.create<mlir::scf::IfOp>(op.getLoc(), result_type,
                                                  adaptor.getCondition());

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
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *parentRegion = currentBlock->getParent();

    // Create the blocks for the 'then', 'else', and continuation ('merge')
    // blocks
    auto *thenBlock = rewriter.createBlock(
        parentRegion, std::next(currentBlock->getIterator()));
    mlir::Block *elseBlock = nullptr;
    if (!op.getElseRegion().empty()) {
      elseBlock = rewriter.createBlock(parentRegion,
                                       std::next(currentBlock->getIterator()));
    }
    // Create the merge block by splitting the current block
    auto *mergeBlock = rewriter.splitBlock(currentBlock, op->getIterator());

    // Insert the conditional branch
    rewriter.setInsertionPointToEnd(currentBlock);
    if (elseBlock) {
      rewriter.create<mlir::cf::CondBranchOp>(loc, op.getCondition(), thenBlock,
                                              elseBlock);
    } else {
      rewriter.create<mlir::cf::CondBranchOp>(loc, op.getCondition(), thenBlock,
                                              mergeBlock);
    }

    // move the contents of the then region to the new block
    rewriter.mergeBlocks(&op.getThenRegion().front(), thenBlock);

    // If 'then' block does not end with a return, branch to the merge block
    if (thenBlock->empty() ||
        !mlir::isa<mlir::lang::ReturnOp, mlir::func::ReturnOp>(
            thenBlock->back())) {
      rewriter.setInsertionPointToEnd(thenBlock);
      rewriter.create<mlir::cf::BranchOp>(loc, mergeBlock);
    }
    // Build the 'else' block if it exists
    if (elseBlock) {
      // move the contents of the else region to the new block
      rewriter.mergeBlocks(&op.getElseRegion().front(), elseBlock);

      // If 'else' block does not end with a return, branch to the merge
      // block
      if (elseBlock->empty() ||
          !mlir::isa<mlir::lang::ReturnOp, mlir::func::ReturnOp>(
              elseBlock->back())) {
        rewriter.setInsertionPointToEnd(elseBlock);
        rewriter.create<mlir::cf::BranchOp>(loc, mergeBlock);
      }
    }

    // Continue building from the merge block
    rewriter.setInsertionPointToStart(mergeBlock);
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
    auto llvmStructType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(new_type);
    if (!llvmStructType) {
      return rewriter.notifyMatchFailure(
          op, "converted type is not an LLVM struct");
    }
    auto one_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    auto alloca_op = rewriter.create<mlir::LLVM::AllocaOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
        llvmStructType, one_val);
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
      rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op,
                                                      adaptor.getOperands());
      return mlir::success();
    }
    // rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
    //                                                   adaptor.getOperands());
    rewriter.setInsertionPointAfter(op);
    auto branch_op = rewriter.create<mlir::cf::BranchOp>(
        op.getLoc(),
        &op->getParentOfType<mlir::func::FuncOp>().getBlocks().back(),
        adaptor.getOperands());

    // rewriter.replaceOp(op, branch_op);
    rewriter.eraseOp(op);

    llvm::errs() << "return op =>\n";
    branch_op->getParentOfType<mlir::func::FuncOp>().dump();
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
    mlir::Type convertedLhsType =
        this->getTypeConverter()->convertType(lhs.getType());
    mlir::Type convertedRhsType =
        this->getTypeConverter()->convertType(rhs.getType());

    if (!convertedLhsType || !convertedRhsType) {
      return rewriter.notifyMatchFailure(op, "failed to convert types");
    }
    if (lhs.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      auto original_lhs =
          lhs.getDefiningOp<mlir::UnrealizedConversionCastOp>().getOperand(0);
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(original_lhs.getType())) {
        lhs = original_lhs;
      }
    }
    if (rhs.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      auto original_rhs =
          rhs.getDefiningOp<mlir::UnrealizedConversionCastOp>().getOperand(0);
      if (mlir::isa<mlir::LLVM::LLVMPointerType>(original_rhs.getType())) {
        rhs = original_rhs;
      }
    }
    // Store the value in the lhs
    if (auto memRefType = mlir::dyn_cast<mlir::MemRefType>(convertedLhsType)) {
      auto storeOp = rewriter.create<mlir::memref::StoreOp>(
          op.getLoc(), rhs, lhs, mlir::ValueRange{});
      rewriter.replaceOp(op, storeOp);
    } else {
      rewriter.create<mlir::LLVM::StoreOp>(op.getLoc(), rhs, lhs);
      rewriter.eraseOp(op);
    }

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

    mlir::Type convertedStructType =
        this->getTypeConverter()->convertType(input.getType());
    if (!convertedStructType)
      return rewriter.notifyMatchFailure(op,
                                         "Failed to convert input struct type");

    auto llvmStructType =
        mlir::dyn_cast<mlir::LLVM::LLVMStructType>(convertedStructType);
    if (!llvmStructType)
      return rewriter.notifyMatchFailure(
          op, "Converted type is not an LLVM struct");

    int64_t fieldIndex = op.getIndex();

    unsigned numFields = llvmStructType.getBody().size();
    if (fieldIndex < 0 || static_cast<unsigned>(fieldIndex) >= numFields)
      return rewriter.notifyMatchFailure(op, "Field index out of bounds");

    if (adaptor.getInput().getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      input = adaptor.getInput()
                  .getDefiningOp<mlir::UnrealizedConversionCastOp>()
                  .getInputs()[0];
    }

    // mlir::Value extractedValue =
    // rewriter.create<mlir::LLVM::ExtractValueOp>(
    //     op.getLoc(), llvmStructType.getBody()[fieldIndex],
    //     adaptor.getInput(), fieldIndex);

    auto zero_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    auto index_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(fieldIndex));

    auto gep = rewriter.create<mlir::LLVM::GEPOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
        llvmStructType, input, mlir::ValueRange{zero_val, index_val});
    auto extractedValue = rewriter.create<mlir::LLVM::LoadOp>(
        op.getLoc(), llvmStructType.getBody()[fieldIndex], gep);

    rewriter.replaceOp(op, extractedValue);

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
    mlir::Type originalType = original.getType();
    mlir::Type convertedType =
        this->getTypeConverter()->convertType(originalType);
    mlir::Type targetType = castOp.getType(0);

    llvm::errs() << "originalType: " << originalType << "\n";
    llvm::errs() << "convertedType: " << convertedType << "\n";
    llvm::errs() << "targetType: " << targetType << "\n";

    if (mlir::isa<mlir::LLVM::LLVMPointerType>(convertedType) &&
        mlir::isa<mlir::LLVM::LLVMStructType>(targetType)) {
      // Load the value from the pointer
      auto loadOp = rewriter.create<mlir::LLVM::LoadOp>(castOp.getLoc(),
                                                        targetType, original);
      rewriter.replaceOp(castOp, loadOp.getResult());
      return mlir::success();
    }
    // if targetType and convertedType are the same, we can erase the cast
    if (targetType == convertedType) {
      rewriter.replaceOp(castOp, original);
      return mlir::success();
    }
    if (!convertedType) {
      return rewriter.notifyMatchFailure(castOp,
                                         "failed to convert operand type");
    }

    // if either the original or converted type is a struct
    // then we can construct using the target struct's constructor function
    if (auto targetStructType =
            mlir::dyn_cast<mlir::lang::StructType>(targetType)) {
      if (auto convertedStructType =
              mlir::dyn_cast<mlir::lang::StructType>(convertedType)) {
        mlir::Operation *op = castOp.getOperation();
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto constructorName = convertedStructType.getName().str() + "_init";
        auto constructor =
            module.lookupSymbol<mlir::lang::FuncOp>(constructorName);
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
    mlir::Type convertedType =
        this->getTypeConverter()->convertType(op.getType());
    if (!convertedType) {
      llvm::errs() << "Failed to convert type: " << op.getType() << "\n";
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");
    }
    auto one_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    auto zero_val = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
    auto ptr_type = mlir::LLVM::LLVMPointerType::get(op.getContext());
    auto alloca_op = rewriter.create<mlir::LLVM::AllocaOp>(
        op.getLoc(), ptr_type, convertedType, one_val);

    // Insert each field into the struct using LLVM::StoreOp
    for (auto it : llvm::enumerate(adaptor.getFields())) {
      mlir::Value field = it.value();

      // Optionally, convert the field type if necessary
      mlir::Type fieldType = field.getType();
      mlir::Type convertedFieldType = typeConverter->convertType(fieldType);
      if (!convertedFieldType)
        return rewriter.notifyMatchFailure(op, "Failed to convert field type");

      auto index_val = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI64Type(),
          rewriter.getI64IntegerAttr(it.index()));
      // Insert the field into the struct
      auto gep_op = rewriter.create<mlir::LLVM::GEPOp>(
          op.getLoc(), ptr_type, convertedType, alloca_op,
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

    mlir::Type varType = op.getVarType().value_or(mlir::Type());

    if (mlir::isa<mlir::lang::StructType>(varType)) {
      // NOTE: Check this
      rewriter.replaceOp(op, adaptor.getInitValue());
      return mlir::success();
    }

    if (mlir::isa<mlir::lang::StringType>(varType)) {
      rewriter.replaceOp(op, adaptor.getInitValue());
      return mlir::success();
    }

    // If the variable type is a TypeValueType, get the aliased type
    if (mlir::isa<mlir::lang::TypeValueType>(varType)) {
      // if the init value is also a TypeValueType, we can erase the VarDeclOp
      if (mlir::isa<mlir::lang::TypeValueType>(
              adaptor.getInitValue().getType())) {
        rewriter.eraseOp(op);
        return mlir::success();
      }
      auto type_value = mlir::cast<mlir::lang::TypeValueType>(varType);
      varType = type_value.getAliasedType();
    }

    // Check if the variable type is a MemRefType
    if (auto memRefType = mlir::dyn_cast<mlir::MemRefType>(varType)) {
      auto allocOp =
          rewriter.create<mlir::memref::AllocOp>(op.getLoc(), memRefType);
      if (adaptor.getInitValue()) {
        rewriter.create<mlir::memref::StoreOp>(op.getLoc(),
                                               adaptor.getInitValue(), allocOp);
      }
      rewriter.replaceOp(op, allocOp.getResult());
      return mlir::success();
    }

    if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::VectorType,
                  mlir::TensorType, mlir::lang::IntLiteralType>(varType)) {
      if (!adaptor.getInitValue()) {
        auto zero = rewriter.getZeroAttr(varType);
        rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(op, varType, zero);
      } else {
        rewriter.replaceOp(op, adaptor.getInitValue());
      }
      return mlir::success();
    }

    return mlir::emitError(op.getLoc(), "lowering of variable type ")
           << varType << " not supported";
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
    auto llvmI8Type = mlir::IntegerType::get(context, 8);
    auto llvmI8PtrType = mlir::LLVM::LLVMPointerType::get(context);

    // Create a global string in the LLVM dialect
    size_t strSize = value.size() + 1; // +1 for null terminator
    auto arrayType = mlir::LLVM::LLVMArrayType::get(llvmI8Type, strSize);

    // Insert global at the module level
    auto module = op->getParentOfType<mlir::ModuleOp>();
    std::string globalName =
        "_str_constant_" +
        std::to_string(reinterpret_cast<uintptr_t>(op.getOperation()));
    if (!module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName)) {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      // Create initial value attribute
      auto initialValue = rewriter.getStringAttr(value.str() + '\0');
      rewriter.create<mlir::LLVM::GlobalOp>(loc, arrayType, /*isConstant=*/true,
                                            mlir::LLVM::Linkage::Internal,
                                            globalName, initialValue);
    }

    auto global_op = module.lookupSymbol<mlir::LLVM::GlobalOp>(globalName);
    // Get pointer to the first character
    auto global = rewriter.create<mlir::LLVM::AddressOfOp>(loc, global_op);
    auto zero = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, rewriter.getIntegerType(32), rewriter.getI32IntegerAttr(0));
    auto ptr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, llvmI8PtrType, arrayType, global,
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
    auto printfSymbol = getOrInsertPrintf(rewriter, module);

    // Get the format string
    auto format_str = op.getFormatAttr().getValue();

    // Create a global string in the LLVM dialect
    static int formatStringCounter = 0;
    auto format_str_val = getOrCreateGlobalString(
        op.getLoc(), rewriter, "_fmt" + std::to_string(formatStringCounter++),
        format_str, module);

    mlir::SmallVector<mlir::Value, 4> args;
    args.push_back(format_str_val);

    mlir::SmallVector<mlir::Type, 4> argTypes;
    argTypes.push_back(format_str_val.getType());

    for (mlir::Value operand : adaptor.getOperands()) {
      mlir::Type llvmType =
          this->getTypeConverter()->convertType(operand.getType());
      if (!llvmType)
        return op.emitError("failed to convert operand type");
      args.push_back(operand);
      argTypes.push_back(llvmType);
    }

    auto i32Ty = rewriter.getIntegerType(32);
    auto funcType = mlir::LLVM::LLVMFunctionType::get(i32Ty, argTypes,
                                                      /*isVarArg=*/true);

    // Call the printf function
    rewriter.create<mlir::LLVM::CallOp>(op.getLoc(), funcType, printfSymbol,
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
    auto llvmI32Ty = mlir::IntegerType::get(context, 32);
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(context);
    auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                        /*isVarArg=*/true);
    return llvmFnType;
  }

  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static mlir::FlatSymbolRefAttr
  getOrInsertPrintf(mlir::PatternRewriter &rewriter, mlir::ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"))
      return mlir::SymbolRefAttr::get(context, "printf");

    // Insert the printf function into the body of the parent module.
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
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
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value),
          /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getIndexAttr(0));
    return builder.create<mlir::LLVM::GEPOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        global.getType(), globalPtr, mlir::ArrayRef<mlir::Value>({cst0, cst0}));
  }
};

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

void LangToAffineLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  LangToLLVMTypeConverter typeConverter(&getContext());

  target
      .addLegalDialect<mlir::affine::AffineDialect, mlir::BuiltinDialect,
                       mlir::arith::ArithDialect, mlir::func::FuncDialect,
                       mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
                       mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect>();

  // Mark all operations illegal.
  target.addIllegalDialect<mlir::lang::LangDialect>();
  target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CreateStructOpLowering, FuncOpLowering, AssignOpLowering,
               ResolveCastPattern, UndefOpLowering, PrintOpLowering,
               StructAccessOpLowering>(typeConverter, &getContext());

  patterns.add<IfOpLowering, CallOpLowering, ReturnOpLowering,
               VarDeclOpLowering, TypeConstOpLowering, StringConstantOpLowering,
               ConstantOpLowering, YieldOpLowering>(&getContext());

  // Apply partial conversion.
  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns)))) {

    llvm::errs() << "Partial conversion failed for\n";
    getOperation().dump();
    signalPassFailure();
  }
}

void ResolveCastPatternPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  LangToLLVMTypeConverter typeConverter(&getContext());

  target.addLegalDialect<mlir::affine::AffineDialect, mlir::BuiltinDialect,
                         mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                         mlir::LLVM::LLVMDialect>();

  // Mark all operations illegal.
  target.addIllegalDialect<mlir::lang::LangDialect>();
  target.addIllegalOp<mlir::UnrealizedConversionCastOp>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ResolveCastPattern>(typeConverter, &getContext());

  // Apply partial conversion.
  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::lang::createLowerToAffinePass() {
  return std::make_unique<LangToAffineLoweringPass>();
}

std::unique_ptr<mlir::Pass>
mlir::lang::createUnrealizedConversionCastResolverPass() {
  return std::make_unique<ResolveCastPatternPass>();
}
