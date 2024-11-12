
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>
#include <utility>

class LangToLLVMTypeConverter : public mlir::LLVMTypeConverter {
public:
  explicit LangToLLVMTypeConverter(mlir::MLIRContext *ctx)
      : LLVMTypeConverter(ctx) {}

  mlir::Type convertType(mlir::Type type) const {
    if (auto structType = mlir::dyn_cast<mlir::lang::StructType>(type)) {
      mlir::SmallVector<mlir::Type, 4> elementTypes;
      for (mlir::Type fieldType : structType.getElementTypes()) {
        mlir::Type convertedType = this->convertType(fieldType);
        if (!convertedType)
          return nullptr;
        elementTypes.push_back(convertedType);
      }
      return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(),
                                                    elementTypes);
    }
    return LLVMTypeConverter::convertType(type);
  }
};

struct FuncOpLowering : public mlir::OpConversionPattern<mlir::lang::FuncOp> {
  using OpConversionPattern<mlir::lang::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ReturnOpLowering
    : public mlir::OpConversionPattern<mlir::lang::ReturnOp> {
  using mlir::OpConversionPattern<mlir::lang::ReturnOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      adaptor.getOperands());
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

struct StructAccessOpLowering
    : public mlir::OpConversionPattern<mlir::lang::StructAccessOp> {
  using mlir::OpConversionPattern<
      mlir::lang::StructAccessOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::StructAccessOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto typeConverter = LangToLLVMTypeConverter(getContext());
    mlir::Type convertedStructType =
        typeConverter.convertType(op.getInput().getType());
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

    mlir::Value extractedValue = rewriter.create<mlir::LLVM::ExtractValueOp>(
        op.getLoc(), llvmStructType.getBody()[fieldIndex], adaptor.getInput(),
        fieldIndex);

    rewriter.replaceOp(op, extractedValue);

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
    const LangToLLVMTypeConverter *converter =
        static_cast<const LangToLLVMTypeConverter *>(this->getTypeConverter());
    mlir::Type convertedType = converter->convertType(op.getType());
    if (!convertedType) {
      llvm::errs() << "Failed to convert type: " << op.getType() << "\n"
                   << this->getTypeConverter();
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");
    }

    // Create an 'undef' value of the target struct type
    mlir::Value llvmUndef =
        rewriter.create<mlir::LLVM::UndefOp>(op.getLoc(), convertedType);

    // Insert each field into the struct using LLVM::InsertValueOp
    mlir::Value structValue = llvmUndef;
    for (auto it : llvm::enumerate(adaptor.getFields())) {
      unsigned index = it.index();
      mlir::Value field = it.value();

      // Optionally, convert the field type if necessary
      mlir::Type fieldType = field.getType();
      mlir::Type convertedFieldType = typeConverter->convertType(fieldType);
      if (!convertedFieldType)
        return rewriter.notifyMatchFailure(op, "Failed to convert field type");

      if (convertedFieldType != fieldType) {
        // Insert a cast if types differ
        field = rewriter
                    .create<mlir::UnrealizedConversionCastOp>(
                        op.getLoc(), convertedFieldType, field)
                    .getResult(0);
      }

      // Insert the field into the struct
      structValue = rewriter.create<mlir::LLVM::InsertValueOp>(
          op.getLoc(), convertedType, structValue, field, index);
    }

    // Replace the original operation with the constructed struct
    rewriter.replaceOp(op, structValue);
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
                  mlir::TensorType>(varType)) {
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
                  mlir::ConversionPatternRewriter &rewriter) const override {
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

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    // Get the printf function
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto printfSymbol = getOrInsertPrintf(rewriter, module);

    // Get the format string
    auto format_str = op.getFormatAttr().getValue();

    auto operands = adaptor.getOperands();

    // Create a global string in the LLVM dialect
    static int formatStringCounter = 0;
    auto format_str_val = getOrCreateGlobalString(
        op.getLoc(), rewriter, "_fmt" + std::to_string(formatStringCounter++),
        format_str, module);

    mlir::SmallVector<mlir::Value, 4> args;
    args.push_back(format_str_val);

    mlir::SmallVector<mlir::Type, 4> argTypes;
    argTypes.push_back(format_str_val.getType());

    LangToLLVMTypeConverter typeConverter(rewriter.getContext());
    for (mlir::Value operand : adaptor.getOperands()) {
      mlir::Type llvmType = typeConverter.convertType(operand.getType());
      if (!llvmType)
        return op.emitError("failed to convert operand type");

      if (auto castOp =
              operand.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        operand = castOp.getResult(0);
      }

      args.push_back(operand);
      argTypes.push_back(llvmType);
    }

    args.append(operands.begin(), operands.end());
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

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
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

void LangToAffineLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());
  LangToLLVMTypeConverter typeConverter(&getContext());

  target
      .addLegalDialect<mlir::affine::AffineDialect, mlir::BuiltinDialect,
                       mlir::arith::ArithDialect, mlir::func::FuncDialect,
                       mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect>();

  // Mark all operations illegal.
  target.addIllegalDialect<mlir::lang::LangDialect>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CreateStructOpLowering>(typeConverter, &getContext());
  patterns.add<FuncOpLowering, ReturnOpLowering, VarDeclOpLowering,
               TypeConstOpLowering, StructAccessOpLowering,
               StringConstantOpLowering, PrintOpLowering>(&getContext());

  // Apply partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
std::unique_ptr<mlir::Pass> mlir::lang::createLowerToAffinePass() {
  return std::make_unique<LangToAffineLoweringPass>();
}
