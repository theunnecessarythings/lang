#include "dialect/LangOps.h"
#include "dialect/LangDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "dialect/LangOps.cpp.inc"

using namespace mlir::lang;

FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  FuncOp::build(builder, state, name, type, attrs);
  return cast<FuncOp>(Operation::create(state));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, llvm::ArrayRef(attrRef));
}
FuncOp FuncOp::create(Location location, StringRef name, FunctionType type,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  FuncOp func = create(location, name, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute("sym_name", builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

mlir::ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

mlir::LogicalResult ArrayOp::verify() {
  // Check all operands are of the same type.
  llvm::SmallVector<Type, 4> operandTypes;
  for (auto operand : getOperands())
    operandTypes.push_back(operand.getType());
  if (!llvm::all_of(operandTypes,
                    [&](Type type) { return type == operandTypes[0]; })) {
    return emitOpError("requires all operands to be of the same type");
  }
  return mlir::success();
}

// mlir::LogicalResult ReturnOp::verify() {
//   auto function = cast<FuncOp>((*this)->getParentOp());
//
//   // The operand number and types must match the function signature.
//   const auto results = function.getFunctionType().getResults();
//   if (getNumOperands() != results.size())
//     return emitOpError("has ")
//            << getNumOperands() << " operands, but enclosing function (@"
//            << function.getName() << ") returns " << results.size();
//
//   for (unsigned i = 0, e = results.size(); i != e; ++i) {
//     // if its a literal then skip -> let the pass handle it
//     if (mlir::isa<mlir::lang::ConstantOp>(getOperand(i).getDefiningOp()))
//       continue;
//     if (getOperand(i).getType() != results[i])
//       return emitError() << "type of return operand " << i + 1 << " "
//                          << getOperand(i).getType()
//                          << " doesn't match function result type " <<
//                          results[i]
//                          << " in function " << function.getName();
//   }
//
//   return success();
// }

void VarDeclOp::build(OpBuilder &builder, OperationState &state,
                      StringRef symName, Type varType = nullptr,
                      Value initValue = nullptr) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symName));
  if (varType) {
    state.addAttribute("var_type", TypeAttr::get(varType));
  } else {
    state.addAttribute("var_type", TypeAttr::get(initValue.getType()));
  }

  if (initValue)
    state.addOperands(initValue);

  Type resultType =
      varType ? varType
              : (initValue ? initValue.getType() : builder.getNoneType());
  state.addTypes(resultType);
}

mlir::LogicalResult VarDeclOp::verify() {
  if (auto varTypeAttr = getVarType()) {
    Type varType = varTypeAttr.value();
    Type initType = getInitValue().getType();
    if (mlir::isa<TypeValueType>(varType) &&
        !mlir::isa<TypeValueType>(initType)) {
      auto type_value = mlir::cast<TypeValueType>(varType);
      varType = type_value.getType();
    }

    // if literals then
    if (mlir::isa<mlir::IntegerType>(varType) &&
        mlir::isa<mlir::IntegerType>(initType)) {
      return success();
    }

    if (mlir::isa<mlir::IntegerType>(varType) &&
        mlir::isa<mlir::lang::IntLiteralType>(initType)) {
      return success();
    }

    if (mlir::isa<mlir::FloatType>(varType) &&
        mlir::isa<mlir::FloatType>(initType)) {
      return success();
    }

    if (mlir::isa<mlir::lang::ArrayType>(varType) &&
        mlir::isa<mlir::lang::ArrayType>(initType)) {
      auto arrayType = mlir::cast<mlir::lang::ArrayType>(varType);
      // if array size is comptime then skip for now
      if (arrayType.getIsComptimeExpr())
        return success();
    }
    // TODO: pass through, for now. Add proper checks later
    // if (varType != initType) {
    //   return emitOpError() << "type of 'init_value' (" << initType
    //                        << ") does not match 'var_type' (" << varType <<
    //                        ")";
    // }
  }
  return success();
}

// void TypeConstOp::build(OpBuilder &builder, OperationState &state, Type type)
// {
//   build(builder, state, TypeValueType::get(builder.getContext(), type),
//         TypeAttr::get(type));
// }

mlir::OpFoldResult mlir::lang::StructAccessOp::fold(FoldAdaptor adaptor) {
  auto structAttr = mlir::dyn_cast_or_null<mlir::ArrayAttr>(adaptor.getInput());
  if (!structAttr)
    return nullptr;

  size_t elementIndex = getIndex();
  return structAttr[elementIndex];
}

mlir::OpFoldResult mlir::lang::ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}

mlir::FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

mlir::OpFoldResult mlir::lang::UndefOp::fold(FoldAdaptor) {
  return mlir::lang::UndefAttr::get(getContext());
}

mlir::LogicalResult
VarDeclOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location> loc,
                            VarDeclOp::Adaptor adaptor,
                            SmallVectorImpl<Type> &results) {
  if (adaptor.getIsMutable()) {
    results.push_back(
        mlir::MemRefType::get({}, adaptor.getInitValue().getType()));
    return success();
  }
  results.push_back(adaptor.getInitValue().getType());
  return success();
}

mlir::LogicalResult DerefOp::inferReturnTypes(MLIRContext *ctx,
                                              std::optional<Location> loc,
                                              DerefOp::Adaptor adaptor,
                                              SmallVectorImpl<Type> &results) {
  auto memref_type = mlir::cast<mlir::MemRefType>(adaptor.getAddr().getType());
  results.push_back(memref_type.getElementType());
  return success();
}

mlir::LogicalResult
IfOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location> loc,
                       IfOp::Adaptor adaptor,
                       SmallVectorImpl<Type> &inferredReturnTypes) {
  if (adaptor.getRegions().empty())
    return failure();
  Region *r = &adaptor.getThenRegion();
  if (r->empty())
    return failure();
  Block &b = r->front();
  if (b.empty())
    return failure();
  if (mlir::isa<mlir::lang::YieldOp>(b.back())) {
    if (b.back().getNumOperands() == 0)
      return success();
    inferredReturnTypes.push_back(b.back().getOperand(0).getType());
    return success();
  }
  if (mlir::isa<mlir::lang::ReturnOp>(b.back())) {
    if (b.back().getNumOperands() == 0)
      return success();
    inferredReturnTypes.push_back(b.back().getOperand(0).getType());
    return success();
  }
  return failure();
}

void IfOp::build(OpBuilder &builder, OperationState &result, Value cond,
                 function_ref<void(OpBuilder &, Location)> thenBuilder,
                 function_ref<void(OpBuilder &, Location)> elseBuilder) {
  assert(thenBuilder && "the builder callback for 'then' must be present");
  result.addOperands(cond);

  // Build then region.
  OpBuilder::InsertionGuard guard(builder);
  Region *thenRegion = result.addRegion();
  builder.createBlock(thenRegion);
  thenBuilder(builder, result.location);

  // Build else region.
  Region *elseRegion = result.addRegion();
  if (elseBuilder) {
    builder.createBlock(elseRegion);
    elseBuilder(builder, result.location);
  }

  // Infer result types.
  SmallVector<Type> inferredReturnTypes;
  MLIRContext *ctx = builder.getContext();
  auto attrDict = DictionaryAttr::get(ctx, result.attributes);
  if (succeeded(inferReturnTypes(ctx, std::nullopt, result.operands, attrDict,
                                 /*properties=*/nullptr, result.regions,
                                 inferredReturnTypes))) {
    result.addTypes(inferredReturnTypes);
  }
}
