#include "dialect/LangOps.h"
#include "dialect/LangDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"

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

mlir::LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}

void VarDeclOp::build(OpBuilder &builder, OperationState &state,
                      StringRef symName, Type varType = nullptr,
                      Value initValue = nullptr) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symName));
  if (varType) {
    state.addAttribute("var_type", TypeAttr::get(varType));
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
      varType = type_value.getAliasedType();
    }
    if (varType != initType) {
      return emitOpError() << "type of 'init_value' (" << initType
                           << ") does not match 'var_type' (" << varType << ")";
    }
  }
  return success();
}

void TypeConstOp::build(OpBuilder &builder, OperationState &state, Type type) {
  build(builder, state, TypeValueType::get(type), TypeAttr::get(type));
}
