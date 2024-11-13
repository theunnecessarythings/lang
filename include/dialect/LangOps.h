#ifndef LANG_LANGDIALECTOPS_H
#define LANG_LANGDIALECTOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "dialect/LangOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "dialect/LangOps.h.inc"

void castIfRequired(mlir::Type literalType, mlir::Type targetType,
                    mlir::arith::ConstantOp constant_op, mlir::Value operand,
                    mlir::OpBuilder &builder, mlir::Operation *op);
#endif // LANG_LANGDIALECTOPS_H
