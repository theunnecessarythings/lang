#ifndef LANG_LANGDIALECTOPS_H
#define LANG_LANGDIALECTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "dialect/LangOps.h.inc"

#endif // LANG_LANGDIALECTOPS_H
