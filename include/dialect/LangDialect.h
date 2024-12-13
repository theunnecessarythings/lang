#ifndef LANG_LANGDIALECT_H
#define LANG_LANGDIALECT_H

#include "mlir/AsmParser/CodeComplete.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"

#include "dialect/LangEnumAttrDefs.h.inc"
#include "dialect/LangOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "dialect/LangOpsAttrDefs.h.inc"

#endif // LANG_LANGDIALECT_H
