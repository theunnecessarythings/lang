//===- MyDialect.td - MyDialect Definition ---------------------*- tablegen -*-===//
//
// This file defines the 'mydialect' dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"

// Define the dialect class.
def MyDialect : Dialect {
  let name = "mydialect";
  let cppNamespace = "mydialect";
}
