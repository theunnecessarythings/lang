//===- MyOps.td - MyDialect Operations -------------------------*- tablegen -*-===//
//
// This file defines operations for the 'mydialect' dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "src/tablegen/dialect/MyDialect.td"

class MyDialect_Op<
    string mnemonic,
    list<Trait> traits = []
> : Op<MyDialect, mnemonic, traits>;

// Define a simple operation called 'add' that adds two integers.
def MyDialect_AddOp : MyDialect_Op<"add", [Commutative, Pure]> {
  let summary = "Addition operation";
  let description = [{
    The `mydialect.add` operation adds two integer values.

    Example:

    ```mlir
    %result = mydialect.add %lhs, %rhs : i32
    ```
  }];

  // The operation has two operands of type `AnyInteger` and one result of the same type.
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);

  // Provide a custom builder to simplify operation creation.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      return builder.create<mydialect::AddOp>(loc, lhs.getType(), lhs, rhs);
    }]>
  ];
}
