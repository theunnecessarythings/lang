
//===----------------------------------------------------------------------===//
// Expr Dialect Definition
//===----------------------------------------------------------------------===//

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def ExprDialect : Dialect {
  let name = "expr";
  let cppNamespace = "::mlir::expr";
}

//===----------------------------------------------------------------------===//
// Expr Types
//===----------------------------------------------------------------------===//

class ExprDialectType<string name, string type_mnemonic, list<Trait> traits = []>
    : TypeDef<ExprDialect, name, traits> {
  let mnemonic = type_mnemonic;
}

def Expr_IntType : ExprDialectType<"Integer", "int"> {
}

def Expr_FloatType : ExprDialectType<"Float", "float"> {
}

//===----------------------------------------------------------------------===//
// Expr Operations
//===----------------------------------------------------------------------===//

class ExprOp<string mnemonic, list<Trait> traits = []> :
    Op<ExprDialect, mnemonic, traits>;

// Variable Declaration Operation
def Lang_VarDeclOp : ExprOp<"var_decl"> {
  let summary = "Variable declaration operation";
  let description = [{
    Declares a variable with an initializer.
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType);
}

// Constant Operation
def Lang_ConstantOp : ExprOp<"constant"> {
  let summary = "Constant value operation";
  let description = [{
    Represents a constant integer or float value.
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs AnyType);
}

// Binary Operation
def Lang_BinaryOp : ExprOp<"binary"> {
  let summary = "Binary operation";
  let description = [{
    Represents a binary arithmetic or logical operation, such as addition or bitwise OR.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$op);
  let results = (outs AnyType);
}

// Unary Operation
def Lang_UnaryOp : ExprOp<"unary"> {
  let summary = "Unary operation";
  let description = [{
    Represents a unary operation like negation.
  }];

  let arguments = (ins AnyType:$operand, StrAttr:$op);
  let results = (outs AnyType);
}
