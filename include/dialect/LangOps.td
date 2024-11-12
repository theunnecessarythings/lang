#ifndef LANG_OPS
#define LANG_OPS

include "LangDialect.td"
include "mlir/IR/Traits.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def FuncOp : Lang_Op<"func", [
  AffineScope, AutomaticAllocationScope,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface, SymbolTable
]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    A `func` operation is a function that contains a single `SSACFG` region.
    The function may have arguments and results, and may be called from other
    functions.
   }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);
  
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }

    static StringRef getDefaultDialect() { return "lang"; }

    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;

}

def ReturnOp : Lang_Op<"return", [Pure, HasParent<"FuncOp">,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `return` operation is used to return control flow from a function.
    The operation may optionally contain a list of return values.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def VarDeclOp : Lang_Op<"var_decl", [IsolatedFromAbove, Symbol]> {
  let summary = "Variable declaration operation with a symbol name";
  let description = [{
    The `var_decl` operation declares a variable with a specified symbol name and an optional initial value.
    The initial value can be a constant or a reference to a previously defined variable.
  }];

  let arguments = (ins OptionalAttr<TypeAttr>:$var_type,
      SymbolNameAttr:$sym_name, 
      Optional<AnyType>:$init_value);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $sym_name attr-dict `:` $var_type `=` $init_value `:` type($init_value) `->` type($result)
  }];

   let builders = [OpBuilder<(ins "StringRef", "Type", "Value")>];


  let hasVerifier = 1;
}

def TypeConstOp : Lang_Op<"type_const", []> {
  let summary = "Operation representing a type as a value";
  let description = [{
    The `type_const` operation allows a type to be represented as a value,
    enabling it to be assigned to variables or passed as arguments.
  }];

  let arguments = (ins TypeAttr:$type);
  let results = (outs TypeValue:$result);

  let hasFolder = 1;

  let assemblyFormat = "$type attr-dict `:` type($result)";
  
  let builders = [OpBuilder<(ins "Type":$type)>];
}


//===----------------------------------------------------------------------===//
// StructAccessOp
//===----------------------------------------------------------------------===//

def StructAccessOp : Lang_Op<"struct_access", [Pure]> {
  let summary = "struct access";
  let description = [{
    Access the Nth element of a value returning a struct type.
  }];

  let arguments = (ins Lang_StructType:$input, I64Attr:$index);
  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
  }];

  // Allow building a StructAccessOp with just a struct value and an index.
  let builders = [
    OpBuilder<(ins "Value":$input, "size_t":$index)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  // Set the folder bit so that we can fold constant accesses.
  // let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructConstantOp
//===----------------------------------------------------------------------===//

def StructConstantOp : Lang_Op<"struct_constant", [ConstantLike, Pure]> {
  let summary = "struct constant";
  let description = [{
    Constant operation turns a literal struct value into an SSA value. The data
    is attached to the operation as an attribute. The struct constant is encoded
    as an array of other constant values. For example:

  }];

  let arguments = (ins ArrayAttr:$value);
  let results = (outs Lang_StructType:$output);

  let assemblyFormat = "$value attr-dict `:` type($output)";

  // Indicate that additional verification for this operation is necessary.
  // let hasVerifier = 1;
  // let hasFolder = 1;
}

def CreateStructOp : Lang_Op<"struct_create"> {
  let summary = "Constructs a struct with dynamic fields";
  let description = [{
    This operation constructs a struct of type `StructType` from a variadic list of operand values.
  }];

  // Define the variadic operands (fields of the struct)
  let arguments = (ins Variadic<AnyType>:$fields);

  // Define the result type
  let results = (outs Lang_StructType:$result);

  // Assembly format for the operation
  let assemblyFormat = "$fields `:` type($fields) attr-dict `->` type($result)";
}


def PrintOp: Lang_Op<"print", [IsolatedFromAbove]> {
  let summary = "Print function -> basicall libc's printf function";
  let description = [{
    The `print` operation is used to print an arbitrary number of values to the console.
    The operation may contain a format string and a list of values to print.
  }];

  let arguments = (ins StrAttr:$format, Variadic<AnyType>:$operands);

  let assemblyFormat = "$format attr-dict `:` $operands type($operands)";
}

//===----------------------------------------------------------------------===//
// string_const Operation
//===----------------------------------------------------------------------===//

def StringConstOp : Lang_Op<"string_const", [Pure]> {
  let summary = "constant string operation";
  let description = [{
    The `string_const` operation produces a constant string value.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs Lang_StringType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];
}
#endif // LANG_OPS
