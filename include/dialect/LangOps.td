#ifndef LANG_OPS
#define LANG_OPS

include "LangDialect.td"
include "mlir/IR/Traits.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
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
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
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


//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : Lang_Op<"call",
    [CallOpInterface, MemRefsNormalizable]> {
  let summary = "call operation";
  let description = [{
    The `lang.call` operation represents a direct call to a function that is
    within the same symbol scope as the call. The operands and result types of
    the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}


def ReturnOp : Lang_Op<"return", [Pure, 
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
  // let hasVerifier = 1;
}

def VarDeclOp : Lang_Op<"var_decl", [IsolatedFromAbove, InferTypeOpAdaptor, InferTypeOpInterface]> {
  let summary = "Variable declaration operation with a symbol name";
  let description = [{
    The `var_decl` operation declares a variable with a specified symbol name and an optional initial value.
    The initial value can be a constant or a reference to a previously defined variable.
  }];

  let arguments = (ins OptionalAttr<TypeAttr>:$var_type,
      SymbolNameAttr:$sym_name, 
      Optional<AnyType>:$init_value,
      BoolAttr:$is_mutable,
      BoolAttr:$is_public);

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

  let arguments = (ins StructType:$input, I64Attr:$index);
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
  let hasFolder = 1;
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
  let results = (outs StructType:$output);

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
  let results = (outs StructType:$result);

  // Assembly format for the operation
  let assemblyFormat = "$fields `:` type($fields) attr-dict `->` type($result)";

}

def ConstantOp : Lang_Op<"constant"> {
  let summary = "constant operation, any constant value";
  let description = [{
    The `constant` operation produces a constant value.
    Wraps a TypedAttr value.
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasFolder = 1;
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


def ImplDeclOp : Lang_Op<"impl_decl", [SymbolTable]> {
  let summary = "Represents an 'impl' block";
  let description = [{
    This operation represents a 'impl' block, which can be used to define
    methods for a type or to implement a trait for a type.
  }];

  let arguments = (ins
    // OptionalAttr<SymbolRefAttr>:$trait, 
    TypeAttr:$selfType                 
  );

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{
    `impl` attr-dict `:`  $region`}`
  }];

  // let hasVerifier = 1;
}

def AssignOp : Lang_Op<"assign", []> {
  let summary = "Assigns a value to a memory location";

  let description = [{
    The `assign` operation assigns a value to a specified memory location (`memref`).
    
    Operands:
      1. `target` (`memref<...>`): The memory location to assign to.
      2. `value` (`!type`): The value to assign.

    Semantics:
      This operation performs a side-effect by modifying the contents of the `memref`.
  }];

  let arguments = (ins
    AnyType:$target,    
    AnyType:$value       
  );

  let results = (outs AnyType:$result);

}


def UndefOp : Lang_Op<"undef", [Pure, ConstantLike]> {
  let summary = "Creates an undefined value of LLVM dialect type.";
  let description = [{
    The `lang.undef` operation creates an undefined value of the specified type.
  }];
  let results = (outs AnyType:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
  let hasFolder = 1;
}

def IfOp : Lang_Op<"if", [InferTypeOpAdaptor, InferTypeOpInterface]> {
  let summary = "Conditional branch operation";
  let description = [{
    The `lang.if` operation is used to conditionally branch based on a
    predicate. The operation contains a region that is executed if the
    predicate is true, and an optional else region that is executed if the
    predicate is false.
  }];

  
  let arguments = (ins I1:$condition);
  let results = (outs Optional<AnyType>:$result);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);

  let builders = [
    OpBuilder<(ins "Value":$cond,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>,
  ];

}


def YieldOp : Lang_Op<"yield", [Pure, Terminator]> {
  let summary = "loop yield and termination operation";
  let description = [{
    The `lang.yield` operation is used to yield control flow from a loop.
    The operation may optionally contain a list of values to yield.
  }];

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let arguments = (ins Optional<AnyType>:$result);
}

def TupleOp : Lang_Op<"tuple", []> {
  let summary = "Tuple operation";
  let description = [{
    The `lang.tuple` operation creates a tuple value from a list of values.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operands `:` type($operands) attr-dict `->` type($result)";
}

def ArrayOp : Lang_Op<"array", []> {
  let summary = "Array operation";
  let description = [{
    The `lang.array` operation creates an array value from a list of values.
  }];

  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$values `:` type($values) attr-dict `->` type($result)";

  let hasVerifier = 1;
}

def LoadOp : Lang_Op<"load", []> {
  let summary = "Load operation";
  let description = [{
    The `lang.load` operation is used to load a value from a memory location.
  }];

  let arguments = (ins AnyType:$memref);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$memref `:` type($memref) attr-dict `->` type($result)";
}

def StoreOp : Lang_Op<"store", []> {
  let summary = "Store operation";
  let description = [{
    The `lang.store` operation is used to store a value to a memory location.
  }];

  let arguments = (ins AnyType:$value, AnyType:$memref);
  let assemblyFormat = "$value `to` $memref `:` type($value) attr-dict `->` type($memref)";
}

def DerefOp: Lang_Op<"deref", [InferTypeOpAdaptor, InferTypeOpInterface]> {
  let summary = "Dereference operation";
  let description = [{
    The `lang.deref` operation is used to dereference a pointer value.
  }];

  let arguments = (ins AnyType:$addr);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$addr `:` type($addr) attr-dict `->` type($result)";
}

def IndexAccessOp: Lang_Op<"index_access", []> {
  let summary = "Index access operation";
  let description = [{
    The `lang.index_access` operation is used to access an element of a container.
  }];

  let arguments = (ins AnyType:$container, AnyType:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$container `[` $index `:` type($index) `]` attr-dict `:` type($container) `->` type($result)";
}


def CreateTypeOp: Lang_Op<"create_type", []> {
  let summary = "Create type operation";
  let description = [{
    The `lang.create_type` operation is used to create a new type from attributes.
  }];

  let arguments = (ins AnyAttr:$attrs);
  let results = (outs TypeValue:$result);

  let assemblyFormat = "attr-dict `->` type($result)";
}

#endif // LANG_OPS
