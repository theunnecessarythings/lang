#ifndef LANG_DIALECT
#define LANG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// lang dialect definition.
//===----------------------------------------------------------------------===//

def Lang_Dialect : Dialect {
    let name = "lang";
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::lang";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// lang dialect types.
//===----------------------------------------------------------------------===//

class Lang_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Lang_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def LangType : Lang_Type<"Lang", "type"> {
    let summary = "Lang type";
    let description = [{
        This type represents a lang type.
    }];
}

def TypeValue: Lang_Type<"TypeValue", "typevalue"> {
    let summary = "Type value type";
    let description = [{
        This type represents a type value.
    }];

    let parameters = (ins "Type":$type);
    let assemblyFormat = "`<` $type `>`";

}

def StructType : Lang_Type<"Struct", "struct"> {
    let summary = "Struct type";
    let description = [{
        This type represents a struct type.
    }];

    let parameters = (ins StringRefParameter<"struct name">:$name, ArrayRefParameter<"Type", "field types">:$element_types);
    let assemblyFormat = "`<` $name `,` $element_types `>`";

    let extraClassDeclaration = [{
       size_t getNumElementTypes() {
           return getElementTypes().size(); 
       }
    }];
}

def StringType : Lang_Type<"String", "string"> {
    let summary = "String type";
    let description = [{
        This type represents a string type.
    }];
}

def IntLiteralType : Lang_Type<"IntLiteral", "int_literal"> {
    let summary = "Int literal type";
    let description = [{
        This type represents an int literal type.
    }];
}


def PointerType : Lang_Type<"Pointer", "ptr"> {
    let summary = "Pointer type";
    let description = [{
        This type represents a pointer type.
    }];
}

def SliceType : Lang_Type<"Slice", "slice"> {
    let summary = "Slice type";
    let description = [{
        This type represents a slice type.
    }];

    let parameters = (ins "Type":$element_type);
    let assemblyFormat = "`<` $element_type `>`";
}

def ArrayType : Lang_Type<"Array", "array"> {
    let summary = "Array type";
    let description = [{
        This type represents an array type.
    }];

    let parameters = (ins "Type":$element_type, "IntegerAttr":$size, "BoolAttr":$is_comptime_expr);
    let assemblyFormat = "`<` $element_type `,` $size `,` $is_comptime_expr `>`";
}

def GenericType : Lang_Type<"Generic", "generic"> {
    let summary = "Generic type";
    let description = [{
        This type represents a generic type.
    }];
    let parameters = (ins "StringAttr":$type_param);
    let assemblyFormat = "`<` $type_param `>`";
}

def ComptimeType: Lang_Type<"Comptime", "comptime"> {
    let summary = "Comptime type";
    let description = [{
        This type represents a comptime type.
    }];
    let parameters = (ins "TypedAttr":$value);
    let assemblyFormat = "`<` $value `>`";
}

// def TypeValue: DialectType<Lang_Dialect, CPred<"isa<TypeValueType>($_self)">,
//                                     "Lang typevalue type">;

// def Lang_StructType :
//     DialectType<Lang_Dialect, CPred<"isa<StructType>($_self)">,
//                 "Lang struct type">;

def Lang_StringType : DialectType<Lang_Dialect, CPred<"isa<StringType>($_self)">,
                                  "Lang string type">;

class Lang_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<Lang_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

def Lang_UndefAttr : Lang_Attr<"Undef", "undef">;

def type: I32EnumAttrCase<"LangType", 0>;
def Void: I32EnumAttrCase<"Void", 1>;
def Bool: I32EnumAttrCase<"Bool", 2>;
def NoReturn: I32EnumAttrCase<"NoReturn", 3>;
def Int: I32EnumAttrCase<"Int", 4>;
def Float: I32EnumAttrCase<"Float", 5>;
def Pointer: I32EnumAttrCase<"Pointer", 6>;
def Array: I32EnumAttrCase<"Array", 7>;
def Struct: I32EnumAttrCase<"Struct", 8>;
def ComptimeFloat: I32EnumAttrCase<"ComptimeFloat", 9>;
def ComptimeInt: I32EnumAttrCase<"ComptimeInt", 10>;
def Undefined: I32EnumAttrCase<"Undefined", 11>;
def Null: I32EnumAttrCase<"Null", 12>;
def Optional: I32EnumAttrCase<"Optional", 13>;
def ErrorUnion: I32EnumAttrCase<"ErrorUnion", 14>;
def ErrorSet: I32EnumAttrCase<"ErrorSet", 15>;
def Enum: I32EnumAttrCase<"Enum", 16>;
def Union: I32EnumAttrCase<"Union", 17>;
def Fn: I32EnumAttrCase<"Fn", 18>;
def Opaque: I32EnumAttrCase<"Opaque", 19>;
def Frame: I32EnumAttrCase<"Frame", 20>;
def AnyFrame: I32EnumAttrCase<"AnyFrame", 21>;
def Vector: I32EnumAttrCase<"Vector", 22>;
def EnumLiteral: I32EnumAttrCase<"EnumLiteral", 23>;

def Lang_TypeKind : I32EnumAttr<"TypeKind", "Type of the value",
                           [type, Void, Bool, NoReturn, Int, Float, Pointer,
                            Array, Struct, ComptimeFloat, ComptimeInt, Undefined,
                            Null, Optional, ErrorUnion, ErrorSet, Enum, Union,
                            Fn, Opaque, Frame, AnyFrame, Vector, EnumLiteral]> {
  let cppNamespace = "::mlir::lang";
}


def Lang_TypeAttr : Lang_Attr<"TypeAttr", "type"> {
    let summary = "Type attribute";
    let description = [{
        This attribute represents a type.
    }];
    let parameters = (ins "TypeKind":$typeKind, "Attribute":$typeValue);
    let assemblyFormat = "`<` $typeKind `,` $typeValue `>`";
}


def Signed: I32EnumAttrCase<"Signed", 1>;
def Unsigned: I32EnumAttrCase<"Unsigned", 0>;

def Lang_IntegerSignedness: I32EnumAttr<"IntegerSignedness", "Signedness of the integer",
                           [Signed, Unsigned]> {
  let cppNamespace = "::mlir::lang";
  // let stringToSymbolFnName = "convertToEnum";
  // let symbolToStringFnName = "convertToString";
}

def Lang_IntegerTypeAttr: Lang_Attr<"IntegerType", "integer_type"> {
    let summary = "Integer type attribute";
    let description = [{
        This attribute represents an integer type.
    }];
    let parameters = (ins "IntegerSignedness":$signedness, "IntegerAttr":$width);
    let assemblyFormat = "`<` $signedness `,` $width `>`";
}

def Lang_FloatTypeAttr: Lang_Attr<"FloatType", "float_type"> {
    let summary = "Float type attribute";
    let description = [{
        This attribute represents a float type.
    }];
    let parameters = (ins "IntegerAttr":$width);
    let assemblyFormat = "`<` $width `>`";
}

def Lang_PointerTypeAttr: Lang_Attr<"PointerType", "pointer_type"> {
    let summary = "Pointer type attribute";
    let description = [{
        This attribute represents a pointer type.
    }];
    // TODO: AddressSpace needs to be an enumattr, fix 'child' as well
    let parameters = (ins "IntegerAttr":$size, "BoolAttr":$isConst, "BoolAttr":$isVolatile,
                       "IntegerAttr":$alignment, "IntegerAttr":$addressSpace,
                       "TypeAttr":$child, "BoolAttr":$isAllowZero);
    let assemblyFormat = "`<` $size `,` $isConst `,` $isVolatile `,` $alignment `,` $addressSpace `,` $child `,` $isAllowZero `>`";
}

def Lang_ArrayTypeAttr: Lang_Attr<"ArrayType", "array_type"> {
    let summary = "Array type attribute";
    let description = [{
        This attribute represents an array type.
    }];
    let parameters = (ins "IntegerAttr":$len, "TypeAttr":$child);
    let assemblyFormat = "`<` $len `,` $child`>`";
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Lang_Op<string mnemonic, list<Trait> traits = []> :
  Op<Lang_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"
def ComptimeAnalysis : Pass<"comptime_analysis"> {
    let summary = "Comptime Analysis Pass";
    let description = ""; 
    let constructor = "mlir::lang::createComptimeAnalysisPass()";
}

#endif // LANG_DIALECT
