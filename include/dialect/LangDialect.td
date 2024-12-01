#ifndef LANG_DIALECT
#define LANG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

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

def TypeValue: DialectType<Lang_Dialect, CPred<"isa<TypeValueType>($_self)">,
                                    "Lang typevalue type">;

def Lang_StructType :
    DialectType<Lang_Dialect, CPred<"isa<StructType>($_self)">,
                "Lang struct type">;

//===----------------------------------------------------------------------===//
// StringType
//===----------------------------------------------------------------------===//

def Lang_StringType : DialectType<Lang_Dialect, CPred<"isa<StringType>($_self)">,
                                  "Lang string type">;

class Lang_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<Lang_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}
def Lang_UndefAttr : Lang_Attr<"Undef", "undef">;

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
