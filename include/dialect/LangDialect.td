#ifndef LANG_DIALECT
#define LANG_DIALECT

include "mlir/IR/OpBase.td"

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
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Lang_Op<string mnemonic, list<Trait> traits = []> :
  Op<Lang_Dialect, mnemonic, traits>;

#endif // LANG_DIALECT
