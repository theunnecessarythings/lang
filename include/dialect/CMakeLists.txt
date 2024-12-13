add_mlir_dialect(LangOps lang)
add_mlir_doc(LangDialect LangDialect dialect/ -gen-dialect-doc)
add_mlir_doc(LangOps LangOps dialect/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS LangDialect.td)
mlir_tablegen(LangOpsAttrDefs.h.inc -gen-attrdef-decls 
  -attrdefs-dialect=lang)
mlir_tablegen(LangOpsAttrDefs.cpp.inc -gen-attrdef-defs 
  -attrdefs-dialect=lang)
mlir_tablegen(LangPasses.h.inc -gen-pass-decls)
mlir_tablegen(LangEnumAttrDefs.h.inc -gen-enum-decls)
mlir_tablegen(LangEnumAttrDefs.cpp.inc -gen-enum-defs)
add_public_tablegen_target(LangOpsAttrDefsIncGen)
