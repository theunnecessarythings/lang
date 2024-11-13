add_mlir_dialect_library(Lang
        LangDialect.cpp
        LangOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/dialect

        DEPENDS
        MLIRLangOpsIncGen
        LangOpsAttrDefsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
