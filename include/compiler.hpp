#pragma once

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

#include "ast.hpp"
#include "lexer.hpp"
#include <memory>
#include <string>

struct CompilerOptions {
  std::string output_file = "output";
  bool report_warnings = false;
  bool convert_warns_to_errors = false;
};

struct Context {
  CompilerOptions options;
  llvm::ScopedHashTable<llvm::StringRef, StructDecl *> struct_table;
  llvm::ScopedHashTable<llvm::StringRef, TupleStructDecl *> tuple_struct_table;
  llvm::ScopedHashTable<llvm::StringRef, EnumDecl *> enum_table;
  llvm::ScopedHashTable<llvm::StringRef, UnionDecl *> union_table;
  llvm::ScopedHashTable<llvm::StringRef, TraitDecl *> trait_table;
  llvm::ScopedHashTable<llvm::StringRef, Type *> var_table;

  llvm::SourceMgr source_mgr;
  mlir::MLIRContext context;
  mlir::DiagnosticEngine &diag_engine = context.getDiagEngine();
  std::unique_ptr<mlir::SourceMgrDiagnosticHandler> source_mgr_handle;

  Context() {
    source_mgr_handle = std::make_unique<mlir::SourceMgrDiagnosticHandler>(
        source_mgr, &context);
  }

  void reportError(llvm::StringRef message, const Token *token = nullptr) {
    if (token && token->kind == TokenKind::Eof) {
      throw std::runtime_error("Unexpected EOF");
    }
    auto loc =
        token ? mlir::FileLineColLoc::get(
                    &context,
                    source_mgr.getBufferInfo(token->span.file_id)
                        .Buffer->getBufferIdentifier(),
                    token->span.line_no, token->span.col_start)
              : mlir::dyn_cast<mlir::Location>(mlir::UnknownLoc::get(&context));
    source_mgr_handle->emitDiagnostic(loc, message,
                                      mlir::DiagnosticSeverity::Error);
  }

  void declareStruct(llvm::StringRef name, StructDecl *decl) {
    if (struct_table.lookup(name)) {
      reportError("Struct " + name.str() + " already declared", &decl->token);
      return;
    }
    struct_table.insert(name, decl);
  }

  void declareTupleStruct(llvm::StringRef name, TupleStructDecl *decl) {
    if (tuple_struct_table.lookup(name)) {
      reportError("Tuple struct " + name.str() + " already declared",
                  &decl->token);
      return;
    }
    tuple_struct_table.insert(name, decl);
  }

  void declareEnum(llvm::StringRef name, EnumDecl *decl) {
    if (enum_table.lookup(name)) {
      reportError("Enum " + name.str() + " already declared", &decl->token);
      return;
    }
    enum_table.insert(name, decl);
  }

  void declareUnion(llvm::StringRef name, UnionDecl *decl) {
    if (union_table.lookup(name)) {
      reportError("Union " + name.str() + " already declared", &decl->token);
      return;
    }
    union_table.insert(name, decl);
  }

  void declareTrait(llvm::StringRef name, TraitDecl *decl) {
    if (trait_table.lookup(name)) {
      reportError("Trait " + name.str() + " already declared", &decl->token);
      return;
    }
    trait_table.insert(name, decl);
  }

  void declareVar(llvm::StringRef name, Type *decl) {
    if (var_table.lookup(name)) {
      reportError("Variable " + name.str() + " already declared", &decl->token);
      return;
    }
    var_table.insert(name, decl);
  }
};
