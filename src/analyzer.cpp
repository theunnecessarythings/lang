#include "analyzer.hpp"
#include "ast.hpp"

#define NEW_SCOPE()                                                            \
  llvm::ScopedHashTableScope<llvm::StringRef, StructDecl *> struct_scope(      \
      context->struct_table);                                                  \
  llvm::ScopedHashTableScope<llvm::StringRef, EnumDecl *> enum_scope(          \
      context->enum_table);                                                    \
  llvm::ScopedHashTableScope<llvm::StringRef, TupleStructDecl *>               \
      tuple_struct_scope(context->tuple_struct_table);                         \
  llvm::ScopedHashTableScope<llvm::StringRef, UnionDecl *> union_scope(        \
      context->union_table);                                                   \
  llvm::ScopedHashTableScope<llvm::StringRef, TraitDecl *> trait_scope(        \
      context->trait_table);                                                   \
  llvm::ScopedHashTableScope<llvm::StringRef, VarDecl *> var_scope(            \
      context->var_table);

void Analyzer::analyze(Program *program) {
  NEW_SCOPE();
  for (auto &item : program->items) {
    analyze(item.get());
  }
}

void Analyzer::analyze(TopLevelDecl *decl) {
  switch (decl->kind()) {
  case AstNodeKind::Function:
    analyze(static_cast<Function *>(decl));
    break;
  case AstNodeKind::StructDecl:
    analyze(static_cast<StructDecl *>(decl));
    break;
  case AstNodeKind::TupleStructDecl:
    analyze(static_cast<TupleStructDecl *>(decl));
    break;
  case AstNodeKind::EnumDecl:
    analyze(static_cast<EnumDecl *>(decl));
    break;
  case AstNodeKind::TraitDecl:
    analyze(static_cast<TraitDecl *>(decl));
    break;
  case AstNodeKind::ImplDecl:
    analyze(static_cast<ImplDecl *>(decl));
    break;
  case AstNodeKind::TopLevelVarDecl:
    analyze(static_cast<TopLevelVarDecl *>(decl));
    break;
  case AstNodeKind::Module:
    analyze(static_cast<Module *>(decl));
    break;
  case AstNodeKind::UnionDecl:
    analyze(static_cast<UnionDecl *>(decl));
    break;
  case AstNodeKind::ImportDecl:
    analyze(static_cast<ImportDecl *>(decl));
    break;
  default:
    assert(false && "not a top level decl or not implemented yet");
  }
}

void Analyzer::analyze(Module *module) {
  for (auto &item : module->items) {
    analyze(item.get());
  }
}

void Analyzer::analyze(ImportDecl *decl) {}

void Analyzer::analyze(Function *func) {
  NEW_SCOPE();
  analyze(func->decl.get());
  analyze(func->body.get());
}

void Analyzer::analyze(FunctionDecl *decl) {
  for (auto &param : decl->parameters) {
    analyze(param.get());
  }
  analyze(decl->return_type.get());
}

void Analyzer::analyze(Parameter *param) {
  analyze(param->pattern.get());
  analyze(param->type.get());

  if (!param->trait_bound.empty()) {
    for (auto &trait : param->trait_bound) {
      analyze(trait.get());
    }
  }
}

void Analyzer::analyze(ImplDecl *impl) {
  NEW_SCOPE();
  for (auto &trait : impl->traits) {
    analyze(trait.get());
  }

  static AstDumper dumper;
  for (auto &func : impl->functions) {
    auto type = dumper.dump<Type>(impl->type.get());
    func->decl->extra.is_method = true;
    func->decl->extra.parent_name = type;
    // check impl->type is a struct, enum or union
    if (context->struct_table.count(type)) {
      func->decl->extra.parent_kind = AstNodeKind::StructDecl;
    } else if (context->enum_table.count(type)) {
      func->decl->extra.parent_kind = AstNodeKind::EnumDecl;
    } else if (context->union_table.count(type)) {
      func->decl->extra.parent_kind = AstNodeKind::UnionDecl;
    }

    if (func->decl->name == "init") {
      // constructor so check the first parameter type is Self or parent_name
      // delete that parameter
      if (func->decl->parameters.size() == 0) {
        context->reportError(
            "constructor must have at least one parameter, self",
            &func->decl->token);
      } else {
        auto &param = func->decl->parameters[0];
        if (param->type->kind() == AstNodeKind::IdentifierType) {
          auto id_type = static_cast<IdentifierType *>(param->type.get());
          if (id_type->name == "Self" || id_type->name == type) {
            func->decl->parameters.erase(func->decl->parameters.begin());
          } else {
            context->reportError(
                "first parameter of constructor must be of type Self",
                &param->token);
          }
        } else {
          context->reportError(
              "first parameter of constructor must be of type Self",
              &param->token);
        }
      }
    }
    analyze(func.get());
  }
}

void Analyzer::analyze(UnionDecl *decl) {}

void Analyzer::analyze(TopLevelVarDecl *decl) {}

void Analyzer::analyze(StructDecl *decl) {}

void Analyzer::analyze(TupleStructDecl *decl) {}

void Analyzer::analyze(EnumDecl *decl) {}

void Analyzer::analyze(TraitDecl *decl) {}

void Analyzer::analyze(Pattern *pattern) {
  switch (pattern->kind()) {
  case AstNodeKind::LiteralPattern:
    analyze(static_cast<LiteralPattern *>(pattern));
    break;
  case AstNodeKind::IdentifierPattern:
    analyze(static_cast<IdentifierPattern *>(pattern));
    break;
  case AstNodeKind::WildcardPattern:
    analyze(static_cast<WildcardPattern *>(pattern));
    break;
  case AstNodeKind::TuplePattern:
    analyze(static_cast<TuplePattern *>(pattern));
    break;
  case AstNodeKind::RestPattern:
    analyze(static_cast<RestPattern *>(pattern));
    break;
  case AstNodeKind::StructPattern:
    analyze(static_cast<StructPattern *>(pattern));
    break;
  case AstNodeKind::SlicePattern:
    analyze(static_cast<SlicePattern *>(pattern));
    break;
  case AstNodeKind::OrPattern:
    analyze(static_cast<OrPattern *>(pattern));
    break;
  case AstNodeKind::ExprPattern:
    analyze(static_cast<ExprPattern *>(pattern));
    break;
  case AstNodeKind::RangePattern:
    analyze(static_cast<RangePattern *>(pattern));
    break;
  case AstNodeKind::VariantPattern:
    analyze(static_cast<VariantPattern *>(pattern));
    break;
  default:
    assert(false && "not a pattern or not implemented yet");
  }
}

void Analyzer::analyze(LiteralPattern *pattern) {}

void Analyzer::analyze(IdentifierPattern *pattern) {}

void Analyzer::analyze(WildcardPattern *pattern) {}

void Analyzer::analyze(TuplePattern *pattern) {}

void Analyzer::analyze(RestPattern *pattern) {}

void Analyzer::analyze(StructPattern *pattern) {}

void Analyzer::analyze(SlicePattern *pattern) {}

void Analyzer::analyze(OrPattern *pattern) {
  for (auto &p : pattern->patterns) {
    analyze(p.get());
  }
}

void Analyzer::analyze(ExprPattern *pattern) { analyze(pattern->expr.get()); }

void Analyzer::analyze(RangePattern *pattern) {
  analyze(pattern->start.get());
  analyze(pattern->end.get());
}

void Analyzer::analyze(VariantPattern *pattern) {}

void Analyzer::analyze(Statement *stmt) {
  switch (stmt->kind()) {
  case AstNodeKind::VarDecl:
    analyze(static_cast<VarDecl *>(stmt));
    break;
  case AstNodeKind::DeferStmt:
    analyze(static_cast<DeferStmt *>(stmt));
    break;
  case AstNodeKind::ExprStmt:
    analyze(static_cast<ExprStmt *>(stmt));
    break;
  case AstNodeKind::TopLevelDeclStmt:
    analyze(static_cast<TopLevelDeclStmt *>(stmt));
    break;
  default:
    assert(false && "not a statement or not implemented yet");
    break;
  }
}

void Analyzer::analyze(VarDecl *decl) {
  if (decl->initializer) {
    analyze(decl->initializer->get());
  }
}

void Analyzer::analyze(DeferStmt *stmt) {}

void Analyzer::analyze(ExprStmt *stmt) {}

void Analyzer::analyze(TopLevelDeclStmt *stmt) {}

void Analyzer::analyze(Expression *expr) {
  switch (expr->kind()) {
  case AstNodeKind::IfExpr:
    analyze(static_cast<IfExpr *>(expr));
    break;
  case AstNodeKind::MatchExpr:
    analyze(static_cast<MatchExpr *>(expr));
    break;
  case AstNodeKind::ForExpr:
    analyze(static_cast<ForExpr *>(expr));
    break;
  case AstNodeKind::WhileExpr:
    analyze(static_cast<WhileExpr *>(expr));
    break;
  case AstNodeKind::IdentifierExpr:
    analyze(static_cast<IdentifierExpr *>(expr));
    break;
  case AstNodeKind::ReturnExpr:
    analyze(static_cast<ReturnExpr *>(expr));
    break;
  case AstNodeKind::BreakExpr:
    analyze(static_cast<BreakExpr *>(expr));
    break;
  case AstNodeKind::ContinueExpr:
    analyze(static_cast<ContinueExpr *>(expr));
    break;
  case AstNodeKind::LiteralExpr:
    analyze(static_cast<LiteralExpr *>(expr));
    break;
  case AstNodeKind::TupleExpr:
    analyze(static_cast<TupleExpr *>(expr));
    break;
  case AstNodeKind::ArrayExpr:
    analyze(static_cast<ArrayExpr *>(expr));
    break;
  case AstNodeKind::BinaryExpr:
    analyze(static_cast<BinaryExpr *>(expr));
    break;
  case AstNodeKind::UnaryExpr:
    analyze(static_cast<UnaryExpr *>(expr));
    break;
  case AstNodeKind::CallExpr:
    analyze(static_cast<CallExpr *>(expr));
    break;
  case AstNodeKind::AssignExpr:
    analyze(static_cast<AssignExpr *>(expr));
    break;
  case AstNodeKind::AssignOpExpr:
    analyze(static_cast<AssignOpExpr *>(expr));
    break;
  case AstNodeKind::FieldAccessExpr:
    analyze(static_cast<FieldAccessExpr *>(expr));
    break;
  case AstNodeKind::IndexExpr:
    analyze(static_cast<IndexExpr *>(expr));
    break;
  case AstNodeKind::RangeExpr:
    analyze(static_cast<RangeExpr *>(expr));
    break;
  case AstNodeKind::ComptimeExpr:
    analyze(static_cast<ComptimeExpr *>(expr));
    break;
  case AstNodeKind::BlockExpression:
    analyze(static_cast<BlockExpression *>(expr));
    break;
  case AstNodeKind::MLIROp:
    analyze(static_cast<MLIROp *>(expr));
    break;
  default:
    assert(false && "not an expression or not implemented yet");
    break;
  }
}

void Analyzer::analyze(IfExpr *expr) {}

void Analyzer::analyze(MatchExpr *expr) {}

void Analyzer::analyze(ForExpr *expr) {}

void Analyzer::analyze(WhileExpr *expr) {}

void Analyzer::analyze(ReturnExpr *expr) {}

void Analyzer::analyze(BreakExpr *expr) {}

void Analyzer::analyze(ContinueExpr *expr) {}

void Analyzer::analyze(LiteralExpr *expr) {}

void Analyzer::analyze(TupleExpr *expr) {}

void Analyzer::analyze(ArrayExpr *expr) {
  // 1. all elements must have the same type (handles in MLIR)
  // 2. if all elements are literals then mark it as a const array

  if (expr->elements.size() == 0) {
    return;
  }
  auto first = expr->elements[0].get();
  if (first->kind() != AstNodeKind::LiteralExpr) {
    return;
  }

  auto literal_kind = static_cast<LiteralExpr *>(first)->type;
  for (auto &elem : expr->elements) {
    if (elem->kind() != AstNodeKind::LiteralExpr) {
      return;
    }
    auto literal = static_cast<LiteralExpr *>(elem.get());
    if (literal->type != literal_kind) {
      return;
    }
  }
  expr->extra.is_const = true;
}

void Analyzer::analyze(BinaryExpr *expr) {}

void Analyzer::analyze(UnaryExpr *expr) {}

void Analyzer::analyze(CallExpr *expr) {}

void Analyzer::analyze(AssignExpr *expr) {}

void Analyzer::analyze(AssignOpExpr *expr) {}

void Analyzer::analyze(FieldAccessExpr *expr) {}

void Analyzer::analyze(IndexExpr *expr) {}

void Analyzer::analyze(RangeExpr *expr) {}

void Analyzer::analyze(ComptimeExpr *expr) {}

void Analyzer::analyze(BlockExpression *expr) {
  NEW_SCOPE();
  for (auto &stmt : expr->statements) {
    analyze(stmt.get());
  }
}

void Analyzer::analyze(IdentifierExpr *expr) {}

void Analyzer::analyze(Type *type) {
  switch (type->kind()) {
  case AstNodeKind::PrimitiveType:
    analyze(static_cast<PrimitiveType *>(type));
    break;
  case AstNodeKind::TupleType:
    analyze(static_cast<TupleType *>(type));
    break;
  case AstNodeKind::FunctionType:
    analyze(static_cast<FunctionType *>(type));
    break;
  case AstNodeKind::ReferenceType:
    analyze(static_cast<ReferenceType *>(type));
    break;
  case AstNodeKind::SliceType:
    analyze(static_cast<SliceType *>(type));
    break;
  case AstNodeKind::ArrayType:
    analyze(static_cast<ArrayType *>(type));
    break;
  case AstNodeKind::TraitType:
    analyze(static_cast<TraitType *>(type));
    break;
  case AstNodeKind::IdentifierType:
    analyze(static_cast<IdentifierType *>(type));
    break;
  case AstNodeKind::StructType:
    analyze(static_cast<StructType *>(type));
    break;
  case AstNodeKind::EnumType:
    analyze(static_cast<EnumType *>(type));
    break;
  case AstNodeKind::UnionType:
    analyze(static_cast<UnionType *>(type));
    break;
  case AstNodeKind::ExprType:
    analyze(static_cast<ExprType *>(type));
    break;
  case AstNodeKind::MLIRType:
    analyze(static_cast<MLIRType *>(type));
    break;
  default:
    assert(false && "not a type or not implemented yet");
    break;
  }
}

void Analyzer::analyze(PrimitiveType *type) {}

void Analyzer::analyze(TupleType *type) {}

void Analyzer::analyze(FunctionType *type) {}

void Analyzer::analyze(ReferenceType *type) {}

void Analyzer::analyze(SliceType *type) {}

void Analyzer::analyze(ArrayType *type) {}

void Analyzer::analyze(TraitType *type) {}

void Analyzer::analyze(IdentifierType *type) {}

void Analyzer::analyze(StructType *type) {}

void Analyzer::analyze(EnumType *type) {}

void Analyzer::analyze(UnionType *type) {}

void Analyzer::analyze(ExprType *type) {}

void Analyzer::analyze(MLIRType *type) {}

void Analyzer::analyze(MLIRAttribute *type) {}

void Analyzer::analyze(MLIROp *type) {}

void Analyzer::analyze(StructField *field) {}

void Analyzer::analyze(FieldsNamed *fields) {}

void Analyzer::analyze(FieldsUnnamed *fields) {}

void Analyzer::analyze(Variant *variant) {}

void Analyzer::analyze(UnionField *field) {}

void Analyzer::analyze(MatchArm *arm) {}

void Analyzer::analyze(PatternField *field) {}
