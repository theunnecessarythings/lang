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
  llvm::ScopedHashTableScope<llvm::StringRef, Type *> var_scope(               \
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
    analyze(decl->as<Function>());
    break;
  case AstNodeKind::StructDecl:
    analyze(decl->as<StructDecl>());
    break;
  case AstNodeKind::TupleStructDecl:
    analyze(decl->as<TupleStructDecl>());
    break;
  case AstNodeKind::EnumDecl:
    analyze(decl->as<EnumDecl>());
    break;
  case AstNodeKind::TraitDecl:
    analyze(decl->as<TraitDecl>());
    break;
  case AstNodeKind::ImplDecl:
    analyze(decl->as<ImplDecl>());
    break;
  case AstNodeKind::TopLevelVarDecl:
    analyze(decl->as<TopLevelVarDecl>());
    break;
  case AstNodeKind::Module:
    analyze(decl->as<Module>());
    break;
  case AstNodeKind::UnionDecl:
    analyze(decl->as<UnionDecl>());
    break;
  case AstNodeKind::ImportDecl:
    analyze(decl->as<ImportDecl>());
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
          auto id_type = param->type->as<IdentifierType>();
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
    analyze(pattern->as<LiteralPattern>());
    break;
  case AstNodeKind::IdentifierPattern:
    analyze(pattern->as<IdentifierPattern>());
    break;
  case AstNodeKind::WildcardPattern:
    analyze(pattern->as<WildcardPattern>());
    break;
  case AstNodeKind::TuplePattern:
    analyze(pattern->as<TuplePattern>());
    break;
  case AstNodeKind::RestPattern:
    analyze(pattern->as<RestPattern>());
    break;
  case AstNodeKind::StructPattern:
    analyze(pattern->as<StructPattern>());
    break;
  case AstNodeKind::SlicePattern:
    analyze(pattern->as<SlicePattern>());
    break;
  case AstNodeKind::OrPattern:
    analyze(pattern->as<OrPattern>());
    break;
  case AstNodeKind::ExprPattern:
    analyze(pattern->as<ExprPattern>());
    break;
  case AstNodeKind::RangePattern:
    analyze(pattern->as<RangePattern>());
    break;
  case AstNodeKind::VariantPattern:
    analyze(pattern->as<VariantPattern>());
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
    analyze(stmt->as<VarDecl>());
    break;
  case AstNodeKind::DeferStmt:
    analyze(stmt->as<DeferStmt>());
    break;
  case AstNodeKind::ExprStmt:
    analyze(stmt->as<ExprStmt>());
    break;
  case AstNodeKind::TopLevelDeclStmt:
    analyze(stmt->as<TopLevelDeclStmt>());
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
    analyze(expr->as<IfExpr>());
    break;
  case AstNodeKind::MatchExpr:
    analyze(expr->as<MatchExpr>());
    break;
  case AstNodeKind::ForExpr:
    analyze(expr->as<ForExpr>());
    break;
  case AstNodeKind::WhileExpr:
    analyze(expr->as<WhileExpr>());
    break;
  case AstNodeKind::IdentifierExpr:
    analyze(expr->as<IdentifierExpr>());
    break;
  case AstNodeKind::ReturnExpr:
    analyze(expr->as<ReturnExpr>());
    break;
  case AstNodeKind::BreakExpr:
    analyze(expr->as<BreakExpr>());
    break;
  case AstNodeKind::ContinueExpr:
    analyze(expr->as<ContinueExpr>());
    break;
  case AstNodeKind::LiteralExpr:
    analyze(expr->as<LiteralExpr>());
    break;
  case AstNodeKind::TupleExpr:
    analyze(expr->as<TupleExpr>());
    break;
  case AstNodeKind::ArrayExpr:
    analyze(expr->as<ArrayExpr>());
    break;
  case AstNodeKind::BinaryExpr:
    analyze(expr->as<BinaryExpr>());
    break;
  case AstNodeKind::UnaryExpr:
    analyze(expr->as<UnaryExpr>());
    break;
  case AstNodeKind::CallExpr:
    analyze(expr->as<CallExpr>());
    break;
  case AstNodeKind::AssignExpr:
    analyze(expr->as<AssignExpr>());
    break;
  case AstNodeKind::AssignOpExpr:
    analyze(expr->as<AssignOpExpr>());
    break;
  case AstNodeKind::FieldAccessExpr:
    analyze(expr->as<FieldAccessExpr>());
    break;
  case AstNodeKind::IndexExpr:
    analyze(expr->as<IndexExpr>());
    break;
  case AstNodeKind::RangeExpr:
    analyze(expr->as<RangeExpr>());
    break;
  case AstNodeKind::ComptimeExpr:
    analyze(expr->as<ComptimeExpr>());
    break;
  case AstNodeKind::BlockExpression:
    analyze(expr->as<BlockExpression>());
    break;
  case AstNodeKind::MLIROp:
    analyze(expr->as<MLIROp>());
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

  auto literal_kind = first->as<LiteralExpr>()->type;
  for (auto &elem : expr->elements) {
    if (elem->kind() != AstNodeKind::LiteralExpr) {
      return;
    }
    auto literal = elem->as<LiteralExpr>();
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
    analyze(type->as<PrimitiveType>());
    break;
  case AstNodeKind::TupleType:
    analyze(type->as<TupleType>());
    break;
  case AstNodeKind::FunctionType:
    analyze(type->as<FunctionType>());
    break;
  case AstNodeKind::ReferenceType:
    analyze(type->as<ReferenceType>());
    break;
  case AstNodeKind::SliceType:
    analyze(type->as<SliceType>());
    break;
  case AstNodeKind::ArrayType:
    analyze(type->as<ArrayType>());
    break;
  case AstNodeKind::TraitType:
    analyze(type->as<TraitType>());
    break;
  case AstNodeKind::IdentifierType:
    analyze(type->as<IdentifierType>());
    break;
  case AstNodeKind::StructType:
    analyze(type->as<StructType>());
    break;
  case AstNodeKind::EnumType:
    analyze(type->as<EnumType>());
    break;
  case AstNodeKind::UnionType:
    analyze(type->as<UnionType>());
    break;
  case AstNodeKind::ExprType:
    analyze(type->as<ExprType>());
    break;
  case AstNodeKind::MLIRType:
    analyze(type->as<MLIRType>());
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
