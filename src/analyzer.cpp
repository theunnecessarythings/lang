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
    if (param->type->kind() == AstNodeKind::PrimitiveType &&
        param->type->as<PrimitiveType>()->type_kind ==
            PrimitiveType::PrimitiveTypeKind::type) {
      decl->extra.is_generic = true;
    }
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

void Analyzer::analyze(TopLevelVarDecl *decl) { analyze(decl->var_decl.get()); }

void Analyzer::analyze(StructDecl *decl) {
  for (auto &field : decl->fields) {
    analyze(field.get());
  }
}

void Analyzer::analyze(TupleStructDecl *decl) {
  for (auto &field : decl->fields) {
    analyze(field.get());
  }
}

void Analyzer::analyze(EnumDecl *decl) {
  for (auto &variant : decl->variants) {
    analyze(variant.get());
  }
}

void Analyzer::analyze(TraitDecl *decl) {
  for (auto &func : decl->functions) {
    if (std::holds_alternative<std::unique_ptr<FunctionDecl>>(func)) {
      auto &f = std::get<std::unique_ptr<FunctionDecl>>(func);
      analyze(f.get());
    } else {
      auto &f = std::get<std::unique_ptr<Function>>(func);
      analyze(f.get());
    }
  }
  for (auto &trait : decl->super_traits) {
    analyze(trait.get());
  }
}

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

void Analyzer::analyze(LiteralPattern *pattern) {
  analyze(pattern->literal.get());
}

void Analyzer::analyze(IdentifierPattern *pattern) {}

void Analyzer::analyze(WildcardPattern *pattern) {}

void Analyzer::analyze(TuplePattern *pattern) {
  for (auto &p : pattern->elements) {
    analyze(p.get());
  }
}

void Analyzer::analyze(RestPattern *pattern) {
  if (pattern->name) {
    analyze(&pattern->name.value());
  }
}

void Analyzer::analyze(StructPattern *pattern) {
  for (auto &field : pattern->fields) {
    if (std::holds_alternative<std::unique_ptr<PatternField>>(field)) {
      auto &p = std::get<std::unique_ptr<PatternField>>(field);
      analyze(p.get());
    } else {
      auto &p = std::get<std::unique_ptr<RestPattern>>(field);
      analyze(p.get());
    }
  }
}

void Analyzer::analyze(SlicePattern *pattern) {
  for (auto &p : pattern->elements) {
    analyze(p.get());
  }
}

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

void Analyzer::analyze(VariantPattern *pattern) {
  if (pattern->field) {
    if (std::holds_alternative<std::unique_ptr<TuplePattern>>(
            pattern->field.value())) {
      auto &p = std::get<std::unique_ptr<TuplePattern>>(pattern->field.value());
      analyze(p.get());
    } else {
      auto &p =
          std::get<std::unique_ptr<StructPattern>>(pattern->field.value());
      analyze(p.get());
    }
  }
}

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
  case AstNodeKind::AssignStatement:
    analyze(stmt->as<AssignStatement>());
    break;
  default:
    assert(false && "not a statement or not implemented yet");
    break;
  }
}

void Analyzer::analyze(VarDecl *decl) {
  analyze(decl->pattern.get());
  if (decl->initializer) {
    analyze(decl->initializer->get());
  }
  if (decl->type) {
    analyze(decl->type.value().get());
  }
}

void Analyzer::analyze(DeferStmt *stmt) {}

void Analyzer::analyze(ExprStmt *stmt) { analyze(stmt->expr.get()); }

void Analyzer::analyze(TopLevelDeclStmt *stmt) { analyze(stmt->decl.get()); }

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
  case AstNodeKind::MLIRAttribute:
    analyze(expr->as<MLIRAttribute>());
    break;
  default:
    assert(false && "not an expression or not implemented yet");
    break;
  }
}

void Analyzer::analyze(IfExpr *expr) {
  analyze(expr->condition.get());
  analyze(expr->then_block.get());
  if (expr->else_block) {
    analyze(expr->else_block->get());
  }
}

void Analyzer::analyze(MatchExpr *expr) {
  analyze(expr->expr.get());
  for (auto &arm : expr->cases) {
    analyze(arm.get());
  }
}

void Analyzer::analyze(ForExpr *expr) {
  analyze(expr->pattern.get());
  analyze(expr->iterable.get());
  analyze(expr->body.get());
}

void Analyzer::analyze(WhileExpr *expr) {
  if (expr->condition) {
    analyze(expr->condition->get());
  }
  analyze(expr->body.get());
  if (expr->continue_expr) {
    analyze(expr->continue_expr->get());
  }
}

void Analyzer::analyze(ReturnExpr *expr) {
  if (expr->value) {
    analyze(expr->value->get());
  }
}

void Analyzer::analyze(BreakExpr *expr) {
  if (expr->value) {
    analyze(expr->value->get());
  }
}

void Analyzer::analyze(ContinueExpr *expr) {
  if (expr->value) {
    analyze(expr->value->get());
  }
}

void Analyzer::analyze(LiteralExpr *expr) {}

void Analyzer::analyze(TupleExpr *expr) {
  for (auto &elem : expr->elements) {
    analyze(elem.get());
  }
}

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

void Analyzer::analyze(BinaryExpr *expr) {
  analyze(expr->lhs.get());
  analyze(expr->rhs.get());
}

void Analyzer::analyze(UnaryExpr *expr) { analyze(expr->operand.get()); }

void Analyzer::analyze(CallExpr *expr) {
  for (auto &arg : expr->arguments) {
    analyze(arg.get());
  }
}

void Analyzer::analyze(AssignStatement *expr) {
  analyze(expr->lhs.get());
  analyze(expr->rhs.get());
}

void Analyzer::analyze(AssignOpExpr *expr) {
  analyze(expr->lhs.get());
  analyze(expr->rhs.get());
}

void Analyzer::analyze(FieldAccessExpr *expr) {
  analyze(expr->base.get());
  if (std::holds_alternative<std::unique_ptr<IdentifierExpr>>(expr->field)) {
    auto &id = std::get<std::unique_ptr<IdentifierExpr>>(expr->field);
    analyze(id.get());
  } else if (std::holds_alternative<std::unique_ptr<LiteralExpr>>(
                 expr->field)) {
    auto &field = std::get<std::unique_ptr<LiteralExpr>>(expr->field);
    analyze(field.get());
  } else {
    auto &field = std::get<std::unique_ptr<CallExpr>>(expr->field);
    analyze(field.get());
  }
}

void Analyzer::analyze(IndexExpr *expr) {
  analyze(expr->base.get());
  analyze(expr->index.get());
}

void Analyzer::analyze(RangeExpr *expr) {
  if (expr->start) {
    analyze(expr->start->get());
  }
  if (expr->end) {
    analyze(expr->end->get());
  }
}

void Analyzer::analyze(ComptimeExpr *expr) { analyze(expr->expr.get()); }

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

void Analyzer::analyze(TupleType *type) {
  for (auto &elem : type->elements) {
    analyze(elem.get());
  }
}

void Analyzer::analyze(FunctionType *type) {
  for (auto &param : type->parameters) {
    analyze(param.get());
  }
  analyze(type->return_type.get());
}

void Analyzer::analyze(ReferenceType *type) { analyze(type->base.get()); }

void Analyzer::analyze(SliceType *type) { analyze(type->base.get()); }

void Analyzer::analyze(ArrayType *type) {
  analyze(type->base.get());
  analyze(type->size.get());
}

void Analyzer::analyze(TraitType *type) {}

void Analyzer::analyze(IdentifierType *type) {}

void Analyzer::analyze(StructType *type) {}

void Analyzer::analyze(EnumType *type) {}

void Analyzer::analyze(UnionType *type) {}

void Analyzer::analyze(ExprType *type) {
  expr_types.push_back(type);
  analyze(type->expr.get());
}

void Analyzer::analyze(MLIRType *type) {}

void Analyzer::analyze(MLIRAttribute *type) {}

void Analyzer::analyze(MLIROp *type) {
  for (auto &operand : type->operands) {
    analyze(operand.get());
  }
}

void Analyzer::analyze(StructField *field) { analyze(field->type.get()); }

void Analyzer::analyze(FieldsNamed *fields) {
  for (auto &field : fields->value) {
    analyze(field.get());
  }
}

void Analyzer::analyze(FieldsUnnamed *fields) {
  for (auto &field : fields->value) {
    analyze(field.get());
  }
}

void Analyzer::analyze(Variant *variant) {
  if (variant->field) {
    if (std::holds_alternative<std::unique_ptr<FieldsNamed>>(
            variant->field.value())) {
      auto &fields =
          std::get<std::unique_ptr<FieldsNamed>>(variant->field.value());
      analyze(fields.get());
    } else if (std::holds_alternative<std::unique_ptr<FieldsUnnamed>>(
                   variant->field.value())) {
      auto &fields =
          std::get<std::unique_ptr<FieldsUnnamed>>(variant->field.value());
      analyze(fields.get());
    } else {
      auto &field =
          std::get<std::unique_ptr<Expression>>(variant->field.value());
      analyze(field.get());
    }
  }
}

void Analyzer::analyze(UnionField *field) {}

void Analyzer::analyze(MatchArm *arm) {
  analyze(arm->pattern.get());
  if (std::holds_alternative<std::unique_ptr<BlockExpression>>(arm->body)) {
    auto &block = std::get<std::unique_ptr<BlockExpression>>(arm->body);
    analyze(block.get());
  } else {
    auto &expr = std::get<std::unique_ptr<Expression>>(arm->body);
    analyze(expr.get());
  }
  if (arm->guard) {
    analyze(arm->guard->get());
  }
}

void Analyzer::analyze(PatternField *field) {
  if (field->pattern) {
    analyze(field->pattern->get());
  }
}
