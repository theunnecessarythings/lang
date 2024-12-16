#include "ast.hpp"
#include <array>
#include <iostream>

#define INDENT() Indent level_(cur_indent);

void AstDumper::dump(Program *program) {
  INDENT();
  for (auto &f : program->items)
    dump(f.get());
}

void AstDumper::dump(TopLevelDecl *decl) {
  switch (decl->kind()) {
  case AstNodeKind::Module:
    dump(decl->as<Module>());
    break;
  case AstNodeKind::ImportDecl:
    dump(decl->as<ImportDecl>());
    break;
  case AstNodeKind::StructDecl:
    dump(decl->as<StructDecl>());
    break;
  case AstNodeKind::TupleStructDecl:
    dump(decl->as<TupleStructDecl>());
    break;
  case AstNodeKind::EnumDecl:
    dump(decl->as<EnumDecl>());
    break;
  case AstNodeKind::UnionDecl:
    dump(decl->as<UnionDecl>());
    break;
  case AstNodeKind::TraitDecl:
    dump(decl->as<TraitDecl>());
    break;
  case AstNodeKind::ImplDecl:
    dump(decl->as<ImplDecl>());
    break;
  case AstNodeKind::TopLevelVarDecl:
    dump(decl->as<TopLevelVarDecl>());
    break;
  case AstNodeKind::Function:
    dump(decl->as<Function>());
    break;
  default:
    std::cerr << "Unknown top level decl kind\n";
  }
}

void AstDumper::dump(Module *mod) {
  output_stream << "module " << mod->name << " {\n";
  INDENT();
  for (auto &item : mod->items) {
    indent();
    dump(item.get());
  }
  output_stream << "}\n";
}

void AstDumper::dump(ComptimeExpr *expr) {
  output_stream << "comptime ";
  dump(expr->expr.get());
}

void AstDumper::dump(BlockExpression *expr) {
  output_stream << "{\n";
  INDENT();
  for (auto &stmt : expr->statements) {
    indent();
    dump(stmt.get());
  }
  output_stream << "}\n";
}

void AstDumper::dump(Parameter *param) {
  output_stream << (param->is_comptime ? "comptime " : "");
  output_stream << (param->is_mut ? "mut " : "");
  dump(param->pattern.get());
  output_stream << ": ";
  dump(param->type.get());
  if (!param->trait_bound.empty()) {
    output_stream << " impl ";
    for (size_t i = 0; i < param->trait_bound.size(); i++) {
      dump(param->trait_bound[i].get());
      if (i != param->trait_bound.size() - 1)
        output_stream << " + ";
    }
  }
}

void AstDumper::dump(StructField *field) {
  output_stream << field->name << ": ";
  dump(field->type.get());
}

// TODO: render_for_enum
void AstDumper::dump(StructDecl *decl) {
  output_stream << (decl->is_pub ? "pub " : "");
  output_stream << "struct " << decl->name << " {\n";
  INDENT();
  for (auto &field : decl->fields) {
    indent();
    dump(field.get());
    if (&field != &decl->fields.back())
      output_stream << ", ";
    output_stream << "\n";
  }
  output_stream << "}\n";
}

// TODO: render_for_enum
void AstDumper::dump(TupleStructDecl *decl) {
  output_stream << (decl->is_pub ? "pub " : "");
  output_stream << "struct " << decl->name << "(";
  for (auto &field : decl->fields) {
    dump(field.get());
    if (&field != &decl->fields.back())
      output_stream << ", ";
  }
  output_stream << ")\n";
}

void AstDumper::dump(IdentifierExpr *expr) { output_stream << expr->name; }

void AstDumper::dump(FieldsNamed *fields) {
  for (size_t i = 0; i < fields->name.size(); i++) {
    output_stream << fields->name[i] << ": ";
    dump(fields->value[i].get());
    if (i != fields->name.size() - 1)
      output_stream << ", ";
  }
}

void AstDumper::dump(FieldsUnnamed *fields) {
  for (auto &val : fields->value) {
    dump(val.get());
    if (&val != &fields->value.back())
      output_stream << ", ";
  }
}

void AstDumper::dump(Variant *variant) {
  output_stream << variant->name;
  if (variant->field.has_value()) {
    if (std::holds_alternative<std::unique_ptr<FieldsUnnamed>>(
            variant->field.value())) {
      output_stream << "(";
      dump(std::get<std::unique_ptr<FieldsUnnamed>>(variant->field.value())
               .get());
      output_stream << ")";
    } else if (std::holds_alternative<std::unique_ptr<FieldsNamed>>(
                   variant->field.value())) {
      output_stream << "{";
      dump(
          std::get<std::unique_ptr<FieldsNamed>>(variant->field.value()).get());
      output_stream << "}";
    } else {
      output_stream << " = ";
      dump(std::get<std::unique_ptr<Expression>>(variant->field.value()).get());
    }
  }
}

void AstDumper::dump(UnionField *field) {
  output_stream << field->name << ": ";
  dump(field->type.get());
}

void AstDumper::dump(FunctionDecl *decl) {
  output_stream << "fn " << decl->name << "(";
  for (size_t i = 0; i < decl->parameters.size(); i++) {
    dump(decl->parameters[i].get());
    if (i != decl->parameters.size() - 1)
      output_stream << ", ";
  }
  output_stream << ") ";
  dump(decl->return_type.get());
}

void AstDumper::dump(Function *func) {
  output_stream << (func->is_pub ? "pub " : "");
  dump(func->decl.get());
  dump(func->body.get());
}

void AstDumper::dump(ImportDecl *decl) {
  if (skip_import)
    return;
  output_stream << "import ";
  for (size_t i = 0; i < decl->paths.size(); i++) {
    output_stream << "\"" << decl->paths[i].first.substr(3) << "\"";
    if (decl->paths[i].second.has_value()) {
      output_stream << " as " << decl->paths[i].second.value();
    }
    if (i != decl->paths.size() - 1)
      output_stream << ", ";
  }
  output_stream << ";\n";
}

void AstDumper::dump(EnumDecl *decl) {
  output_stream << (decl->is_pub ? "pub " : "");
  output_stream << "enum " << decl->name << " {\n";
  INDENT();
  for (auto &variant : decl->variants) {
    indent();
    dump(variant.get());
    if (&variant != &decl->variants.back())
      output_stream << ", ";
    output_stream << "\n";
  }
  output_stream << "}\n";
}

void AstDumper::dump(UnionDecl *decl) {
  output_stream << (decl->is_pub ? "pub " : "");
  output_stream << "union " << decl->name << " {\n";
  INDENT();
  for (auto &field : decl->fields) {
    indent();
    dump(field.get());
    if (&field != &decl->fields.back())
      output_stream << ", ";
    output_stream << "\n";
  }
  output_stream << "}\n";
}

void AstDumper::dump(TraitDecl *decl) {
  output_stream << (decl->is_pub ? "pub " : "");
  output_stream << "trait " << decl->name;
  if (!decl->super_traits.empty()) {
    output_stream << " : ";
    for (size_t i = 0; i < decl->super_traits.size(); i++) {
      dump(decl->super_traits[i].get());
      if (i != decl->super_traits.size() - 1)
        output_stream << " + ";
    }
  }
  output_stream << " {\n";
  INDENT();
  for (auto &func : decl->functions) {
    indent();
    if (std::holds_alternative<std::unique_ptr<Function>>(func)) {
      dump(std::get<std::unique_ptr<Function>>(func).get());
    } else {
      dump(std::get<std::unique_ptr<FunctionDecl>>(func).get());
      output_stream << ";\n";
    }
  }
  output_stream << "}\n";
}

void AstDumper::dump(ImplDecl *decl) {
  output_stream << "impl ";
  dump(decl->type.get());
  if (!decl->traits.empty()) {
    output_stream << " : ";
    for (size_t i = 0; i < decl->traits.size(); i++) {
      dump(decl->traits[i].get());
      if (i != decl->traits.size() - 1)
        output_stream << " + ";
    }
  }
  output_stream << " {\n";
  INDENT();
  for (auto &func : decl->functions) {
    indent();
    dump(func.get());
  }
  output_stream << "}\n";
}

void AstDumper::dump(VarDecl *decl) {
  output_stream << (decl->is_pub ? "pub " : "");
  output_stream << (decl->is_mut ? "var " : "const ");
  dump(decl->pattern.get());
  if (decl->type.has_value()) {
    output_stream << ": ";
    dump(decl->type.value().get());
  }
  if (decl->initializer.has_value()) {
    output_stream << " = ";
    dump(decl->initializer.value().get());
  }
  output_stream << ";\n";
}

void AstDumper::dump(TopLevelVarDecl *decl) { dump(decl->var_decl.get()); }

void AstDumper::dump(IfExpr *expr) {
  output_stream << "if ";
  dump(expr->condition.get());
  dump(expr->then_block.get());
  if (expr->else_block.has_value()) {
    output_stream << "else ";
    dump(expr->else_block.value().get());
  }
}

void AstDumper::dump(MatchArm *arm) {
  output_stream << "is ";
  dump(arm->pattern.get());
  if (arm->guard.has_value()) {
    output_stream << " if ";
    dump(arm->guard.value().get());
  }
  output_stream << " => ";
  if (std::holds_alternative<std::unique_ptr<BlockExpression>>(arm->body)) {
    dump(std::get<std::unique_ptr<BlockExpression>>(arm->body).get());
  } else {
    dump(std::get<std::unique_ptr<Expression>>(arm->body).get());
  }
}

void AstDumper::dump(MatchExpr *expr) {
  output_stream << "match ";
  dump(expr->expr.get());
  output_stream << " {\n";
  INDENT();
  for (auto &arm : expr->cases) {
    dump(arm.get());
    output_stream << ",\n";
  }
  output_stream << "}\n";
}

void AstDumper::dump(ForExpr *expr) {
  if (expr->label.has_value()) {
    output_stream << expr->label.value() << ": ";
  }
  output_stream << "for ";
  dump(expr->pattern.get());
  output_stream << " in ";
  dump(expr->iterable.get());
  dump(expr->body.get());
}
void AstDumper::dump(WhileExpr *expr) {
  if (expr->label.has_value()) {
    output_stream << expr->label.value() << ": ";
  }
  output_stream << "while ";
  if (expr->condition.has_value()) {
    dump(expr->condition.value().get());
  }
  if (expr->continue_expr.has_value()) {
    output_stream << " : ";
    dump(expr->continue_expr.value().get());
  }
  dump(expr->body.get());
}

void AstDumper::dump(ReturnExpr *expr) {
  output_stream << "return ";
  if (expr->value.has_value()) {
    dump(expr->value.value().get());
  }
}

void AstDumper::dump(DeferStmt *stmt) {
  output_stream << "defer ";
  if (std::holds_alternative<std::unique_ptr<BlockExpression>>(stmt->body)) {
    dump(std::get<std::unique_ptr<BlockExpression>>(stmt->body).get());
  } else {
    dump(std::get<std::unique_ptr<Expression>>(stmt->body).get());
  }
}

void AstDumper::dump(BreakExpr *expr) {
  output_stream << "break";
  if (expr->label.has_value()) {
    output_stream << ":" << expr->label.value();
  }
  if (expr->value.has_value()) {
    dump(expr->value.value().get());
  }
}

void AstDumper::dump(YieldExpr *expr) {
  output_stream << "yield ";
  if (expr->label.has_value()) {
    output_stream << ":" << expr->label.value();
  }
  dump(expr->value.get());
}

void AstDumper::dump(ContinueExpr *expr) {
  output_stream << "continue";
  if (expr->label.has_value()) {
    output_stream << ":" << expr->label.value();
  }
  if (expr->value.has_value()) {
    dump(expr->value.value().get());
  }
}

void AstDumper::dump(ExprStmt *stmt) {
  dump(stmt->expr.get());
  switch (stmt->expr->kind()) {
  case AstNodeKind::IfExpr:
  case AstNodeKind::MatchExpr:
  case AstNodeKind::ForExpr:
  case AstNodeKind::WhileExpr:
  case AstNodeKind::BlockExpression:
    break;
  default:
    output_stream << ";\n";
    break;
  }
}

void AstDumper::dump(LiteralExpr *expr) {
  switch (expr->type) {
  case LiteralExpr::LiteralType::Int:
    output_stream << std::get<int>(expr->value);
    break;
  case LiteralExpr::LiteralType::Float:
    output_stream << std::get<double>(expr->value);
    break;
  case LiteralExpr::LiteralType::String:
    output_stream << std::get<std::string>(expr->value);
    break;
  case LiteralExpr::LiteralType::Char:
    output_stream << "'" << std::get<char>(expr->value) << "'";
    break;
  case LiteralExpr::LiteralType::Bool:
    output_stream << (std::get<bool>(expr->value) ? "true" : "false");
    break;
  }
}

void AstDumper::dump(TupleExpr *expr) {
  output_stream << "(";
  for (size_t i = 0; i < expr->elements.size(); i++) {
    dump(expr->elements[i].get());
    if (i != expr->elements.size() - 1 || expr->elements.size() == 1)
      output_stream << ", ";
  }
  output_stream << ")";
}

void AstDumper::dump(ArrayExpr *expr) {
  output_stream << "[";
  for (size_t i = 0; i < expr->elements.size(); i++) {
    dump(expr->elements[i].get());
    if (i != expr->elements.size() - 1)
      output_stream << ", ";
  }
  output_stream << "]";
}

void AstDumper::dump(BinaryExpr *expr) {
  output_stream << "(";
  dump(expr->lhs.get());
  switch (expr->op) {
  case Operator::Add:
    output_stream << " + ";
    break;
  case Operator::Sub:
    output_stream << " - ";
    break;
  case Operator::Mul:
    output_stream << " * ";
    break;
  case Operator::Div:
    output_stream << " / ";
    break;
  case Operator::Mod:
    output_stream << " % ";
    break;
  case Operator::And:
    output_stream << " and ";
    break;
  case Operator::Or:
    output_stream << " or ";
    break;
  case Operator::Not:
    output_stream << " not ";
    break;
  case Operator::Eq:
    output_stream << " == ";
    break;
  case Operator::Ne:
    output_stream << " != ";
    break;
  case Operator::Lt:
    output_stream << " < ";
    break;
  case Operator::Le:
    output_stream << " <= ";
    break;
  case Operator::Gt:
    output_stream << " > ";
    break;
  case Operator::Ge:
    output_stream << " >= ";
    break;
  case Operator::BitAnd:
    output_stream << " & ";
    break;
  case Operator::BitOr:
    output_stream << " | ";
    break;
  case Operator::BitXor:
    output_stream << " ^ ";
    break;
  case Operator::BitShl:
    output_stream << " << ";
    break;
  case Operator::BitShr:
    output_stream << " >> ";
    break;
  case Operator::Invalid:
    output_stream << " invalid ";
    break;
  case Operator::Pow:
    output_stream << " ** ";
    break;
  default:
    break;
  }
  dump(expr->rhs.get());
  output_stream << ")";
}

void AstDumper::dump(UnaryExpr *expr) {
  switch (expr->op) {
  case Operator::Not:
    output_stream << "not ";
    break;
  case Operator::BitNot:
    output_stream << "~";
    break;
  case Operator::Sub:
    output_stream << "-";
    break;
  case Operator::Add:
    output_stream << "+";
    break;
  default:
    break;
  }
  dump(expr->operand.get());
}

void AstDumper::dump(CallExpr *expr) {
  output_stream << expr->callee << "(";
  for (size_t i = 0; i < expr->arguments.size(); i++) {
    dump(expr->arguments[i].get());
    if (i != expr->arguments.size() - 1)
      output_stream << ", ";
  }
  output_stream << ")";
}

void AstDumper::dump(AssignExpr *expr) {
  dump(expr->lhs.get());
  output_stream << " = ";
  dump(expr->rhs.get());
}

void AstDumper::dump(AssignOpExpr *expr) {
  dump(expr->lhs.get());
  switch (expr->op) {
  case Operator::Add:
    output_stream << " += ";
    break;
  case Operator::Sub:
    output_stream << " -= ";
    break;
  case Operator::Mul:
    output_stream << " *= ";
    break;
  case Operator::Div:
    output_stream << " /= ";
    break;
  case Operator::Mod:
    output_stream << " %= ";
    break;
  case Operator::BitAnd:
    output_stream << " &= ";
    break;
  case Operator::BitOr:
    output_stream << " |= ";
    break;
  case Operator::BitXor:
    output_stream << " ^= ";
    break;
  case Operator::BitShl:
    output_stream << " <<= ";
    break;
  case Operator::BitShr:
    output_stream << " >>= ";
    break;
  default:
    break;
  }
  dump(expr->rhs.get());
}

void AstDumper::dump(FieldAccessExpr *expr) {
  dump(expr->base.get());
  output_stream << ".";
  if (std::holds_alternative<std::unique_ptr<LiteralExpr>>(expr->field)) {
    dump(std::get<std::unique_ptr<LiteralExpr>>(expr->field).get());
  } else if (std::holds_alternative<std::unique_ptr<IdentifierExpr>>(
                 expr->field)) {
    dump(std::get<std::unique_ptr<IdentifierExpr>>(expr->field).get());
  } else {
    dump(std::get<std::unique_ptr<CallExpr>>(expr->field).get());
  }
}

void AstDumper::dump(IndexExpr *expr) {
  dump(expr->base.get());
  output_stream << "[";
  dump(expr->index.get());
}

void AstDumper::dump(RangeExpr *expr) {
  if (expr->start.has_value())
    dump(expr->start.value().get());
  if (expr->inclusive)
    output_stream << "..=";
  else
    output_stream << "..";
  if (expr->end.has_value())
    dump(expr->end.value().get());
}

void AstDumper::dump(PrimitiveType *type) {
  switch (type->type_kind) {
  case PrimitiveType::PrimitiveTypeKind::String:
    output_stream << "string";
    break;
  case PrimitiveType::PrimitiveTypeKind::Char:
    output_stream << "char";
    break;
  case PrimitiveType::PrimitiveTypeKind::Bool:
    output_stream << "bool";
    break;
  case PrimitiveType::PrimitiveTypeKind::Void:
    output_stream << "void";
    break;
  case PrimitiveType::PrimitiveTypeKind::I8:
    output_stream << "i8";
    break;
  case PrimitiveType::PrimitiveTypeKind::I16:
    output_stream << "i16";
    break;
  case PrimitiveType::PrimitiveTypeKind::I32:
    output_stream << "i32";
    break;
  case PrimitiveType::PrimitiveTypeKind::I64:
    output_stream << "i64";
    break;
  case PrimitiveType::PrimitiveTypeKind::U8:
    output_stream << "u8";
    break;
  case PrimitiveType::PrimitiveTypeKind::U16:
    output_stream << "u16";
    break;
  case PrimitiveType::PrimitiveTypeKind::U32:
    output_stream << "u32";
    break;
  case PrimitiveType::PrimitiveTypeKind::U64:
    output_stream << "u64";
    break;
  case PrimitiveType::PrimitiveTypeKind::F32:
    output_stream << "f32";
    break;
  case PrimitiveType::PrimitiveTypeKind::F64:
    output_stream << "f64";
    break;
  case PrimitiveType::PrimitiveTypeKind::type:
    output_stream << "type";
    break;
  }
}

void AstDumper::dump(TupleType *type) {
  output_stream << (as_type ? "tuple<" : "(");
  for (size_t i = 0; i < type->elements.size(); i++) {
    dump(type->elements[i].get());
    if (i != type->elements.size() - 1)
      output_stream << ", ";
  }
  output_stream << (as_type ? ">" : ")");
}

void AstDumper::dump(FunctionType *type) {
  output_stream << "(";
  for (size_t i = 0; i < type->parameters.size(); i++) {
    dump(type->parameters[i].get());
    if (i != type->parameters.size() - 1)
      output_stream << ", ";
  }
  output_stream << ") -> ";
  dump(type->return_type.get());
}

void AstDumper::dump(ReferenceType *type) {
  output_stream << "&";
  dump(type->base.get());
}

void AstDumper::dump(SliceType *type) {
  output_stream << "[";
  output_stream << "]";
  dump(type->base.get());
}

void AstDumper::dump(ArrayType *type) {
  output_stream << (as_type ? "array<" : "[");
  dump(type->size.get());
  output_stream << (as_type ? "x" : "]");
  dump(type->base.get());
  output_stream << (as_type ? ">" : "");
}

void AstDumper::dump(TraitType *type) { output_stream << type->name; }

void AstDumper::dump(IdentifierType *type) { output_stream << type->name; }

void AstDumper::dump(StructType *type) { output_stream << type->name; }

void AstDumper::dump(EnumType *type) { output_stream << type->name; }

void AstDumper::dump(UnionType *type) { output_stream << type->name; }

void AstDumper::dump(ExprType *type) { dump(type->expr.get()); }

void AstDumper::dump(LiteralPattern *pattern) { dump(pattern->literal.get()); }

void AstDumper::dump(IdentifierPattern *pattern) {
  output_stream << pattern->name;
}

void AstDumper::dump(WildcardPattern *pattern) { output_stream << "_"; }

void AstDumper::dump(TuplePattern *pattern) {
  output_stream << "(";
  for (size_t i = 0; i < pattern->elements.size(); i++) {
    dump(pattern->elements[i].get());
    if (i != pattern->elements.size() - 1)
      output_stream << ", ";
  }
  output_stream << ")";
}

void AstDumper::dump(PatternField *field) {
  output_stream << field->name;
  if (field->pattern.has_value()) {
    output_stream << ": ";
    dump(field->pattern.value().get());
  }
}

void AstDumper::dump(RestPattern *pattern) {
  output_stream << "..";
  if (pattern->name.has_value()) {
    output_stream << " as ";
    dump(&pattern->name.value());
  }
}

void AstDumper::dump(StructPattern *pattern) {
  if (pattern->name.has_value()) {
    output_stream << pattern->name.value();
  }
  output_stream << " {";
  for (size_t i = 0; i < pattern->fields.size(); i++) {
    if (std::holds_alternative<std::unique_ptr<PatternField>>(
            pattern->fields[i])) {
      dump(std::get<std::unique_ptr<PatternField>>(pattern->fields[i]).get());
    } else {
      dump(std::get<std::unique_ptr<RestPattern>>(pattern->fields[i]).get());
    }
    if (i != pattern->fields.size() - 1)
      output_stream << ", ";
  }
  output_stream << "}";
}

void AstDumper::dump(SlicePattern *pattern) {
  output_stream << "[";
  for (size_t i = 0; i < pattern->elements.size(); i++) {
    dump(pattern->elements[i].get());
    if (i != pattern->elements.size() - 1)
      output_stream << ", ";
  }
  output_stream << "]";
}

void AstDumper::dump(OrPattern *pattern) {
  for (size_t i = 0; i < pattern->patterns.size(); i++) {
    dump(pattern->patterns[i].get());
    if (i != pattern->patterns.size() - 1)
      output_stream << " | ";
  }
}

void AstDumper::dump(ExprPattern *pattern) { dump(pattern->expr.get()); }

void AstDumper::dump(RangePattern *pattern) {
  dump(pattern->start.get());
  if (pattern->inclusive)
    output_stream << "..=";
  else
    output_stream << "..";
  dump(pattern->end.get());
}

void AstDumper::dump(VariantPattern *pattern) {
  output_stream << "." << pattern->name;
  if (pattern->field.has_value()) {
    if (std::holds_alternative<std::unique_ptr<TuplePattern>>(
            pattern->field.value())) {
      dump(std::get<std::unique_ptr<TuplePattern>>(pattern->field.value())
               .get());
    } else {
      dump(std::get<std::unique_ptr<StructPattern>>(pattern->field.value())
               .get());
    }
  }
}

void AstDumper::dump(TopLevelDeclStmt *stmt) { dump(stmt->decl.get()); }

void AstDumper::dump(MLIRType *type) {
  if (as_type)
    output_stream << type->type.substr(1, type->type.size() - 2);
  else
    output_stream << "@mlir_type(" << type->type << ")";
}

void AstDumper::dump(MLIRAttribute *attr) {
  output_stream << "@mlir_attr(" << attr->attribute << ")";
}

void AstDumper::dump(MLIROp *op) {
  output_stream << "@mlir_op(" << op->op << ", [";
  for (auto &operand : op->operands) {
    dump(operand.get());
    if (&operand != &op->operands.back())
      output_stream << ", ";
  }
  output_stream << "], {";
  int i = 0;
  for (auto &attr : op->attributes) {
    output_stream << attr.first << ": " << attr.second;
    if (i != (int)op->attributes.size() - 1)
      output_stream << ", ";
    i++;
  }
  output_stream << "}, [";
  for (auto &result : op->result_types) {
    output_stream << result;
    if (&result != &op->result_types.back())
      output_stream << ", ";
  }
  output_stream << "])";
}

void AstDumper::dump(Pattern *pattern) {
  switch (pattern->kind()) {
  case AstNodeKind::LiteralPattern:
    dump(pattern->as<LiteralPattern>());
    break;
  case AstNodeKind::IdentifierPattern:
    dump(pattern->as<IdentifierPattern>());
    break;
  case AstNodeKind::WildcardPattern:
    dump(pattern->as<WildcardPattern>());
    break;
  case AstNodeKind::TuplePattern:
    dump(pattern->as<TuplePattern>());
    break;
  case AstNodeKind::StructPattern:
    dump(pattern->as<StructPattern>());
    break;
  case AstNodeKind::SlicePattern:
    dump(pattern->as<SlicePattern>());
    break;
  case AstNodeKind::OrPattern:
    dump(pattern->as<OrPattern>());
    break;
  case AstNodeKind::ExprPattern:
    dump(pattern->as<ExprPattern>());
    break;
  case AstNodeKind::RangePattern:
    dump(pattern->as<RangePattern>());
    break;
  case AstNodeKind::VariantPattern:
    dump(pattern->as<VariantPattern>());
    break;
  case AstNodeKind::RestPattern:
    dump(pattern->as<RestPattern>());
    break;
  default:
    std::cerr << "Unknown pattern kind\n";
    break;
  }
}

void AstDumper::dump(Statement *stmt) {
  switch (stmt->kind()) {
  case AstNodeKind::ExprStmt:
    dump(stmt->as<ExprStmt>());
    break;
  case AstNodeKind::TopLevelDeclStmt:
    dump(stmt->as<TopLevelDeclStmt>());
    break;

  case AstNodeKind::VarDecl:
    dump(stmt->as<VarDecl>());
    break;
  case AstNodeKind::DeferStmt:
    dump(stmt->as<DeferStmt>());
    break;
  default:
    std::cerr << "Unknown statement kind\n";
    break;
  }
}

void AstDumper::dump(Expression *expr) {
  output_stream << "(";
  switch (expr->kind()) {
  case AstNodeKind::LiteralExpr:
    dump(expr->as<LiteralExpr>());
    break;
  case AstNodeKind::IdentifierExpr:
    dump(expr->as<IdentifierExpr>());
    break;
  case AstNodeKind::TupleExpr:
    dump(expr->as<TupleExpr>());
    break;
  case AstNodeKind::ArrayExpr:
    dump(expr->as<ArrayExpr>());
    break;
  case AstNodeKind::BinaryExpr:
    dump(expr->as<BinaryExpr>());
    break;
  case AstNodeKind::UnaryExpr:
    dump(expr->as<UnaryExpr>());
    break;
  case AstNodeKind::CallExpr:
    dump(expr->as<CallExpr>());
    break;
  case AstNodeKind::AssignExpr:
    dump(expr->as<AssignExpr>());
    break;
  case AstNodeKind::AssignOpExpr:
    dump(expr->as<AssignOpExpr>());
    break;
  case AstNodeKind::FieldAccessExpr:
    dump(expr->as<FieldAccessExpr>());
    break;
  case AstNodeKind::IndexExpr:
    dump(expr->as<IndexExpr>());
    break;
  case AstNodeKind::RangeExpr:
    dump(expr->as<RangeExpr>());
    break;
  case AstNodeKind::IfExpr:
    dump(expr->as<IfExpr>());
    break;
  case AstNodeKind::MatchExpr:
    dump(expr->as<MatchExpr>());
    break;
  case AstNodeKind::ForExpr:
    dump(expr->as<ForExpr>());
    break;
  case AstNodeKind::WhileExpr:
    dump(expr->as<WhileExpr>());
    break;
  case AstNodeKind::ReturnExpr:
    dump(expr->as<ReturnExpr>());
    break;
  case AstNodeKind::BreakExpr:
    dump(expr->as<BreakExpr>());
    break;
  case AstNodeKind::ContinueExpr:
    dump(expr->as<ContinueExpr>());
    break;
  case AstNodeKind::BlockExpression:
    dump(expr->as<BlockExpression>());
    break;
  case AstNodeKind::ComptimeExpr:
    dump(expr->as<ComptimeExpr>());
    break;
  case AstNodeKind::YieldExpr:
    dump(expr->as<YieldExpr>());
    break;
  case AstNodeKind::MLIROp:
    dump(expr->as<MLIROp>());
    break;
  case AstNodeKind::MLIRAttribute:
    dump(expr->as<MLIRAttribute>());
    break;
  default:
    std::cerr << "Unknown expression kind\n";
    break;
  }
  output_stream << ")";
}

void AstDumper::dump(Type *type) {
  switch (type->kind()) {
  case AstNodeKind::PrimitiveType:
    dump(type->as<PrimitiveType>());
    break;
  case AstNodeKind::TupleType:
    dump(type->as<TupleType>());
    break;
  case AstNodeKind::FunctionType:
    dump(type->as<FunctionType>());
    break;
  case AstNodeKind::ReferenceType:
    dump(type->as<ReferenceType>());
    break;
  case AstNodeKind::SliceType:
    dump(type->as<SliceType>());
    break;
  case AstNodeKind::ArrayType:
    dump(type->as<ArrayType>());
    break;
  case AstNodeKind::TraitType:
    dump(type->as<TraitType>());
    break;
  case AstNodeKind::IdentifierType:
    dump(type->as<IdentifierType>());
    break;
  case AstNodeKind::StructType:
    dump(type->as<StructType>());
    break;
  case AstNodeKind::EnumType:
    dump(type->as<EnumType>());
    break;
  case AstNodeKind::UnionType:
    dump(type->as<UnionType>());
    break;
  case AstNodeKind::ExprType:
    dump(type->as<ExprType>());
    break;
  case AstNodeKind::MLIRType:
    dump(type->as<MLIRType>());
    break;
  default:
    std::cerr << "Unknown type kind " << ::toString(type->kind()) << "\n";
    break;
  }
}

void AstDumper::indent() {
  for (int i = 0; i < cur_indent; i++)
    output_stream << "  ";
}

std::string &toString(AstNodeKind kind) {
  static std::array<std::string, 86> names = {
      "Program",
      "Module",
      "Expression",
      "Statement",
      "Type",
      "Pattern",
      "ComptimeExpr",
      "BlockExpression",
      "BlockStatement",
      "Parameter",
      "StructField",
      "StructDecl",
      "TupleStructDecl",
      "IdentifierExpr",
      "FieldsNamed",
      "FieldsUnnamed",
      "Variant",
      "UnionField",
      "FunctionDecl",
      "Function",
      "ImportDecl",
      "EnumDecl",
      "UnionDecl",
      "TraitDecl",
      "ImplDecl",
      "VarDecl",
      "TopLevelVarDecl",
      "IfExpr",
      "MatchArm",
      "MatchExpr",
      "ForExpr",
      "WhileExpr",
      "ReturnExpr",
      "DeferStmt",
      "BreakExpr",
      "ContinueExpr",
      "ExprStmt",
      "LiteralExpr",
      "TupleExpr",
      "ArrayExpr",
      "BinaryExpr",
      "UnaryExpr",
      "CallExpr",
      "AssignExpr",
      "AssignOpExpr",
      "FieldAccessExpr",
      "IndexExpr",
      "RangeExpr",
      "PrimitiveType",
      "PointerType",
      "ArrayType",
      "TupleType",
      "FunctionType",
      "StructType",
      "EnumType",
      "UnionType",
      "TraitType",
      "ImplType",
      "SliceType",
      "ReferenceType",
      "IdentifierType",
      "ExprType",
      "LiteralPattern",
      "IdentifierPattern",
      "TuplePattern",
      "StructPattern",
      "EnumPattern",
      "SlicePattern",
      "WildcardPattern",
      "RestPattern",
      "OrPattern",
      "ExprPattern",
      "PatternField",
      "RangePattern",
      "VariantPattern",
      "TopLevelDeclStmt",
      "YieldExpr",

      "MLIRAttribute",
      "MLIRType",
      "MLIROp",

      "InvalidNode",
      "InvalidExpression",
      "InvalidStatement",
      "InvalidTopLevelDecl",
      "InvalidType",
      "InvalidPattern",
  };
  return names[static_cast<int>(kind)];
}
