#include "json_dumper.hpp"

std::string JsonDumper::tokenKindToString(TokenKind kind) {
  switch (kind) {
  case TokenKind::Dummy:
    return "Dummy";
  case TokenKind::StringLiteral:
    return "StringLiteral";
  case TokenKind::CharLiteral:
    return "CharLiteral";
  case TokenKind::NumberLiteral:
    return "NumberLiteral";
  case TokenKind::Identifier:
    return "Identifier";
  case TokenKind::Bang:
    return "Bang";
  case TokenKind::BangEqual:
    return "BangEqual";
  case TokenKind::Pipe:
    return "Pipe";
  case TokenKind::PipePipe:
    return "PipePipe";
  case TokenKind::PipeEqual:
    return "PipeEqual";
  case TokenKind::Equal:
    return "Equal";
  case TokenKind::EqualEqual:
    return "EqualEqual";
  case TokenKind::Caret:
    return "Caret";
  case TokenKind::CaretEqual:
    return "CaretEqual";
  case TokenKind::Plus:
    return "Plus";
  case TokenKind::PlusEqual:
    return "PlusEqual";
  case TokenKind::PlusPlus:
    return "PlusPlus";
  case TokenKind::Minus:
    return "Minus";
  case TokenKind::MinusEqual:
    return "MinusEqual";
  case TokenKind::Star:
    return "Star";
  case TokenKind::StarEqual:
    return "StarEqual";
  case TokenKind::StarStar:
    return "StarStar";
  case TokenKind::Percent:
    return "Percent";
  case TokenKind::PercentEqual:
    return "PercentEqual";
  case TokenKind::Slash:
    return "Slash";
  case TokenKind::SlashEqual:
    return "SlashEqual";
  case TokenKind::Ampersand:
    return "Ampersand";
  case TokenKind::AmpersandEqual:
    return "AmpersandEqual";
  case TokenKind::Tilde:
    return "Tilde";
  case TokenKind::Less:
    return "Less";
  case TokenKind::LessEqual:
    return "LessEqual";
  case TokenKind::LessLess:
    return "LessLess";
  case TokenKind::LessLessEqual:
    return "LessLessEqual";
  case TokenKind::Greater:
    return "Greater";
  case TokenKind::GreaterEqual:
    return "GreaterEqual";
  case TokenKind::GreaterGreater:
    return "GreaterGreater";
  case TokenKind::GreaterGreaterEqual:
    return "GreaterGreaterEqual";
  case TokenKind::Question:
    return "Question";
  case TokenKind::LParen:
    return "LParen";
  case TokenKind::RParen:
    return "RParen";
  case TokenKind::LBrace:
    return "LBrace";
  case TokenKind::RBrace:
    return "RBrace";
  case TokenKind::LBracket:
    return "LBracket";
  case TokenKind::RBracket:
    return "RBracket";
  case TokenKind::Comma:
    return "Comma";
  case TokenKind::Dot:
    return "Dot";
  case TokenKind::DotDot:
    return "DotDot";
  case TokenKind::DotDotEqual:
    return "DotDotEqual";
  case TokenKind::Colon:
    return "Colon";
  case TokenKind::Semicolon:
    return "Semicolon";
  case TokenKind::EqualGreater:
    return "EqualGreater";
  case TokenKind::MinusGreater:
    return "MinusGreater";
  case TokenKind::KeywordMut:
    return "KeywordMut";
  case TokenKind::KeywordImpl:
    return "KeywordImpl";
  case TokenKind::KeywordUnion:
    return "KeywordUnion";
  case TokenKind::KeywordTrait:
    return "KeywordTrait";
  case TokenKind::KeywordMatch:
    return "KeywordMatch";
  case TokenKind::KeywordIs:
    return "KeywordIs";
  case TokenKind::KeywordAs:
    return "KeywordAs";
  case TokenKind::KeywordAnd:
    return "KeywordAnd";
  case TokenKind::KeywordBreak:
    return "KeywordBreak";
  case TokenKind::KeywordConst:
    return "KeywordConst";
  case TokenKind::KeywordContinue:
    return "KeywordContinue";
  case TokenKind::KeywordElse:
    return "KeywordElse";
  case TokenKind::KeywordEnum:
    return "KeywordEnum";
  case TokenKind::KeywordFn:
    return "KeywordFn";
  case TokenKind::KeywordFor:
    return "KeywordFor";
  case TokenKind::KeywordWhile:
    return "KeywordWhile";
  case TokenKind::KeywordIf:
    return "KeywordIf";
  case TokenKind::KeywordImport:
    return "KeywordImport";
  case TokenKind::KeywordIn:
    return "KeywordIn";
  case TokenKind::KeywordNot:
    return "KeywordNot";
  case TokenKind::KeywordOr:
    return "KeywordOr";
  case TokenKind::KeywordPub:
    return "KeywordPub";
  case TokenKind::KeywordReturn:
    return "KeywordReturn";
  case TokenKind::KeywordStruct:
    return "KeywordStruct";
  case TokenKind::KeywordVar:
    return "KeywordVar";
  case TokenKind::KeywordComptime:
    return "KeywordComptime";
  case TokenKind::KeywordModule:
    return "KeywordModule";
  case TokenKind::Eof:
    return "Eof";
  case TokenKind::Invalid:
    return "Invalid";
  default:
    return "Unknown";
  }
}

void JsonDumper::dump(const TokenSpan &span) {
  output_stream << "{\n";
  cur_indent++;
  indent();
  output_stream << "\"file_id\": " << span.file_id << ",\n";
  indent();
  output_stream << "\"line_no\": " << span.line_no << ",\n";
  indent();
  output_stream << "\"col_start\": " << span.col_start << ",\n";
  indent();
  output_stream << "\"start\": " << span.start << ",\n";
  indent();
  output_stream << "\"end\": " << span.end << "\n";
  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(const Token &token) {
  output_stream << "{\n";
  cur_indent++;
  indent();
  output_stream << "\"kind\": \"" << tokenKindToString(token.kind) << "\",\n";
  indent();
  output_stream << "\"span\": ";
  dump(token.span);
  output_stream << "\n";
  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dumpNodeToken(Node *node) {
  indent();
  output_stream << "\"token\": ";
  dump(node->token);
  output_stream << ",\n";
}

void JsonDumper::dump(Program *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"Program\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"items\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->items.size(); ++i) {
    indent();
    dump(node->items[i].get());
    if (i + 1 != node->items.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(TopLevelDecl *node) {
  switch (node->kind()) {
  case AstNodeKind::Module:
    dump(node->as<Module>());
    break;
  case AstNodeKind::Function:
    dump(node->as<Function>());
    break;
  case AstNodeKind::TopLevelVarDecl:
    dump(node->as<TopLevelVarDecl>());
    break;
  case AstNodeKind::StructDecl:
    dump(node->as<StructDecl>());
    break;
  case AstNodeKind::TupleStructDecl:
    dump(node->as<TupleStructDecl>());
    break;
  case AstNodeKind::EnumDecl:
    dump(node->as<EnumDecl>());
    break;
  case AstNodeKind::UnionDecl:
    dump(node->as<UnionDecl>());
    break;
  case AstNodeKind::TraitDecl:
    dump(node->as<TraitDecl>());
    break;
  case AstNodeKind::ImplDecl:
    dump(node->as<ImplDecl>());
    break;
  case AstNodeKind::ImportDecl:
    if (!skip_import) {
      dump(node->as<ImportDecl>());
    }
    break;
  default:
    output_stream << "{\n";
    cur_indent++;
    indent();
    output_stream << "\"kind\": \"InvalidTopLevelDecl\"\n";
    cur_indent--;
    indent();
    output_stream << "}";
    break;
  }
}

void JsonDumper::dump(Module *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"Module\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"items\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->items.size(); ++i) {
    indent();
    dump(node->items[i].get());
    if (i + 1 != node->items.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(Function *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"Function\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"is_pub\": " << (node->is_pub ? "true" : "false") << ",\n";

  indent();
  output_stream << "\"decl\": ";
  dump(node->decl.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"body\": ";
  dump(node->body.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(FunctionDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"FunctionDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"parameters\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->parameters.size(); ++i) {
    indent();
    dump(node->parameters[i].get());
    if (i + 1 != node->parameters.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"return_type\": ";
  if (node->return_type)
    dump(node->return_type.get());
  else
    output_stream << "null";
  output_stream << ",\n";

  indent();
  output_stream << "\"extra\": {\n";
  cur_indent++;

  indent();
  output_stream << "\"is_method\": "
                << (node->extra.is_method ? "true" : "false") << ",\n";

  indent();
  output_stream << "\"parent_name\": ";
  // if (node->extra.parent_name)
  //   output_stream << "\"" << *node->extra.parent_name << "\"";
  // else
  output_stream << "null";
  output_stream << ",\n";

  indent();
  output_stream << "\"parent_kind\": \"" << toString(node->extra.parent_kind)
                << "\"\n";

  cur_indent--;
  indent();
  output_stream << "}\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ImplDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ImplDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"type\": ";
  dump(node->type.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"traits\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->traits.size(); ++i) {
    indent();
    dump(node->traits[i].get());
    if (i + 1 != node->traits.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"functions\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->functions.size(); ++i) {
    indent();
    dump(node->functions[i].get());
    if (i + 1 != node->functions.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(VarDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"VarDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"pattern\": ";
  dump(node->pattern.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"type\": ";
  if (node->type) {
    dump(node->type->get());
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"initializer\": ";
  if (node->initializer) {
    dump(node->initializer->get());
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"is_mut\": " << (node->is_mut ? "true" : "false") << ",\n";

  indent();
  output_stream << "\"is_pub\": " << (node->is_pub ? "true" : "false") << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(TopLevelVarDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"TopLevelVarDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"var_decl\": ";
  dump(node->var_decl.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(IfExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"IfExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"condition\": ";
  dump(node->condition.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"then_block\": ";
  dump(node->then_block.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"else_block\": ";
  if (node->else_block) {
    dump(node->else_block->get());
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"structured\": "
                << (node->extra.structured ? "true" : "false") << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(MatchArm *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"MatchArm\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"pattern\": ";
  dump(node->pattern.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"body\": ";
  std::visit([this](auto &arg) { this->dump(arg.get()); }, node->body);
  output_stream << ",\n";

  indent();
  output_stream << "\"guard\": ";
  if (node->guard) {
    dump(node->guard->get());
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(MatchExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"MatchExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"expr\": ";
  dump(node->expr.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"cases\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->cases.size(); ++i) {
    indent();
    dump(node->cases[i].get());
    if (i + 1 != node->cases.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ForExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ForExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"pattern\": ";
  dump(node->pattern.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"iterable\": ";
  dump(node->iterable.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"body\": ";
  dump(node->body.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"label\": ";
  if (node->label) {
    output_stream << "\"" << *node->label << "\"";
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(WhileExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"WhileExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"condition\": ";
  if (node->condition) {
    dump(node->condition->get());
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"continue_expr\": ";
  if (node->continue_expr) {
    dump(node->continue_expr->get());
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"body\": ";
  dump(node->body.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"label\": ";
  if (node->label) {
    output_stream << "\"" << *node->label << "\"";
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ReturnExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ReturnExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"value\": ";
  if (node->value) {
    dump(node->value->get());
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(DeferStmt *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"DeferStmt\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"body\": ";
  std::visit([this](auto &arg) { this->dump(arg.get()); }, node->body);
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}
void JsonDumper::dump(YieldExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"YieldExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"label\": ";
  if (node->label) {
    output_stream << "\"" << *node->label << "\"";
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"value\": ";
  dump(node->value.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(BreakExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"BreakExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"label\": ";
  if (node->label) {
    output_stream << "\"" << *node->label << "\"";
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"value\": ";
  if (node->value) {
    dump(node->value->get());
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ContinueExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ContinueExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"label\": ";
  if (node->label) {
    output_stream << "\"" << *node->label << "\"";
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"value\": ";
  if (node->value) {
    dump(node->value->get());
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ExprStmt *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ExprStmt\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"expr\": ";
  dump(node->expr.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(LiteralExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"LiteralExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"type\": \"";
  dumpLiteralType(node->type);
  output_stream << "\",\n";

  indent();
  output_stream << "\"value\": ";
  std::visit(
      [this](auto &val) {
        if constexpr (std::is_same_v<std::decay_t<decltype(val)>, int>) {
          output_stream << val;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(val)>,
                                            double>) {
          output_stream << val;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(val)>,
                                            std::string>) {
          output_stream << val;
        } else if constexpr (std::is_same_v<std::decay_t<decltype(val)>,
                                            char>) {
          output_stream << "\"" << val << "\"";
        } else if constexpr (std::is_same_v<std::decay_t<decltype(val)>,
                                            bool>) {
          output_stream << (val ? "true" : "false");
        }
      },
      node->value);
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(TupleExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"TupleExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"elements\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->elements.size(); ++i) {
    indent();
    dump(node->elements[i].get());
    if (i + 1 != node->elements.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ArrayExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ArrayExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"elements\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->elements.size(); ++i) {
    indent();
    dump(node->elements[i].get());
    if (i + 1 != node->elements.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(BinaryExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"BinaryExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"op\": \"";
  dumpOperator(node->op);
  output_stream << "\",\n";

  indent();
  output_stream << "\"lhs\": ";
  dump(node->lhs.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"rhs\": ";
  dump(node->rhs.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(UnaryExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"UnaryExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"op\": \"";
  dumpOperator(node->op);
  output_stream << "\",\n";

  indent();
  output_stream << "\"operand\": ";
  dump(node->operand.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(CallExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"CallExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"callee\": \"" << node->callee << "\",\n";

  indent();
  output_stream << "\"arguments\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->arguments.size(); ++i) {
    indent();
    dump(node->arguments[i].get());
    if (i + 1 != node->arguments.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(AssignStatement *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"AssignExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"lhs\": ";
  dump(node->lhs.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"rhs\": ";
  dump(node->rhs.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(AssignOpExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"AssignOpExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"op\": \"";
  dumpOperator(node->op);
  output_stream << "\",\n";

  indent();
  output_stream << "\"lhs\": ";
  dump(node->lhs.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"rhs\": ";
  dump(node->rhs.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(FieldAccessExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"FieldAccessExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"base\": ";
  dump(node->base.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"field\": ";
  std::visit([this](auto &arg) { this->dump(arg.get()); }, node->field);
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(IndexExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"IndexExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"base\": ";
  dump(node->base.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"index\": ";
  dump(node->index.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(RangeExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"RangeExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"start\": ";
  if (node->start) {
    dump(node->start->get());
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"end\": ";
  if (node->end) {
    dump(node->end->get());
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"inclusive\": " << (node->inclusive ? "true" : "false")
                << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(PrimitiveType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"PrimitiveType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"type_kind\": \"";
  dumpPrimitiveTypeKind(node->type_kind);
  output_stream << "\"\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(TupleType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"TupleType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"elements\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->elements.size(); ++i) {
    indent();
    dump(node->elements[i].get());
    if (i + 1 != node->elements.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(FunctionType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"FunctionType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"parameters\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->parameters.size(); ++i) {
    indent();
    dump(node->parameters[i].get());
    if (i + 1 != node->parameters.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"return_type\": ";
  dump(node->return_type.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ReferenceType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ReferenceType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"base\": ";
  dump(node->base.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(SliceType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"SliceType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"base\": ";
  dump(node->base.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ArrayType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ArrayType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"base\": ";
  dump(node->base.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"size\": ";
  dump(node->size.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(TraitType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"TraitType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\"\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(IdentifierType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"IdentifierType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\"\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(StructType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"StructType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\"\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(EnumType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"EnumType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\"\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(UnionType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"UnionType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\"\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ExprType *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ExprType\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"expr\": ";
  dump(node->expr.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(LiteralPattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"LiteralPattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"literal\": ";
  dump(node->literal.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(IdentifierPattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"IdentifierPattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\"\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(WildcardPattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"WildcardPattern\"\n";
  dumpNodeToken(node);

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(TuplePattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"TuplePattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"elements\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->elements.size(); ++i) {
    indent();
    dump(node->elements[i].get());
    if (i + 1 != node->elements.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(PatternField *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"PatternField\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"pattern\": ";
  if (node->pattern) {
    dump(node->pattern->get());
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(RestPattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"RestPattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": ";
  if (node->name) {
    output_stream << "{\n";
    cur_indent++;
    indent();
    output_stream << "\"kind\": \"IdentifierExpr\",\n";
    indent();
    output_stream << "\"name\": \"" << node->name->name << "\"\n";
    cur_indent--;
    indent();
    output_stream << "}";
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(StructPattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"StructPattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": ";
  if (node->name) {
    output_stream << "\"" << *node->name << "\"";
  } else {
    output_stream << "null";
  }
  output_stream << ",\n";

  indent();
  output_stream << "\"fields\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->fields.size(); ++i) {
    indent();
    std::visit([this](auto &arg) { this->dump(arg.get()); }, node->fields[i]);
    if (i + 1 != node->fields.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(SlicePattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"SlicePattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"elements\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->elements.size(); ++i) {
    indent();
    dump(node->elements[i].get());
    if (i + 1 != node->elements.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"is_exhaustive\": "
                << (node->is_exhaustive ? "true" : "false") << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(OrPattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"OrPattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"patterns\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->patterns.size(); ++i) {
    indent();
    dump(node->patterns[i].get());
    if (i + 1 != node->patterns.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ExprPattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ExprPattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"expr\": ";
  dump(node->expr.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(RangePattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"RangePattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"start\": ";
  dump(node->start.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"end\": ";
  dump(node->end.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"inclusive\": " << (node->inclusive ? "true" : "false")
                << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(VariantPattern *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"VariantPattern\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"field\": ";
  if (node->field) {
    std::visit([this](auto &arg) { dump(arg.get()); }, *node->field);
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(TopLevelDeclStmt *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"TopLevelDeclStmt\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"decl\": ";
  dump(node->decl.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ComptimeExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ComptimeExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"expr\": ";
  dump(node->expr.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(BlockExpression *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"BlockExpression\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"statements\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->statements.size(); ++i) {
    indent();
    dump(node->statements[i].get());
    if (i + 1 != node->statements.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(Parameter *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"Parameter\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"pattern\": ";
  dump(node->pattern.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"type\": ";
  dump(node->type.get());
  output_stream << ",\n";

  indent();
  output_stream << "\"trait_bound\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->trait_bound.size(); ++i) {
    indent();
    dump(node->trait_bound[i].get());
    if (i + 1 != node->trait_bound.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"is_mut\": " << (node->is_mut ? "true" : "false") << ",\n";

  indent();
  output_stream << "\"is_comptime\": " << (node->is_comptime ? "true" : "false")
                << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(Expression *node) {
  switch (node->kind()) {
  case AstNodeKind::IfExpr:
    dump(node->as<IfExpr>());
    break;
  case AstNodeKind::MatchExpr:
    dump(node->as<MatchExpr>());
    break;
  case AstNodeKind::ForExpr:
    dump(node->as<ForExpr>());
    break;
  case AstNodeKind::WhileExpr:
    dump(node->as<WhileExpr>());
    break;
  case AstNodeKind::ReturnExpr:
    dump(node->as<ReturnExpr>());
    break;
  case AstNodeKind::BreakExpr:
    dump(node->as<BreakExpr>());
    break;
  case AstNodeKind::ContinueExpr:
    dump(node->as<ContinueExpr>());
    break;
  case AstNodeKind::LiteralExpr:
    dump(node->as<LiteralExpr>());
    break;
  case AstNodeKind::TupleExpr:
    dump(node->as<TupleExpr>());
    break;
  case AstNodeKind::ArrayExpr:
    dump(node->as<ArrayExpr>());
    break;
  case AstNodeKind::BinaryExpr:
    dump(node->as<BinaryExpr>());
    break;
  case AstNodeKind::UnaryExpr:
    dump(node->as<UnaryExpr>());
    break;
  case AstNodeKind::CallExpr:
    dump(node->as<CallExpr>());
    break;
  case AstNodeKind::AssignOpExpr:
    dump(node->as<AssignOpExpr>());
    break;
  case AstNodeKind::FieldAccessExpr:
    dump(node->as<FieldAccessExpr>());
    break;
  case AstNodeKind::IndexExpr:
    dump(node->as<IndexExpr>());
    break;
  case AstNodeKind::RangeExpr:
    dump(node->as<RangeExpr>());
    break;
  case AstNodeKind::IdentifierExpr:
    dump(node->as<IdentifierExpr>());
    break;
  case AstNodeKind::BlockExpression:
    dump(node->as<BlockExpression>());
    break;
  case AstNodeKind::ComptimeExpr:
    dump(node->as<ComptimeExpr>());
    break;
  case AstNodeKind::MLIRAttribute:
    dump(node->as<MLIRAttribute>());
    break;
  case AstNodeKind::MLIROp:
    dump(node->as<MLIROp>());
    break;
  case AstNodeKind::Type:
    dump(node->as<Type>());
    break;
  case AstNodeKind::YieldExpr:
    dump(node->as<YieldExpr>());
    break;
  default:
    output_stream << "{\n";
    cur_indent++;
    indent();
    output_stream << "\"kind\": \"InvalidExpression\"\n";
    cur_indent--;
    indent();
    output_stream << "}";
    break;
  }
}

void JsonDumper::dump(Statement *node) {
  switch (node->kind()) {
  case AstNodeKind::VarDecl:
    dump(node->as<VarDecl>());
    break;
  case AstNodeKind::ExprStmt:
    dump(node->as<ExprStmt>());
    break;
  case AstNodeKind::DeferStmt:
    dump(node->as<DeferStmt>());
    break;
  case AstNodeKind::TopLevelDeclStmt:
    dump(node->as<TopLevelDeclStmt>());
    break;
  case AstNodeKind::AssignStatement:
    dump(node->as<AssignStatement>());
    break;
  default:
    output_stream << "{\n";
    cur_indent++;
    indent();
    output_stream << "\"kind\": \"InvalidStatement\"\n";
    cur_indent--;
    indent();
    output_stream << "}";
    break;
  }
}

void JsonDumper::dump(Type *node) {
  switch (node->kind()) {
  case AstNodeKind::PrimitiveType:
    dump(node->as<PrimitiveType>());
    break;
  case AstNodeKind::TupleType:
    dump(node->as<TupleType>());
    break;
  case AstNodeKind::FunctionType:
    dump(node->as<FunctionType>());
    break;
  case AstNodeKind::ReferenceType:
    dump(node->as<ReferenceType>());
    break;
  case AstNodeKind::SliceType:
    dump(node->as<SliceType>());
    break;
  case AstNodeKind::ArrayType:
    dump(node->as<ArrayType>());
    break;
  case AstNodeKind::TraitType:
    dump(node->as<TraitType>());
    break;
  case AstNodeKind::IdentifierType:
    dump(node->as<IdentifierType>());
    break;
  case AstNodeKind::StructType:
    dump(node->as<StructType>());
    break;
  case AstNodeKind::EnumType:
    dump(node->as<EnumType>());
    break;
  case AstNodeKind::UnionType:
    dump(node->as<UnionType>());
    break;
  case AstNodeKind::ExprType:
    dump(node->as<ExprType>());
    break;
  case AstNodeKind::MLIRType:
    dump(node->as<MLIRType>());
    break;
  default:
    output_stream << "{\n";
    cur_indent++;
    indent();
    output_stream << "\"kind\": \"InvalidType\"\n";
    cur_indent--;
    indent();
    output_stream << "}";
    break;
  }
}

void JsonDumper::dump(Pattern *node) {
  switch (node->kind()) {
  case AstNodeKind::LiteralPattern:
    dump(node->as<LiteralPattern>());
    break;
  case AstNodeKind::IdentifierPattern:
    dump(node->as<IdentifierPattern>());
    break;
  case AstNodeKind::WildcardPattern:
    dump(node->as<WildcardPattern>());
    break;
  case AstNodeKind::TuplePattern:
    dump(node->as<TuplePattern>());
    break;
  case AstNodeKind::StructPattern:
    dump(node->as<StructPattern>());
    break;
  case AstNodeKind::SlicePattern:
    dump(node->as<SlicePattern>());
    break;
  case AstNodeKind::OrPattern:
    dump(node->as<OrPattern>());
    break;
  case AstNodeKind::ExprPattern:
    dump(node->as<ExprPattern>());
    break;
  case AstNodeKind::RangePattern:
    dump(node->as<RangePattern>());
    break;
  case AstNodeKind::VariantPattern:
    dump(node->as<VariantPattern>());
    break;
  default:
    output_stream << "{\n";
    cur_indent++;
    indent();
    output_stream << "\"kind\": \"InvalidPattern\"\n";
    cur_indent--;
    indent();
    output_stream << "}";
    break;
  }
}

void JsonDumper::dump(StructField *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"StructField\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"type\": ";
  dump(node->type.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(StructDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"StructDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"fields\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->fields.size(); ++i) {
    indent();
    dump(node->fields[i].get());
    if (i + 1 != node->fields.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"is_pub\": " << (node->is_pub ? "true" : "false") << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(TupleStructDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"TupleStructDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"fields\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->fields.size(); ++i) {
    indent();
    dump(node->fields[i].get());
    if (i + 1 != node->fields.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"is_pub\": " << (node->is_pub ? "true" : "false") << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(IdentifierExpr *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"IdentifierExpr\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\"\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(FieldsNamed *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"FieldsNamed\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->name.size(); ++i) {
    indent();
    output_stream << "\"" << node->name[i] << "\"";
    if (i + 1 != node->name.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"value\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->value.size(); ++i) {
    indent();
    dump(node->value[i].get());
    if (i + 1 != node->value.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(FieldsUnnamed *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"FieldsUnnamed\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"value\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->value.size(); ++i) {
    indent();
    dump(node->value[i].get());
    if (i + 1 != node->value.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(Variant *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"Variant\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"field\": ";
  if (node->field) {
    std::visit([this](auto &arg) { dump(arg.get()); }, *node->field);
  } else {
    output_stream << "null";
  }
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(UnionDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"UnionDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"fields\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->fields.size(); ++i) {
    indent();
    dump(node->fields[i].get());
    if (i + 1 != node->fields.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"is_pub\": " << (node->is_pub ? "true" : "false") << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(UnionField *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"UnionField\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"type\": ";
  dump(node->type.get());
  output_stream << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(EnumDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"EnumDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"variants\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->variants.size(); ++i) {
    indent();
    dump(node->variants[i].get());
    if (i + 1 != node->variants.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"is_pub\": " << (node->is_pub ? "true" : "false") << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(ImportDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"ImportDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"paths\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->paths.size(); ++i) {
    indent();
    output_stream << "{\n";
    cur_indent++;

    indent();
    output_stream << "\"module\": \"" << node->paths[i].first << "\",\n";

    indent();
    output_stream << "\"alias\": ";
    if (node->paths[i].second) {
      output_stream << "\"" << *node->paths[i].second << "\"";
    } else {
      output_stream << "null";
    }
    output_stream << "\n";

    cur_indent--;
    indent();
    output_stream << "}";
    if (i + 1 != node->paths.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(TraitDecl *node) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"TraitDecl\",\n";
  dumpNodeToken(node);

  indent();
  output_stream << "\"name\": \"" << node->name << "\",\n";

  indent();
  output_stream << "\"functions\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->functions.size(); ++i) {
    indent();
    std::visit([this](auto &arg) { dump(arg.get()); }, node->functions[i]);
    if (i + 1 != node->functions.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"super_traits\": [\n";
  cur_indent++;
  for (size_t i = 0; i < node->super_traits.size(); ++i) {
    indent();
    dump(node->super_traits[i].get());
    if (i + 1 != node->super_traits.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"is_pub\": " << (node->is_pub ? "true" : "false") << "\n";

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(MLIRType *type) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"MLIRType\",\n";
  dumpNodeToken(type);

  indent();
  output_stream << "\"type\": ";
  output_stream << type->type;

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(MLIRAttribute *attr) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"MLIRAttribute\",\n";
  dumpNodeToken(attr);

  indent();
  output_stream << "\"attr\": ";
  output_stream << attr->attribute;

  cur_indent--;
  indent();
  output_stream << "}";
}

void JsonDumper::dump(MLIROp *op) {
  output_stream << "{\n";
  cur_indent++;

  indent();
  output_stream << "\"kind\": \"MLIROp\",\n";
  dumpNodeToken(op);

  indent();
  output_stream << "\"op\": ";
  output_stream << op->op << ",\n";

  indent();
  output_stream << "\"operands\": [\n";
  cur_indent++;
  for (size_t i = 0; i < op->operands.size(); ++i) {
    indent();
    dump(op->operands[i].get());
    if (i + 1 != op->operands.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"result_types\": [\n";
  cur_indent++;
  for (size_t i = 0; i < op->result_types.size(); ++i) {
    indent();
    output_stream << (op->result_types[i]);
    if (i + 1 != op->result_types.size())
      output_stream << ",";
    output_stream << "\n";
  }
  cur_indent--;
  indent();
  output_stream << "],\n";

  indent();
  output_stream << "\"attributes\": [\n";
  cur_indent++;
  int i = 0;
  for (auto &attr : op->attributes) {
    indent();
    output_stream << "{\n";
    cur_indent++;
    indent();
    output_stream << "\"name\": \"" << attr.first << "\",\n";
    indent();
    output_stream << "\"value\": " << attr.second << "\n";
    cur_indent--;
    indent();
    output_stream << "}";
    if (i + 1 != (int)op->attributes.size())
      output_stream << ",";
    output_stream << "\n";
    ++i;
  }
  cur_indent--;
  indent();
  output_stream << "]\n";

  cur_indent--;
  indent();
  output_stream << "}";
}
