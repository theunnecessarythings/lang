#include "json_dumper.hpp"

std::string JsonDumper::token_kind_to_string(TokenKind kind) {
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
  output_stream << "\"kind\": \"" << token_kind_to_string(token.kind)
                << "\",\n";
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
    dump(static_cast<Module *>(node));
    break;
  case AstNodeKind::Function:
    dump(static_cast<Function *>(node));
    break;
  case AstNodeKind::TopLevelVarDecl:
    dump(static_cast<TopLevelVarDecl *>(node));
    break;
  case AstNodeKind::StructDecl:
    dump(static_cast<StructDecl *>(node));
    break;
  case AstNodeKind::TupleStructDecl:
    dump(static_cast<TupleStructDecl *>(node));
    break;
  case AstNodeKind::EnumDecl:
    dump(static_cast<EnumDecl *>(node));
    break;
  case AstNodeKind::UnionDecl:
    dump(static_cast<UnionDecl *>(node));
    break;
  case AstNodeKind::TraitDecl:
    dump(static_cast<TraitDecl *>(node));
    break;
  case AstNodeKind::ImplDecl:
    dump(static_cast<ImplDecl *>(node));
    break;
  case AstNodeKind::ImportDecl:
    if (!skip_import) {
      dump(static_cast<ImportDecl *>(node));
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
  if (node->extra.parent_name)
    output_stream << "\"" << *node->extra.parent_name << "\"";
  else
    output_stream << "null";
  output_stream << ",\n";

  indent();
  output_stream << "\"parent_kind\": \"" << to_string(node->extra.parent_kind)
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
  output_stream << "\"type\": \"" << node->type << "\",\n";

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
  std::visit([this](auto &arg) { this->dump(arg.get()); }, node->then_block);
  output_stream << ",\n";

  indent();
  output_stream << "\"else_block\": ";
  if (node->else_block) {
    std::visit([this](auto &arg) { this->dump(arg.get()); },
               node->else_block.value());
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

void JsonDumper::dump(AssignExpr *node) {
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
    dump(static_cast<IfExpr *>(node));
    break;
  case AstNodeKind::MatchExpr:
    dump(static_cast<MatchExpr *>(node));
    break;
  case AstNodeKind::ForExpr:
    dump(static_cast<ForExpr *>(node));
    break;
  case AstNodeKind::WhileExpr:
    dump(static_cast<WhileExpr *>(node));
    break;
  case AstNodeKind::ReturnExpr:
    dump(static_cast<ReturnExpr *>(node));
    break;
  case AstNodeKind::BreakExpr:
    dump(static_cast<BreakExpr *>(node));
    break;
  case AstNodeKind::ContinueExpr:
    dump(static_cast<ContinueExpr *>(node));
    break;
  case AstNodeKind::LiteralExpr:
    dump(static_cast<LiteralExpr *>(node));
    break;
  case AstNodeKind::TupleExpr:
    dump(static_cast<TupleExpr *>(node));
    break;
  case AstNodeKind::ArrayExpr:
    dump(static_cast<ArrayExpr *>(node));
    break;
  case AstNodeKind::BinaryExpr:
    dump(static_cast<BinaryExpr *>(node));
    break;
  case AstNodeKind::UnaryExpr:
    dump(static_cast<UnaryExpr *>(node));
    break;
  case AstNodeKind::CallExpr:
    dump(static_cast<CallExpr *>(node));
    break;
  case AstNodeKind::AssignExpr:
    dump(static_cast<AssignExpr *>(node));
    break;
  case AstNodeKind::AssignOpExpr:
    dump(static_cast<AssignOpExpr *>(node));
    break;
  case AstNodeKind::FieldAccessExpr:
    dump(static_cast<FieldAccessExpr *>(node));
    break;
  case AstNodeKind::IndexExpr:
    dump(static_cast<IndexExpr *>(node));
    break;
  case AstNodeKind::RangeExpr:
    dump(static_cast<RangeExpr *>(node));
    break;
  case AstNodeKind::IdentifierExpr:
    dump(static_cast<IdentifierExpr *>(node));
    break;
  case AstNodeKind::BlockExpression:
    dump(static_cast<BlockExpression *>(node));
    break;
  case AstNodeKind::ComptimeExpr:
    dump(static_cast<ComptimeExpr *>(node));
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
    dump(static_cast<VarDecl *>(node));
    break;
  case AstNodeKind::ExprStmt:
    dump(static_cast<ExprStmt *>(node));
    break;
  case AstNodeKind::DeferStmt:
    dump(static_cast<DeferStmt *>(node));
    break;
  case AstNodeKind::TopLevelDeclStmt:
    dump(static_cast<TopLevelDeclStmt *>(node));
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
    dump(static_cast<PrimitiveType *>(node));
    break;
  case AstNodeKind::TupleType:
    dump(static_cast<TupleType *>(node));
    break;
  case AstNodeKind::FunctionType:
    dump(static_cast<FunctionType *>(node));
    break;
  case AstNodeKind::ReferenceType:
    dump(static_cast<ReferenceType *>(node));
    break;
  case AstNodeKind::SliceType:
    dump(static_cast<SliceType *>(node));
    break;
  case AstNodeKind::ArrayType:
    dump(static_cast<ArrayType *>(node));
    break;
  case AstNodeKind::TraitType:
    dump(static_cast<TraitType *>(node));
    break;
  case AstNodeKind::IdentifierType:
    dump(static_cast<IdentifierType *>(node));
    break;
  case AstNodeKind::StructType:
    dump(static_cast<StructType *>(node));
    break;
  case AstNodeKind::EnumType:
    dump(static_cast<EnumType *>(node));
    break;
  case AstNodeKind::UnionType:
    dump(static_cast<UnionType *>(node));
    break;
  case AstNodeKind::ExprType:
    dump(static_cast<ExprType *>(node));
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
    dump(static_cast<LiteralPattern *>(node));
    break;
  case AstNodeKind::IdentifierPattern:
    dump(static_cast<IdentifierPattern *>(node));
    break;
  case AstNodeKind::WildcardPattern:
    dump(static_cast<WildcardPattern *>(node));
    break;
  case AstNodeKind::TuplePattern:
    dump(static_cast<TuplePattern *>(node));
    break;
  case AstNodeKind::StructPattern:
    dump(static_cast<StructPattern *>(node));
    break;
  case AstNodeKind::SlicePattern:
    dump(static_cast<SlicePattern *>(node));
    break;
  case AstNodeKind::OrPattern:
    dump(static_cast<OrPattern *>(node));
    break;
  case AstNodeKind::ExprPattern:
    dump(static_cast<ExprPattern *>(node));
    break;
  case AstNodeKind::RangePattern:
    dump(static_cast<RangePattern *>(node));
    break;
  case AstNodeKind::VariantPattern:
    dump(static_cast<VariantPattern *>(node));
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
