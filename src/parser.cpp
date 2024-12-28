#include "parser.hpp"
#include "ast.hpp"
#include "lexer.hpp"
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>

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

static std::unordered_map<std::string, PrimitiveType::PrimitiveTypeKind>
    str_to_primitive_type = {
        {"i8", PrimitiveType::PrimitiveTypeKind::I8},
        {"i16", PrimitiveType::PrimitiveTypeKind::I16},
        {"i32", PrimitiveType::PrimitiveTypeKind::I32},
        {"i64", PrimitiveType::PrimitiveTypeKind::I64},
        {"u8", PrimitiveType::PrimitiveTypeKind::U8},
        {"u16", PrimitiveType::PrimitiveTypeKind::U16},
        {"u32", PrimitiveType::PrimitiveTypeKind::U32},
        {"u64", PrimitiveType::PrimitiveTypeKind::U64},
        {"f32", PrimitiveType::PrimitiveTypeKind::F32},
        {"f64", PrimitiveType::PrimitiveTypeKind::F64},
        {"bool", PrimitiveType::PrimitiveTypeKind::Bool},
        {"char", PrimitiveType::PrimitiveTypeKind::Char},
        {"String", PrimitiveType::PrimitiveTypeKind::String},
        {"type", PrimitiveType::PrimitiveTypeKind::type},
        {"void", PrimitiveType::PrimitiveTypeKind::Void},
};

inline std::unique_ptr<Type> cloneType(Type *type) {
  switch (type->kind()) {
  case AstNodeKind::PrimitiveType: {
    auto t = type->as<PrimitiveType>();
    return std::make_unique<PrimitiveType>(t->token, t->type_kind);
  }
  case AstNodeKind::TupleType: {
    auto t = type->as<TupleType>();
    std::vector<std::unique_ptr<Type>> elements;
    for (auto &elem : t->elements) {
      elements.emplace_back(cloneType(elem.get()));
    }
    return std::make_unique<TupleType>(t->token, std::move(elements));
  }
  case AstNodeKind::FunctionType: {
    auto t = type->as<FunctionType>();
    std::vector<std::unique_ptr<Type>> params;
    auto ret = cloneType(t->return_type.get());
    return std::make_unique<FunctionType>(t->token, std::move(params),
                                          std::move(ret));
  }
  // case AstNodeKind::ArrayType: {
  //   auto t = type->as<ArrayType>();
  //   auto elem = cloneType(t->element_type.get());
  //   return std::make_unique<ArrayType>(t->token, std::move(elem));
  // }
  case AstNodeKind::SliceType: {
    auto t = type->as<SliceType>();
    auto base = cloneType(t->base.get());
    return std::make_unique<SliceType>(t->token, std::move(base));
  }
  case AstNodeKind::StructType: {
    auto t = type->as<StructType>();
    return std::make_unique<StructType>(t->token, t->name);
  }
  case AstNodeKind::EnumType: {
    auto t = type->as<EnumType>();
    return std::make_unique<EnumType>(t->token, t->name);
  }
  case AstNodeKind::UnionType: {
    auto t = type->as<UnionType>();
    return std::make_unique<UnionType>(t->token, t->name);
  }
  case AstNodeKind::TraitType: {
    auto t = type->as<TraitType>();
    return std::make_unique<TraitType>(t->token, t->name);
  }
  case AstNodeKind::ReferenceType: {
    auto t = type->as<ReferenceType>();
    auto base = cloneType(t->base.get());
    return std::make_unique<ReferenceType>(t->token, std::move(base));
  }
  // case AstNodeKind::ExprType: {
  //   auto t = type->as<ExprType>();
  //   return std::make_unique<ExprType>(t->token, t->expr);
  // }
  default:
    break;
  }
}

Token Parser::getErrorToken() {
  TokenSpan span;
  return Token{TokenKind::Dummy, span};
}

Token Parser::unexpectedTokenError(TokenKind &expected, Token &found) {
  context->reportError("Expected " + Lexer::lexeme(expected) + ", got " +
                           Lexer::lexeme(found.kind),
                       &found);
  skipToNextStmt();
  throw std::runtime_error("Unexpected token");
}

Token Parser::invalidTokenError(Token &found) {
  context->reportError("Invalid token " + Lexer::lexeme(found.kind), &found);
  skipToNextStmt();
  throw std::runtime_error("Invalid token");
}

std::optional<Token> Parser::consume() {
  prev_token = current_token;
  current_token = next_token;
  next_token = next2_token;
  next2_token = lexer->next();
  return current_token;
}

Token Parser::consumeKind(TokenKind kind) {
  auto token = peek();
  if (token.has_value() && token.value().kind == kind) {
    return consume().value();
  }
  return unexpectedTokenError(kind, token.value());
}

std::optional<Token> Parser::peek() { return next_token; }

std::optional<Token> Parser::peek2() { return next2_token; }

bool Parser::isPeek(TokenKind kind) {
  return peek().has_value() && peek().value().kind == kind;
}

bool Parser::isPeek2(TokenKind kind) {
  return peek2().has_value() && peek2().value().kind == kind;
}

void Parser::consumeOptionalSemicolon() {
  if (isPeek(TokenKind::Semicolon)) {
    consume();
  }
}

int Parser::bindingPow(const Token &token) {
  switch (token.kind) {
  case TokenKind::PlusEqual:
  case TokenKind::MinusEqual:
  case TokenKind::StarEqual:
  case TokenKind::SlashEqual:
  case TokenKind::PercentEqual:
    return 5;
  case TokenKind::KeywordOr:
    return 10;
  case TokenKind::KeywordAnd:
    return 20;
  case TokenKind::DotDot:
  case TokenKind::DotDotEqual:
    return 25;
  case TokenKind::EqualEqual:
  case TokenKind::BangEqual:
  case TokenKind::Less:
  case TokenKind::Greater:
  case TokenKind::LessEqual:
  case TokenKind::GreaterEqual:
    return 30;
  case TokenKind::Ampersand:
  case TokenKind::Caret:
  case TokenKind::Pipe:
    return 40;
  case TokenKind::LessLess:
  case TokenKind::GreaterGreater:
    return 50;
  case TokenKind::Plus:
  case TokenKind::Minus:
  case TokenKind::PlusPlus:
    return 60;
  case TokenKind::PipePipe:
    return 70;
  case TokenKind::Star:
  case TokenKind::Slash:
  case TokenKind::Percent:
  case TokenKind::StarStar:
    return 70;
  case TokenKind::KeywordNot:
    return 80;
  case TokenKind::Dot:
  case TokenKind::LBracket:
    return 90;
  case TokenKind::LParen:
    return 100;
  default:
    return 0;
  }
}

std::unique_ptr<LiteralExpr>
Parser::parseNumberLiteralExpr(const Token &token) {
  auto number_str = lexer->tokenToString(token);
  auto number = parseNumberLiteral(number_str);
  if (auto val = std::get_if<int>(&number)) {
    return std::make_unique<LiteralExpr>(token, LiteralExpr::LiteralType::Int,
                                         *val);
  } else {
    return std::make_unique<LiteralExpr>(token, LiteralExpr::LiteralType::Float,
                                         std::get<double>(number));
  }
}

std::unique_ptr<Expression> Parser::nud(const Token &token) {
  switch (token.kind) {
  case TokenKind::NumberLiteral: {
    return parseNumberLiteralExpr(token);
  }
  case TokenKind::At: {
    if (isPeek(TokenKind::Identifier) &&
        lexer->tokenToString(peek().value()) == "mlir_attr") {
      return parseMlirAttr();
    } else if (isPeek(TokenKind::Identifier) &&
               lexer->tokenToString(peek().value()) == "mlir_op") {
      return parseMlirOp();
    }
    return parseMlirType(false);
  }
  case TokenKind::StringLiteral:
    return std::make_unique<LiteralExpr>(
        token, LiteralExpr::LiteralType::String, lexer->tokenToString(token));
  case TokenKind::CharLiteral:
    return std::make_unique<LiteralExpr>(token, LiteralExpr::LiteralType::Char,
                                         lexer->tokenToString(token)[1]);
  case TokenKind::LParen: {
    auto expr = parseExpr(0);
    auto comma = peek();
    if (comma.has_value() && comma.value().kind == TokenKind::Comma) {
      return parseTupleExpr(std::move(expr));
    } else {
      consumeKind(TokenKind::RParen);
      return expr;
    }
    break;
  }
  case TokenKind::LBracket:
    return parseArrayExpr();
  case TokenKind::LBrace: {
    return parseBlock(false);
  }
  case TokenKind::Plus:
    return parseExpr(bindingPow(token));
  case TokenKind::Minus:
    return std::make_unique<UnaryExpr>(token, Operator::Sub,
                                       std::move(parseExpr(bindingPow(token))));
  case TokenKind::KeywordNot:
    return std::make_unique<UnaryExpr>(token, Operator::Not,
                                       std::move(parseExpr(bindingPow(token))));
  case TokenKind::Identifier: {
    if (isPeek(TokenKind::LParen)) {
      return parseCallExpr();
    } else {
      auto str = lexer->tokenToString(token);
      if (str == "true" || str == "false") {
        return std::make_unique<LiteralExpr>(
            token, LiteralExpr::LiteralType::Bool, str == "true");
      }
      return std::make_unique<IdentifierExpr>(token,
                                              lexer->tokenToString(token));
    }
  }
  case TokenKind::DotDot:
  case TokenKind::DotDotEqual: {
    std::optional<std::unique_ptr<Expression>> right_expr = std::nullopt;
    if (!isPeek(TokenKind::RBracket) && !isPeek(TokenKind::LBrace) &&
        !isPeek(TokenKind::RParen)) {
      right_expr = parseExpr(bindingPow(token));
    }
    return std::make_unique<RangeExpr>(token, std::nullopt,
                                       std::move(right_expr),
                                       token.kind == TokenKind::DotDotEqual);
  }
  case TokenKind::KeywordMatch:
    return parseMatchExpr(false);
  case TokenKind::KeywordIf:
    return parseIfExpr(false);
  case TokenKind::KeywordWhile:
    return parseWhileExpr(false);
  case TokenKind::KeywordFor:
    return parseForExpr(false);
  case TokenKind::KeywordBreak:
    return parseBreakExpr(false);
  case TokenKind::KeywordContinue:
    return parseContinueExpr(false);
  case TokenKind::KeywordYield:
    return parseYieldExpr(false);
  case TokenKind::KeywordComptime: {
    auto expr = parseExpr(0);
    return std::make_unique<ComptimeExpr>(token, std::move(expr));
  }
  case TokenKind::KeywordReturn:
    return parseReturnExpr(false);
  default:
    context->reportError("Invalid token " + Lexer::lexeme(token.kind), &token);
    return std::make_unique<InvalidExpression>(token);
  }
}

std::unique_ptr<Expression> Parser::led(std::unique_ptr<Expression> left,
                                        Token &op) {
  auto precedence = bindingPow(op);
  switch (op.kind) {
  case TokenKind::PlusEqual:
  case TokenKind::MinusEqual:
  case TokenKind::StarEqual:
  case TokenKind::SlashEqual:
  case TokenKind::PercentEqual:
  case TokenKind::AmpersandEqual:
  case TokenKind::PipeEqual:
  case TokenKind::CaretEqual:
  case TokenKind::LessLessEqual:
  case TokenKind::GreaterGreaterEqual:
    return std::make_unique<AssignOpExpr>(
        op, tokenToOperator(op), std::move(left), parseExpr(precedence - 1));
  case TokenKind::Dot: {
    auto right = consume();
    if (right->kind == TokenKind::NumberLiteral) {
      auto value = parseNumberLiteral(lexer->tokenToString(right.value()));
      if (auto val = std::get_if<int>(&value)) {
        auto expr = std::make_unique<LiteralExpr>(
            op, LiteralExpr::LiteralType::Int, *val);
        return std::make_unique<FieldAccessExpr>(op, std::move(left),
                                                 std::move(expr));
      }
      context->reportError("Expected integer literal found " +
                               lexer->tokenToString(right.value()),
                           &right.value());
      return std::make_unique<InvalidExpression>(op);
    } else if (right->kind == TokenKind::Identifier) {
      if (isPeek(TokenKind::LParen)) {
        auto call_expr = parseCallExpr();
        return std::make_unique<FieldAccessExpr>(op, std::move(left),
                                                 std::move(call_expr));
      } else if (isPeek(TokenKind::NumberLiteral)) {
        auto number = consume();
        auto value = parseNumberLiteral(lexer->tokenToString(number.value()));
        if (auto val = std::get_if<int>(&value)) {
          auto expr = std::make_unique<LiteralExpr>(
              op, LiteralExpr::LiteralType::Int, *val);
          return std::make_unique<FieldAccessExpr>(op, std::move(left),
                                                   std::move(expr));
        }
        context->reportError("Expected integer literal found " +
                                 lexer->tokenToString(number.value()),
                             &number.value());
        return std::make_unique<InvalidExpression>(op);
      } else {
        auto right_expr =
            std::make_unique<IdentifierExpr>(op, lexer->tokenToString(*right));
        return std::make_unique<FieldAccessExpr>(op, std::move(left),
                                                 std::move(right_expr));
      }
    }
    context->reportError("Expected identifier or number literal found " +
                             Lexer::lexeme(right->kind),
                         &right.value());
    return std::make_unique<InvalidExpression>(op);
  }
  case TokenKind::LBracket: {
    auto index_expr = parseExpr(0);
    consumeKind(TokenKind::RBracket);
    return std::make_unique<IndexExpr>(op, std::move(left),
                                       std::move(index_expr));
  }
  case TokenKind::DotDot:
  case TokenKind::DotDotEqual: {
    std::optional<std::unique_ptr<Expression>> right_expr = std::nullopt;
    if (!isPeek(TokenKind::RBracket) && !isPeek(TokenKind::LBrace) &&
        !isPeek(TokenKind::RParen)) {
      right_expr = parseExpr(bindingPow(op));
    }
    bool is_inclusive = op.kind == TokenKind::DotDotEqual;
    return std::make_unique<RangeExpr>(op, std::move(left),
                                       std::move(right_expr), is_inclusive);
  }
  default: {
    auto right = parseExpr(precedence);
    Operator op_kind = tokenToOperator(op);
    return std::make_unique<BinaryExpr>(op, op_kind, std::move(left),
                                        std::move(right));
  }
  }
}

std::unique_ptr<Program> Parser::parseSingleSource(std::string &path) {
  static int file_id = 1;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (std::error_code ec = fileOrErr.getError()) {
    context->reportError("Could not open file " + path);
    std::vector<std::unique_ptr<TopLevelDecl>> top_level_decls;
    return std::make_unique<Program>(getErrorToken(),
                                     std::move(top_level_decls));
  }
  auto buffer = fileOrErr->get()->getBuffer();
  context->source_mgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  std::unique_ptr<Lexer> lexer =
      std::make_unique<Lexer>(buffer.str(), file_id + 1);
  std::unique_ptr<Parser> parser =
      std::make_unique<Parser>(std::move(lexer), context, file_id + 1);
  return parser->parseProgram();
}

void Parser::loadBuiltins() {
  namespace fs = std::filesystem;
  std::string path =
      "/mnt/ubuntu/home/sreeraj/Documents/mlir-lang-cpp/std/builtin";
  for (const auto &entry : fs::directory_iterator(path)) {
    auto file_path = entry.path().string();
    if (fs::is_regular_file(entry.path())) {
      if (!isFileLoaded(context->source_mgr, file_path)) {
        auto tree = parseSingleSource(file_path);
        for (auto &decl : tree->items) {
          top_level_decls.emplace_back(std::move(decl));
        }
      }
    }
  }
}

// program = top_level_decl*
std::unique_ptr<Program> Parser::parseProgram(bool load_builtins) {
  NEW_SCOPE();
  if (load_builtins)
    loadBuiltins();
  while (true) {
    if (isPeek(TokenKind::Eof)) {
      break;
    }
    std::unique_ptr<TopLevelDecl> top_level_decl;
    try {
      top_level_decl = parseTopLevelDecl();
    } catch (std::runtime_error &e) {
      return std::make_unique<Program>(current_token.value(),
                                       std::move(top_level_decls));
    }
    top_level_decls.emplace_back(std::move(top_level_decl));
  }

  return std::make_unique<Program>(next_token.value(),
                                   std::move(top_level_decls));
}

void Parser::skipToNextTopLevelDecl() {
  while (!isPeek(TokenKind::Eof)) {
    if (isPeek(TokenKind::KeywordFn) || isPeek(TokenKind::KeywordStruct) ||
        isPeek(TokenKind::KeywordEnum) || isPeek(TokenKind::KeywordImpl) ||
        isPeek(TokenKind::KeywordUnion) || isPeek(TokenKind::KeywordTrait)) {
      break;
    }
    consume();
  }
}

void Parser::skipToNextStmt() {
  while (!isPeek(TokenKind::Eof)) {
    if (isPeek(TokenKind::Semicolon)) {
      consume();
      break;
    }
    consume();
  }
}

bool Parser::isFileLoaded(llvm::SourceMgr &sourceMgr,
                          const std::string &filePath) {
  for (unsigned i = 1; i <= sourceMgr.getNumBuffers(); ++i) {
    llvm::StringRef bufferIdentifier =
        sourceMgr.getMemoryBuffer(i)->getBufferIdentifier();
    if (bufferIdentifier == filePath)
      return true;
  }
  return false;
}

// import_decl -> 'import' (ident ('.' ident)* ('as' ident)?)*
// eg: import std.io, std.math.rand as rand, std.fs
std::unique_ptr<ImportDecl> Parser::parseImportDecl() {
  auto import_token = consumeKind(TokenKind::KeywordImport);
  std::vector<ImportDecl::Path> paths;
  while (true) {
    auto path_token = consumeKind(TokenKind::StringLiteral);
    auto path = lexer->tokenToString(path_token);
    path = "../" + path.substr(1, path.size() - 2);

    if (isPeek(TokenKind::KeywordAs)) {
      consume();
      auto alias = consumeKind(TokenKind::Identifier);
      paths.emplace_back(ImportDecl::Path{path, lexer->tokenToString(alias)});
    } else {
      paths.emplace_back(ImportDecl::Path{path, std::nullopt});
    }

    // Import the parsed path
    path += ".lang";
    if (!isFileLoaded(context->source_mgr, path)) {
      auto tree = parseSingleSource(path);
      for (auto &decl : tree->items) {
        top_level_decls.emplace_back(std::move(decl));
      }
    }
    if (!isPeek(TokenKind::Comma))
      break;
    else
      consume();
  }
  consumeKind(TokenKind::Semicolon);
  return std::make_unique<ImportDecl>(import_token, std::move(paths));
}

std::unordered_set<Attribute> Parser::parseAttributes() {
  std::unordered_set<Attribute> attrs;
  if (isPeek(TokenKind::At)) {
    // @[attrs, ...]
    consumeKind(TokenKind::At);
    consumeKind(TokenKind::LBracket);
    while (!isPeek(TokenKind::RBracket)) {
      auto token = consumeKind(TokenKind::Identifier);
      if (!isPeek(TokenKind::RBracket)) {
        consumeKind(TokenKind::Comma);
      }
      attrs.insert(tokenToAttribute(token));
    }
    consumeKind(TokenKind::RBracket);
  }
  return attrs;
}

// top_level_decl = function_decl | struct_decl | enum_decl | impl_decl |
// union_decl | trait_decl
std::unique_ptr<TopLevelDecl> Parser::parseTopLevelDecl() {
  auto attrs = parseAttributes();
  bool is_pub = false;
  if (isPeek(TokenKind::KeywordPub)) {
    consume();
    is_pub = true;
  }
  if (isPeek(TokenKind::KeywordFn)) {
    auto fn = parseFunction(is_pub, std::move(attrs));
    return fn;
  } else if (isPeek(TokenKind::KeywordStruct)) {
    consume();
    if (isPeek2(TokenKind::LParen)) {
      auto tuple_struct = parseTupleStructDecl(is_pub);
      context->declareTupleStruct(tuple_struct->name, tuple_struct.get());
      return tuple_struct;
    }
    auto struct_decl = parseStructDecl(is_pub);
    context->declareStruct(struct_decl->name, struct_decl.get());
    return struct_decl;
  } else if (isPeek(TokenKind::KeywordImport)) {
    return parseImportDecl();
  } else if (isPeek(TokenKind::KeywordEnum)) {
    auto enum_decl = parseEnumDecl(is_pub);
    context->declareEnum(enum_decl->name, enum_decl.get());
    return enum_decl;
  } else if (isPeek(TokenKind::KeywordImpl)) {
    return parseImplDecl();
  } else if (isPeek(TokenKind::KeywordVar) || isPeek(TokenKind::KeywordConst)) {
    auto token = peek();
    auto var_decl = parseVarDecl(is_pub);
    return std::make_unique<TopLevelVarDecl>(token.value(),
                                             std::move(var_decl));
  }
  // else if (is_peek(TokenKind::KeywordUnion)) {
  //   return parse_union_decl();
  // }
  else if (isPeek(TokenKind::KeywordTrait)) {
    auto trait_decl = parseTraitDecl(is_pub);
    context->declareTrait(trait_decl->name, trait_decl.get());
    return trait_decl;
  } else {
    auto token = consume();
    context->reportError("Expected top level decl found " +
                             Lexer::lexeme(token->kind),
                         &token.value());
    skipToNextTopLevelDecl();
    return std::make_unique<InvalidTopLevelDecl>(token.value());
  }
}

// pratt parser
std::unique_ptr<Expression> Parser::parseExpr(int precedence) {
  auto token = consume();
  auto left = nud(token.value());
  if (left->kind() == AstNodeKind::BlockExpression) {
    return left;
  }
  while (precedence < bindingPow(peek().value())) {
    token = consume();
    auto new_left = led(std::move(left), token.value());
    left = std::move(new_left);
  }
  return left;
}

// tuple_expr = '(' expr (',' expr)* ')'
std::unique_ptr<Expression>
Parser::parseTupleExpr(std::unique_ptr<Expression> first_expr) {
  std::vector<std::unique_ptr<Expression>> exprs;
  exprs.emplace_back(std::move(first_expr));
  while (isPeek(TokenKind::Comma)) {
    consume();
    if (isPeek(TokenKind::RParen)) {
      break;
    }
    auto expr = parseExpr(0);
    exprs.emplace_back(std::move(expr));
  }
  auto token = consumeKind(TokenKind::RParen);
  return std::make_unique<TupleExpr>(token, std::move(exprs));
}

// array_expr = '[' expr (',' expr)* ']'
std::unique_ptr<Expression> Parser::parseArrayExpr() {
  std::vector<std::unique_ptr<Expression>> exprs;
  while (!isPeek(TokenKind::RBracket)) {
    auto expr = parseExpr(0);
    exprs.emplace_back(std::move(expr));
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  auto token = consumeKind(TokenKind::RBracket);
  return std::make_unique<ArrayExpr>(token, std::move(exprs));
}

// function = 'fn' identifier '(' params ')' type block
std::unique_ptr<Function>
Parser::parseFunction(bool is_pub, std::unordered_set<Attribute> attrs) {
  NEW_SCOPE();
  consumeKind(TokenKind::KeywordFn);
  auto name = consumeKind(TokenKind::Identifier);
  auto name_str = lexer->tokenToString(name);
  auto params = parseParams();
  auto return_type = parseType();
  auto block = parseBlock();
  return std::make_unique<Function>(name, std::move(name_str),
                                    std::move(params), std::move(return_type),
                                    std::move(block), std::move(attrs), is_pub);
}

TraitDecl::Method Parser::parseTraitMethod() {
  auto attrs = parseAttributes();
  auto fn_token = consumeKind(TokenKind::KeywordFn);
  auto name = consumeKind(TokenKind::Identifier);
  auto name_str = lexer->tokenToString(name);
  auto params = parseParams();
  auto return_type = parseType();
  if (isPeek(TokenKind::Semicolon)) {
    consume();
    return std::make_unique<FunctionDecl>(fn_token, std::move(name_str),
                                          std::move(params),
                                          std::move(return_type));
  }
  auto block = parseBlock();
  return std::make_unique<Function>(fn_token, std::move(name_str),
                                    std::move(params), std::move(return_type),
                                    std::move(block), std::move(attrs));
}

// params = '(' (param (',' param)*)? ')'
std::vector<std::unique_ptr<Parameter>> Parser::parseParams() {
  std::vector<std::unique_ptr<Parameter>> params;
  consumeKind(TokenKind::LParen);
  int comptime_idx = -1;
  while (!isPeek(TokenKind::RParen)) {
    if (isPeek(TokenKind::Star)) {
      consume();
      if (comptime_idx != -1) {
        context->reportError("Multiple comptime separator in function params",
                             &current_token.value());
      }
      comptime_idx = params.size();
    } else {
      auto param = parseParam();
      params.emplace_back(std::move(param));
    }
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  consumeKind(TokenKind::RParen);

  // Set comptime flag for all parameters before comptime separator
  for (int i = 0; i < comptime_idx; i++) {
    params[i]->is_comptime = true;
  }
  return params;
}

// param -> ("mut")? identifier ":" type
std::unique_ptr<Parameter> Parser::parseParam() {
  bool is_mut = false;
  if (isPeek(TokenKind::KeywordMut)) {
    consume();
    is_mut = true;
  }
  auto token = peek();
  auto pattern = parsePattern();
  consumeKind(TokenKind::Colon);
  auto type = parseType();

  std::vector<std::unique_ptr<Type>> trait_bounds;
  // if type is primitive type and it is "type" then check for impl Trait
  if (type->kind() == AstNodeKind::PrimitiveType) {
    auto primitive_type = type->as<PrimitiveType>();
    if (primitive_type->type_kind == PrimitiveType::PrimitiveTypeKind::type) {
      if (isPeek(TokenKind::KeywordImpl)) {
        consume();
        while (true) {
          auto trait = parseType();
          trait_bounds.emplace_back(std::move(trait));
          if (isPeek(TokenKind::Plus))
            consume();
          else
            break;
        }
        return std::make_unique<Parameter>(
            token.value(), std::move(pattern), std::move(type),
            std::move(trait_bounds), is_mut, false);
      }
    }
  }
  return std::make_unique<Parameter>(token.value(), std::move(pattern),
                                     std::move(type), std::move(trait_bounds),
                                     is_mut, false);
}

// block_expr = '{' stmt* '}'
std::unique_ptr<BlockExpression> Parser::parseBlock(bool consume_lbrace) {
  NEW_SCOPE();
  std::vector<std::unique_ptr<Statement>> stmts;
  if (consume_lbrace)
    consumeKind(TokenKind::LBrace);
  while (!isPeek(TokenKind::RBrace)) {
    auto stmt = parseStatement();
    stmts.emplace_back(std::move(stmt));
  }
  auto token = consumeKind(TokenKind::RBrace);
  return std::make_unique<BlockExpression>(token, std::move(stmts));
}

// struct_decl = 'struct' identifier '{' struct_field* '}'
std::unique_ptr<StructDecl> Parser::parseStructDecl(bool is_pub) {
  auto name = consumeKind(TokenKind::Identifier);
  consumeKind(TokenKind::LBrace);
  std::vector<std::unique_ptr<StructField>> fields;
  while (!isPeek(TokenKind::RBrace)) {
    auto field = parseStructField();
    fields.emplace_back(std::move(field));
  }
  consumeKind(TokenKind::RBrace);
  return std::make_unique<StructDecl>(name, lexer->tokenToString(name),
                                      std::move(fields), is_pub);
}

// struct_field = identifier ':' type ','
std::unique_ptr<StructField> Parser::parseStructField() {
  auto name = consumeKind(TokenKind::Identifier);
  consumeKind(TokenKind::Colon);
  auto type = parseType();
  if (isPeek(TokenKind::Comma)) {
    consume();
  }
  return std::make_unique<StructField>(name, lexer->tokenToString(name),
                                       std::move(type));
}

// tuple_struct_decl = 'struct' identifier '(' type (',' type)* ')'
std::unique_ptr<TupleStructDecl> Parser::parseTupleStructDecl(bool is_pub) {
  auto name = consumeKind(TokenKind::Identifier);
  consumeKind(TokenKind::LParen);
  std::vector<std::unique_ptr<Type>> fields;
  while (true) {
    auto field = parseType();
    fields.emplace_back(std::move(field));
    if (isPeek(TokenKind::Comma)) {
      consume();
    } else {
      break;
    }
  }
  consumeKind(TokenKind::RParen);
  return std::make_unique<TupleStructDecl>(name, lexer->tokenToString(name),
                                           std::move(fields), is_pub);
}

// // union_decl = 'union' identifier '{' union_field* '}'
// std::shared_ptr<UnionDecl> parse_union_decl() {
//   auto union_token = consume_kind(TokenKind::KeywordUnion);
//   auto name = consume_kind(TokenKind::Identifier);
//   consume_kind(TokenKind::LBrace);
//   std::vector<std::shared_ptr<UnionField>> fields;
//   while (!is_peek(TokenKind::RBrace)) {
//     auto field = parse_union_field();
//     fields.push_back(field);
//   }
//   consume_kind(TokenKind::RBrace);
//   return std::make_shared<UnionDecl>(union_token, name, fields);
// }
//
// // union_field = identifier ':' type ','
// std::shared_ptr<UnionField> parse_union_field() {
//   auto name = consume_kind(TokenKind::Identifier);
//   consume_kind(TokenKind::Colon);
//   auto type = parse_type();
//   if (is_peek(TokenKind::Comma)) {
//     consume();
//   }
//   return std::make_shared<UnionField>(name, type);
// }

// enum_decl = 'enum' identifier '{' enum_variant* '}'
std::unique_ptr<EnumDecl> Parser::parseEnumDecl(bool is_pub) {
  consumeKind(TokenKind::KeywordEnum);
  auto name = consumeKind(TokenKind::Identifier);
  consumeKind(TokenKind::LBrace);
  std::vector<std::unique_ptr<Variant>> variants;
  while (!isPeek(TokenKind::RBrace)) {
    auto variant = parseEnumVariant();
    variants.emplace_back(std::move(variant));
    if (!isPeek(TokenKind::RBrace))
      consumeKind(TokenKind::Comma);
  }
  consumeKind(TokenKind::RBrace);
  return std::make_unique<EnumDecl>(name, std::move(lexer->tokenToString(name)),
                                    std::move(variants), is_pub);
}

std::unique_ptr<FieldsUnnamed> Parser::parseFieldUnnamed() {
  std::vector<std::unique_ptr<Type>> fields;
  auto token = consumeKind(TokenKind::LParen);
  while (!isPeek(TokenKind::RParen)) {
    auto field = parseType();
    fields.emplace_back(std::move(field));
    if (!isPeek(TokenKind::RParen))
      consumeKind(TokenKind::Comma);
  }
  consumeKind(TokenKind::RParen);
  return std::make_unique<FieldsUnnamed>(token, std::move(fields));
}

std::unique_ptr<FieldsNamed> Parser::parseFieldNamed() {
  std::vector<std::string> names;
  std::vector<std::unique_ptr<Type>> fields;
  auto token = consumeKind(TokenKind::LBrace);
  while (!isPeek(TokenKind::RBrace)) {
    auto name = consumeKind(TokenKind::Identifier);
    names.emplace_back(lexer->tokenToString(name));
    consumeKind(TokenKind::Colon);
    auto field = parseType();
    fields.emplace_back(std::move(field));
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  consumeKind(TokenKind::RBrace);
  return std::make_unique<FieldsNamed>(token, std::move(names),
                                       std::move(fields));
}

// enum_variant = (type | tuple_struct_decl | struct_decl)
std::unique_ptr<Variant> Parser::parseEnumVariant() {
  auto name = consumeKind(TokenKind::Identifier);
  if (isPeek(TokenKind::LBrace)) {
    return std::make_unique<Variant>(name, lexer->tokenToString(name),
                                     parseFieldNamed());
  } else if (isPeek(TokenKind::LParen)) {
    return std::make_unique<Variant>(name, lexer->tokenToString(name),
                                     parseFieldUnnamed());
  } else if (isPeek(TokenKind::Equal)) {
    consume();
    auto expr = parseExpr(0);
    return std::make_unique<Variant>(name, lexer->tokenToString(name),
                                     std::move(expr));
  } else {
    return std::make_unique<Variant>(name, lexer->tokenToString(name),
                                     std::nullopt);
  }
}

// impl_decl = 'impl' type '{' function_decl* '}'
std::unique_ptr<ImplDecl> Parser::parseImplDecl() {
  NEW_SCOPE();
  auto impl_token = consumeKind(TokenKind::KeywordImpl);
  auto type = parseType();
  // auto type = consumeKind(TokenKind::Identifier);

  std::vector<std::unique_ptr<Type>> traits;
  if (isPeek(TokenKind::Colon)) {
    consume();
    auto trait = parseType();
    traits.emplace_back(std::move(trait));
    // while (!isPeek(TokenKind::LBrace)) {
    //   auto trait = parseType();
    //   traits.emplace_back(std::move(trait));
    //   if (isPeek(TokenKind::Plus)) {
    //     consume();
    //   }
    // }
  }
  consumeKind(TokenKind::LBrace);
  std::vector<std::unique_ptr<Function>> functions;
  while (!isPeek(TokenKind::RBrace)) {
    auto attrs = parseAttributes();
    bool is_pub = false;
    if (isPeek(TokenKind::KeywordPub)) {
      consume();
      is_pub = true;
    }
    auto function = parseFunction(is_pub, std::move(attrs));
    functions.emplace_back(std::move(function));
  }
  consumeKind(TokenKind::RBrace);
  return std::make_unique<ImplDecl>(impl_token, std::move(type),
                                    std::move(traits), std::move(functions));
}

// trait_decl = 'trait' identifier '{' function_decl* '}'
std::unique_ptr<TraitDecl> Parser::parseTraitDecl(bool is_pub) {
  consumeKind(TokenKind::KeywordTrait);
  auto name = consumeKind(TokenKind::Identifier);
  std::vector<std::unique_ptr<Type>> traits;
  if (isPeek(TokenKind::Colon)) {
    consume();
    while (!isPeek(TokenKind::LBrace)) {
      auto trait = parseType();
      traits.emplace_back(std::move(trait));
      if (isPeek(TokenKind::Plus)) {
        consume();
      }
    }
  }
  consumeKind(TokenKind::LBrace);
  std::vector<TraitDecl::Method> functions;
  while (!isPeek(TokenKind::RBrace)) {
    auto function = parseTraitMethod();
    functions.emplace_back(std::move(function));
  }
  consumeKind(TokenKind::RBrace);
  return std::make_unique<TraitDecl>(name, lexer->tokenToString(name),
                                     std::move(functions), std::move(traits),
                                     is_pub);
}

// type = primitive_type | tuple_type | array_type | function_type |
//          pointer_type | reference_type | identifier | expr_type
std::unique_ptr<Type> Parser::parseType() {
  // if (is_peek(TokenKind::KeywordFn)) {
  //   return parse_function_type();
  // } else
  if (isPeek(TokenKind::At)) {
    return parseMlirType();
  } else if (isPeek(TokenKind::KeywordTuple)) {
    return parseTupleType();
  } else if (isPeek(TokenKind::LBracket)) {
    return parseArrayType();
  }
  if (isPeek(TokenKind::Identifier)) {
    auto token_str = lexer->tokenToString(peek().value());
    if (str_to_primitive_type.find(token_str) != str_to_primitive_type.end()) {
      auto token = consume();
      return std::make_unique<PrimitiveType>(
          token.value(), str_to_primitive_type.at(token_str));
    }
    if (isPeek2(TokenKind::RParen) || isPeek2(TokenKind::Comma) ||
        isPeek2(TokenKind::LBrace) || isPeek2(TokenKind::Equal) ||
        isPeek2(TokenKind::Colon) || isPeek2(TokenKind::RBrace)) {
      auto token = consume();
      auto type = context->var_table.lookup(token_str);
      if (type) {
        if (type->kind() == AstNodeKind::PrimitiveType &&
            type->as<PrimitiveType>()->type_kind ==
                PrimitiveType::PrimitiveTypeKind::type) {
          return std::make_unique<IdentifierType>(token.value(), token_str);
        }
        return cloneType(type);
      }
      return std::make_unique<IdentifierType>(token.value(), token_str);
    }
  }
  auto expr = parseExpr(0);
  return std::make_unique<ExprType>(expr->token, std::move(expr));
}

// mlir_type = "@mlir_type(" type_str ")"
std::unique_ptr<Type> Parser::parseMlirType(bool consume_at) {
  if (consume_at)
    consumeKind(TokenKind::At);
  std::vector<std::unique_ptr<Expression>> parameters;
  auto mlir_type = consumeKind(TokenKind::Identifier);
  if (lexer->tokenToString(mlir_type) != "mlir_type") {
    context->reportError("Expected mlir_type found " +
                             lexer->tokenToString(mlir_type),
                         &mlir_type);
  }
  consumeKind(TokenKind::LParen);
  auto type_str = consumeKind(TokenKind::StringLiteral);
  if (isPeek(TokenKind::Comma)) {
    consume();
    // parse parameters -> [expr, ...]
    consumeKind(TokenKind::LBracket);
    while (!isPeek(TokenKind::RBracket)) {
      auto expr = parseExpr(0);
      parameters.emplace_back(std::move(expr));
      if (isPeek(TokenKind::Comma)) {
        consume();
      }
    }
    consumeKind(TokenKind::RBracket);
  }
  consumeKind(TokenKind::RParen);
  return std::make_unique<MLIRType>(type_str, lexer->tokenToString(type_str),
                                    std::move(parameters));
}

// mlir_attr = "@mlir_attr(" attr_str ")"
std::unique_ptr<MLIRAttribute> Parser::parseMlirAttr() {
  // consume_kind(TokenKind::At);
  auto mlir_attr = consumeKind(TokenKind::Identifier);
  if (lexer->tokenToString(mlir_attr) != "mlir_attr") {
    context->reportError("Expected mlir_attr found " +
                             lexer->tokenToString(mlir_attr),
                         &mlir_attr);
  }
  consumeKind(TokenKind::LParen);
  auto attr_str = consumeKind(TokenKind::StringLiteral);
  consumeKind(TokenKind::RParen);
  return std::make_unique<MLIRAttribute>(attr_str,
                                         lexer->tokenToString(attr_str));
}

// eg: @mlir_op("addi", [lhs, rhs], [attr1], ["i32"])
std::unique_ptr<MLIROp> Parser::parseMlirOp() {
  // consume_kind(TokenKind::At);
  auto mlir_op = consumeKind(TokenKind::Identifier);
  if (lexer->tokenToString(mlir_op) != "mlir_op") {
    context->reportError(
        "Expected mlir_op found " + lexer->tokenToString(mlir_op), &mlir_op);
  }
  consumeKind(TokenKind::LParen);
  auto op_str = consumeKind(TokenKind::StringLiteral);
  std::vector<std::unique_ptr<Expression>> operands;
  std::unordered_map<std::string, std::string> attributes;
  std::vector<std::unique_ptr<Type>> result_types;

  consumeKind(TokenKind::Comma);
  consumeKind(TokenKind::LBracket);
  while (!isPeek(TokenKind::RBracket)) {
    auto operand = parseExpr(0);
    operands.emplace_back(std::move(operand));
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  consumeKind(TokenKind::RBracket);
  consumeKind(TokenKind::Comma);
  consumeKind(TokenKind::LBrace);
  while (!isPeek(TokenKind::RBrace)) {
    auto name = consumeKind(TokenKind::Identifier);
    consumeKind(TokenKind::Colon);
    auto attr = consumeKind(TokenKind::StringLiteral);
    attributes[lexer->tokenToString(name)] = lexer->tokenToString(attr);
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  consumeKind(TokenKind::RBrace);
  consumeKind(TokenKind::Comma);
  consumeKind(TokenKind::LBracket);
  while (!isPeek(TokenKind::RBracket)) {
    auto type = parseType();
    result_types.emplace_back(std::move(type));
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  consumeKind(TokenKind::RBracket);
  consumeKind(TokenKind::RParen);
  return std::make_unique<MLIROp>(op_str, lexer->tokenToString(op_str),
                                  std::move(operands), std::move(attributes),
                                  std::move(result_types));
}

// tuple_type = '(' type (',' type)* ')'
std::unique_ptr<Type> Parser::parseTupleType() {
  consumeKind(TokenKind::KeywordTuple);
  auto token = consumeKind(TokenKind::LParen);
  if (!isPeek2(TokenKind::Comma)) {
    // not a tuple type
    auto type = parseType();
    consumeKind(TokenKind::RParen);
    return type;
  }
  std::vector<std::unique_ptr<Type>> types;
  while (!isPeek(TokenKind::RParen)) {
    auto type = parseType();
    types.emplace_back(std::move(type));
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  consumeKind(TokenKind::RParen);
  return std::make_unique<TupleType>(token, std::move(types));
}

// array_type = slice_type | fixed_array_type
// slice_type = '[]' type
// fixed_array_type = '['expr']' type
std::unique_ptr<Type> Parser::parseArrayType() {
  if (isPeek2(TokenKind::RBracket)) {
    auto token = consume();
    consumeKind(TokenKind::RBracket);
    auto type = parseType();
    return std::make_unique<SliceType>(token.value(), std::move(type));
  } else {
    auto token = consume();
    auto expr = parseExpr(0);
    consumeKind(TokenKind::RBracket);
    auto type = parseType();
    return std::make_unique<ArrayType>(token.value(), std::move(type),
                                       std::move(expr));
  }
}

// // function_type = 'fn' '(' type (',' type)* ')' type
// std::shared_ptr<Type> parse_function_type() {
//   auto fn_token = consume_kind(TokenKind::KeywordFn);
//   consume_kind(TokenKind::LParen);
//   std::vector<std::shared_ptr<Type>> parameters;
//   while (!is_peek(TokenKind::RParen)) {
//     auto type = parse_type();
//     parameters.push_back(type);
//     if (is_peek(TokenKind::Comma)) {
//       consume();
//     }
//   }
//   consume_kind(TokenKind::RParen);
//   auto return_type = parse_type();
//   return std::make_shared<FunctionType>(parameters, return_type);
// }

// match_expr = 'match' expr '{' match_arm* '}'
std::unique_ptr<MatchExpr> Parser::parseMatchExpr(bool consume_match) {
  if (consume_match)
    consumeKind(TokenKind::KeywordMatch);
  auto token = current_token.value();
  auto expr = parseExpr(0);
  consumeKind(TokenKind::LBrace);
  std::vector<std::unique_ptr<MatchArm>> arms;
  while (!isPeek(TokenKind::RBrace)) {
    auto arm = parseMatchArm();
    arms.emplace_back(std::move(arm));
    consumeKind(TokenKind::Comma);
  }
  consumeKind(TokenKind::RBrace);
  return std::make_unique<MatchExpr>(token, std::move(expr), std::move(arms));
}

// match_arm = 'is' pattern (if expr)? '=>' expr | block
std::unique_ptr<MatchArm> Parser::parseMatchArm() {
  auto is_token = consumeKind(TokenKind::KeywordIs);
  auto pattern = parsePattern();
  std::optional<std::unique_ptr<Expression>> guard = std::nullopt;
  if (isPeek(TokenKind::KeywordIf)) {
    consume();
    guard = parseExpr(0);
  }
  consumeKind(TokenKind::EqualGreater);
  if (isPeek(TokenKind::LBrace)) {
    auto block = parseBlock();
    return std::make_unique<MatchArm>(is_token, std::move(pattern),
                                      std::move(block), std::move(guard));
  }
  auto expr = parseExpr(0);
  return std::make_unique<MatchArm>(is_token, std::move(pattern),
                                    std::move(expr), std::move(guard));
}

std::unique_ptr<Pattern> Parser::parsePattern() {
  auto pattern = parseSinglePattern();
  if (isPeek(TokenKind::Pipe)) {
    return parseOrPattern(std::move(pattern));
  }
  if (isPeek(TokenKind::DotDot) || isPeek(TokenKind::DotDotEqual)) {
    return parseRangePattern(std::move(pattern));
  }
  return pattern;
}

// rest_pattern = '..' (as identifier)?
std::unique_ptr<RestPattern> Parser::parseRestPattern() {
  auto token = consumeKind(TokenKind::DotDot);
  std::optional<IdentifierExpr> name = std::nullopt;
  if (isPeek(TokenKind::KeywordAs)) {
    consume();
    auto ident = consumeKind(TokenKind::Identifier);
    name = IdentifierExpr(ident, lexer->tokenToString(ident));
  }
  return std::make_unique<RestPattern>(token, std::move(name));
}

// variant_pattern = '.' identifier (tuple_pattern | struct_pattern)
std::unique_ptr<VariantPattern> Parser::parseVariantPattern() {
  consumeKind(TokenKind::Dot);
  auto name = consumeKind(TokenKind::Identifier);
  if (isPeek(TokenKind::LParen)) {
    auto tuple = parseTuplePattern();
    return std::make_unique<VariantPattern>(name, lexer->tokenToString(name),
                                            std::move(tuple));
  } else if (isPeek(TokenKind::LBrace)) {
    auto struct_pattern = parseStructPattern();
    return std::make_unique<VariantPattern>(name, lexer->tokenToString(name),
                                            std::move(struct_pattern));
  }
  return std::make_unique<VariantPattern>(name, lexer->tokenToString(name),
                                          std::nullopt);
}

// pattern = literal_pattern | identifier_pattern | wildcard_pattern |
//          tuple_pattern | struct_pattern | enum_variant_pattern |
//          slice_pattern | or_pattern | range_pattern
std::unique_ptr<Pattern> Parser::parseSinglePattern() {
  if (isPeek(TokenKind::NumberLiteral) || isPeek(TokenKind::StringLiteral) ||
      isPeek(TokenKind::CharLiteral)) {
    return parseLiteralPattern();
  } else if (isPeek(TokenKind::Identifier)) {
    if (isPeek2(TokenKind::LBrace)) {
      auto name = consume();
      return parseStructPattern(lexer->tokenToString(name.value()));
    }
    auto str = lexer->tokenToString(peek().value());
    if (str == "_") {
      return parseWildcardPattern();
    }
    return parseIdentifierPattern();
  } else if (isPeek(TokenKind::LParen)) {
    // parse single pattern
    consume();
    auto pattern = parsePattern();
    if (isPeek(TokenKind::Comma)) {
      consume();
      return parseTuplePattern(std::move(pattern));
    }
    consumeKind(TokenKind::RParen);
    return pattern;
  } else if (isPeek(TokenKind::Dot)) {
    return parseVariantPattern();
  } else if (isPeek(TokenKind::DotDot)) {
    return parseRestPattern();
  }
  // else if (is_peek(TokenKind::LBrace)) {
  //   return parse_struct_pattern();
  // }
  else if (isPeek(TokenKind::LBracket)) {
    return parseSlicePattern();
  } else {
    auto expr = parseExpr(0);
    return std::make_unique<ExprPattern>(expr->token, std::move(expr));
  }
}

// literal_pattern = number_literal | string_literal | char_literal
std::unique_ptr<LiteralPattern> Parser::parseLiteralPattern() {
  auto token = consume();
  std::unique_ptr<LiteralExpr> expr;
  switch (token->kind) {
  case TokenKind::NumberLiteral:
    expr = parseNumberLiteralExpr(token.value());
    break;
  case TokenKind::StringLiteral:
    expr = std::make_unique<LiteralExpr>(token.value(),
                                         LiteralExpr::LiteralType::String,
                                         lexer->tokenToString(token.value()));
    break;
  case TokenKind::CharLiteral:
    expr = std::make_unique<LiteralExpr>(
        token.value(), LiteralExpr::LiteralType::Char,
        lexer->tokenToString(token.value())[0]);
    break;
  default:
    break;
  }
  return std::make_unique<LiteralPattern>(expr->token, std::move(expr));
}

// identifier_pattern = identifier
std::unique_ptr<IdentifierPattern> Parser::parseIdentifierPattern() {
  auto token = consumeKind(TokenKind::Identifier);
  return std::make_unique<IdentifierPattern>(token,
                                             lexer->tokenToString(token));
}

// wildcard_pattern = '_'
std::unique_ptr<WildcardPattern> Parser::parseWildcardPattern() {
  auto token = consumeKind(TokenKind::Identifier);
  return std::make_unique<WildcardPattern>(token);
}

// tuple_pattern = '(' pattern (',' pattern)* ')'
std::unique_ptr<TuplePattern> Parser::parseTuplePattern(
    std::optional<std::unique_ptr<Pattern>> first_pattern) {
  Token token;
  std::vector<std::unique_ptr<Pattern>> patterns;
  if (!first_pattern.has_value())
    token = consumeKind(TokenKind::LParen);
  else {
    token = first_pattern.value()->token;
    patterns.emplace_back(std::move(first_pattern.value()));
  }
  while (!isPeek(TokenKind::RParen)) {
    auto pattern = parsePattern();
    patterns.emplace_back(std::move(pattern));
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  consumeKind(TokenKind::RParen);
  return std::make_unique<TuplePattern>(token, std::move(patterns));
}

// struct_pattern = '{' pattern_field* '}'
std::unique_ptr<StructPattern>
Parser::parseStructPattern(std::optional<std::string> name) {
  auto token = consumeKind(TokenKind::LBrace);
  std::vector<StructPattern::Field> fields;
  while (!isPeek(TokenKind::RBrace)) {
    if (isPeek(TokenKind::DotDot)) {
      auto rest = parseRestPattern();
      fields.emplace_back(std::move(rest));
      break;
    }
    auto field = parsePatternField();
    fields.emplace_back(std::move(field));
  }
  consumeKind(TokenKind::RBrace);
  return std::make_unique<StructPattern>(token, std::move(name),
                                         std::move(fields));
}

// pattern_field = identifier ':' pattern ','?
std::unique_ptr<PatternField> Parser::parsePatternField() {
  auto name = consumeKind(TokenKind::Identifier);
  std::optional<std::unique_ptr<Pattern>> pattern = std::nullopt;
  if (isPeek(TokenKind::Colon)) {
    consume();
    pattern = parsePattern();
  }
  if (isPeek(TokenKind::Comma)) {
    consume();
  }
  return std::make_unique<PatternField>(name, lexer->tokenToString(name),
                                        std::move(pattern));
}

// slice_pattern = '[' pattern (',' pattern)* ']'
std::unique_ptr<SlicePattern> Parser::parseSlicePattern() {
  auto token = consumeKind(TokenKind::LBracket);
  std::vector<std::unique_ptr<Pattern>> patterns;
  while (!isPeek(TokenKind::RBracket)) {
    auto pattern = parsePattern();
    patterns.emplace_back(std::move(pattern));
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  consumeKind(TokenKind::RBracket);
  return std::make_unique<SlicePattern>(token, std::move(patterns), true);
}

// or_pattern = pattern '|' pattern
std::unique_ptr<Pattern>
Parser::parseOrPattern(std::unique_ptr<Pattern> first_pattern) {
  std::vector<std::unique_ptr<Pattern>> alternatives;
  auto span = first_pattern->token;
  alternatives.emplace_back(std::move(first_pattern));

  while (isPeek(TokenKind::Pipe)) {
    consume();
    alternatives.emplace_back(parseSinglePattern());
  }

  return std::make_unique<OrPattern>(span, std::move(alternatives));
}

// range_pattern = pattern '..' expr
// range_pattern = expr '..=' expr
std::unique_ptr<Pattern>
Parser::parseRangePattern(std::unique_ptr<Pattern> start_pattern) {
  auto range_type = consume(); // Consume the `..` or `..=`
  bool inclusive = (range_type->kind == TokenKind::DotDotEqual);

  // Parse the end pattern
  auto end_pattern = parseSinglePattern();
  if (!end_pattern) {
    context->reportError("Expected a pattern after range operator",
                         &range_type.value());
    return std::make_unique<InvalidPattern>(range_type.value());
  }

  return std::make_unique<RangePattern>(range_type.value(),
                                        std::move(start_pattern),
                                        std::move(end_pattern), inclusive);
}

// if_expr = 'if' expr block ('else' block)?
std::unique_ptr<IfExpr> Parser::parseIfExpr(bool consume_if) {
  if (consume_if)
    consumeKind(TokenKind::KeywordIf);
  auto token = current_token.value();
  auto condition = parseExpr(0);
  std::unique_ptr<BlockExpression> then_branch;
  if (isPeek(TokenKind::LBrace)) {
    then_branch = parseBlock();
  } else {
    auto expr = parseExpr(0);
    auto token = expr->token;
    auto expr_stmt = std::make_unique<ExprStmt>(expr->token, std::move(expr));
    std::vector<std::unique_ptr<Statement>> stmts;
    stmts.emplace_back(std::move(expr_stmt));
    then_branch = std::make_unique<BlockExpression>(token, std::move(stmts));
  }
  std::optional<std::unique_ptr<BlockExpression>> else_branch = std::nullopt;
  if (isPeek(TokenKind::KeywordElse)) {
    consume();
    if (isPeek(TokenKind::LBrace))
      else_branch = parseBlock();
    else {
      auto expr = parseExpr(0);
      auto token = expr->token;
      auto expr_stmt = std::make_unique<ExprStmt>(expr->token, std::move(expr));
      std::vector<std::unique_ptr<Statement>> stmts;
      stmts.emplace_back(std::move(expr_stmt));
      else_branch = std::make_unique<BlockExpression>(token, std::move(stmts));
    }
  }
  return std::make_unique<IfExpr>(token, std::move(condition),
                                  std::move(then_branch),
                                  std::move(else_branch));
}

// while_expr = 'while' expr : expr block
std::unique_ptr<WhileExpr> Parser::parseWhileExpr(bool consume_while,
                                                  std::optional<Token> label) {
  std::optional<std::string> label_name = std::nullopt;
  if (label.has_value()) {
    label_name = lexer->tokenToString(label.value());
  }
  if (consume_while)
    consumeKind(TokenKind::KeywordWhile);
  auto token = current_token.value();
  std::optional<std::unique_ptr<Expression>> condition = std::nullopt;
  if (!isPeek(TokenKind::LBrace)) {
    condition = parseExpr(0);
  }
  std::optional<std::unique_ptr<Expression>> expr = std::nullopt;
  if (isPeek(TokenKind::Colon)) {
    consume();
    expr = parseExpr(0);
  }
  auto block = parseBlock();
  return std::make_unique<WhileExpr>(token, std::move(condition),
                                     std::move(expr), std::move(block),
                                     std::move(label_name));
}

// for_expr = 'for' identifier 'in' expr block
std::unique_ptr<ForExpr> Parser::parseForExpr(bool consume_for,
                                              std::optional<Token> label) {
  std::optional<std::string> label_name = std::nullopt;
  if (label.has_value()) {
    label_name = lexer->tokenToString(label.value());
  }
  if (consume_for)
    consumeKind(TokenKind::KeywordFor);
  auto token = current_token.value();
  auto pattern = parsePattern();
  consumeKind(TokenKind::KeywordIn);
  auto iterable = parseExpr(0);
  auto block = parseBlock();
  return std::make_unique<ForExpr>(token, std::move(pattern),
                                   std::move(iterable), std::move(block),
                                   std::move(label_name));
}

// call_expr = identifier '(' expr (',' expr)* ')'
std::unique_ptr<CallExpr> Parser::parseCallExpr() {
  auto name = current_token.value();
  consumeKind(TokenKind::LParen);
  std::vector<std::unique_ptr<Expression>> args;
  int comptime_index = -1;
  while (!isPeek(TokenKind::RParen)) {
    if (isPeek(TokenKind::Star)) {
      consume();
      if (comptime_index != -1) {
        context->reportError("Multiple comptime arguments are not allowed",
                             &current_token.value());
      }
      comptime_index = args.size();
    } else {
      auto arg = parseExpr(0);
      args.emplace_back(std::move(arg));
    }
    if (isPeek(TokenKind::Comma)) {
      consume();
    }
  }
  consumeKind(TokenKind::RParen);
  auto call_expr = std::make_unique<CallExpr>(name, lexer->tokenToString(name),
                                              std::move(args), comptime_index);
  return call_expr;
}

// return_stmt = 'return' expr? ';'?
std::unique_ptr<ReturnExpr> Parser::parseReturnExpr(bool consume_return) {
  if (consume_return)
    consumeKind(TokenKind::KeywordReturn);
  if (isPeek(TokenKind::Semicolon)) {
    return std::make_unique<ReturnExpr>(current_token.value(), std::nullopt);
  }
  auto expr = parseExpr(0);
  return std::make_unique<ReturnExpr>(expr->token, std::move(expr));
}

// stmt = expr_stmt | var_decl | return_stmt | block_stmt |
// if_stmt |
//       for_stmt | while_stmt | break_stmt | continue_stmt | match_stmt
std::unique_ptr<Statement> Parser::parseStatement() {
  if (isPeek(TokenKind::KeywordVar) || isPeek(TokenKind::KeywordConst)) {
    return parseVarDecl();
  } else if (isPeek(TokenKind::KeywordReturn)) {
    auto expr = parseReturnExpr();
    consumeKind(TokenKind::Semicolon);
    return std::make_unique<ExprStmt>(expr->token, std::move(expr));
  } else if (isPeek(TokenKind::KeywordIf)) {
    auto expr = parseIfExpr();
    return std::make_unique<ExprStmt>(expr->token, std::move(expr));
  } else if (isPeek(TokenKind::KeywordFor)) {
    auto expr = parseForExpr();
    return std::make_unique<ExprStmt>(expr->token, std::move(expr));
  } else if (isPeek(TokenKind::KeywordWhile)) {
    auto expr = parseWhileExpr();
    return std::make_unique<ExprStmt>(expr->token, std::move(expr));
  } else if (isPeek(TokenKind::KeywordComptime)) {
    consume();
    auto token = peek().value();
    auto expr = parseExpr(0);
    if (token.kind != TokenKind::LBrace)
      consumeKind(TokenKind::Semicolon);
    return std::make_unique<ExprStmt>(expr->token, std::move(expr));
  } else if (isPeek(TokenKind::KeywordBreak)) {
    auto break_expr = parseBreakExpr();
    consumeKind(TokenKind::Semicolon);
    return std::make_unique<ExprStmt>(break_expr->token, std::move(break_expr));
  } else if (isPeek(TokenKind::KeywordContinue)) {
    auto continue_expr = parseContinueExpr();
    consumeKind(TokenKind::Semicolon);
    return std::make_unique<ExprStmt>(continue_expr->token,
                                      std::move(continue_expr));
  } else if (isPeek(TokenKind::KeywordYield)) {
    auto yield_expr = parseYieldExpr();
    consumeKind(TokenKind::Semicolon);
    return std::make_unique<ExprStmt>(yield_expr->token, std::move(yield_expr));
  } else if (isPeek(TokenKind::KeywordMatch)) {
    auto match_expr = parseMatchExpr();
    return std::make_unique<ExprStmt>(match_expr->token, std::move(match_expr));
  } else if (isPeek(TokenKind::Identifier) && isPeek2(TokenKind::Colon)) {
    // labeled statements
    auto label = consume();
    consume();
    if (isPeek(TokenKind::KeywordWhile)) {
      auto expr = parseWhileExpr(true, label);
      return std::make_unique<ExprStmt>(expr->token, std::move(expr));
    }
  } else if (isPeek(TokenKind::KeywordStruct)) {
    consume();
    auto struct_decl = parseStructDecl();
    context->declareStruct(struct_decl->name, struct_decl.get());
    return std::make_unique<TopLevelDeclStmt>(struct_decl->token,
                                              std::move(struct_decl));
  } else if (isPeek(TokenKind::KeywordEnum)) {
    auto enum_decl = parseEnumDecl();
    context->declareEnum(enum_decl->name, enum_decl.get());
    return std::make_unique<TopLevelDeclStmt>(enum_decl->token,
                                              std::move(enum_decl));
  } else if (isPeek(TokenKind::KeywordImpl)) {
    auto impl_decl = parseImplDecl();
    return std::make_unique<TopLevelDeclStmt>(impl_decl->token,
                                              std::move(impl_decl));
  } else if (isPeek(TokenKind::KeywordTrait)) {
    auto trait_decl = parseTraitDecl();
    context->declareTrait(trait_decl->name, trait_decl.get());
    return std::make_unique<TopLevelDeclStmt>(trait_decl->token,
                                              std::move(trait_decl));
  } else if (isPeek(TokenKind::KeywordFn)) {
    auto function = parseFunction();
    return std::make_unique<TopLevelDeclStmt>(function->token,
                                              std::move(function));
  } else {
    auto token = peek().value();
    auto expr = parseExpr(0);
    auto kind = expr->kind();
    if ((kind == AstNodeKind::IndexExpr ||
         kind == AstNodeKind::FieldAccessExpr ||
         kind == AstNodeKind::IdentifierExpr) &&
        isPeek(TokenKind::Equal)) {
      // assignment statement
      consumeKind(TokenKind::Equal);
      auto rhs = parseExpr(0);
      consumeKind(TokenKind::Semicolon);
      return std::make_unique<AssignStatement>(token, std::move(expr),
                                               std::move(rhs));
    }
    if (kind != AstNodeKind::BlockExpression && !isPeek(TokenKind::Semicolon)) {
      // if not block expression, then it is an implicit yield
      expr = std::make_unique<YieldExpr>(token, std::nullopt, std::move(expr));
      return std::make_unique<ExprStmt>(token, std::move(expr));
    }
    // if block statement then no need for semicolon
    if (token.kind != TokenKind::LBrace)
      consumeKind(TokenKind::Semicolon);
    return std::make_unique<ExprStmt>(expr->token, std::move(expr));
  }
  return std::make_unique<InvalidStatement>(current_token.value());
}

// var_decl = 'var' | 'const' identifier ('=' expr)? ';'
std::unique_ptr<VarDecl> Parser::parseVarDecl(bool is_pub) {
  bool is_mut = false;
  if (peek()->kind == TokenKind::KeywordConst) {
    consumeKind(TokenKind::KeywordConst);
  } else {
    is_mut = true;
    consumeKind(TokenKind::KeywordVar);
  }
  auto token = current_token.value();
  auto pattern = parsePattern();
  std::optional<std::unique_ptr<Type>> type = std::nullopt;
  if (peek()->kind == TokenKind::Colon) {
    consume();
    type = parseType();
  }
  std::optional<std::unique_ptr<Expression>> expr = std::nullopt;
  if (isPeek(TokenKind::Equal)) {
    consume();
    expr = parseExpr(0);
  }
  consumeKind(TokenKind::Semicolon);

  return std::make_unique<VarDecl>(token, std::move(pattern), std::move(type),
                                   std::move(expr), is_mut, is_pub);
}

// break_expr = 'break' (':' identifier)? (expr)?;
std::unique_ptr<BreakExpr> Parser::parseBreakExpr(bool consume_break) {
  if (consume_break)
    consumeKind(TokenKind::KeywordBreak);
  auto token = current_token.value();
  std::optional<std::string> label = std::nullopt;
  if (isPeek(TokenKind::Colon)) {
    consume();
    auto name = consumeKind(TokenKind::Identifier);
    label = lexer->tokenToString(name);
  }
  std::optional<std::unique_ptr<Expression>> expr = std::nullopt;
  if (!isPeek(TokenKind::Semicolon) && !isPeek(TokenKind::Comma)) {
    expr = parseExpr(0);
  }
  return std::make_unique<BreakExpr>(token, std::move(label), std::move(expr));
}

// yield_expr = 'yield' (':' identifier)? expr;
std::unique_ptr<YieldExpr> Parser::parseYieldExpr(bool consume_yield) {
  if (consume_yield)
    consumeKind(TokenKind::KeywordYield);
  auto token = current_token.value();
  std::optional<std::string> label = std::nullopt;
  if (isPeek(TokenKind::Colon)) {
    consume();
    auto name = consumeKind(TokenKind::Identifier);
    label = lexer->tokenToString(name);
  }
  auto expr = parseExpr(0);
  return std::make_unique<YieldExpr>(token, std::move(label), std::move(expr));
}

// continue_expr = 'continue' (':' identifier)? (expr)?;
std::unique_ptr<ContinueExpr> Parser::parseContinueExpr(bool consume_continue) {
  if (consume_continue)
    consumeKind(TokenKind::KeywordContinue);
  auto token = current_token.value();
  std::optional<std::string> label = std::nullopt;
  if (isPeek(TokenKind::Colon)) {
    consume();
    auto name = consumeKind(TokenKind::Identifier);
    label = lexer->tokenToString(name);
  }
  std::optional<std::unique_ptr<Expression>> expr = std::nullopt;
  if (!isPeek(TokenKind::Semicolon) && !isPeek(TokenKind::Comma)) {
    expr = parseExpr(0);
  }
  return std::make_unique<ContinueExpr>(token, std::move(label),
                                        std::move(expr));
}

std::variant<int, double>
Parser::parseNumberLiteral(const std::basic_string<char> &bytes) {
  size_t i = 0;
  uint8_t base = 10;
  if (bytes.size() >= 2 && bytes[0] == '0') {
    switch (bytes[1]) {
    case 'b':
      base = 2;
      i = 2;
      break;
    case 'o':
      base = 8;
      i = 2;
      break;
    case 'x':
      base = 16;
      i = 2;
      break;
    case 'B':
    case 'O':
    case 'X':
      context->reportError("Upper case base not allowed",
                           &current_token.value());
      break;
    case '.':
    case 'e':
    case 'E':
      break;
    default:
      context->reportError("Leading zero not allowed", &current_token.value());
      break;
    }
  }
  if (bytes.size() == 2 && base != 10) {
    context->reportError("Digit after base not allowed",
                         &current_token.value());
  }

  uint64_t x = 0;
  bool overflow = false;
  bool underscore = false;
  bool period = false;
  uint8_t special = 0;
  bool exponent = false;
  bool floating = false;
  while (i < bytes.size()) {
    const unsigned char c = bytes[i];
    switch (c) {
    case '_':
      if (i == 2 && base != 10) {
        context->reportError("Invalid underscore after special",
                             &current_token.value());
      }
      if (special != 0) {
        context->reportError("Invalid underscore after special",
                             &current_token.value());
      }
      if (underscore) {
        context->reportError("Repeated underscore", &current_token.value());
      }
      underscore = true;
      ++i;
      continue;
    case 'e':
    case 'E':
      if (base == 10) {
        floating = true;
        if (exponent) {
          context->reportError("Duplicate exponent", &current_token.value());
        }
        if (underscore) {
          context->reportError("Exponent after underscore",
                               &current_token.value());
        }
        special = c;
        exponent = true;
        ++i;
        continue;
      }
      break;
    case 'p':
    case 'P':
      if (base == 16) {
        floating = true;
        if (exponent) {
          context->reportError("Duplicate exponent", &current_token.value());
        }
        if (underscore) {
          context->reportError("Exponent after underscore",
                               &current_token.value());
        }
        special = c;
        exponent = true;
        ++i;
        continue;
      }
      break;
    case '.':
      floating = true;
      if (base != 10 && base != 16) {
        context->reportError("Invalid float base", &current_token.value());
      }
      if (period) {
        context->reportError("Duplicate period", &current_token.value());
      }
      period = true;
      if (underscore) {
        context->reportError("Special after underscore",
                             &current_token.value());
      }
      special = c;
      ++i;
      continue;
    case '+':
    case '-':
      switch (special) {
      case 'p':
      case 'P':
        break;
      case 'e':
      case 'E':
        if (base != 10) {
          context->reportError("Invalid exponent sign", &current_token.value());
        }
        break;
      default:
        context->reportError("Invalid exponent sign", &current_token.value());
        break;
      }
      special = c;
      ++i;
      continue;
    default:
      break;
    }
    const unsigned char digit = [&]() {
      if (c >= '0' && c <= '9') {
        return c - '0';
      } else if (c >= 'A' && c <= 'Z') {
        return c - 'A' + 10;
      } else if (c >= 'a' && c <= 'z') {
        return c - 'a' + 10;
      }
      context->reportError("Invalid character", &current_token.value());
      return 0;
    }();
    if (digit >= base) {
      context->reportError("Invalid digit", &current_token.value());
    }

    if (exponent && digit >= 10) {
      context->reportError("Invalid digit exponent", &current_token.value());
    }

    underscore = false;
    special = 0;

    if (floating) {
      ++i;
      continue;
    }

    if (x != 0) {
      x = x * base;
    }

    if (digit > UINT64_MAX - x) {
      overflow = true;
    }

    x += digit;
    if (x > UINT64_MAX - digit) {
      overflow = true;
    }

    ++i;
  }

  if (underscore) {
    context->reportError("Trailing underscore", &current_token.value());
  }
  if (special != 0) {
    context->reportError("Trailing special", &current_token.value());
  }

  if (floating) {
    return stod(lexer->tokenToString(current_token.value()));
  }
  if (overflow) {
    // bigint
    context->reportError("Overflow", &current_token.value());
  }
  return (int)x;
}

Attribute Parser::tokenToAttribute(const Token &token) {
  auto str = lexer->tokenToString(token);
  if (str == "inline") {
    return Attribute::Inline;
  } else if (str == "noinline") {
    return Attribute::NoInline;
  } else if (str == "alwaysinline") {
    return Attribute::AlwaysInline;
  }
  context->reportError("Invalid attribute " + str, &token);
  return Attribute::None;
}

Operator Parser::tokenToOperator(const Token &op) {
  switch (op.kind) {
  case TokenKind::Plus:
  case TokenKind::PlusEqual:
    return Operator::Add;
  case TokenKind::Minus:
  case TokenKind::MinusEqual:
    return Operator::Sub;
  case TokenKind::Star:
  case TokenKind::StarEqual:
    return Operator::Mul;
  case TokenKind::Slash:
  case TokenKind::SlashEqual:
    return Operator::Div;
  case TokenKind::Percent:
  case TokenKind::PercentEqual:
    return Operator::Mod;
  case TokenKind::EqualEqual:
    return Operator::Eq;
  case TokenKind::BangEqual:
    return Operator::Ne;
  case TokenKind::Less:
    return Operator::Lt;
  case TokenKind::LessEqual:
    return Operator::Le;
  case TokenKind::Greater:
    return Operator::Gt;
  case TokenKind::GreaterEqual:
    return Operator::Ge;
  case TokenKind::Ampersand:
  case TokenKind::AmpersandEqual:
    return Operator::BitAnd;
  case TokenKind::Pipe:
  case TokenKind::PipeEqual:
    return Operator::BitOr;
  case TokenKind::KeywordAnd:
    return Operator::And;
  case TokenKind::KeywordOr:
    return Operator::Or;
  case TokenKind::Caret:
  case TokenKind::CaretEqual:
    return Operator::BitXor;
  case TokenKind::LessLess:
  case TokenKind::LessLessEqual:
    return Operator::BitShl;
  case TokenKind::GreaterGreater:
  case TokenKind::GreaterGreaterEqual:
    return Operator::BitShr;
  case TokenKind::StarStar:
    return Operator::Pow;
  default:
    context->reportError("Invalid operator " + Lexer::lexeme(op.kind), &op);
    return Operator::Invalid;
  }
}
