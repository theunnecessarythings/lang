#include "ast.hpp"
#include "compiler.hpp"
#include "lexer.hpp"
#include <fstream>
#include <memory>
#include <optional>

Token get_error_token(TokenSpan &span) { return Token{TokenKind::Dummy, span}; }

struct Parser {
  std::unique_ptr<Lexer> lexer;
  std::optional<Token> current_token;
  std::optional<Token> next_token;
  std::optional<Token> next2_token;
  std::optional<Token> prev_token;
  std::shared_ptr<Context> context;

  std::vector<std::unique_ptr<TopLevelDecl>> top_level_decls;

  Parser(std::unique_ptr<Lexer> _lexer, std::shared_ptr<Context> _context)
      : lexer(std::move(_lexer)), context(std::move(_context)) {
    next_token = lexer->next();
    next2_token = lexer->next();
  }

  Token unexpected_token_error(TokenKind &expected, Token &found) {
    context->diagnostics.report_error(found,
                                      "Expected " + Lexer::lexeme(expected) +
                                          ", got " + Lexer::lexeme(found.kind));
    skip_to_next_stmt();
    return get_error_token(found.span);
  }

  Token invalid_token_error(Token &found) {
    context->diagnostics.report_error(found, "Invalid token " +
                                                 Lexer::lexeme(found.kind));
    skip_to_next_stmt();
    return get_error_token(found.span);
  }

  std::optional<Token> consume() {
    prev_token = current_token;
    current_token = next_token;
    next_token = next2_token;
    next2_token = lexer->next();
    return current_token;
  }

  Token consume_kind(TokenKind kind) {
    auto token = peek();
    if (token.has_value() && token.value().kind == kind) {
      return consume().value();
    }
    return unexpected_token_error(kind, token.value());
  }

  std::optional<Token> peek() { return next_token; }

  std::optional<Token> peek2() { return next2_token; }

  bool is_peek(TokenKind kind) {
    return peek().has_value() && peek().value().kind == kind;
  }

  bool is_peek2(TokenKind kind) {
    return peek2().has_value() && peek2().value().kind == kind;
  }

  void consume_optional_semicolon() {
    if (is_peek(TokenKind::Semicolon)) {
      consume();
    }
  }

  int binding_pow(const Token &token) {
    switch (token.kind) {
    case TokenKind::Equal:
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

  std::unique_ptr<LiteralExpr> parse_number_literal_expr(const Token &token) {
    auto number_str = lexer->token_to_string(token);
    auto number = parse_number_literal(number_str);
    if (auto val = std::get_if<int>(&number)) {
      return std::make_unique<LiteralExpr>(LiteralExpr::LiteralType::Int, *val);
    } else {
      return std::make_unique<LiteralExpr>(LiteralExpr::LiteralType::Float,
                                           std::get<double>(number));
    }
  }

  std::unique_ptr<Expression> nud(const Token &token) {
    switch (token.kind) {
    case TokenKind::NumberLiteral: {
      return parse_number_literal_expr(token);
    }
    case TokenKind::StringLiteral:
      return std::make_unique<LiteralExpr>(LiteralExpr::LiteralType::String,
                                           lexer->token_to_string(token));
    case TokenKind::CharLiteral:
      return std::make_unique<LiteralExpr>(LiteralExpr::LiteralType::Char,
                                           lexer->token_to_string(token)[1]);
    case TokenKind::LParen: {
      auto expr = parse_expr(0);
      auto comma = peek();
      if (comma.has_value() && comma.value().kind == TokenKind::Comma) {
        return parse_tuple_expr(std::move(expr));
      } else {
        consume_kind(TokenKind::RParen);
        return expr;
      }
      break;
    }
    case TokenKind::LBracket:
      return parse_array_expr();
    case TokenKind::LBrace: {
      return parse_block(false);
    }
    case TokenKind::Plus:
      return parse_expr(binding_pow(token));
    case TokenKind::Minus:
      return std::make_unique<UnaryExpr>(
          Operator::Sub, std::move(parse_expr(binding_pow(token))));
    case TokenKind::KeywordNot:
      return std::make_unique<UnaryExpr>(
          Operator::Not, std::move(parse_expr(binding_pow(token))));
    case TokenKind::Identifier: {
      if (is_peek(TokenKind::LParen)) {
        return parse_call_expr();
      } else {
        auto str = lexer->token_to_string(token);
        if (str == "true" || str == "false") {
          return std::make_unique<LiteralExpr>(LiteralExpr::LiteralType::Bool,
                                               str == "true");
        }
        return std::make_unique<IdentifierExpr>(lexer->token_to_string(token));
      }
    }
    case TokenKind::DotDot:
    case TokenKind::DotDotEqual: {
      std::optional<std::unique_ptr<Expression>> right_expr = std::nullopt;
      if (!is_peek(TokenKind::RBracket) && !is_peek(TokenKind::LBrace)) {
        right_expr = parse_expr(binding_pow(token));
      }
      return std::make_unique<RangeExpr>(std::nullopt, std::move(right_expr),
                                         token.kind == TokenKind::DotDotEqual);
    }
    case TokenKind::KeywordMatch:
      return parse_match_expr(false);
    case TokenKind::KeywordIf:
      return parse_if_expr(false);
    case TokenKind::KeywordWhile:
      return parse_while_expr(false);
    case TokenKind::KeywordFor:
      return parse_for_expr(false);
    case TokenKind::KeywordBreak:
      return parse_break_expr(false);
    case TokenKind::KeywordContinue:
      return parse_continue_expr(false);
    default:
      context->diagnostics.report_error(token, "Invalid character");
      return std::make_unique<InvalidExpression>();
    }
  }

  std::unique_ptr<Expression> led(std::unique_ptr<Expression> left, Token &op) {
    auto precedence = binding_pow(op);
    switch (op.kind) {
    case TokenKind::Equal:
      return std::make_unique<AssignExpr>(std::move(left),
                                          parse_expr(precedence - 1));
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
          token_to_operator(op), std::move(left), parse_expr(precedence - 1));
    case TokenKind::Dot: {
      auto right = consume_kind(TokenKind::Identifier);
      if (is_peek(TokenKind::LParen)) {
        auto call_expr = parse_call_expr();
        return std::make_unique<FieldAccessExpr>(std::move(left),
                                                 std::move(call_expr));
      } else if (is_peek(TokenKind::NumberLiteral)) {
        auto number = consume();
        auto value =
            parse_number_literal(lexer->token_to_string(number.value()));
        if (auto val = std::get_if<int>(&value)) {
          auto expr = std::make_unique<LiteralExpr>(
              LiteralExpr::LiteralType::Int, *val);
          return std::make_unique<FieldAccessExpr>(std::move(left),
                                                   std::move(expr));
        }
        context->diagnostics.report_error(
            number.value(), "Expected integer literal found " +
                                lexer->token_to_string(number.value()));
        return std::make_unique<InvalidExpression>();
      } else {
        auto right_expr =
            std::make_unique<IdentifierExpr>(lexer->token_to_string(right));
        return std::make_unique<FieldAccessExpr>(std::move(left),
                                                 std::move(right_expr));
      }
    }
    case TokenKind::LBracket: {
      auto index_expr = parse_expr(0);
      consume_kind(TokenKind::RBracket);
      return std::make_unique<IndexExpr>(std::move(left),
                                         std::move(index_expr));
    }
    case TokenKind::DotDot:
    case TokenKind::DotDotEqual: {
      std::optional<std::unique_ptr<Expression>> right_expr = std::nullopt;
      if (!is_peek(TokenKind::RBracket) && !is_peek(TokenKind::LBrace)) {
        right_expr = parse_expr(binding_pow(op));
      }
      bool is_inclusive = op.kind == TokenKind::DotDotEqual;
      return std::make_unique<RangeExpr>(std::move(left), std::move(right_expr),
                                         is_inclusive);
    }
    default: {
      auto right = parse_expr(precedence);
      Operator op_kind = token_to_operator(op);
      return std::make_unique<BinaryExpr>(op_kind, std::move(left),
                                          std::move(right));
    }
    }
  }

  std::unique_ptr<Program> parse_single_source(std::string &path) {
    // Try to open the file
    std::ifstream file(path);
    if (!file.is_open()) {
      context->diagnostics.report_error(current_token.value(),
                                        "Could not open file " + path);
      std::vector<std::unique_ptr<TopLevelDecl>> top_level_decls;
      return std::make_unique<Program>(std::move(top_level_decls));
    }
    // Read contents of the file into a string
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    int file_id = context->source_manager->add_path(path, source);
    std::unique_ptr<Lexer> lexer = std::make_unique<Lexer>(source, file_id);
    std::unique_ptr<Parser> parser =
        std::make_unique<Parser>(std::move(lexer), context);
    auto program = parser->parse_program();

    return program;
  }

  // program = top_level_decl*
  std::unique_ptr<Program> parse_program() {
    while (true) {
      if (is_peek(TokenKind::Eof)) {
        break;
      }
      std::unique_ptr<TopLevelDecl> top_level_decl;
      try {
        top_level_decl = parse_top_level_decl();
      } catch (std::runtime_error &e) {
        return std::make_unique<Program>(std::move(top_level_decls));
      }
      top_level_decls.emplace_back(std::move(top_level_decl));
    }
    return std::make_unique<Program>(std::move(top_level_decls));
  }

  void skip_to_next_top_level_decl() {
    while (!is_peek(TokenKind::Eof)) {
      if (is_peek(TokenKind::KeywordFn) || is_peek(TokenKind::KeywordStruct) ||
          is_peek(TokenKind::KeywordEnum) || is_peek(TokenKind::KeywordImpl) ||
          is_peek(TokenKind::KeywordUnion) ||
          is_peek(TokenKind::KeywordTrait)) {
        break;
      }
      consume();
    }
  }

  void skip_to_next_stmt() {
    while (!is_peek(TokenKind::Eof)) {
      if (is_peek(TokenKind::Semicolon)) {
        consume();
        break;
      }
      consume();
    }
  }

  // import_decl -> 'import' (ident ('.' ident)* ('as' ident)?)*
  // eg: import std.io, std.math.rand as rand, std.fs
  std::unique_ptr<ImportDecl> parse_import_decl() {
    auto import_token = consume_kind(TokenKind::KeywordImport);
    std::vector<ImportDecl::Path> paths;
    while (true) {
      std::string path = "";
      while (is_peek2(TokenKind::Dot)) {
        auto ident = consume_kind(TokenKind::Identifier);
        path += lexer->token_to_string(ident) + "/";
        consume();
      }
      auto ident = consume_kind(TokenKind::Identifier);
      path += lexer->token_to_string(ident);

      if (is_peek(TokenKind::KeywordAs)) {
        consume();
        auto alias = consume_kind(TokenKind::Identifier);
        paths.emplace_back(
            ImportDecl::Path{path, lexer->token_to_string(alias)});
      } else {
        paths.emplace_back(ImportDecl::Path{path, std::nullopt});
      }

      // Import the parsed path
      path += ".lang";
      if (!context->source_manager->contains_path(path)) {
        auto tree = parse_single_source(path);
        for (auto &decl : tree->items) {
          top_level_decls.emplace_back(std::move(decl));
        }
      }
      if (!is_peek(TokenKind::Comma))
        break;
      else
        consume();
    }
    // consume_optional_semicolon();
    consume_kind(TokenKind::Semicolon);
    return std::make_unique<ImportDecl>(std::move(paths));
  }

  // top_level_decl = function_decl | struct_decl | enum_decl | impl_decl |
  // union_decl | trait_decl
  std::unique_ptr<TopLevelDecl> parse_top_level_decl() {
    if (is_peek(TokenKind::KeywordFn)) {
      return parse_function();
    } else if (is_peek(TokenKind::KeywordStruct)) {
      consume();
      if (is_peek2(TokenKind::LParen)) {
        return parse_tuple_struct_decl();
      }
      return parse_struct_decl();
    } else if (is_peek(TokenKind::KeywordImport)) {
      return parse_import_decl();
    } else if (is_peek(TokenKind::KeywordEnum)) {
      return parse_enum_decl();
    } else if (is_peek(TokenKind::KeywordImpl)) {
      return parse_impl_decl();
    } else if (is_peek(TokenKind::KeywordVar) ||
               is_peek(TokenKind::KeywordConst)) {
      auto var_decl = parse_var_decl();
      return std::make_unique<TopLevelVarDecl>(std::move(var_decl));
    }
    // else if (is_peek(TokenKind::KeywordUnion)) {
    //   return parse_union_decl();
    // } else if (is_peek(TokenKind::KeywordTrait)) {
    //   return parse_trait_decl();
    //  }
    else {
      auto token = consume();
      context->diagnostics.report_error(token.value(),
                                        "Expected top level decl found " +
                                            Lexer::lexeme(token->kind));
      skip_to_next_top_level_decl();
      return std::make_unique<InvalidTopLevelDecl>();
    }
  }

  // pratt parser
  std::unique_ptr<Expression> parse_expr(int precedence) {
    auto token = consume();
    auto left = nud(token.value());
    while (precedence < binding_pow(peek().value())) {
      token = consume();
      auto new_left = led(std::move(left), token.value());
      left = std::move(new_left);
    }
    return left;
  }

  // tuple_expr = '(' expr (',' expr)* ')'
  std::unique_ptr<Expression>
  parse_tuple_expr(std::unique_ptr<Expression> first_expr) {
    std::vector<std::unique_ptr<Expression>> exprs;
    exprs.emplace_back(std::move(first_expr));
    while (is_peek(TokenKind::Comma)) {
      consume();
      if (is_peek(TokenKind::RParen)) {
        break;
      }
      auto expr = parse_expr(0);
      exprs.emplace_back(std::move(expr));
    }
    consume_kind(TokenKind::RParen);
    return std::make_unique<TupleExpr>(std::move(exprs));
  }

  // array_expr = '[' expr (',' expr)* ']'
  std::unique_ptr<Expression> parse_array_expr() {
    std::vector<std::unique_ptr<Expression>> exprs;
    while (!is_peek(TokenKind::RBracket)) {
      auto expr = parse_expr(0);
      exprs.emplace_back(std::move(expr));
      if (is_peek(TokenKind::Comma)) {
        consume();
      }
    }
    consume_kind(TokenKind::RBracket);
    return std::make_unique<ArrayExpr>(std::move(exprs));
  }

  // function = 'fn' identifier '(' params ')' type block
  std::unique_ptr<Function> parse_function() {
    auto fn_token = consume_kind(TokenKind::KeywordFn);
    auto name = consume_kind(TokenKind::Identifier);
    auto name_str = lexer->token_to_string(name);
    auto params = parse_params();
    auto return_type = parse_type();
    auto block = parse_block();
    return std::make_unique<Function>(std::move(name_str), std::move(params),
                                      std::move(return_type), std::move(block));
  }

  // params = '(' (param (',' param)*)? ')'
  std::vector<std::unique_ptr<Parameter>> parse_params() {
    std::vector<std::unique_ptr<Parameter>> params;
    consume_kind(TokenKind::LParen);
    while (!is_peek(TokenKind::RParen)) {
      auto param = parse_param();
      params.emplace_back(std::move(param));
      if (is_peek(TokenKind::Comma)) {
        consume();
      }
    }
    consume_kind(TokenKind::RParen);
    return params;
  }

  // param -> ("mut")? identifier ":" type
  std::unique_ptr<Parameter> parse_param() {
    bool is_mut = false;
    if (is_peek(TokenKind::KeywordMut)) {
      consume();
      is_mut = true;
    }
    auto pattern = parse_pattern();
    consume_kind(TokenKind::Colon);
    auto type = parse_type();
    return std::make_unique<Parameter>(std::move(pattern), std::move(type),
                                       is_mut);
  }

  // block_expr = '{' stmt* '}'
  std::unique_ptr<BlockExpression> parse_block(bool consume_lbrace = true) {
    std::vector<std::unique_ptr<Statement>> stmts;
    if (consume_lbrace)
      consume_kind(TokenKind::LBrace);
    while (!is_peek(TokenKind::RBrace)) {
      auto stmt = parse_stmt();
      stmts.emplace_back(std::move(stmt));
    }
    consume_kind(TokenKind::RBrace);
    return std::make_unique<BlockExpression>(std::move(stmts),
                                             std::move(std::nullopt));
  }

  // struct_decl = 'struct' identifier '{' struct_field* '}'
  std::unique_ptr<StructDecl> parse_struct_decl() {
    auto name = consume_kind(TokenKind::Identifier);
    consume_kind(TokenKind::LBrace);
    std::vector<std::unique_ptr<StructField>> fields;
    while (!is_peek(TokenKind::RBrace)) {
      auto field = parse_struct_field();
      fields.emplace_back(std::move(field));
    }
    consume_kind(TokenKind::RBrace);
    return std::make_unique<StructDecl>(lexer->token_to_string(name),
                                        std::move(fields));
  }

  // struct_field = identifier ':' type ','
  std::unique_ptr<StructField> parse_struct_field() {
    auto name = consume_kind(TokenKind::Identifier);
    consume_kind(TokenKind::Colon);
    auto type = parse_type();
    if (is_peek(TokenKind::Comma)) {
      consume();
    }
    return std::make_unique<StructField>(lexer->token_to_string(name),
                                         std::move(type));
  }

  // tuple_struct_decl = 'struct' identifier '(' type (',' type)* ')'
  std::unique_ptr<TupleStructDecl> parse_tuple_struct_decl() {
    auto name = consume_kind(TokenKind::Identifier);
    consume_kind(TokenKind::LParen);
    std::vector<std::unique_ptr<Type>> fields;
    while (true) {
      auto field = parse_type();
      fields.emplace_back(std::move(field));
      if (is_peek(TokenKind::Comma)) {
        consume();
      } else {
        break;
      }
    }
    consume_kind(TokenKind::RParen);
    return std::make_unique<TupleStructDecl>(lexer->token_to_string(name),
                                             std::move(fields));
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
  std::unique_ptr<EnumDecl> parse_enum_decl() {
    auto enum_token = consume_kind(TokenKind::KeywordEnum);
    auto name = consume_kind(TokenKind::Identifier);
    consume_kind(TokenKind::LBrace);
    std::vector<std::unique_ptr<Variant>> variants;
    while (!is_peek(TokenKind::RBrace)) {
      auto variant = parse_enum_variant();
      variants.emplace_back(std::move(variant));
      if (!is_peek(TokenKind::RBrace))
        consume_kind(TokenKind::Comma);
    }
    consume_kind(TokenKind::RBrace);
    return std::make_unique<EnumDecl>(std::move(lexer->token_to_string(name)),
                                      std::move(variants));
  }

  std::unique_ptr<FieldsUnnamed> parse_field_unnamed() {
    std::vector<std::unique_ptr<Type>> fields;
    consume_kind(TokenKind::LParen);
    while (!is_peek(TokenKind::RParen)) {
      auto field = parse_type();
      fields.emplace_back(std::move(field));
      if (!is_peek(TokenKind::RParen))
        consume_kind(TokenKind::Comma);
    }
    consume_kind(TokenKind::RParen);
    return std::make_unique<FieldsUnnamed>(std::move(fields));
  }

  std::unique_ptr<FieldsNamed> parse_field_named() {
    std::vector<std::string> names;
    std::vector<std::unique_ptr<Type>> fields;
    consume_kind(TokenKind::LBrace);
    while (!is_peek(TokenKind::RBrace)) {
      auto name = consume_kind(TokenKind::Identifier);
      names.emplace_back(lexer->token_to_string(name));
      consume_kind(TokenKind::Colon);
      auto field = parse_type();
      fields.emplace_back(std::move(field));
      if (is_peek(TokenKind::Comma)) {
        consume();
      }
    }
    consume_kind(TokenKind::RBrace);
    return std::make_unique<FieldsNamed>(std::move(names), std::move(fields));
  }

  // enum_variant = (type | tuple_struct_decl | struct_decl)
  std::unique_ptr<Variant> parse_enum_variant() {
    auto name = consume_kind(TokenKind::Identifier);
    if (is_peek(TokenKind::LBrace)) {
      return std::make_unique<Variant>(lexer->token_to_string(name),
                                       parse_field_named());
    } else if (is_peek(TokenKind::LParen)) {
      return std::make_unique<Variant>(lexer->token_to_string(name),
                                       parse_field_unnamed());
    } else if (is_peek(TokenKind::Equal)) {
      consume();
      auto expr = parse_expr(0);
      return std::make_unique<Variant>(lexer->token_to_string(name),
                                       std::move(expr));
    } else {
      return std::make_unique<Variant>(lexer->token_to_string(name),
                                       std::nullopt);
    }
  }

  // impl_decl = 'impl' type '{' function_decl* '}'
  std::unique_ptr<ImplDecl> parse_impl_decl() {
    auto impl_token = consume_kind(TokenKind::KeywordImpl);
    auto type = parse_type();

    std::vector<std::unique_ptr<Type>> traits;
    if (is_peek(TokenKind::Colon)) {
      consume();
      while (!is_peek(TokenKind::LBrace)) {
        auto trait = parse_type();
        traits.emplace_back(std::move(trait));
        if (is_peek(TokenKind::Comma)) {
          consume();
        }
      }
    }
    consume_kind(TokenKind::LBrace);
    std::vector<std::unique_ptr<Function>> functions;
    while (!is_peek(TokenKind::RBrace)) {
      auto function = parse_function();
      functions.emplace_back(std::move(function));
    }
    consume_kind(TokenKind::RBrace);
    return std::make_unique<ImplDecl>(std::move(type), std::move(traits),
                                      std::move(functions));
  }

  // // trait_decl = 'trait' identifier '{' function_decl* '}'
  // std::shared_ptr<TraitDecl> parse_trait_decl() {
  //   auto trait_token = consume_kind(TokenKind::KeywordTrait);
  //   auto name = consume_kind(TokenKind::Identifier);
  //   consume_kind(TokenKind::LBrace);
  //   std::vector<std::shared_ptr<FunctionDecl>> functions;
  //   while (!is_peek(TokenKind::RBrace)) {
  //     auto function = parse_function(true, false);
  //     functions.push_back(function);
  //   }
  //   consume_kind(TokenKind::RBrace);
  //   return std::make_shared<TraitDecl>(trait_token, name, functions);
  // }
  //
  // type = primitive_type | tuple_type | array_type | function_type |
  //          pointer_type | reference_type | identifier
  std::unique_ptr<Type> parse_type() {
    // if (is_peek(TokenKind::KeywordFn)) {
    //   return parse_function_type();
    // } else
    if (is_peek(TokenKind::LParen)) {
      return parse_tuple_type();
    } else if (is_peek(TokenKind::LBracket)) {
      return parse_array_type();
    }
    return parse_primitive_type();
  }

  // primitive_type = 'i8' | 'i16' | 'i32' | 'i64' | 'u8' | 'u16' | 'u32' |
  //                 'u64' | 'f32' | 'f64' | 'bool' | 'char' | 'str' | 'void'
  std::unique_ptr<Type> parse_primitive_type() {
    auto token = consume_kind(TokenKind::Identifier);
    auto token_str = lexer->token_to_string(token);
    PrimitiveType::PrimitiveTypeKind kind =
        PrimitiveType::PrimitiveTypeKind::Void;
    if (token_str == "bool")
      kind = PrimitiveType::PrimitiveTypeKind::Bool;
    else if (token_str == "char")
      kind = PrimitiveType::PrimitiveTypeKind::Char;
    else if (token_str == "i8")
      kind = PrimitiveType::PrimitiveTypeKind::I8;
    else if (token_str == "i16")
      kind = PrimitiveType::PrimitiveTypeKind::I16;
    else if (token_str == "i32")
      kind = PrimitiveType::PrimitiveTypeKind::I32;
    else if (token_str == "i64")
      kind = PrimitiveType::PrimitiveTypeKind::I64;
    else if (token_str == "u8")
      kind = PrimitiveType::PrimitiveTypeKind::U8;
    else if (token_str == "u16")
      kind = PrimitiveType::PrimitiveTypeKind::U16;
    else if (token_str == "u32")
      kind = PrimitiveType::PrimitiveTypeKind::U32;
    else if (token_str == "u64")
      kind = PrimitiveType::PrimitiveTypeKind::U64;
    else if (token_str == "f32")
      kind = PrimitiveType::PrimitiveTypeKind::F32;
    else if (token_str == "f64")
      kind = PrimitiveType::PrimitiveTypeKind::F64;
    else if (token_str == "string")
      kind = PrimitiveType::PrimitiveTypeKind::String;
    else
      return std::make_unique<IdentifierType>(std::move(token_str));
    return std::make_unique<PrimitiveType>(kind);
  }

  // tuple_type = '(' type (',' type)* ')'
  std::unique_ptr<Type> parse_tuple_type() {
    consume_kind(TokenKind::LParen);
    std::vector<std::unique_ptr<Type>> types;
    while (!is_peek(TokenKind::RParen)) {
      auto type = parse_type();
      types.emplace_back(std::move(type));
      if (is_peek(TokenKind::Comma)) {
        consume();
      }
    }
    consume_kind(TokenKind::RParen);
    return std::make_unique<TupleType>(std::move(types));
  }

  // array_type = slice_type | fixed_array_type
  // slice_type = '[]' type
  // fixed_array_type = '['expr']' type
  std::unique_ptr<Type> parse_array_type() {
    if (is_peek2(TokenKind::RBracket)) {
      consume();
      consume();
      auto type = parse_type();
      return std::make_unique<SliceType>(std::move(type));
    } else {
      consume();
      auto expr = parse_expr(0);
      consume_kind(TokenKind::RBracket);
      auto type = parse_type();
      return std::make_unique<ArrayType>(std::move(type), std::move(expr));
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
  std::unique_ptr<MatchExpr> parse_match_expr(bool consume_match = true) {
    if (consume_match)
      consume_kind(TokenKind::KeywordMatch);
    auto expr = parse_expr(0);
    consume_kind(TokenKind::LBrace);
    std::vector<std::unique_ptr<MatchArm>> arms;
    while (!is_peek(TokenKind::RBrace)) {
      auto arm = parse_match_arm();
      arms.emplace_back(std::move(arm));
      consume_kind(TokenKind::Comma);
    }
    consume_kind(TokenKind::RBrace);
    return std::make_unique<MatchExpr>(std::move(expr), std::move(arms));
  }

  // match_arm = 'is' pattern (if expr)? '=>' expr | block
  std::unique_ptr<MatchArm> parse_match_arm() {
    auto is_token = consume_kind(TokenKind::KeywordIs);
    auto pattern = parse_pattern();
    std::optional<std::unique_ptr<Expression>> guard = std::nullopt;
    if (is_peek(TokenKind::KeywordIf)) {
      consume();
      guard = parse_expr(0);
    }
    consume_kind(TokenKind::EqualGreater);
    if (is_peek(TokenKind::LBrace)) {
      auto block = parse_block();
      return std::make_unique<MatchArm>(std::move(pattern), std::move(block),
                                        std::move(guard));
    }
    auto expr = parse_expr(0);
    return std::make_unique<MatchArm>(std::move(pattern), std::move(expr),
                                      std::move(guard));
  }

  std::unique_ptr<Pattern> parse_pattern() {
    auto pattern = parse_single_pattern();
    if (is_peek(TokenKind::Pipe)) {
      return parse_or_pattern(std::move(pattern));
    }
    if (is_peek(TokenKind::DotDot) || is_peek(TokenKind::DotDotEqual)) {
      return parse_range_pattern(std::move(pattern));
    }
    return pattern;
  }

  // rest_pattern = '..' (as identifier)?
  std::unique_ptr<RestPattern> parse_rest_pattern() {
    consume_kind(TokenKind::DotDot);
    std::optional<IdentifierExpr> name = std::nullopt;
    if (is_peek(TokenKind::KeywordAs)) {
      consume();
      auto ident = consume_kind(TokenKind::Identifier);
      name = IdentifierExpr(lexer->token_to_string(ident));
    }
    return std::make_unique<RestPattern>(std::move(name));
  }

  // variant_pattern = '.' identifier (tuple_pattern | struct_pattern)
  std::unique_ptr<VariantPattern> parse_variant_pattern() {
    consume_kind(TokenKind::Dot);
    auto name = consume_kind(TokenKind::Identifier);
    if (is_peek(TokenKind::LParen)) {
      auto tuple = parse_tuple_pattern();
      return std::make_unique<VariantPattern>(lexer->token_to_string(name),
                                              std::move(tuple));
    } else if (is_peek(TokenKind::LBrace)) {
      auto struct_pattern = parse_struct_pattern();
      return std::make_unique<VariantPattern>(lexer->token_to_string(name),
                                              std::move(struct_pattern));
    }
    return std::make_unique<VariantPattern>(lexer->token_to_string(name),
                                            std::nullopt);
  }

  // pattern = literal_pattern | identifier_pattern | wildcard_pattern |
  //          tuple_pattern | struct_pattern | enum_variant_pattern |
  //          slice_pattern | or_pattern | range_pattern
  std::unique_ptr<Pattern> parse_single_pattern() {
    if (is_peek(TokenKind::NumberLiteral) ||
        is_peek(TokenKind::StringLiteral) || is_peek(TokenKind::CharLiteral)) {
      return parse_literal_pattern();
    } else if (is_peek(TokenKind::Identifier)) {
      if (is_peek2(TokenKind::LBrace)) {
        auto name = consume();
        return parse_struct_pattern(lexer->token_to_string(name.value()));
      }
      auto str = lexer->token_to_string(peek().value());
      if (str == "_") {
        return parse_wildcard_pattern();
      }
      return parse_identifier_pattern();
    } else if (is_peek(TokenKind::LParen)) {
      return parse_tuple_pattern();
    } else if (is_peek(TokenKind::Dot)) {
      return parse_variant_pattern();
    } else if (is_peek(TokenKind::DotDot)) {
      return parse_rest_pattern();
    }
    // else if (is_peek(TokenKind::LBrace)) {
    //   return parse_struct_pattern();
    // }
    else if (is_peek(TokenKind::LBracket)) {
      return parse_slice_pattern();
    } else {
      auto expr = parse_expr(0);
      return std::make_unique<ExprPattern>(std::move(expr));
    }
  }

  // literal_pattern = number_literal | string_literal | char_literal
  std::unique_ptr<LiteralPattern> parse_literal_pattern() {
    auto token = consume();
    std::unique_ptr<LiteralExpr> expr;
    switch (token->kind) {
    case TokenKind::NumberLiteral:
      expr = parse_number_literal_expr(token.value());
      break;
    case TokenKind::StringLiteral:
      expr =
          std::make_unique<LiteralExpr>(LiteralExpr::LiteralType::String,
                                        lexer->token_to_string(token.value()));
      break;
    case TokenKind::CharLiteral:
      expr = std::make_unique<LiteralExpr>(
          LiteralExpr::LiteralType::Char,
          lexer->token_to_string(token.value())[0]);
      break;
    default:
      break;
    }
    return std::make_unique<LiteralPattern>(std::move(expr));
  }

  // identifier_pattern = identifier
  std::unique_ptr<IdentifierPattern> parse_identifier_pattern() {
    auto token = consume_kind(TokenKind::Identifier);
    return std::make_unique<IdentifierPattern>(lexer->token_to_string(token));
  }

  // wildcard_pattern = '_'
  std::unique_ptr<WildcardPattern> parse_wildcard_pattern() {
    consume_kind(TokenKind::Identifier);
    return std::make_unique<WildcardPattern>();
  }

  // tuple_pattern = '(' pattern (',' pattern)* ')'
  std::unique_ptr<TuplePattern> parse_tuple_pattern() {
    consume_kind(TokenKind::LParen);
    std::vector<std::unique_ptr<Pattern>> patterns;
    while (!is_peek(TokenKind::RParen)) {
      auto pattern = parse_pattern();
      patterns.emplace_back(std::move(pattern));
      if (is_peek(TokenKind::Comma)) {
        consume();
      }
    }
    consume_kind(TokenKind::RParen);
    return std::make_unique<TuplePattern>(std::move(patterns));
  }

  // struct_pattern = '{' pattern_field* '}'
  std::unique_ptr<StructPattern>
  parse_struct_pattern(std::optional<std::string> name = std::nullopt) {
    consume_kind(TokenKind::LBrace);
    std::vector<StructPattern::Field> fields;
    while (!is_peek(TokenKind::RBrace)) {
      if (is_peek(TokenKind::DotDot)) {
        auto rest = parse_rest_pattern();
        fields.emplace_back(std::move(rest));
        break;
      }
      auto field = parse_pattern_field();
      fields.emplace_back(std::move(field));
    }
    consume_kind(TokenKind::RBrace);
    return std::make_unique<StructPattern>(std::move(name), std::move(fields));
  }

  // pattern_field = identifier ':' pattern ','?
  std::unique_ptr<PatternField> parse_pattern_field() {
    auto name = consume_kind(TokenKind::Identifier);
    std::optional<std::unique_ptr<Pattern>> pattern = std::nullopt;
    if (is_peek(TokenKind::Colon)) {
      consume();
      pattern = parse_pattern();
    }
    if (is_peek(TokenKind::Comma)) {
      consume();
    }
    return std::make_unique<PatternField>(lexer->token_to_string(name),
                                          std::move(pattern));
  }

  // slice_pattern = '[' pattern (',' pattern)* ']'
  std::unique_ptr<SlicePattern> parse_slice_pattern() {
    consume_kind(TokenKind::LBracket);
    std::vector<std::unique_ptr<Pattern>> patterns;
    while (!is_peek(TokenKind::RBracket)) {
      auto pattern = parse_pattern();
      patterns.emplace_back(std::move(pattern));
      if (is_peek(TokenKind::Comma)) {
        consume();
      }
    }
    consume_kind(TokenKind::RBracket);
    return std::make_unique<SlicePattern>(std::move(patterns), true);
  }

  // or_pattern = pattern '|' pattern
  std::unique_ptr<Pattern>
  parse_or_pattern(std::unique_ptr<Pattern> first_pattern) {
    std::vector<std::unique_ptr<Pattern>> alternatives;
    alternatives.emplace_back(std::move(first_pattern));

    while (is_peek(TokenKind::Pipe)) {
      consume();
      alternatives.emplace_back(parse_single_pattern());
    }

    return std::make_unique<OrPattern>(std::move(alternatives));
  }

  // range_pattern = pattern '..' expr
  // range_pattern = expr '..=' expr
  std::unique_ptr<Pattern>
  parse_range_pattern(std::unique_ptr<Pattern> start_pattern) {
    auto range_type = consume(); // Consume the `..` or `..=`
    bool inclusive = (range_type->kind == TokenKind::DotDotEqual);

    // Parse the end pattern
    auto end_pattern = parse_single_pattern();
    if (!end_pattern) {
      context->diagnostics.report_error(
          range_type.value(), "Expected a pattern after range operator");
      return std::make_unique<InvalidPattern>();
    }

    return std::make_unique<RangePattern>(std::move(start_pattern),
                                          std::move(end_pattern), inclusive);
  }

  // if_expr = 'if' expr block ('else' block)?
  std::unique_ptr<IfExpr> parse_if_expr(bool consume_if = true) {
    if (consume_if)
      consume_kind(TokenKind::KeywordIf);
    auto condition = parse_expr(0);
    IfExpr::Branch then_branch;
    if (is_peek(TokenKind::LBrace)) {
      then_branch = parse_block();
    } else {
      then_branch = parse_expr(0);
    }
    std::optional<IfExpr::Branch> else_branch = std::nullopt;
    if (is_peek(TokenKind::KeywordElse)) {
      consume();
      if (is_peek(TokenKind::LBrace))
        else_branch = parse_block();
      else
        else_branch = parse_expr(0);
    }
    return std::make_unique<IfExpr>(
        std::move(condition), std::move(then_branch), std::move(else_branch));
  }

  // while_expr = 'while' expr : expr block
  std::unique_ptr<WhileExpr>
  parse_while_expr(bool consume_while = true,
                   std::optional<Token> label = std::nullopt) {
    std::optional<std::string> label_name = std::nullopt;
    if (label.has_value()) {
      label_name = lexer->token_to_string(label.value());
    }
    if (consume_while)
      consume_kind(TokenKind::KeywordWhile);
    std::optional<std::unique_ptr<Expression>> condition = std::nullopt;
    if (!is_peek(TokenKind::LBrace)) {
      condition = parse_expr(0);
    }
    std::optional<std::unique_ptr<Expression>> expr = std::nullopt;
    if (is_peek(TokenKind::Colon)) {
      consume();
      expr = parse_expr(0);
    }
    auto block = parse_block();
    return std::make_unique<WhileExpr>(std::move(condition), std::move(expr),
                                       std::move(block), std::move(label_name));
  }

  // for_expr = 'for' identifier 'in' expr block
  std::unique_ptr<ForExpr>
  parse_for_expr(bool consume_for = true,
                 std::optional<Token> label = std::nullopt) {
    std::optional<std::string> label_name = std::nullopt;
    if (label.has_value()) {
      label_name = lexer->token_to_string(label.value());
    }
    if (consume_for)
      consume_kind(TokenKind::KeywordFor);
    auto pattern = parse_pattern();
    consume_kind(TokenKind::KeywordIn);
    auto iterable = parse_expr(0);
    auto block = parse_block();
    return std::make_unique<ForExpr>(std::move(pattern), std::move(iterable),
                                     std::move(block), std::move(label_name));
  }

  // call_expr = identifier '(' expr (',' expr)* ')'
  std::unique_ptr<CallExpr> parse_call_expr() {
    auto name = current_token.value();
    consume_kind(TokenKind::LParen);
    std::vector<std::unique_ptr<Expression>> args;
    while (!is_peek(TokenKind::RParen)) {
      auto arg = parse_expr(0);
      args.emplace_back(std::move(arg));
      if (is_peek(TokenKind::Comma)) {
        consume();
      }
    }
    consume_kind(TokenKind::RParen);
    auto name_expr =
        std::make_unique<IdentifierExpr>(lexer->token_to_string(name));
    return std::make_unique<CallExpr>(std::move(name_expr), std::move(args));
  }

  // return_stmt = 'return' expr? ';'?
  std::unique_ptr<ReturnExpr> parse_return_expr() {
    auto return_token = consume_kind(TokenKind::KeywordReturn);
    auto expr = parse_expr(0);
    return std::make_unique<ReturnExpr>(std::move(expr));
  }

  // stmt = expr_stmt | var_decl | return_stmt | block_stmt |
  // if_stmt |
  //       for_stmt | while_stmt | break_stmt | continue_stmt | match_stmt
  std::unique_ptr<Statement> parse_stmt() {
    if (is_peek(TokenKind::KeywordVar) || is_peek(TokenKind::KeywordConst)) {
      return parse_var_decl();
    } else if (is_peek(TokenKind::KeywordReturn)) {
      auto expr = parse_return_expr();
      consume_kind(TokenKind::Semicolon);
      return std::make_unique<ExprStmt>(std::move(expr));
    } else if (is_peek(TokenKind::KeywordIf)) {
      auto expr = parse_if_expr();
      return std::make_unique<ExprStmt>(std::move(expr));
    } else if (is_peek(TokenKind::KeywordFor)) {
      auto expr = parse_for_expr();
      return std::make_unique<ExprStmt>(std::move(expr));
    } else if (is_peek(TokenKind::KeywordWhile)) {
      auto expr = parse_while_expr();
      return std::make_unique<ExprStmt>(std::move(expr));
    } else if (is_peek(TokenKind::KeywordBreak)) {
      auto break_expr = parse_break_expr();
      consume_kind(TokenKind::Semicolon);
      return std::make_unique<ExprStmt>(std::move(break_expr));
    } else if (is_peek(TokenKind::KeywordContinue)) {
      auto continue_expr = parse_continue_expr();
      consume_kind(TokenKind::Semicolon);
      return std::make_unique<ExprStmt>(std::move(continue_expr));
    } else if (is_peek(TokenKind::KeywordMatch)) {
      auto match_expr = parse_match_expr();
      return std::make_unique<ExprStmt>(std::move(match_expr));
    } else if (is_peek(TokenKind::Identifier) && is_peek2(TokenKind::Colon)) {
      // labeled statements
      auto label = consume();
      consume();
      if (is_peek(TokenKind::KeywordWhile)) {
        auto expr = parse_while_expr(true, label);
        return std::make_unique<ExprStmt>(std::move(expr));
      }
    } else if (is_peek(TokenKind::KeywordStruct)) {
      consume();
      auto struct_decl = parse_struct_decl();
      return std::make_unique<TopLevelDeclStmt>(std::move(struct_decl));
    } else if (is_peek(TokenKind::KeywordFn)) {
      auto function = parse_function();
      return std::make_unique<TopLevelDeclStmt>(std::move(function));
    } else {
      auto token = peek().value();
      auto expr = parse_expr(0);
      // if block statement then no need for semicolon
      if (token.kind != TokenKind::LBrace)
        consume_kind(TokenKind::Semicolon);
      return std::make_unique<ExprStmt>(std::move(expr));
    }
    return std::make_unique<InvalidStatement>();
  }

  // var_decl = 'var' | 'const' identifier ('=' expr)? ';'
  std::unique_ptr<VarDecl> parse_var_decl() {
    bool is_mut = false;
    if (peek()->kind == TokenKind::KeywordConst) {
      consume_kind(TokenKind::KeywordConst);
    } else {
      is_mut = true;
      consume_kind(TokenKind::KeywordVar);
    }
    auto pattern = parse_pattern();
    std::optional<std::unique_ptr<Type>> type = std::nullopt;
    if (peek()->kind == TokenKind::Colon) {
      consume();
      type = parse_type();
    }
    std::optional<std::unique_ptr<Expression>> expr = std::nullopt;
    if (is_peek(TokenKind::Equal)) {
      consume();
      expr = parse_expr(0);
    }
    consume_kind(TokenKind::Semicolon);
    return std::make_unique<VarDecl>(std::move(pattern), std::move(type),
                                     std::move(expr), is_mut);
  }

  // break_expr = 'break' (':' identifier)? (expr)?;
  std::unique_ptr<BreakExpr> parse_break_expr(bool consume_break = true) {
    if (consume_break)
      consume_kind(TokenKind::KeywordBreak);
    std::optional<std::string> label = std::nullopt;
    if (is_peek(TokenKind::Colon)) {
      consume();
      auto name = consume_kind(TokenKind::Identifier);
      label = lexer->token_to_string(name);
    }
    std::optional<std::unique_ptr<Expression>> expr = std::nullopt;
    if (!is_peek(TokenKind::Semicolon) && !is_peek(TokenKind::Comma)) {
      expr = parse_expr(0);
    }
    return std::make_unique<BreakExpr>(std::move(label), std::move(expr));
  }

  // continue_expr = 'continue' (':' identifier)? (expr)?;
  std::unique_ptr<ContinueExpr>
  parse_continue_expr(bool consume_continue = true) {
    if (consume_continue)
      consume_kind(TokenKind::KeywordContinue);
    std::optional<std::string> label = std::nullopt;
    if (is_peek(TokenKind::Colon)) {
      consume();
      auto name = consume_kind(TokenKind::Identifier);
      label = lexer->token_to_string(name);
    }
    std::optional<std::unique_ptr<Expression>> expr = std::nullopt;
    if (!is_peek(TokenKind::Semicolon) && !is_peek(TokenKind::Comma)) {
      expr = parse_expr(0);
    }
    return std::make_unique<ContinueExpr>(std::move(label), std::move(expr));
  }

  std::variant<int, double>
  parse_number_literal(const std::basic_string<char> &bytes) {
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
        // return {.failure = {.upper_case_base = 1}};
        context->diagnostics.report_error(current_token.value(),
                                          "Upper case base not allowed");
        break;
      case '.':
      case 'e':
      case 'E':
        break;
      default:
        // return {.failure = {.leading_zero}};
        context->diagnostics.report_error(current_token.value(),
                                          "Leading zero not allowed");
        break;
      }
    }
    if (bytes.size() == 2 && base != 10) {
      // return {.failure = {.digit_after_base}};
      context->diagnostics.report_error(current_token.value(),
                                        "Digit after base not allowed");
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
          // return {.failure = {.invalid_underscore_after_special = i}};
          context->diagnostics.report_error(current_token.value(),
                                            "Invalid underscore after special");
        }
        if (special != 0) {
          // return {.failure = {.invalid_underscore_after_special = i}};
          context->diagnostics.report_error(current_token.value(),
                                            "Invalid underscore after special");
        }
        if (underscore) {
          // return {.failure = {.repeated_underscore = i}};
          context->diagnostics.report_error(current_token.value(),
                                            "Repeated underscore");
        }
        underscore = true;
        ++i;
        continue;
      case 'e':
      case 'E':
        if (base == 10) {
          floating = true;
          if (exponent) {
            // return {.failure = {.duplicate_exponent = i}};
            context->diagnostics.report_error(current_token.value(),
                                              "Duplicate exponent");
          }
          if (underscore) {
            // return {.failure = {.exponent_after_underscore = i}};
            context->diagnostics.report_error(current_token.value(),
                                              "Exponent after underscore");
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
            // return {.failure = {.duplicate_exponent = i}};
            context->diagnostics.report_error(current_token.value(),
                                              "Duplicate exponent");
          }
          if (underscore) {
            // return {.failure = {.exponent_after_underscore = i}};
            context->diagnostics.report_error(current_token.value(),
                                              "Exponent after underscore");
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
          // return {.failure = {.invalid_float_base = 2}};
          context->diagnostics.report_error(current_token.value(),
                                            "Invalid float base");
        }
        if (period) {
          // return {.failure = {.duplicate_period}};
          context->diagnostics.report_error(current_token.value(),
                                            "Duplicate period");
        }
        period = true;
        if (underscore) {
          // return {.failure = {.special_after_underscore = i}};
          context->diagnostics.report_error(current_token.value(),
                                            "Special after underscore");
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
            // return {.failure = {.invalid_exponent_sign = i}};
            context->diagnostics.report_error(current_token.value(),
                                              "Invalid exponent sign");
          }
          break;
        default:
          // return {.failure = {.invalid_exponent_sign = i}};
          context->diagnostics.report_error(current_token.value(),
                                            "Invalid exponent sign");
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
        // return {.failure = {.invalid_character = i}};
        context->diagnostics.report_error(current_token.value(),
                                          "Invalid character");
        return 0;
      }();
      if (digit >= base) {
        // return {.failure = {.invalid_digit = {.i = i, .base = base}};
        context->diagnostics.report_error(current_token.value(),
                                          "Invalid digit");
      }

      if (exponent && digit >= 10) {
        // return {.failure = {.invalid_digit_exponent = i}};
        context->diagnostics.report_error(current_token.value(),
                                          "Invalid digit exponent");
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
      // return {.failure = {.trailing_underscore}};
      context->diagnostics.report_error(current_token.value(),
                                        "Trailing underscore");
    }
    if (special != 0) {
      // return {.failure = {.trailing_special}};
      context->diagnostics.report_error(current_token.value(),
                                        "Trailing special");
    }

    if (floating) {
      // return {.success = {.floating = {.value = x}}};
      return stod(lexer->token_to_string(current_token.value()));
    }
    if (overflow) {
      // bigint
      context->diagnostics.report_error(current_token.value(), "Overflow");
    }
    return (int)x;
  }

  Operator token_to_operator(const Token &op) {
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
      return Operator::Xor;
    case TokenKind::LessLess:
    case TokenKind::LessLessEqual:
      return Operator::BitShl;
    case TokenKind::GreaterGreater:
    case TokenKind::GreaterGreaterEqual:
      return Operator::BitShr;
    case TokenKind::StarStar:
      return Operator::Pow;
    default:
      context->diagnostics.report_error(op, "Invalid operator");
      return Operator::Invalid;
    }
  }
};
