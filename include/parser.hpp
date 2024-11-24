#pragma once

#include <memory>
#include <optional>

#include "ast.hpp"
#include "compiler.hpp"
#include "lexer.hpp"

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

  bool isFileLoaded(llvm::SourceMgr &sourceMgr, const std::string &filePath);
  void load_builtins();
  Token get_error_token();
  Token unexpected_token_error(TokenKind &expected, Token &found);
  Token invalid_token_error(Token &found);
  std::optional<Token> consume();
  Token consume_kind(TokenKind kind);
  std::optional<Token> peek();
  std::optional<Token> peek2();
  bool is_peek(TokenKind kind);
  bool is_peek2(TokenKind kind);
  void consume_optional_semicolon();
  int binding_pow(const Token &token);
  std::unique_ptr<LiteralExpr> parse_number_literal_expr(const Token &token);
  std::unique_ptr<Expression> nud(const Token &token);
  std::unique_ptr<Expression> led(std::unique_ptr<Expression> left, Token &op);
  std::unique_ptr<Program> parse_single_source(std::string &path);
  std::unique_ptr<Program> parse_program();
  void skip_to_next_top_level_decl();
  void skip_to_next_stmt();
  std::unique_ptr<ImportDecl> parse_import_decl();
  std::unique_ptr<TopLevelDecl> parse_top_level_decl();
  std::unique_ptr<Expression> parse_expr(int precedence);
  std::unique_ptr<Expression>
  parse_tuple_expr(std::unique_ptr<Expression> first_expr);
  std::unique_ptr<Expression> parse_array_expr();
  std::unique_ptr<Function> parse_function(bool is_pub = false);
  TraitDecl::Method parse_trait_method();
  std::vector<std::unique_ptr<Parameter>> parse_params();
  std::unique_ptr<Parameter> parse_param();
  std::unique_ptr<BlockExpression> parse_block(bool consume_lbrace = true);
  std::unique_ptr<StructDecl> parse_struct_decl(bool is_pub = false);
  std::unique_ptr<StructField> parse_struct_field();
  std::unique_ptr<TupleStructDecl> parse_tuple_struct_decl(bool is_pub = false);
  std::unique_ptr<EnumDecl> parse_enum_decl(bool is_pub = false);
  std::unique_ptr<FieldsUnnamed> parse_field_unnamed();
  std::unique_ptr<FieldsNamed> parse_field_named();
  std::unique_ptr<Variant> parse_enum_variant();
  std::unique_ptr<ImplDecl> parse_impl_decl();
  std::unique_ptr<TraitDecl> parse_trait_decl(bool is_pub = false);
  std::unique_ptr<Type> parse_type();
  std::unique_ptr<Type> parse_tuple_type();
  std::unique_ptr<Type> parse_array_type();
  std::unique_ptr<MatchExpr> parse_match_expr(bool consume_match = true);
  std::unique_ptr<MatchArm> parse_match_arm();
  std::unique_ptr<Pattern> parse_pattern();
  std::unique_ptr<RestPattern> parse_rest_pattern();
  std::unique_ptr<VariantPattern> parse_variant_pattern();
  std::unique_ptr<Pattern> parse_single_pattern();
  std::unique_ptr<LiteralPattern> parse_literal_pattern();
  std::unique_ptr<IdentifierPattern> parse_identifier_pattern();
  std::unique_ptr<WildcardPattern> parse_wildcard_pattern();
  std::unique_ptr<TuplePattern> parse_tuple_pattern();
  std::unique_ptr<StructPattern>
  parse_struct_pattern(std::optional<std::string> name = std::nullopt);
  std::unique_ptr<PatternField> parse_pattern_field();
  std::unique_ptr<SlicePattern> parse_slice_pattern();
  std::unique_ptr<Pattern>
  parse_or_pattern(std::unique_ptr<Pattern> first_pattern);
  std::unique_ptr<Pattern>
  parse_range_pattern(std::unique_ptr<Pattern> start_pattern);
  std::unique_ptr<IfExpr> parse_if_expr(bool consume_if = true);
  std::unique_ptr<WhileExpr>
  parse_while_expr(bool consume_while = true,
                   std::optional<Token> label = std::nullopt);
  std::unique_ptr<ForExpr>
  parse_for_expr(bool consume_for = true,
                 std::optional<Token> label = std::nullopt);
  std::unique_ptr<CallExpr> parse_call_expr();
  std::unique_ptr<ReturnExpr> parse_return_expr();
  std::unique_ptr<Statement> parse_stmt();
  std::unique_ptr<VarDecl> parse_var_decl(bool is_pub = false);
  std::unique_ptr<BreakExpr> parse_break_expr(bool consume_break = true);
  std::unique_ptr<ContinueExpr>
  parse_continue_expr(bool consume_continue = true);
  std::unique_ptr<Type> parse_mlir_type(bool consume_at = true);
  std::unique_ptr<MLIRAttribute> parse_mlir_attr();
  std::unique_ptr<MLIROp> parse_mlir_op();
  std::unique_ptr<YieldExpr> parse_yield_expr(bool consume_yield = true);
  std::variant<int, double>
  parse_number_literal(const std::basic_string<char> &bytes);
  Operator token_to_operator(const Token &op);
};
