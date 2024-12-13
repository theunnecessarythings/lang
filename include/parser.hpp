#pragma once

#include <memory>
#include <optional>
#include <unordered_set>

#include "ast.hpp"
#include "compiler.hpp"
#include "lexer.hpp"

using PatternBindings = std::vector<std::pair<std::string, Type *>>;

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

  Attribute tokenToAttribute(const Token &token);
  bool isFileLoaded(llvm::SourceMgr &sourceMgr, const std::string &filePath);
  void loadBuiltins();
  Token getErrorToken();
  Token unexpectedTokenError(TokenKind &expected, Token &found);
  Token invalidTokenError(Token &found);
  std::optional<Token> consume();
  Token consumeKind(TokenKind kind);
  std::optional<Token> peek();
  std::optional<Token> peek2();
  bool isPeek(TokenKind kind);
  bool isPeek2(TokenKind kind);
  void consumeOptionalSemicolon();
  int bindingPow(const Token &token);
  std::unique_ptr<LiteralExpr> parseNumberLiteralExpr(const Token &token);
  std::unique_ptr<Expression> nud(const Token &token);
  std::unique_ptr<Expression> led(std::unique_ptr<Expression> left, Token &op);
  std::unique_ptr<Program> parseSingleSource(std::string &path);
  std::unique_ptr<Program> parseProgram();
  void skipToNextTopLevelDecl();
  void skipToNextStmt();
  std::unique_ptr<ImportDecl> parseImportDecl();
  std::unique_ptr<TopLevelDecl> parseTopLevelDecl();
  std::unique_ptr<Expression> parseExpr(int precedence);
  std::unique_ptr<Expression>
  parseTupleExpr(std::unique_ptr<Expression> first_expr);
  std::unique_ptr<Expression> parseArrayExpr();
  std::unique_ptr<Function>
  parseFunction(bool is_pub = false, std::unordered_set<Attribute> attrs = {});
  TraitDecl::Method parseTraitMethod();
  std::vector<std::unique_ptr<Parameter>> parseParams();
  std::unique_ptr<Parameter> parseParam();
  std::unique_ptr<BlockExpression> parseBlock(bool consume_lbrace = true);
  std::unique_ptr<StructDecl> parseStructDecl(bool is_pub = false);
  std::unique_ptr<StructField> parseStructField();
  std::unique_ptr<TupleStructDecl> parseTupleStructDecl(bool is_pub = false);
  std::unique_ptr<EnumDecl> parseEnumDecl(bool is_pub = false);
  std::unique_ptr<FieldsUnnamed> parseFieldUnnamed();
  std::unique_ptr<FieldsNamed> parseFieldNamed();
  std::unique_ptr<Variant> parseEnumVariant();
  std::unique_ptr<ImplDecl> parseImplDecl();
  std::unique_ptr<TraitDecl> parseTraitDecl(bool is_pub = false);
  std::unique_ptr<Type> parseType();
  std::unique_ptr<Type> parseTupleType();
  std::unique_ptr<Type> parseArrayType();
  std::unique_ptr<MatchExpr> parseMatchExpr(bool consume_match = true);
  std::unique_ptr<MatchArm> parseMatchArm();
  std::unique_ptr<Pattern> parsePattern();
  std::unique_ptr<RestPattern> parseRestPattern();
  std::unique_ptr<VariantPattern> parseVariantPattern();
  std::unique_ptr<Pattern> parseSinglePattern();
  std::unique_ptr<LiteralPattern> parseLiteralPattern();
  std::unique_ptr<IdentifierPattern> parseIdentifierPattern();
  std::unique_ptr<WildcardPattern> parseWildcardPattern();
  std::unique_ptr<TuplePattern> parseTuplePattern();
  std::unique_ptr<StructPattern>
  parseStructPattern(std::optional<std::string> name = std::nullopt);
  std::unique_ptr<PatternField> parsePatternField();
  std::unique_ptr<SlicePattern> parseSlicePattern();
  std::unique_ptr<Pattern>
  parseOrPattern(std::unique_ptr<Pattern> first_pattern);
  std::unique_ptr<Pattern>
  parseRangePattern(std::unique_ptr<Pattern> start_pattern);
  std::unique_ptr<IfExpr> parseIfExpr(bool consume_if = true);
  std::unique_ptr<WhileExpr>
  parseWhileExpr(bool consume_while = true,
                 std::optional<Token> label = std::nullopt);
  std::unique_ptr<ForExpr>
  parseForExpr(bool consume_for = true,
               std::optional<Token> label = std::nullopt);
  std::unique_ptr<CallExpr> parseCallExpr();
  std::unique_ptr<ReturnExpr> parseReturnExpr();
  std::unique_ptr<Statement> parseStmt();
  std::unique_ptr<VarDecl> parseVarDecl(bool is_pub = false);
  std::unique_ptr<BreakExpr> parseBreakExpr(bool consume_break = true);
  std::unique_ptr<ContinueExpr> parseContinueExpr(bool consume_continue = true);
  std::unique_ptr<Type> parseMlirType(bool consume_at = true);
  std::unique_ptr<MLIRAttribute> parseMlirAttr();
  std::unique_ptr<MLIROp> parseMlirOp();
  std::unique_ptr<YieldExpr> parseYieldExpr(bool consume_yield = true);
  std::variant<int, double>
  parseNumberLiteral(const std::basic_string<char> &bytes);
  Operator tokenToOperator(const Token &op);
  std::unordered_set<Attribute> parseAttributes();

  PatternBindings destructurePattern(Pattern *pattern, Type *type);
};
