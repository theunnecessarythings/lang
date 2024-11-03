#pragma once

#include "lexer.hpp"
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace Ast {

enum class AstNodeKind {
  Program,
  Module,
  ComptimeExpr,
  BlockExpression,
  BlockStatement,
  Parameter,
  StructField,
  StructDecl,
  TupleStructDecl,
  IdentifierExpr,
  FieldsNamed,
  FieldsUnnamed,
  Variant,
  UnionField,
  FunctionDecl,
  Function,
  ImportDecl,
  EnumDecl,
  UnionDecl,
  TraitDecl,
  ImplDecl,
  VarDecl,
  TopLevelVarDecl,
  IfExpr,
  MatchArm,
  MatchExpr,
  ForExpr,
  WhileExpr,
  ReturnExpr,
  DeferStmt,
  BreakExpr,
  ContinueExpr,
  ExprStmt,
  LiteralExpr,
  TupleExpr,
  ArrayExpr,
  BinaryExpr,
  UnaryExpr,
  CallExpr,
  AssignExpr,
  AssignOpExpr,
  FieldAccessExpr,
  IndexExpr,
  RangeExpr,
  PrimitiveType,
  PointerType,
  ArrayType,
  TupleType,
  FunctionType,
  StructType,
  EnumType,
  UnionType,
  TraitType,
  ImplType,
  SliceType,
  ReferenceType,
  IdentifierType,
  ExprType,
  LiteralPattern,
  IdentifierPattern,
  TuplePattern,
  StructPattern,
  EnumPattern,
  SlicePattern,
  WildcardPattern,
  RestPattern,
  OrPattern,
  ExprPattern,
  PatternField,
  RangePattern,
  VariantPattern,
  TopLevelDeclStmt,

  InvalidNode,
  InvalidExpression,
  InvalidStatement,
  InvalidTopLevelDecl,
  InvalidType,
  InvalidPattern,
};

struct Node {
  Token token;
  Node(Token token) : token(std::move(token)) {}
  virtual ~Node() = default;
  virtual void render() const = 0;
  virtual AstNodeKind kind() const = 0;
};

struct Expression : public Node {
  Expression(Token token) : Node(std::move(token)) {}
  virtual ~Expression() = default;
  virtual void render() const = 0;
};
struct Statement : public Node {
  Statement(Token token) : Node(std::move(token)) {}
  virtual ~Statement() = default;
  virtual void render() const = 0;
};
struct TopLevelDecl : public Node {
  TopLevelDecl(Token token) : Node(std::move(token)) {}
  virtual ~TopLevelDecl() = default;
  virtual void render() const = 0;
};
struct Type : public Node {
  Type(Token token) : Node(std::move(token)) {}
  virtual ~Type() = default;
  virtual void render() const = 0;
};
struct Pattern : public Node {
  Pattern(Token token) : Node(std::move(token)) {}
  virtual ~Pattern() = default;
  virtual void render() const = 0;
};

struct InvalidNode : public Node {
  InvalidNode(Token token) : Node(std::move(token)) {}
  void render() const override {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidNode; }
};
struct InvalidExpression : public Expression {
  InvalidExpression(Token token) : Expression(std::move(token)) {}
  void render() const override {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidExpression; }
};
struct InvalidStatement : public Statement {
  InvalidStatement(Token token) : Statement(std::move(token)) {}
  void render() const override {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidStatement; }
};
struct InvalidTopLevelDecl : public TopLevelDecl {
  InvalidTopLevelDecl(Token token) : TopLevelDecl(std::move(token)) {}
  void render() const override {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidTopLevelDecl; }
};
struct InvalidType : public Type {
  InvalidType(Token token) : Type(std::move(token)) {}
  void render() const override {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidType; }
};
struct InvalidPattern : public Pattern {
  InvalidPattern(Token token) : Pattern(std::move(token)) {}
  void render() const override {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidPattern; }
};

struct Program : public Node {
  std::vector<std::unique_ptr<TopLevelDecl>> items;

  Program(Token token, std::vector<std::unique_ptr<TopLevelDecl>> items)
      : items(std::move(items)), Node(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::Program; }

  void render() const override {
    for (auto &item : items) {
      item->render();
    }
  }
};

struct Module : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<TopLevelDecl>> items;

  Module(Token token, std::string name,
         std::vector<std::unique_ptr<TopLevelDecl>> items)
      : name(std::move(name)), items(std::move(items)),
        TopLevelDecl(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::Module; }

  void render() const override {
    std::cout << "module " << name << " {" << std::endl;
    for (auto &item : items) {
      item->render();
    }
    std::cout << "}" << std::endl;
  }
};

struct ComptimeExpr : public Expression {
  std::unique_ptr<Expression> expr;

  ComptimeExpr(Token token, std::unique_ptr<Expression> expr)
      : expr(std::move(expr)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ComptimeExpr; }

  void render() const override {
    std::cout << "comptime ";
    expr->render();
  }
};

struct BlockExpression : public Expression {
  std::vector<std::unique_ptr<Statement>> statements;

  BlockExpression(Token token,
                  std::vector<std::unique_ptr<Statement>> statements)
      : statements(std::move(statements)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::BlockExpression; }

  void render() const override {
    std::cout << "{" << std::endl;
    for (auto &stmt : statements) {
      stmt->render();
    }
    std::cout << "}" << std::endl;
  }
};

struct Parameter : public Node {
  std::unique_ptr<Pattern> pattern;
  std::unique_ptr<Type> type;
  std::optional<std::unique_ptr<Type>> trait_bound;
  bool is_mut;
  bool is_comptime;

  Parameter(Token token, std::unique_ptr<Pattern> pattern,
            std::unique_ptr<Type> type,
            std::optional<std::unique_ptr<Type>> trait_bound, bool is_mut,
            bool is_comptime)
      : pattern(std::move(pattern)), type(std::move(type)),
        trait_bound(std::move(trait_bound)), is_mut(is_mut),
        is_comptime(is_comptime), Node(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::Parameter; }

  void render() const override {
    if (is_comptime)
      std::cout << "comptime ";
    if (is_mut)
      std::cout << "mut ";
    pattern->render();
    std::cout << ": ";
    type->render();
    if (trait_bound.has_value()) {
      std::cout << " impl ";
      trait_bound.value()->render();
    }
  }
};

struct StructField : public Node {
  std::string name;
  std::unique_ptr<Type> type;

  StructField(Token token, std::string name, std::unique_ptr<Type> type)
      : name(std::move(name)), type(std::move(type)), Node(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::StructField; }

  void render() const override {
    std::cout << name << ": ";
    type->render();
  }
};

struct StructDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<StructField>> fields;
  bool is_pub;

  StructDecl(Token token, std::string name,
             std::vector<std::unique_ptr<StructField>> fields, bool is_pub)
      : name(std::move(name)), fields(std::move(fields)), is_pub(is_pub),
        TopLevelDecl(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::StructDecl; }

  void render() const override {
    if (is_pub)
      std::cout << "pub ";
    std::cout << "struct " << name << " {" << std::endl;
    for (auto &field : fields) {
      field->render();
      std::cout << "," << std::endl;
    }
    std::cout << "}" << std::endl;
  }

  void render_for_enum() const {
    std::cout << name << " {";
    for (auto &field : fields) {
      field->render();
      if (&field != &fields.back())
        std::cout << ", ";
    }
    std::cout << "}";
  }
};

struct TupleStructDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<Type>> fields;
  bool is_pub;

  TupleStructDecl(Token token, std::string name,
                  std::vector<std::unique_ptr<Type>> fields, bool is_pub)
      : name(std::move(name)), fields(std::move(fields)), is_pub(is_pub),
        TopLevelDecl(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::TupleStructDecl; }

  void render() const override {
    if (is_pub)
      std::cout << "pub ";
    std::cout << "struct " << name << "(";
    for (auto &field : fields) {
      field->render();
      if (&field != &fields.back()) {
        std::cout << ", ";
      }
    }
    std::cout << ");" << std::endl;
  }

  void render_for_enum() const {
    std::cout << name << "(";
    for (auto &field : fields) {
      field->render();
      if (&field != &fields.back())
        std::cout << ", ";
    }
    std::cout << ")";
  }
};

struct IdentifierExpr : public Expression {
  std::string name;

  IdentifierExpr(Token token, std::string name)
      : name(std::move(name)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::IdentifierExpr; }

  void render() const override { std::cout << name; }
};

struct FieldsNamed : public Node {
  std::vector<std::string> name;
  std::vector<std::unique_ptr<Type>> value;

  FieldsNamed(Token token, std::vector<std::string> name,
              std::vector<std::unique_ptr<Type>> value)
      : name(std::move(name)), value(std::move(value)), Node(std::move(token)) {
  }

  AstNodeKind kind() const override { return AstNodeKind::FieldsNamed; }

  void render() const override {
    for (size_t i = 0; i < name.size(); i++) {
      std::cout << name[i] << ": ";
      value[i]->render();
      if (i != name.size() - 1)
        std::cout << ", ";
    }
  }
};

struct FieldsUnnamed : public Node {
  std::vector<std::unique_ptr<Type>> value;

  FieldsUnnamed(Token token, std::vector<std::unique_ptr<Type>> value)
      : value(std::move(value)), Node(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::FieldsUnnamed; }

  void render() const override {
    for (auto &val : value) {
      val->render();
      if (&val != &value.back())
        std::cout << ", ";
    }
  }
};

struct Variant : public Node {
  std::string name;
  std::optional<
      std::variant<std::unique_ptr<FieldsUnnamed>, std::unique_ptr<FieldsNamed>,
                   std::unique_ptr<Expression>>>
      field;

  Variant(Token token, std::string name,
          std::optional<std::variant<std::unique_ptr<FieldsUnnamed>,
                                     std::unique_ptr<FieldsNamed>,
                                     std::unique_ptr<Expression>>>
              field)
      : name(std::move(name)), field(std::move(field)), Node(std::move(token)) {
  }

  AstNodeKind kind() const override { return AstNodeKind::Variant; }

  void render() const override {
    std::cout << name;
    if (field.has_value()) {
      if (std::holds_alternative<std::unique_ptr<FieldsUnnamed>>(
              field.value())) {
        std::cout << "(";
        std::get<std::unique_ptr<FieldsUnnamed>>(field.value())->render();
        std::cout << ")";
      } else if (std::holds_alternative<std::unique_ptr<FieldsNamed>>(
                     field.value())) {
        std::cout << "{";
        std::get<std::unique_ptr<FieldsNamed>>(field.value())->render();
        std::cout << "}";
      } else {
        std::cout << " = ";
        std::get<std::unique_ptr<Expression>>(field.value())->render();
      }
    }
  }
};

struct UnionField : public Node {
  std::string name;
  std::unique_ptr<Type> type;

  UnionField(Token token, std::string name, std::unique_ptr<Type> type)
      : name(std::move(name)), type(std::move(type)), Node(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::UnionField; }

  void render() const override { type->render(); }
};

struct FunctionDecl : public Node {
  std::string name;
  std::vector<std::unique_ptr<Parameter>> parameters;
  std::unique_ptr<Type> return_type;

  FunctionDecl(Token token, std::string name,
               std::vector<std::unique_ptr<Parameter>> parameters,
               std::unique_ptr<Type> return_type)
      : name(std::move(name)), parameters(std::move(parameters)),
        return_type(std::move(return_type)), Node(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::FunctionDecl; }

  void render() const override {
    std::cout << "fn " << name << "(";
    for (auto &param : parameters) {
      param->render();
      if (&param != &parameters.back())
        std::cout << ", ";
    }
    std::cout << ") ";
    return_type->render();
  }
};

struct Function : public TopLevelDecl {
  std::unique_ptr<FunctionDecl> decl;
  std::unique_ptr<BlockExpression> body;
  bool is_pub;

  Function(Token token, std::string name,
           std::vector<std::unique_ptr<Parameter>> parameters,
           std::unique_ptr<Type> return_type,
           std::unique_ptr<BlockExpression> body, bool is_pub = false)
      : body(std::move(body)), is_pub(is_pub), TopLevelDecl(token) {
    decl = std::make_unique<FunctionDecl>(token, name, std::move(parameters),
                                          std::move(return_type));
  }

  AstNodeKind kind() const override { return AstNodeKind::Function; }

  void render() const override {
    if (is_pub)
      std::cout << "pub ";
    std::cout << "fn " << decl->name << "(";
    for (auto &param : decl->parameters) {
      param->render();
      if (&param != &decl->parameters.back())
        std::cout << ", ";
    }
    std::cout << ") ";
    decl->return_type->render();
    body->render();
  }
};

struct ImportDecl : public TopLevelDecl {
  using Path = std::pair<std::string, std::optional<std::string>>;
  std::vector<Path> paths;

  ImportDecl(Token token, std::vector<Path> paths)
      : paths(std::move(paths)), TopLevelDecl(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ImportDecl; }

  void render() const override {
    size_t i = 0;
    std::cout << "import ";
    for (auto &[path, alias] : paths) {
      std::cout << path;
      if (alias.has_value()) {
        std::cout << " as " << alias.value();
      }
      if (i++ != paths.size() - 1)
        std::cout << ", ";
    }
    std::cout << ";" << std::endl;
  }
};

struct EnumDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<Variant>> variants;
  bool is_pub;

  EnumDecl(Token token, std::string name,
           std::vector<std::unique_ptr<Variant>> variants, bool is_pub)
      : name(std::move(name)), variants(std::move(variants)), is_pub(is_pub),
        TopLevelDecl(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::EnumDecl; }

  void render() const override {
    if (is_pub)
      std::cout << "pub ";
    std::cout << "enum " << name << " {" << std::endl;
    for (auto &variant : variants) {
      variant->render();
      if (&variant != &variants.back())
        std::cout << ", ";
      std::cout << std::endl;
    }
    std::cout << "}" << std::endl;
  }
};

struct UnionDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<UnionField>> fields;
  bool is_pub;

  UnionDecl(Token token, std::string name,
            std::vector<std::unique_ptr<UnionField>> fields, bool is_pub)
      : name(std::move(name)), fields(std::move(fields)), is_pub(is_pub),
        TopLevelDecl(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::UnionDecl; }

  void render() const override {
    if (is_pub)
      std::cout << "pub ";
    std::cout << "union " << name << " {" << std::endl;
    for (auto &field : fields) {
      field->render();
      if (&field != &fields.back())
        std::cout << ", ";
    }
    std::cout << "}" << std::endl;
  }
};

struct TraitDecl : public TopLevelDecl {
  using Method =
      std::variant<std::unique_ptr<Function>, std::unique_ptr<FunctionDecl>>;
  std::string name;
  std::vector<Method> functions;
  std::vector<std::unique_ptr<Type>> super_traits;
  bool is_pub;

  TraitDecl(Token token, std::string name, std::vector<Method> functions,
            std::vector<std::unique_ptr<Type>> super_traits, bool is_pub)
      : name(std::move(name)), functions(std::move(functions)), is_pub(is_pub),
        super_traits(std::move(super_traits)), TopLevelDecl(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::TraitDecl; }

  void render() const override {
    if (is_pub)
      std::cout << "pub ";
    std::cout << "trait " << name;
    if (super_traits.size() > 0) {
      std::cout << " : ";
      for (auto &trait : super_traits) {
        trait->render();
        if (&trait != &super_traits.back())
          std::cout << ", ";
      }
    }
    std::cout << " {" << std::endl;
    for (auto &func : functions) {
      if (std::holds_alternative<std::unique_ptr<Function>>(func)) {
        std::get<std::unique_ptr<Function>>(func)->render();
      } else {
        std::get<std::unique_ptr<FunctionDecl>>(func)->render();
        std::cout << ";" << std::endl;
      }
    }
    std::cout << "}" << std::endl;
  }
};

struct ImplDecl : public TopLevelDecl {
  std::unique_ptr<Type> type;
  std::vector<std::unique_ptr<Type>> traits;
  std::vector<std::unique_ptr<Function>> functions;

  ImplDecl(Token token, std::unique_ptr<Type> type,
           std::vector<std::unique_ptr<Type>> traits,
           std::vector<std::unique_ptr<Function>> functions)
      : type(std::move(type)), traits(std::move(traits)),
        functions(std::move(functions)), TopLevelDecl(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ImplDecl; }

  void render() const override {
    std::cout << "impl ";
    type->render();
    if (!traits.empty()) {
      std::cout << " : ";
      for (auto &trait : traits) {
        trait->render();
      }
    }
    std::cout << " {" << std::endl;
    for (auto &func : functions) {
      func->render();
    }
    std::cout << "}" << std::endl;
  }
};

struct VarDecl : public Statement {
  std::unique_ptr<Pattern> pattern;
  std::optional<std::unique_ptr<Type>> type;
  std::optional<std::unique_ptr<Expression>> initializer;
  bool is_mut;
  bool is_pub;

  VarDecl(Token token, std::unique_ptr<Pattern> pattern,
          std::optional<std::unique_ptr<Type>> type,
          std::optional<std::unique_ptr<Expression>> initializer, bool is_mut,
          bool is_pub)
      : pattern(std::move(pattern)), type(std::move(type)),
        initializer(std::move(initializer)), is_mut(is_mut), is_pub(is_pub),
        Statement(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::VarDecl; }

  void render() const override {
    if (is_pub)
      std::cout << "pub ";
    if (is_mut)
      std::cout << "var ";
    else
      std::cout << "const ";
    pattern->render();
    if (type.has_value()) {
      std::cout << ": ";
      type.value()->render();
    }
    if (initializer.has_value()) {
      std::cout << " = ";
      initializer.value()->render();
    }
    std::cout << ";" << std::endl;
  }
};

struct TopLevelVarDecl : public TopLevelDecl {
  std::unique_ptr<VarDecl> var_decl;

  TopLevelVarDecl(Token token, std::unique_ptr<VarDecl> var_decl)
      : var_decl(std::move(var_decl)), TopLevelDecl(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::TopLevelVarDecl; }

  void render() const override { var_decl->render(); }
};

struct IfExpr : public Expression {
  using Branch = std::variant<std::unique_ptr<BlockExpression>,
                              std::unique_ptr<Expression>>;

  std::unique_ptr<Expression> condition;
  Branch then_block;
  std::optional<Branch> else_block;

  IfExpr(Token token, std::unique_ptr<Expression> condition, Branch then_block,
         std::optional<Branch> else_block)
      : condition(std::move(condition)), then_block(std::move(then_block)),
        else_block(std::move(else_block)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::IfExpr; }

  void render() const override {
    std::cout << "if ";
    condition->render();
    std::visit([](auto &&arg) { arg->render(); }, then_block);
    if (else_block.has_value()) {
      std::cout << "else ";
      std::visit([](auto &&arg) { arg->render(); }, else_block.value());
    }
  }
};

struct MatchArm : public Node {
  using Branch = std::variant<std::unique_ptr<BlockExpression>,
                              std::unique_ptr<Expression>>;
  std::unique_ptr<Pattern> pattern;
  Branch body;
  std::optional<std::unique_ptr<Expression>> guard;

  MatchArm(Token token, std::unique_ptr<Pattern> pattern, Branch body,
           std::optional<std::unique_ptr<Expression>> guard)
      : pattern(std::move(pattern)), body(std::move(body)),
        guard(std::move(guard)), Node(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::MatchArm; }

  void render() const override {
    std::cout << "is ";
    pattern->render();
    if (guard.has_value()) {
      std::cout << " if ";
      guard.value()->render();
    }
    std::cout << " => ";
    std::visit([](auto &&arg) { arg->render(); }, body);
  }
};

struct MatchExpr : public Expression {
  std::unique_ptr<Expression> expr;
  std::vector<std::unique_ptr<MatchArm>> cases;

  MatchExpr(Token token, std::unique_ptr<Expression> expr,
            std::vector<std::unique_ptr<MatchArm>> cases)
      : expr(std::move(expr)), cases(std::move(cases)),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::MatchExpr; }

  void render() const override {
    std::cout << "match ";
    expr->render();
    std::cout << "{" << std::endl;
    for (auto &arm : cases) {
      arm->render();
      std::cout << "," << std::endl;
    }
    std::cout << "}" << std::endl;
  }
};

struct ForExpr : public Expression {
  std::unique_ptr<Pattern> pattern;
  std::unique_ptr<Expression> iterable;
  std::unique_ptr<BlockExpression> body;
  std::optional<std::string> label;

  ForExpr(Token token, std::unique_ptr<Pattern> pattern,
          std::unique_ptr<Expression> iterable,
          std::unique_ptr<BlockExpression> body,
          std::optional<std::string> label)
      : pattern(std::move(pattern)), iterable(std::move(iterable)),
        body(std::move(body)), label(std::move(label)),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ForExpr; }

  void render() const override {
    if (label.has_value()) {
      std::cout << label.value() << ": ";
    }
    std::cout << "for ";
    pattern->render();
    std::cout << " in ";
    iterable->render();
    body->render();
  }
};

struct WhileExpr : public Expression {
  std::optional<std::unique_ptr<Expression>> condition;
  std::optional<std::unique_ptr<Expression>> continue_expr;
  std::unique_ptr<BlockExpression> body;
  std::optional<std::string> label;

  WhileExpr(Token token, std::optional<std::unique_ptr<Expression>> condition,
            std::optional<std::unique_ptr<Expression>> continue_expr,
            std::unique_ptr<BlockExpression> body,
            std::optional<std::string> label)
      : condition(std::move(condition)),
        continue_expr(std::move(continue_expr)), body(std::move(body)),
        label(std::move(label)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::WhileExpr; }

  void render() const override {
    if (label.has_value()) {
      std::cout << label.value() << ": ";
    }
    std::cout << "while ";
    if (condition.has_value()) {
      condition.value()->render();
    }
    if (continue_expr.has_value()) {
      std::cout << " : ";
      continue_expr.value()->render();
    }
    body->render();
  }
};

struct ReturnExpr : public Expression {
  std::optional<std::unique_ptr<Expression>> value;

  ReturnExpr(Token token, std::optional<std::unique_ptr<Expression>> value)
      : value(std::move(value)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ReturnExpr; }

  void render() const override {
    std::cout << "return ";
    if (value.has_value()) {
      value.value()->render();
    }
  }
};

struct DeferStmt : public Statement {
  using Branch = std::variant<std::unique_ptr<BlockExpression>,
                              std::unique_ptr<Expression>>;
  Branch body;

  DeferStmt(Token token, Branch body)
      : body(std::move(body)), Statement(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::DeferStmt; }

  void render() const override {
    std::visit([](auto &&arg) { arg->render(); }, body);
  }
};

struct BreakExpr : public Expression {
  std::optional<std::string> label;
  std::optional<std::unique_ptr<Expression>> value;

  BreakExpr(Token token, std::optional<std::string> label,
            std::optional<std::unique_ptr<Expression>> value)
      : label(std::move(label)), value(std::move(value)),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::BreakExpr; }

  void render() const override {
    std::cout << "break ";
    if (label.has_value()) {
      std::cout << ":" << label.value();
    }
    if (value.has_value()) {
      value.value()->render();
    }
  }
};

struct ContinueExpr : public Expression {
  std::optional<std::string> label;
  std::optional<std::unique_ptr<Expression>> value;

  ContinueExpr(Token token, std::optional<std::string> label,
               std::optional<std::unique_ptr<Expression>> value)
      : label(std::move(label)), value(std::move(value)),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ContinueExpr; }

  void render() const override {
    std::cout << "continue ";
    if (label.has_value()) {
      std::cout << ":" << label.value();
    }
    if (value.has_value()) {
      value.value()->render();
    }
  }
};

struct ExprStmt : public Statement {
  std::unique_ptr<Expression> expr;

  ExprStmt(Token token, std::unique_ptr<Expression> expr)
      : expr(std::move(expr)), Statement(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ExprStmt; }

  void render() const override {
    expr->render();
    // if expr is while, if, for or match then no need to add ;
    if (dynamic_cast<WhileExpr *>(expr.get()) ||
        dynamic_cast<IfExpr *>(expr.get()) ||
        dynamic_cast<ForExpr *>(expr.get()) ||
        dynamic_cast<MatchExpr *>(expr.get())) {
    } else {
      std::cout << ";" << std::endl;
    }
  }
};

struct LiteralExpr : public Expression {
  enum class LiteralType {
    Int,
    Float,
    String,
    Char,
    Bool,
  };
  LiteralType type;
  using Value = std::variant<int, double, std::string, char, bool>;
  Value value;

  LiteralExpr(Token token, LiteralType type, Value value)
      : type(type), value(std::move(value)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::LiteralExpr; }

  void render() const override {
    switch (type) {
    case LiteralType::Int:
      std::cout << std::get<int>(value) << "i";
      break;
    case LiteralType::Float:
      std::cout << std::get<double>(value) << "f";
      break;
    case LiteralType::String:
      std::cout << std::get<std::string>(value);
      break;
    case LiteralType::Char:
      std::cout << "'" << std::get<char>(value) << "'";
      break;
    case LiteralType::Bool:
      if (std::get<bool>(value))
        std::cout << "true";
      else
        std::cout << "false";
      break;
    }
  }
};

struct TupleExpr : public Expression {
  std::vector<std::unique_ptr<Expression>> elements;

  TupleExpr(Token token, std::vector<std::unique_ptr<Expression>> elements)
      : elements(std::move(elements)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::TupleExpr; }

  void render() const override {
    std::cout << "(";
    for (auto &elem : elements) {
      elem->render();
      std::cout << ", ";
    }
    std::cout << ")";
  }
};

struct ArrayExpr : public Expression {
  std::vector<std::unique_ptr<Expression>> elements;

  ArrayExpr(Token token, std::vector<std::unique_ptr<Expression>> elements)
      : elements(std::move(elements)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ArrayExpr; }

  void render() const override {
    std::cout << "[";
    for (auto &elem : elements) {
      elem->render();
      if (&elem != &elements.back())
        std::cout << ", ";
    }
    std::cout << "]";
  }
};

enum class Operator {
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  And,
  Or,
  Not,
  Eq,
  Ne,
  Lt,
  Le,
  Gt,
  Ge,
  Xor,
  BitAnd,
  BitOr,
  BitXor,
  BitShl,
  BitShr,
  BitNot,
  Pow,

  Invalid,
};

struct BinaryExpr : public Expression {
  Operator op;
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;

  BinaryExpr(Token token, Operator op, std::unique_ptr<Expression> lhs,
             std::unique_ptr<Expression> rhs)
      : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::BinaryExpr; }

  void render() const override {
    std::cout << "(";
    lhs->render();
    switch (op) {
    case Operator::Add:
      std::cout << " + ";
      break;
    case Operator::Sub:
      std::cout << " - ";
      break;
    case Operator::Mul:
      std::cout << " * ";
      break;
    case Operator::Div:
      std::cout << " / ";
      break;
    case Operator::Mod:
      std::cout << " % ";
      break;
    case Operator::And:
      std::cout << " and ";
      break;
    case Operator::Or:
      std::cout << " or ";
      break;
    case Operator::Not:
      std::cout << " not ";
      break;
    case Operator::Eq:
      std::cout << " == ";
      break;
    case Operator::Ne:
      std::cout << " != ";
      break;
    case Operator::Lt:
      std::cout << " < ";
      break;
    case Operator::Le:
      std::cout << " <= ";
      break;
    case Operator::Gt:
      std::cout << " > ";
      break;
    case Operator::Ge:
      std::cout << " >= ";
      break;
    case Operator::BitAnd:
      std::cout << " & ";
      break;
    case Operator::BitOr:
      std::cout << " | ";
      break;
    case Operator::BitXor:
      std::cout << " ^ ";
      break;
    case Operator::BitShl:
      std::cout << " << ";
      break;
    case Operator::BitShr:
      std::cout << " >> ";
      break;
    case Operator::Xor:
      std::cout << " ^ ";
      break;
    case Operator::Invalid:
      std::cout << " invalid ";
      break;
    case Operator::Pow:
      std::cout << " ** ";
      break;
    default:
      break;
    }
    rhs->render();
    std::cout << ")";
  }
};

struct UnaryExpr : public Expression {
  Operator op;
  std::unique_ptr<Expression> operand;

  UnaryExpr(Token token, Operator op, std::unique_ptr<Expression> operand)
      : op(op), operand(std::move(operand)), Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::UnaryExpr; }

  void render() const override {
    switch (op) {
    case Operator::Not:
      std::cout << "not ";
      break;
    case Operator::BitNot:
      std::cout << "~";
      break;
    case Operator::Sub:
      std::cout << "-";
      break;
    case Operator::Add:
      std::cout << "+";
      break;
    default:
      break;
    }
    operand->render();
  }
};

struct CallExpr : public Expression {
  std::unique_ptr<Expression> callee;
  std::vector<std::unique_ptr<Expression>> arguments;

  CallExpr(Token token, std::unique_ptr<Expression> callee,
           std::vector<std::unique_ptr<Expression>> arguments)
      : callee(std::move(callee)), arguments(std::move(arguments)),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::CallExpr; }

  void render() const override {
    callee->render();
    std::cout << "(";
    for (auto &arg : arguments) {
      arg->render();
      if (&arg != &arguments.back()) {
        std::cout << ", ";
      }
    }
    std::cout << ")";
  }
};

struct AssignExpr : public Expression {
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;

  AssignExpr(Token token, std::unique_ptr<Expression> lhs,
             std::unique_ptr<Expression> rhs)
      : lhs(std::move(lhs)), rhs(std::move(rhs)), Expression(std::move(token)) {
  }

  AstNodeKind kind() const override { return AstNodeKind::AssignExpr; }

  void render() const override {
    lhs->render();
    std::cout << " = ";
    rhs->render();
  }
};

struct AssignOpExpr : public Expression {
  Operator op;
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;

  AssignOpExpr(Token token, Operator op, std::unique_ptr<Expression> lhs,
               std::unique_ptr<Expression> rhs)
      : op(op), lhs(std::move(lhs)), rhs(std::move(rhs)),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::AssignOpExpr; }

  void render() const override {
    lhs->render();
    switch (op) {
    case Operator::Add:
      std::cout << " += ";
      break;
    case Operator::Sub:
      std::cout << " -= ";
      break;
    case Operator::Mul:
      std::cout << " *= ";
      break;
    case Operator::Div:
      std::cout << " /= ";
      break;
    case Operator::Mod:
      std::cout << " %= ";
      break;
    case Operator::BitAnd:
      std::cout << " &= ";
      break;
    case Operator::BitOr:
      std::cout << " |= ";
      break;
    case Operator::BitXor:
      std::cout << " ^= ";
      break;
    case Operator::BitShl:
      std::cout << " <<= ";
      break;
    case Operator::BitShr:
      std::cout << " >>= ";
      break;
    default:
      break;
    }
    rhs->render();
  }
};

struct FieldAccessExpr : public Expression {
  std::unique_ptr<Expression> base;
  // std::string field;
  using Field =
      std::variant<std::unique_ptr<LiteralExpr>,
                   std::unique_ptr<IdentifierExpr>, std::unique_ptr<CallExpr>>;
  Field field;

  FieldAccessExpr(Token token, std::unique_ptr<Expression> base, Field field)
      : base(std::move(base)), field(std::move(field)),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::FieldAccessExpr; }

  void render() const override {
    base->render();
    std::cout << ".";
    std::visit([](auto &&arg) { arg->render(); }, field);
  }
};

struct IndexExpr : public Expression {
  std::unique_ptr<Expression> base;
  std::unique_ptr<Expression> index;

  IndexExpr(Token token, std::unique_ptr<Expression> base,
            std::unique_ptr<Expression> index)
      : base(std::move(base)), index(std::move(index)),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::IndexExpr; }

  void render() const override {
    base->render();
    std::cout << "[";
    index->render();
    std::cout << "]";
  }
};

struct RangeExpr : public Expression {
  std::optional<std::unique_ptr<Expression>> start;
  std::optional<std::unique_ptr<Expression>> end;
  bool inclusive;

  RangeExpr(Token token, std::optional<std::unique_ptr<Expression>> start,
            std::optional<std::unique_ptr<Expression>> end, bool inclusive)
      : start(std::move(start)), end(std::move(end)), inclusive(inclusive),
        Expression(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::RangeExpr; }

  void render() const override {
    if (start.has_value())
      start.value()->render();
    if (inclusive)
      std::cout << "..=";
    else
      std::cout << "..";
    if (end.has_value())
      end.value()->render();
  }
};

struct PrimitiveType : public Type {
  enum class PrimitiveTypeKind {
    String,
    Char,
    Bool,
    Void,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    type,
  };

  PrimitiveTypeKind type_kind;

  PrimitiveType(Token token, PrimitiveTypeKind kind)
      : type_kind(kind), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::PrimitiveType; }

  void render() const override {
    switch (type_kind) {
    case PrimitiveTypeKind::String:
      std::cout << "string";
      break;
    case PrimitiveTypeKind::Char:
      std::cout << "char";
      break;
    case PrimitiveTypeKind::Bool:
      std::cout << "bool";
      break;
    case PrimitiveTypeKind::Void:
      std::cout << "void";
      break;
    case PrimitiveTypeKind::I8:
      std::cout << "i8";
      break;
    case PrimitiveTypeKind::I16:
      std::cout << "i16";
      break;
    case PrimitiveTypeKind::I32:
      std::cout << "i32";
      break;
    case PrimitiveTypeKind::I64:
      std::cout << "i64";
      break;
    case PrimitiveTypeKind::U8:
      std::cout << "u8";
      break;
    case PrimitiveTypeKind::U16:
      std::cout << "u16";
      break;
    case PrimitiveTypeKind::U32:
      std::cout << "u32";
      break;
    case PrimitiveTypeKind::U64:
      std::cout << "u64";
      break;
    case PrimitiveTypeKind::F32:
      std::cout << "f32";
      break;
    case PrimitiveTypeKind::F64:
      std::cout << "f64";
      break;
    case PrimitiveTypeKind::type:
      std::cout << "type";
      break;
    }
  }
};

struct TupleType : public Type {
  std::vector<std::unique_ptr<Type>> elements;

  TupleType(Token token, std::vector<std::unique_ptr<Type>> elements)
      : elements(std::move(elements)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::TupleType; }

  void render() const override {
    std::cout << "(";
    for (auto &elem : elements) {
      elem->render();
      if (&elem != &elements.back())
        std::cout << ", ";
    }
    std::cout << ")";
  }
};

struct FunctionType : public Type {
  std::vector<std::unique_ptr<Type>> parameters;
  std::unique_ptr<Type> return_type;

  FunctionType(Token token, std::vector<std::unique_ptr<Type>> parameters,
               std::unique_ptr<Type> return_type)
      : parameters(std::move(parameters)), return_type(std::move(return_type)),
        Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::FunctionType; }

  void render() const override {
    for (auto &param : parameters) {
      param->render();
    }
    return_type->render();
  }
};

struct ReferenceType : public Type {
  std::unique_ptr<Type> base;

  ReferenceType(Token token, std::unique_ptr<Type> base)
      : base(std::move(base)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ReferenceType; }

  void render() const override { base->render(); }
};

struct SliceType : public Type {
  std::unique_ptr<Type> base;

  SliceType(Token token, std::unique_ptr<Type> base)
      : base(std::move(base)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::SliceType; }

  void render() const override {
    std::cout << "[]";
    base->render();
  }
};

struct ArrayType : public Type {
  std::unique_ptr<Type> base;
  std::unique_ptr<Expression> size;

  ArrayType(Token token, std::unique_ptr<Type> base,
            std::unique_ptr<Expression> size)
      : base(std::move(base)), size(std::move(size)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ArrayType; }

  void render() const override {
    std::cout << "[";
    size->render();
    std::cout << "]";
    base->render();
  }
};

struct TraitType : public Type {
  std::string name;

  TraitType(Token token, std::string name)
      : name(std::move(name)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::TraitType; }

  void render() const override { std::cout << name; }
};

struct IdentifierType : public Type {
  std::string name;

  IdentifierType(Token token, std::string name)
      : name(std::move(name)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::IdentifierType; }

  void render() const override { std::cout << name; }
};

struct StructType : public Type {
  std::string name;

  StructType(Token token, std::string name)
      : name(std::move(name)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::StructType; }

  void render() const override { std::cout << name; }
};

struct EnumType : public Type {
  std::string name;

  EnumType(Token token, std::string name)
      : name(std::move(name)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::EnumType; }

  void render() const override { std::cout << name; }
};

struct UnionType : public Type {
  std::string name;

  UnionType(Token token, std::string name)
      : name(std::move(name)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::UnionType; }

  void render() const override { std::cout << name; }
};

struct ExprType : public Type {
  std::unique_ptr<Expression> expr;

  ExprType(Token token, std::unique_ptr<Expression> expr)
      : expr(std::move(expr)), Type(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ExprType; }

  void render() const override { expr->render(); }
};

struct LiteralPattern : public Pattern {
  std::unique_ptr<LiteralExpr> literal;

  LiteralPattern(Token token, std::unique_ptr<LiteralExpr> literal)
      : literal(std::move(literal)), Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::LiteralPattern; }

  void render() const override { literal->render(); }
};

struct IdentifierPattern : public Pattern {
  std::string name;

  IdentifierPattern(Token token, std::string name)
      : name(std::move(name)), Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::IdentifierPattern; }

  void render() const override { std::cout << name; }
};

struct WildcardPattern : public Pattern {
  WildcardPattern(Token token) : Pattern(std::move(token)) {}
  AstNodeKind kind() const override { return AstNodeKind::WildcardPattern; }
  void render() const override { std::cout << "_"; }
};

struct TuplePattern : public Pattern {
  std::vector<std::unique_ptr<Pattern>> elements;

  TuplePattern(Token token, std::vector<std::unique_ptr<Pattern>> elements)
      : elements(std::move(elements)), Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::TuplePattern; }

  void render() const override {
    std::cout << "(";
    for (auto &elem : elements) {
      elem->render();
      if (&elem != &elements.back())
        std::cout << ", ";
    }
    std::cout << ")";
  }
};

struct PatternField : public Node {
  std::string name;
  std::optional<std::unique_ptr<Pattern>> pattern;

  PatternField(Token token, std::string name,
               std::optional<std::unique_ptr<Pattern>> pattern)
      : name(std::move(name)), pattern(std::move(pattern)),
        Node(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::PatternField; }

  void render() const override {
    std::cout << name;
    if (pattern.has_value()) {
      std::cout << ": ";
      pattern.value()->render();
    }
  }
};

struct RestPattern : public Pattern {
  std::optional<IdentifierExpr> name;

  RestPattern(Token token, std::optional<IdentifierExpr> name)
      : name(std::move(name)), Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::RestPattern; }

  void render() const override {
    std::cout << "..";
    if (name.has_value()) {
      std::cout << " as ";
      name.value().render();
    }
  }
};

struct StructPattern : public Pattern {
  std::optional<std::string> name;
  using Field =
      std::variant<std::unique_ptr<PatternField>, std::unique_ptr<RestPattern>>;
  std::vector<Field> fields;

  StructPattern(Token token, std::optional<std::string> name,
                std::vector<Field> fields)
      : name(std::move(name)), fields(std::move(fields)),
        Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::StructPattern; }

  void render() const override {
    if (name.has_value()) {
      std::cout << name.value();
    }
    std::cout << " {";
    for (auto &field : fields) {
      std::visit([](auto &&arg) { arg->render(); }, field);
      if (&field != &fields.back())
        std::cout << ", ";
    }
    std::cout << "}";
  }
};

struct SlicePattern : public Pattern {
  std::vector<std::unique_ptr<Pattern>> elements;
  bool is_exhaustive;

  SlicePattern(Token token, std::vector<std::unique_ptr<Pattern>> elements,
               bool is_exhaustive)
      : elements(std::move(elements)), is_exhaustive(is_exhaustive),
        Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::SlicePattern; }

  void render() const override {
    std::cout << "[";
    for (auto &elem : elements) {
      elem->render();
      if (&elem != &elements.back())
        std::cout << ", ";
    }
    std::cout << "]";
  }
};

struct OrPattern : public Pattern {
  std::vector<std::unique_ptr<Pattern>> patterns;

  OrPattern(Token token, std::vector<std::unique_ptr<Pattern>> patterns)
      : patterns(std::move(patterns)), Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::OrPattern; }

  void render() const override {
    for (auto &pattern : patterns) {
      pattern->render();
      if (&pattern != &patterns.back())
        std::cout << " | ";
    }
  }
};

struct ExprPattern : public Pattern {
  std::unique_ptr<Expression> expr;

  ExprPattern(Token token, std::unique_ptr<Expression> expr)
      : expr(std::move(expr)), Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::ExprPattern; }

  void render() const override { expr->render(); }
};

struct RangePattern : public Pattern {
  std::unique_ptr<Pattern> start;
  std::unique_ptr<Pattern> end;
  bool inclusive;

  RangePattern(Token token, std::unique_ptr<Pattern> start,
               std::unique_ptr<Pattern> end, bool inclusive)
      : start(std::move(start)), end(std::move(end)), inclusive(inclusive),
        Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::RangePattern; }

  void render() const override {
    start->render();
    if (inclusive)
      std::cout << "..=";
    else
      std::cout << "..";
    end->render();
  }
};

struct VariantPattern : public Pattern {
  std::string name;
  std::optional<std::variant<std::unique_ptr<TuplePattern>,
                             std::unique_ptr<StructPattern>>>
      field;

  VariantPattern(Token token, std::string name,
                 std::optional<std::variant<std::unique_ptr<TuplePattern>,
                                            std::unique_ptr<StructPattern>>>
                     field)
      : name(std::move(name)), field(std::move(field)),
        Pattern(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::VariantPattern; }

  void render() const override {
    std::cout << "." << name;
    if (field.has_value()) {
      if (std::holds_alternative<std::unique_ptr<TuplePattern>>(
              field.value())) {
        std::get<std::unique_ptr<TuplePattern>>(field.value())->render();
      } else {
        std::get<std::unique_ptr<StructPattern>>(field.value())->render();
      }
    }
  }
};

struct TopLevelDeclStmt : public Statement {
  std::unique_ptr<TopLevelDecl> decl;

  TopLevelDeclStmt(Token token, std::unique_ptr<TopLevelDecl> decl)
      : decl(std::move(decl)), Statement(std::move(token)) {}

  AstNodeKind kind() const override { return AstNodeKind::TopLevelDeclStmt; }

  void render() const override { decl->render(); }
};
} // namespace Ast
