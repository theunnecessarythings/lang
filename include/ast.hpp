#pragma once

#include "lexer.hpp"
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

enum class AstNodeKind {
  Program,
  Module,
  Expression,
  Statement,
  Type,
  Pattern,
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
  YieldExpr,

  MLIRAttribute,
  MLIRType,
  MLIROp,

  InvalidNode,
  InvalidExpression,
  InvalidStatement,
  InvalidTopLevelDecl,
  InvalidType,
  InvalidPattern,
};

std::string &toString(AstNodeKind kind);

struct Node {
  Token token;
  Node(Token token) : token(std::move(token)) {}
  virtual ~Node() = default;
  virtual AstNodeKind kind() const = 0;
};

struct Expression : public Node {
  Expression(Token token) : Node(std::move(token)) {}
  virtual ~Expression() = default;
};
struct Statement : public Node {
  Statement(Token token) : Node(std::move(token)) {}
  virtual ~Statement() = default;
};
struct TopLevelDecl : public Node {
  TopLevelDecl(Token token) : Node(std::move(token)) {}
  virtual ~TopLevelDecl() = default;
};
struct Type : public Expression {
  Type(Token token) : Expression(std::move(token)) {}
  virtual ~Type() = default;
};
struct Pattern : public Node {
  Pattern(Token token) : Node(std::move(token)) {}
  virtual ~Pattern() = default;
};

struct InvalidNode : public Node {
  InvalidNode(Token token) : Node(std::move(token)) {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidNode; }
};
struct InvalidExpression : public Expression {
  InvalidExpression(Token token) : Expression(std::move(token)) {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidExpression; }
};
struct InvalidStatement : public Statement {
  InvalidStatement(Token token) : Statement(std::move(token)) {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidStatement; }
};
struct InvalidTopLevelDecl : public TopLevelDecl {
  InvalidTopLevelDecl(Token token) : TopLevelDecl(std::move(token)) {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidTopLevelDecl; }
};
struct InvalidType : public Type {
  InvalidType(Token token) : Type(std::move(token)) {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidType; }
};
struct InvalidPattern : public Pattern {
  InvalidPattern(Token token) : Pattern(std::move(token)) {}
  AstNodeKind kind() const override { return AstNodeKind::InvalidPattern; }
};

struct Program : public Node {
  std::vector<std::unique_ptr<TopLevelDecl>> items;

  Program(Token token, std::vector<std::unique_ptr<TopLevelDecl>> items)
      : Node(std::move(token)), items(std::move(items)) {}

  AstNodeKind kind() const override { return AstNodeKind::Program; }
};

struct Module : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<TopLevelDecl>> items;

  Module(Token token, std::string name,
         std::vector<std::unique_ptr<TopLevelDecl>> items)
      : TopLevelDecl(std::move(token)), name(std::move(name)),
        items(std::move(items)) {}

  AstNodeKind kind() const override { return AstNodeKind::Module; }
};

struct ComptimeExpr : public Expression {
  std::unique_ptr<Expression> expr;

  ComptimeExpr(Token token, std::unique_ptr<Expression> expr)
      : Expression(std::move(token)), expr(std::move(expr)) {}

  AstNodeKind kind() const override { return AstNodeKind::ComptimeExpr; }
};

struct BlockExpression : public Expression {
  std::vector<std::unique_ptr<Statement>> statements;

  BlockExpression(Token token,
                  std::vector<std::unique_ptr<Statement>> statements)
      : Expression(std::move(token)), statements(std::move(statements)) {}

  AstNodeKind kind() const override { return AstNodeKind::BlockExpression; }
};

struct Parameter : public Node {
  std::unique_ptr<Pattern> pattern;
  std::unique_ptr<Type> type;
  std::vector<std::unique_ptr<Type>> trait_bound;
  bool is_mut;
  bool is_comptime;

  Parameter(Token token, std::unique_ptr<Pattern> pattern,
            std::unique_ptr<Type> type,
            std::vector<std::unique_ptr<Type>> trait_bound, bool is_mut,
            bool is_comptime)
      : Node(std::move(token)), pattern(std::move(pattern)),
        type(std::move(type)), trait_bound(std::move(trait_bound)),
        is_mut(is_mut), is_comptime(is_comptime) {}

  AstNodeKind kind() const override { return AstNodeKind::Parameter; }
};

struct StructField : public Node {
  std::string name;
  std::unique_ptr<Type> type;

  StructField(Token token, std::string name, std::unique_ptr<Type> type)
      : Node(std::move(token)), name(std::move(name)), type(std::move(type)) {}

  AstNodeKind kind() const override { return AstNodeKind::StructField; }
};

struct StructDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<StructField>> fields;
  bool is_pub;

  StructDecl(Token token, std::string name,
             std::vector<std::unique_ptr<StructField>> fields, bool is_pub)
      : TopLevelDecl(std::move(token)), name(std::move(name)),
        fields(std::move(fields)), is_pub(is_pub) {}

  AstNodeKind kind() const override { return AstNodeKind::StructDecl; }
};

struct TupleStructDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<Type>> fields;
  bool is_pub;

  TupleStructDecl(Token token, std::string name,
                  std::vector<std::unique_ptr<Type>> fields, bool is_pub)
      : TopLevelDecl(std::move(token)), name(std::move(name)),
        fields(std::move(fields)), is_pub(is_pub) {}

  AstNodeKind kind() const override { return AstNodeKind::TupleStructDecl; }
};

struct IdentifierExpr : public Expression {
  std::string name;

  IdentifierExpr(Token token, std::string name)
      : Expression(std::move(token)), name(std::move(name)) {}

  AstNodeKind kind() const override { return AstNodeKind::IdentifierExpr; }
};

struct FieldsNamed : public Node {
  std::vector<std::string> name;
  std::vector<std::unique_ptr<Type>> value;

  FieldsNamed(Token token, std::vector<std::string> name,
              std::vector<std::unique_ptr<Type>> value)
      : Node(std::move(token)), name(std::move(name)), value(std::move(value)) {
  }

  AstNodeKind kind() const override { return AstNodeKind::FieldsNamed; }
};

struct FieldsUnnamed : public Node {
  std::vector<std::unique_ptr<Type>> value;

  FieldsUnnamed(Token token, std::vector<std::unique_ptr<Type>> value)
      : Node(std::move(token)), value(std::move(value)) {}

  AstNodeKind kind() const override { return AstNodeKind::FieldsUnnamed; }
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
      : Node(std::move(token)), name(std::move(name)), field(std::move(field)) {
  }

  AstNodeKind kind() const override { return AstNodeKind::Variant; }
};

struct UnionField : public Node {
  std::string name;
  std::unique_ptr<Type> type;

  UnionField(Token token, std::string name, std::unique_ptr<Type> type)
      : Node(std::move(token)), name(std::move(name)), type(std::move(type)) {}

  AstNodeKind kind() const override { return AstNodeKind::UnionField; }
};

struct FunctionDecl : public Node {
  std::string name;
  std::vector<std::unique_ptr<Parameter>> parameters;
  std::unique_ptr<Type> return_type;

  struct ExtraData {
    bool is_method = false;
    std::optional<std::string> parent_name = std::nullopt;
    AstNodeKind parent_kind = AstNodeKind::InvalidNode;
  } extra;

  FunctionDecl(Token token, std::string name,
               std::vector<std::unique_ptr<Parameter>> parameters,
               std::unique_ptr<Type> return_type)
      : Node(std::move(token)), name(std::move(name)),
        parameters(std::move(parameters)), return_type(std::move(return_type)) {
  }

  AstNodeKind kind() const override { return AstNodeKind::FunctionDecl; }
};

struct Function : public TopLevelDecl {
  std::unique_ptr<FunctionDecl> decl;
  std::unique_ptr<BlockExpression> body;
  bool is_pub;

  Function(Token token, std::string name,
           std::vector<std::unique_ptr<Parameter>> parameters,
           std::unique_ptr<Type> return_type,
           std::unique_ptr<BlockExpression> body, bool is_pub = false)
      : TopLevelDecl(token), body(std::move(body)), is_pub(is_pub) {
    decl = std::make_unique<FunctionDecl>(token, name, std::move(parameters),
                                          std::move(return_type));
  }

  AstNodeKind kind() const override { return AstNodeKind::Function; }
};

struct ImportDecl : public TopLevelDecl {
  using Path = std::pair<std::string, std::optional<std::string>>;
  std::vector<Path> paths;

  ImportDecl(Token token, std::vector<Path> paths)
      : TopLevelDecl(std::move(token)), paths(std::move(paths)) {}

  AstNodeKind kind() const override { return AstNodeKind::ImportDecl; }
};

struct EnumDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<Variant>> variants;
  bool is_pub;

  EnumDecl(Token token, std::string name,
           std::vector<std::unique_ptr<Variant>> variants, bool is_pub)
      : TopLevelDecl(std::move(token)), name(std::move(name)),
        variants(std::move(variants)), is_pub(is_pub) {}

  AstNodeKind kind() const override { return AstNodeKind::EnumDecl; }
};

struct UnionDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<UnionField>> fields;
  bool is_pub;

  UnionDecl(Token token, std::string name,
            std::vector<std::unique_ptr<UnionField>> fields, bool is_pub)
      : TopLevelDecl(std::move(token)), name(std::move(name)),
        fields(std::move(fields)), is_pub(is_pub) {}

  AstNodeKind kind() const override { return AstNodeKind::UnionDecl; }
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
      : TopLevelDecl(std::move(token)), name(std::move(name)),
        functions(std::move(functions)), super_traits(std::move(super_traits)),
        is_pub(is_pub) {}

  AstNodeKind kind() const override { return AstNodeKind::TraitDecl; }
};

struct ImplDecl : public TopLevelDecl {
  std::string type;
  std::vector<std::unique_ptr<Type>> traits;
  std::vector<std::unique_ptr<Function>> functions;

  ImplDecl(Token token, std::string type,
           std::vector<std::unique_ptr<Type>> traits,
           std::vector<std::unique_ptr<Function>> functions)
      : TopLevelDecl(std::move(token)), type(std::move(type)),
        traits(std::move(traits)), functions(std::move(functions)) {}

  AstNodeKind kind() const override { return AstNodeKind::ImplDecl; }
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
      : Statement(std::move(token)), pattern(std::move(pattern)),
        type(std::move(type)), initializer(std::move(initializer)),
        is_mut(is_mut), is_pub(is_pub) {}

  AstNodeKind kind() const override { return AstNodeKind::VarDecl; }
};

struct TopLevelVarDecl : public TopLevelDecl {
  std::unique_ptr<VarDecl> var_decl;

  TopLevelVarDecl(Token token, std::unique_ptr<VarDecl> var_decl)
      : TopLevelDecl(std::move(token)), var_decl(std::move(var_decl)) {}

  AstNodeKind kind() const override { return AstNodeKind::TopLevelVarDecl; }
};

struct YieldExpr : public Expression {
  std::unique_ptr<Expression> value;
  std::optional<std::string> label;

  YieldExpr(Token token, std::optional<std::string> label,
            std::unique_ptr<Expression> value)
      : Expression(std::move(token)), value(std::move(value)),
        label(std::move(label)) {}

  AstNodeKind kind() const override { return AstNodeKind::YieldExpr; }
};

struct IfExprData {
  bool structured = true;
};

struct IfExpr : public Expression {
  std::unique_ptr<Expression> condition;
  std::unique_ptr<BlockExpression> then_block;
  std::optional<std::unique_ptr<BlockExpression>> else_block;

  IfExprData extra;

  IfExpr(Token token, std::unique_ptr<Expression> condition,
         std::unique_ptr<BlockExpression> then_block,
         std::optional<std::unique_ptr<BlockExpression>> else_block)
      : Expression(std::move(token)), condition(std::move(condition)),
        then_block(std::move(then_block)), else_block(std::move(else_block)) {}

  AstNodeKind kind() const override { return AstNodeKind::IfExpr; }
};

struct MatchArm : public Node {
  using Branch = std::variant<std::unique_ptr<BlockExpression>,
                              std::unique_ptr<Expression>>;
  std::unique_ptr<Pattern> pattern;
  Branch body;
  std::optional<std::unique_ptr<Expression>> guard;

  MatchArm(Token token, std::unique_ptr<Pattern> pattern, Branch body,
           std::optional<std::unique_ptr<Expression>> guard)
      : Node(std::move(token)), pattern(std::move(pattern)),
        body(std::move(body)), guard(std::move(guard)) {}

  AstNodeKind kind() const override { return AstNodeKind::MatchArm; }
};

struct MatchExpr : public Expression {
  std::unique_ptr<Expression> expr;
  std::vector<std::unique_ptr<MatchArm>> cases;

  MatchExpr(Token token, std::unique_ptr<Expression> expr,
            std::vector<std::unique_ptr<MatchArm>> cases)
      : Expression(std::move(token)), expr(std::move(expr)),
        cases(std::move(cases)) {}

  AstNodeKind kind() const override { return AstNodeKind::MatchExpr; }
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
      : Expression(std::move(token)), pattern(std::move(pattern)),
        iterable(std::move(iterable)), body(std::move(body)),
        label(std::move(label)) {}

  AstNodeKind kind() const override { return AstNodeKind::ForExpr; }
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
      : Expression(std::move(token)), condition(std::move(condition)),
        continue_expr(std::move(continue_expr)), body(std::move(body)),
        label(std::move(label)) {}

  AstNodeKind kind() const override { return AstNodeKind::WhileExpr; }
};

struct ReturnExpr : public Expression {
  std::optional<std::unique_ptr<Expression>> value;

  ReturnExpr(Token token, std::optional<std::unique_ptr<Expression>> value)
      : Expression(std::move(token)), value(std::move(value)) {}

  AstNodeKind kind() const override { return AstNodeKind::ReturnExpr; }
};

struct DeferStmt : public Statement {
  using Branch = std::variant<std::unique_ptr<BlockExpression>,
                              std::unique_ptr<Expression>>;
  Branch body;

  DeferStmt(Token token, Branch body)
      : Statement(std::move(token)), body(std::move(body)) {}

  AstNodeKind kind() const override { return AstNodeKind::DeferStmt; }
};

struct BreakExpr : public Expression {
  std::optional<std::string> label;
  std::optional<std::unique_ptr<Expression>> value;

  BreakExpr(Token token, std::optional<std::string> label,
            std::optional<std::unique_ptr<Expression>> value)
      : Expression(std::move(token)), label(std::move(label)),
        value(std::move(value)) {}

  AstNodeKind kind() const override { return AstNodeKind::BreakExpr; }
};

struct ContinueExpr : public Expression {
  std::optional<std::string> label;
  std::optional<std::unique_ptr<Expression>> value;

  ContinueExpr(Token token, std::optional<std::string> label,
               std::optional<std::unique_ptr<Expression>> value)
      : Expression(std::move(token)), label(std::move(label)),
        value(std::move(value)) {}

  AstNodeKind kind() const override { return AstNodeKind::ContinueExpr; }
};

struct ExprStmt : public Statement {
  std::unique_ptr<Expression> expr;

  ExprStmt(Token token, std::unique_ptr<Expression> expr)
      : Statement(std::move(token)), expr(std::move(expr)) {}

  AstNodeKind kind() const override { return AstNodeKind::ExprStmt; }
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
      : Expression(std::move(token)), type(type), value(std::move(value)) {}

  AstNodeKind kind() const override { return AstNodeKind::LiteralExpr; }
};

struct TupleExpr : public Expression {
  std::vector<std::unique_ptr<Expression>> elements;

  TupleExpr(Token token, std::vector<std::unique_ptr<Expression>> elements)
      : Expression(std::move(token)), elements(std::move(elements)) {}

  AstNodeKind kind() const override { return AstNodeKind::TupleExpr; }
};

struct ArrayExpr : public Expression {
  std::vector<std::unique_ptr<Expression>> elements;

  ArrayExpr(Token token, std::vector<std::unique_ptr<Expression>> elements)
      : Expression(std::move(token)), elements(std::move(elements)) {}

  AstNodeKind kind() const override { return AstNodeKind::ArrayExpr; }
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
      : Expression(std::move(token)), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  AstNodeKind kind() const override { return AstNodeKind::BinaryExpr; }
};

struct UnaryExpr : public Expression {
  Operator op;
  std::unique_ptr<Expression> operand;

  UnaryExpr(Token token, Operator op, std::unique_ptr<Expression> operand)
      : Expression(std::move(token)), op(op), operand(std::move(operand)) {}

  AstNodeKind kind() const override { return AstNodeKind::UnaryExpr; }
};

struct CallExpr : public Expression {
  std::string callee;
  std::vector<std::unique_ptr<Expression>> arguments;

  CallExpr(Token token, std::string callee,
           std::vector<std::unique_ptr<Expression>> arguments)
      : Expression(std::move(token)), callee(std::move(callee)),
        arguments(std::move(arguments)) {}

  AstNodeKind kind() const override { return AstNodeKind::CallExpr; }
};

struct AssignExpr : public Expression {
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;

  AssignExpr(Token token, std::unique_ptr<Expression> lhs,
             std::unique_ptr<Expression> rhs)
      : Expression(std::move(token)), lhs(std::move(lhs)), rhs(std::move(rhs)) {
  }

  AstNodeKind kind() const override { return AstNodeKind::AssignExpr; }
};

struct AssignOpExpr : public Expression {
  Operator op;
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;

  AssignOpExpr(Token token, Operator op, std::unique_ptr<Expression> lhs,
               std::unique_ptr<Expression> rhs)
      : Expression(std::move(token)), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  AstNodeKind kind() const override { return AstNodeKind::AssignOpExpr; }
};

struct FieldAccessExpr : public Expression {
  std::unique_ptr<Expression> base;
  using Field =
      std::variant<std::unique_ptr<LiteralExpr>,
                   std::unique_ptr<IdentifierExpr>, std::unique_ptr<CallExpr>>;
  Field field;

  FieldAccessExpr(Token token, std::unique_ptr<Expression> base, Field field)
      : Expression(std::move(token)), base(std::move(base)),
        field(std::move(field)) {}

  AstNodeKind kind() const override { return AstNodeKind::FieldAccessExpr; }
};

struct IndexExpr : public Expression {
  std::unique_ptr<Expression> base;
  std::unique_ptr<Expression> index;

  IndexExpr(Token token, std::unique_ptr<Expression> base,
            std::unique_ptr<Expression> index)
      : Expression(std::move(token)), base(std::move(base)),
        index(std::move(index)) {}

  AstNodeKind kind() const override { return AstNodeKind::IndexExpr; }
};

struct RangeExpr : public Expression {
  std::optional<std::unique_ptr<Expression>> start;
  std::optional<std::unique_ptr<Expression>> end;
  bool inclusive;

  RangeExpr(Token token, std::optional<std::unique_ptr<Expression>> start,
            std::optional<std::unique_ptr<Expression>> end, bool inclusive)
      : Expression(std::move(token)), start(std::move(start)),
        end(std::move(end)), inclusive(inclusive) {}

  AstNodeKind kind() const override { return AstNodeKind::RangeExpr; }
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
      : Type(std::move(token)), type_kind(kind) {}

  AstNodeKind kind() const override { return AstNodeKind::PrimitiveType; }
};

struct MLIRAttribute : public Expression {
  std::string attribute;

  MLIRAttribute(Token token, std::string attribute)
      : Expression(std::move(token)), attribute(std::move(attribute)) {}

  AstNodeKind kind() const override { return AstNodeKind::MLIRAttribute; }
};

struct MLIRType : public Type {
  std::string type;

  MLIRType(Token token, std::string type)
      : Type(std::move(token)), type(std::move(type)) {}

  AstNodeKind kind() const override { return AstNodeKind::MLIRType; }
};

struct MLIROp : public Expression {
  std::string op;
  std::vector<std::unique_ptr<Expression>> operands;
  std::unordered_map<std::string, std::string> attributes;
  std::vector<std::string> result_types;

  MLIROp(Token token, std::string op,
         std::vector<std::unique_ptr<Expression>> operands,
         std::unordered_map<std::string, std::string> attributes,
         std::vector<std::string> result_types)
      : Expression(std::move(token)), op(std::move(op)),
        operands(std::move(operands)), attributes(std::move(attributes)),
        result_types(std::move(result_types)) {}

  AstNodeKind kind() const override { return AstNodeKind::MLIROp; }
};

struct TupleType : public Type {
  std::vector<std::unique_ptr<Type>> elements;

  TupleType(Token token, std::vector<std::unique_ptr<Type>> elements)
      : Type(std::move(token)), elements(std::move(elements)) {}

  AstNodeKind kind() const override { return AstNodeKind::TupleType; }
};

struct FunctionType : public Type {
  std::vector<std::unique_ptr<Type>> parameters;
  std::unique_ptr<Type> return_type;

  FunctionType(Token token, std::vector<std::unique_ptr<Type>> parameters,
               std::unique_ptr<Type> return_type)
      : Type(std::move(token)), parameters(std::move(parameters)),
        return_type(std::move(return_type)) {}

  AstNodeKind kind() const override { return AstNodeKind::FunctionType; }
};

struct ReferenceType : public Type {
  std::unique_ptr<Type> base;

  ReferenceType(Token token, std::unique_ptr<Type> base)
      : Type(std::move(token)), base(std::move(base)) {}

  AstNodeKind kind() const override { return AstNodeKind::ReferenceType; }
};

struct SliceType : public Type {
  std::unique_ptr<Type> base;

  SliceType(Token token, std::unique_ptr<Type> base)
      : Type(std::move(token)), base(std::move(base)) {}

  AstNodeKind kind() const override { return AstNodeKind::SliceType; }
};

struct ArrayType : public Type {
  std::unique_ptr<Type> base;
  std::unique_ptr<Expression> size;

  ArrayType(Token token, std::unique_ptr<Type> base,
            std::unique_ptr<Expression> size)
      : Type(std::move(token)), base(std::move(base)), size(std::move(size)) {}

  AstNodeKind kind() const override { return AstNodeKind::ArrayType; }
};

struct TraitType : public Type {
  std::string name;

  TraitType(Token token, std::string name)
      : Type(std::move(token)), name(std::move(name)) {}

  AstNodeKind kind() const override { return AstNodeKind::TraitType; }
};

struct IdentifierType : public Type {
  std::string name;

  IdentifierType(Token token, std::string name)
      : Type(std::move(token)), name(std::move(name)) {}

  AstNodeKind kind() const override { return AstNodeKind::IdentifierType; }
};

struct StructType : public Type {
  std::string name;

  StructType(Token token, std::string name)
      : Type(std::move(token)), name(std::move(name)) {}

  AstNodeKind kind() const override { return AstNodeKind::StructType; }
};

struct EnumType : public Type {
  std::string name;

  EnumType(Token token, std::string name)
      : Type(std::move(token)), name(std::move(name)) {}

  AstNodeKind kind() const override { return AstNodeKind::EnumType; }
};

struct UnionType : public Type {
  std::string name;

  UnionType(Token token, std::string name)
      : Type(std::move(token)), name(std::move(name)) {}

  AstNodeKind kind() const override { return AstNodeKind::UnionType; }
};

struct ExprType : public Type {
  std::unique_ptr<Expression> expr;

  ExprType(Token token, std::unique_ptr<Expression> expr)
      : Type(std::move(token)), expr(std::move(expr)) {}

  AstNodeKind kind() const override { return AstNodeKind::ExprType; }
};

struct LiteralPattern : public Pattern {
  std::unique_ptr<LiteralExpr> literal;

  LiteralPattern(Token token, std::unique_ptr<LiteralExpr> literal)
      : Pattern(std::move(token)), literal(std::move(literal)) {}

  AstNodeKind kind() const override { return AstNodeKind::LiteralPattern; }
};

struct IdentifierPattern : public Pattern {
  std::string name;

  IdentifierPattern(Token token, std::string name)
      : Pattern(std::move(token)), name(std::move(name)) {}

  AstNodeKind kind() const override { return AstNodeKind::IdentifierPattern; }
};

struct WildcardPattern : public Pattern {
  WildcardPattern(Token token) : Pattern(std::move(token)) {}
  AstNodeKind kind() const override { return AstNodeKind::WildcardPattern; }
};

struct TuplePattern : public Pattern {
  std::vector<std::unique_ptr<Pattern>> elements;

  TuplePattern(Token token, std::vector<std::unique_ptr<Pattern>> elements)
      : Pattern(std::move(token)), elements(std::move(elements)) {}

  AstNodeKind kind() const override { return AstNodeKind::TuplePattern; }
};

struct PatternField : public Node {
  std::string name;
  std::optional<std::unique_ptr<Pattern>> pattern;

  PatternField(Token token, std::string name,
               std::optional<std::unique_ptr<Pattern>> pattern)
      : Node(std::move(token)), name(std::move(name)),
        pattern(std::move(pattern)) {}

  AstNodeKind kind() const override { return AstNodeKind::PatternField; }
};

struct RestPattern : public Pattern {
  std::optional<IdentifierExpr> name;

  RestPattern(Token token, std::optional<IdentifierExpr> name)
      : Pattern(std::move(token)), name(std::move(name)) {}

  AstNodeKind kind() const override { return AstNodeKind::RestPattern; }
};

struct StructPattern : public Pattern {
  std::optional<std::string> name;
  using Field =
      std::variant<std::unique_ptr<PatternField>, std::unique_ptr<RestPattern>>;
  std::vector<Field> fields;

  StructPattern(Token token, std::optional<std::string> name,
                std::vector<Field> fields)
      : Pattern(std::move(token)), name(std::move(name)),
        fields(std::move(fields)) {}

  AstNodeKind kind() const override { return AstNodeKind::StructPattern; }
};

struct SlicePattern : public Pattern {
  std::vector<std::unique_ptr<Pattern>> elements;
  bool is_exhaustive;

  SlicePattern(Token token, std::vector<std::unique_ptr<Pattern>> elements,
               bool is_exhaustive)
      : Pattern(std::move(token)), elements(std::move(elements)),
        is_exhaustive(is_exhaustive) {}

  AstNodeKind kind() const override { return AstNodeKind::SlicePattern; }
};

struct OrPattern : public Pattern {
  std::vector<std::unique_ptr<Pattern>> patterns;

  OrPattern(Token token, std::vector<std::unique_ptr<Pattern>> patterns)
      : Pattern(std::move(token)), patterns(std::move(patterns)) {}

  AstNodeKind kind() const override { return AstNodeKind::OrPattern; }
};

struct ExprPattern : public Pattern {
  std::unique_ptr<Expression> expr;

  ExprPattern(Token token, std::unique_ptr<Expression> expr)
      : Pattern(std::move(token)), expr(std::move(expr)) {}

  AstNodeKind kind() const override { return AstNodeKind::ExprPattern; }
};

struct RangePattern : public Pattern {
  std::unique_ptr<Pattern> start;
  std::unique_ptr<Pattern> end;
  bool inclusive;

  RangePattern(Token token, std::unique_ptr<Pattern> start,
               std::unique_ptr<Pattern> end, bool inclusive)
      : Pattern(std::move(token)), start(std::move(start)), end(std::move(end)),
        inclusive(inclusive) {}

  AstNodeKind kind() const override { return AstNodeKind::RangePattern; }
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
      : Pattern(std::move(token)), name(std::move(name)),
        field(std::move(field)) {}

  AstNodeKind kind() const override { return AstNodeKind::VariantPattern; }
};

struct TopLevelDeclStmt : public Statement {
  std::unique_ptr<TopLevelDecl> decl;

  TopLevelDeclStmt(Token token, std::unique_ptr<TopLevelDecl> decl)
      : Statement(std::move(token)), decl(std::move(decl)) {}

  AstNodeKind kind() const override { return AstNodeKind::TopLevelDeclStmt; }
};

struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

struct AstDumper {
  void dump(Program *);
  void dump(TopLevelDecl *);
  void dump(Module *);
  void dump(Function *);
  void dump(FunctionDecl *);
  void dump(ImplDecl *);
  void dump(VarDecl *);
  void dump(TopLevelVarDecl *);
  void dump(IfExpr *);
  void dump(MatchArm *);
  void dump(MatchExpr *);
  void dump(ForExpr *);
  void dump(WhileExpr *);
  void dump(ReturnExpr *);
  void dump(DeferStmt *);
  void dump(BreakExpr *);
  void dump(ContinueExpr *);
  void dump(ExprStmt *);
  void dump(LiteralExpr *);
  void dump(TupleExpr *);
  void dump(ArrayExpr *);
  void dump(BinaryExpr *);
  void dump(UnaryExpr *);
  void dump(CallExpr *);
  void dump(AssignExpr *);
  void dump(AssignOpExpr *);
  void dump(FieldAccessExpr *);
  void dump(IndexExpr *);
  void dump(RangeExpr *);
  void dump(PrimitiveType *);
  void dump(TupleType *);
  void dump(FunctionType *);
  void dump(ReferenceType *);
  void dump(SliceType *);
  void dump(ArrayType *);
  void dump(TraitType *);
  void dump(IdentifierType *);
  void dump(StructType *);
  void dump(EnumType *);
  void dump(UnionType *);
  void dump(ExprType *);
  void dump(LiteralPattern *);
  void dump(IdentifierPattern *);
  void dump(WildcardPattern *);
  void dump(TuplePattern *);
  void dump(PatternField *);
  void dump(RestPattern *);
  void dump(StructPattern *);
  void dump(SlicePattern *);
  void dump(OrPattern *);
  void dump(ExprPattern *);
  void dump(RangePattern *);
  void dump(VariantPattern *);
  void dump(TopLevelDeclStmt *);
  void dump(ComptimeExpr *);
  void dump(BlockExpression *);
  void dump(Parameter *);
  void dump(Expression *);
  void dump(Statement *);
  void dump(Type *);
  void dump(Pattern *);
  void dump(StructField *);
  void dump(StructDecl *);
  void dump(TupleStructDecl *);
  void dump(IdentifierExpr *);
  void dump(FieldsNamed *);
  void dump(FieldsUnnamed *);
  void dump(Variant *);
  void dump(UnionDecl *);
  void dump(UnionField *);
  void dump(EnumDecl *);
  void dump(ImportDecl *);
  void dump(TraitDecl *);
  void dump(MLIRAttribute *);
  void dump(MLIRType *);
  void dump(MLIROp *);
  void dump(YieldExpr *);

  void indent();

  int cur_indent = 0;
  std::ostringstream output_stream;
  bool skip_import = false;

  std::string toString() const { return output_stream.str(); }

  template <typename T> std::string dump(T *node) {
    dump(node);
    auto str = toString();
    output_stream.str("");
    return str;
  }

  AstDumper(bool skip_import = false) : skip_import(skip_import) {}
};
