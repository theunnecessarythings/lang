#include <memory>
#include <optional>
#include <string>
#include <vector>

struct Node {
  virtual ~Node() = default;
};

struct Expression : public Node {};
struct Statement : public Node {};
struct TopLevelDecl : public Node {};
struct Type : public Node {};
struct Pattern : public Node {};

struct Program : public Node {
  std::vector<std::unique_ptr<TopLevelDecl>> items;
};

struct Module : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<TopLevelDecl>> items;
};

struct BlockExpression : public Expression {
  std::vector<std::unique_ptr<Statement>> statements;
  std::optional<std::unique_ptr<Type>> return_type;
};

struct BlockStatement : public Statement {
  BlockExpression block;
};

struct Parameter : public Node {
  std::unique_ptr<Pattern> pattern;
  std::unique_ptr<Type> type;
};

struct StructField : public Node {
  std::string name;
  std::unique_ptr<Type> type;
};

struct StructDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<StructField>> fields;
};

struct TupleStructDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<Type>> fields;
};

struct EnumVariant : public Node {
  std::string name;

  union VariantType {
    Type type;
    TupleStructDecl tuple_struct;
    StructDecl struct_decl;
  };
  std::vector<std::unique_ptr<VariantType>> fields;
};

struct UnionField : public Node {
  std::string name;
  std::unique_ptr<Type> type;
};

struct Function : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<Parameter>> parameters;
  std::unique_ptr<Type> return_type;
  std::unique_ptr<BlockStatement> body;
};

struct ImportDecl : public TopLevelDecl {
  std::string path;
  std::optional<std::string> alias;
};

struct EnumDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<EnumVariant>> variants;
};

struct UnionDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<UnionField>> fields;
};

struct TraitDecl : public TopLevelDecl {
  std::string name;
  std::vector<std::unique_ptr<Function>> functions;
};

struct ImplDecl : public TopLevelDecl {
  std::optional<std::string> trait;
  std::string for_type;
  std::vector<std::unique_ptr<Function>> functions;
};

struct VarDecl : public Statement {
  Pattern pattern;
  std::optional<Type> type;
  std::optional<Expression> initializer;
  bool is_mut;
};

struct IfExpr : public Expression {
  union Branch {
    std::unique_ptr<BlockExpression> block;
    std::unique_ptr<Expression> expr;
  };

  std::unique_ptr<Expression> condition;
  std::unique_ptr<Branch> then_block;
  std::optional<std::unique_ptr<Branch>> else_block;
  std::optional<std::unique_ptr<Type>> type;
};

struct MatchArm : public Node {
  std::unique_ptr<Pattern> pattern;
  union Branch {
    std::unique_ptr<BlockExpression> block;
    std::unique_ptr<Expression> expr;
  };
  std::unique_ptr<Branch> body;
  std::optional<std::unique_ptr<Expression>> guard;
};

struct MatchExpr : public Expression {
  std::unique_ptr<Expression> expr;
  std::vector<std::unique_ptr<MatchArm>> cases;
};

struct ForExpr : public Expression {
  std::unique_ptr<Pattern> pattern;
  std::unique_ptr<Expression> iterable;
  std::unique_ptr<BlockStatement> body;
  std::optional<std::unique_ptr<Type>> type;
};

struct WhileExpr : public Expression {
  std::unique_ptr<Expression> condition;
  std::unique_ptr<BlockStatement> body;
  std::optional<std::unique_ptr<Type>> type;
};

struct ReturnExpr : public Expression {
  std::optional<std::unique_ptr<Expression>> value;
  std::unique_ptr<Type> type;
};

struct DeferStmt : public Statement {
  union Branch {
    std::unique_ptr<BlockStatement> block;
    std::unique_ptr<Expression> expr;
  };
  std::unique_ptr<Branch> body;
};

struct BreakExpr : public Expression {
  std::optional<std::string> label;
  std::optional<std::unique_ptr<Type>> type;
  std::optional<std::unique_ptr<Expression>> value;
};

struct ContinueExpr : public Expression {
  std::optional<std::string> label;
  std::optional<std::unique_ptr<Type>> type;
  std::optional<std::unique_ptr<Expression>> value;
};

struct ExprStmt : public Statement {
  std::unique_ptr<Expression> expr;
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
  union Value {
    int int_val;
    float float_val;
    std::string string_val;
    char char_val;
    bool bool_val;
  };
  std::shared_ptr<Value> value;
};

struct IdentifierExpr : public Expression {
  std::string name;
};

struct TupleExpr : public Expression {
  std::vector<std::unique_ptr<Expression>> elements;
};

struct ArrayExpr : public Expression {
  std::vector<std::unique_ptr<Expression>> elements;
};

enum class Operator {
  Add,
  Sub,
  Mul,
  Div,
  Mod,
  And,
  Or,
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
};
struct BinaryExpr : public Expression {
  Operator op;
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;
};

struct UnaryExpr : public Expression {
  Operator op;
  std::unique_ptr<Expression> operand;
};

struct CallExpr : public Expression {
  std::unique_ptr<Expression> callee;
  std::vector<std::unique_ptr<Expression>> arguments;
};

struct AssignExpr : public Expression {
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;
};

struct AssignOpExpr : public Expression {
  Operator op;
  std::unique_ptr<Expression> lhs;
  std::unique_ptr<Expression> rhs;
};

struct FieldAccessExpr : public Expression {
  std::unique_ptr<Expression> base;
  std::string field;
};

struct IndexExpr : public Expression {
  std::unique_ptr<Expression> base;
  std::unique_ptr<Expression> index;
};

struct RangeExpr : public Expression {
  std::unique_ptr<Expression> start;
  std::unique_ptr<Expression> end;
  bool inclusive;
};

struct PrimitiveType : public Type {
  enum class PrimitiveTypeKind {
    Int,
    Float,
    String,
    Char,
    Bool,
    Unit,
  };
  PrimitiveTypeKind kind;
};

struct TupleType : public Type {
  std::vector<std::unique_ptr<Type>> elements;
};

struct FunctionType : public Type {
  std::vector<std::unique_ptr<Type>> parameters;
  std::unique_ptr<Type> return_type;
};

struct ReferenceType : public Type {
  std::unique_ptr<Type> base;
};

struct SliceType : public Type {
  std::unique_ptr<Type> base;
};

struct ArrayType : public Type {
  std::unique_ptr<Type> base;
  std::unique_ptr<Expression> size;
};

struct StructType : public Type {
  std::string name;
};

struct EnumType : public Type {
  std::string name;
};

struct UnionType : public Type {
  std::string name;
};

struct TraitType : public Type {
  std::string name;
};

struct LiteralPattern : public Pattern {
  LiteralExpr literal;
};

struct IdentifierPattern : public Pattern {
  std::string name;
};

struct WildcardPattern : public Pattern {};

struct TuplePattern : public Pattern {
  std::vector<std::unique_ptr<Pattern>> elements;
};

struct StructPattern : public Pattern {
  std::string name;
  std::vector<std::pair<std::string, std::unique_ptr<Pattern>>> fields;
  bool is_exhaustive;
};

struct EnumVariantPattern : public Pattern {
  std::string enum_name;
  std::string variant_name;
  std::vector<std::pair<std::string, std::unique_ptr<Pattern>>> fields;
};

struct SlicePattern : public Pattern {
  std::vector<std::unique_ptr<Pattern>> elements;
  bool is_exhaustive;
};

struct OrPattern : public Pattern {
  std::vector<std::unique_ptr<Pattern>> patterns;
};

struct RangePattern : public Pattern {
  std::unique_ptr<Pattern> start;
  std::unique_ptr<Pattern> end;
  bool inclusive;
};
