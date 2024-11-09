#include "ast.hpp"
#include "compiler.hpp"
#include <llvm/ADT/ScopedHashTable.h>
#include <memory>

struct Analyzer {

  std::shared_ptr<Context> context;

  Analyzer(std::shared_ptr<Context> context) : context(context) {}

  void analyze(Program *);
  void analyze(TopLevelDecl *);
  void analyze(Module *);
  void analyze(Function *);
  void analyze(FunctionDecl *);
  void analyze(ImplDecl *);
  void analyze(VarDecl *);
  void analyze(TopLevelVarDecl *);
  void analyze(IfExpr *);
  void analyze(MatchArm *);
  void analyze(MatchExpr *);
  void analyze(ForExpr *);
  void analyze(WhileExpr *);
  void analyze(ReturnExpr *);
  void analyze(DeferStmt *);
  void analyze(BreakExpr *);
  void analyze(ContinueExpr *);
  void analyze(ExprStmt *);
  void analyze(LiteralExpr *);
  void analyze(TupleExpr *);
  void analyze(ArrayExpr *);
  void analyze(BinaryExpr *);
  void analyze(UnaryExpr *);
  void analyze(CallExpr *);
  void analyze(AssignExpr *);
  void analyze(AssignOpExpr *);
  void analyze(FieldAccessExpr *);
  void analyze(IndexExpr *);
  void analyze(RangeExpr *);
  void analyze(PrimitiveType *);
  void analyze(TupleType *);
  void analyze(FunctionType *);
  void analyze(ReferenceType *);
  void analyze(SliceType *);
  void analyze(ArrayType *);
  void analyze(TraitType *);
  void analyze(IdentifierType *);
  void analyze(StructType *);
  void analyze(EnumType *);
  void analyze(UnionType *);
  void analyze(ExprType *);
  void analyze(LiteralPattern *);
  void analyze(IdentifierPattern *);
  void analyze(WildcardPattern *);
  void analyze(TuplePattern *);
  void analyze(PatternField *);
  void analyze(RestPattern *);
  void analyze(StructPattern *);
  void analyze(SlicePattern *);
  void analyze(OrPattern *);
  void analyze(ExprPattern *);
  void analyze(RangePattern *);
  void analyze(VariantPattern *);
  void analyze(TopLevelDeclStmt *);
  void analyze(ComptimeExpr *);
  void analyze(BlockExpression *);
  void analyze(Parameter *);
  void analyze(Expression *);
  void analyze(Statement *);
  void analyze(Type *);
  void analyze(Pattern *);
  void analyze(StructField *);
  void analyze(StructDecl *);
  void analyze(TupleStructDecl *);
  void analyze(IdentifierExpr *);
  void analyze(FieldsNamed *);
  void analyze(FieldsUnnamed *);
  void analyze(Variant *);
  void analyze(UnionDecl *);
  void analyze(UnionField *);
  void analyze(EnumDecl *);
  void analyze(ImportDecl *);
  void analyze(TraitDecl *);
};
