#pragma once

#include "ast.hpp"

#include <sstream>
#include <string>

struct JsonDumper {
  JsonDumper(bool skip_import = false) : skip_import(skip_import) {}

  std::string to_string() const { return output_stream.str(); }

  void dump(const Token &token);
  void dump(Program *node);
  void dump(TopLevelDecl *node);
  void dump(Module *node);
  void dump(Function *node);
  void dump(FunctionDecl *node);
  void dump(ImplDecl *node);
  void dump(VarDecl *node);
  void dump(TopLevelVarDecl *node);
  void dump(IfExpr *node);
  void dump(MatchArm *node);
  void dump(MatchExpr *node);
  void dump(ForExpr *node);
  void dump(WhileExpr *node);
  void dump(ReturnExpr *node);
  void dump(DeferStmt *node);
  void dump(BreakExpr *node);
  void dump(ContinueExpr *node);
  void dump(ExprStmt *node);
  void dump(LiteralExpr *node);
  void dump(TupleExpr *node);
  void dump(ArrayExpr *node);
  void dump(BinaryExpr *node);
  void dump(UnaryExpr *node);
  void dump(CallExpr *node);
  void dump(AssignExpr *node);
  void dump(AssignOpExpr *node);
  void dump(FieldAccessExpr *node);
  void dump(IndexExpr *node);
  void dump(RangeExpr *node);
  void dump(PrimitiveType *node);
  void dump(TupleType *node);
  void dump(FunctionType *node);
  void dump(ReferenceType *node);
  void dump(SliceType *node);
  void dump(ArrayType *node);
  void dump(TraitType *node);
  void dump(IdentifierType *node);
  void dump(StructType *node);
  void dump(EnumType *node);
  void dump(UnionType *node);
  void dump(ExprType *node);
  void dump(LiteralPattern *node);
  void dump(IdentifierPattern *node);
  void dump(WildcardPattern *node);
  void dump(TuplePattern *node);
  void dump(PatternField *node);
  void dump(RestPattern *node);
  void dump(StructPattern *node);
  void dump(SlicePattern *node);
  void dump(OrPattern *node);
  void dump(ExprPattern *node);
  void dump(RangePattern *node);
  void dump(VariantPattern *node);
  void dump(TopLevelDeclStmt *node);
  void dump(ComptimeExpr *node);
  void dump(BlockExpression *node);
  void dump(Parameter *node);
  void dump(Expression *node);
  void dump(Statement *node);
  void dump(Type *node);
  void dump(Pattern *node);
  void dump(StructField *node);
  void dump(StructDecl *node);
  void dump(TupleStructDecl *node);
  void dump(IdentifierExpr *node);
  void dump(FieldsNamed *node);
  void dump(FieldsUnnamed *node);
  void dump(Variant *node);
  void dump(UnionDecl *node);
  void dump(UnionField *node);
  void dump(EnumDecl *node);
  void dump(ImportDecl *node);
  void dump(TraitDecl *node);
  void dump(MLIRType *node);
  void dump(MLIRAttribute *node);
  void dump(MLIROp *node);

  std::string token_kind_to_string(TokenKind kind);
  void dump(const TokenSpan &span);
  void dump(const TokenKind &span);
  void dumpNodeToken(Node *node);

private:
  void indent() {
    for (int i = 0; i < cur_indent; ++i) {
      output_stream << "  ";
    }
  }

  std::string to_string(AstNodeKind kind) {
    switch (kind) {
    case AstNodeKind::InvalidNode:
      return "InvalidNode";
    case AstNodeKind::InvalidExpression:
      return "InvalidExpression";
    case AstNodeKind::InvalidStatement:
      return "InvalidStatement";
    case AstNodeKind::InvalidTopLevelDecl:
      return "InvalidTopLevelDecl";
    case AstNodeKind::InvalidType:
      return "InvalidType";
    case AstNodeKind::InvalidPattern:
      return "InvalidPattern";
    case AstNodeKind::Program:
      return "Program";
    case AstNodeKind::Module:
      return "Module";
    case AstNodeKind::ComptimeExpr:
      return "ComptimeExpr";
    case AstNodeKind::BlockExpression:
      return "BlockExpression";
    case AstNodeKind::Parameter:
      return "Parameter";
    case AstNodeKind::Expression:
      return "Expression";
    case AstNodeKind::Statement:
      return "Statement";
    case AstNodeKind::Type:
      return "Type";
    case AstNodeKind::Pattern:
      return "Pattern";
    case AstNodeKind::StructField:
      return "StructField";
    case AstNodeKind::StructDecl:
      return "StructDecl";
    case AstNodeKind::TupleStructDecl:
      return "TupleStructDecl";
    case AstNodeKind::IdentifierExpr:
      return "IdentifierExpr";
    case AstNodeKind::FieldsNamed:
      return "FieldsNamed";
    case AstNodeKind::FieldsUnnamed:
      return "FieldsUnnamed";
    case AstNodeKind::Variant:
      return "Variant";
    case AstNodeKind::UnionDecl:
      return "UnionDecl";
    case AstNodeKind::UnionField:
      return "UnionField";
    case AstNodeKind::EnumDecl:
      return "EnumDecl";
    case AstNodeKind::ImportDecl:
      return "ImportDecl";
    case AstNodeKind::TraitDecl:
      return "TraitDecl";
    case AstNodeKind::FunctionDecl:
      return "FunctionDecl";
    case AstNodeKind::Function:
      return "Function";
    case AstNodeKind::ImplDecl:
      return "ImplDecl";
    case AstNodeKind::VarDecl:
      return "VarDecl";
    case AstNodeKind::TopLevelVarDecl:
      return "TopLevelVarDecl";
    case AstNodeKind::IfExpr:
      return "IfExpr";
    case AstNodeKind::MatchArm:
      return "MatchArm";
    case AstNodeKind::MatchExpr:
      return "MatchExpr";
    case AstNodeKind::ForExpr:
      return "ForExpr";
    case AstNodeKind::WhileExpr:
      return "WhileExpr";
    case AstNodeKind::ReturnExpr:
      return "ReturnExpr";
    case AstNodeKind::DeferStmt:
      return "DeferStmt";
    case AstNodeKind::BreakExpr:
      return "BreakExpr";
    case AstNodeKind::ContinueExpr:
      return "ContinueExpr";
    case AstNodeKind::ExprStmt:
      return "ExprStmt";
    case AstNodeKind::LiteralExpr:
      return "LiteralExpr";
    case AstNodeKind::TupleExpr:
      return "TupleExpr";
    case AstNodeKind::ArrayExpr:
      return "ArrayExpr";
    case AstNodeKind::BinaryExpr:
      return "BinaryExpr";
    case AstNodeKind::UnaryExpr:
      return "UnaryExpr";
    case AstNodeKind::CallExpr:
      return "CallExpr";
    case AstNodeKind::AssignExpr:
      return "AssignExpr";
    case AstNodeKind::AssignOpExpr:
      return "AssignOpExpr";
    case AstNodeKind::FieldAccessExpr:
      return "FieldAccessExpr";
    case AstNodeKind::IndexExpr:
      return "IndexExpr";
    case AstNodeKind::RangeExpr:
      return "RangeExpr";
    case AstNodeKind::PrimitiveType:
      return "PrimitiveType";
    case AstNodeKind::TupleType:
      return "TupleType";
    case AstNodeKind::FunctionType:
      return "FunctionType";
    case AstNodeKind::ReferenceType:
      return "ReferenceType";
    case AstNodeKind::SliceType:
      return "SliceType";
    case AstNodeKind::ArrayType:
      return "ArrayType";
    case AstNodeKind::TraitType:
      return "TraitType";
    case AstNodeKind::IdentifierType:
      return "IdentifierType";
    case AstNodeKind::StructType:
      return "StructType";
    case AstNodeKind::EnumType:
      return "EnumType";
    case AstNodeKind::UnionType:
      return "UnionType";
    case AstNodeKind::ExprType:
      return "ExprType";
    case AstNodeKind::LiteralPattern:
      return "LiteralPattern";
    case AstNodeKind::IdentifierPattern:
      return "IdentifierPattern";
    case AstNodeKind::WildcardPattern:
      return "WildcardPattern";
    case AstNodeKind::TuplePattern:
      return "TuplePattern";
    case AstNodeKind::PatternField:
      return "PatternField";
    case AstNodeKind::RestPattern:
      return "RestPattern";
    case AstNodeKind::StructPattern:
      return "StructPattern";
    case AstNodeKind::SlicePattern:
      return "SlicePattern";
    case AstNodeKind::OrPattern:
      return "OrPattern";
    case AstNodeKind::ExprPattern:
      return "ExprPattern";
    case AstNodeKind::RangePattern:
      return "RangePattern";
    case AstNodeKind::VariantPattern:
      return "VariantPattern";
    case AstNodeKind::TopLevelDeclStmt:
      return "TopLevelDeclStmt";
    case AstNodeKind::MLIRType:
      return "MLIRType";
    case AstNodeKind::MLIRAttribute:
      return "MLIRAttribute";
    case AstNodeKind::MLIROp:
      return "MLIROp";
    default:
      return "Unknown";
    }
  }

  void dumpOperator(Operator op) {
    switch (op) {
    case Operator::Add:
      output_stream << "Add";
      break;
    case Operator::Sub:
      output_stream << "Sub";
      break;
    case Operator::Mul:
      output_stream << "Mul";
      break;
    case Operator::Div:
      output_stream << "Div";
      break;
    case Operator::Mod:
      output_stream << "Mod";
      break;
    case Operator::And:
      output_stream << "And";
      break;
    case Operator::Or:
      output_stream << "Or";
      break;
    case Operator::Not:
      output_stream << "Not";
      break;
    case Operator::Eq:
      output_stream << "Eq";
      break;
    case Operator::Ne:
      output_stream << "Ne";
      break;
    case Operator::Lt:
      output_stream << "Lt";
      break;
    case Operator::Le:
      output_stream << "Le";
      break;
    case Operator::Gt:
      output_stream << "Gt";
      break;
    case Operator::Ge:
      output_stream << "Ge";
      break;
    case Operator::BitAnd:
      output_stream << "BitAnd";
      break;
    case Operator::BitOr:
      output_stream << "BitOr";
      break;
    case Operator::BitXor:
      output_stream << "BitXor";
      break;
    case Operator::BitShl:
      output_stream << "BitShl";
      break;
    case Operator::BitShr:
      output_stream << "BitShr";
      break;
    case Operator::BitNot:
      output_stream << "BitNot";
      break;
    case Operator::Pow:
      output_stream << "Pow";
      break;
    default:
      output_stream << "Invalid";
      break;
    }
  }

  void dumpLiteralType(LiteralExpr::LiteralType type) {
    switch (type) {
    case LiteralExpr::LiteralType::Int:
      output_stream << "Int";
      break;
    case LiteralExpr::LiteralType::Float:
      output_stream << "Float";
      break;
    case LiteralExpr::LiteralType::String:
      output_stream << "String";
      break;
    case LiteralExpr::LiteralType::Char:
      output_stream << "Char";
      break;
    case LiteralExpr::LiteralType::Bool:
      output_stream << "Bool";
      break;
    }
  }

  void dumpPrimitiveTypeKind(PrimitiveType::PrimitiveTypeKind kind) {
    switch (kind) {
    case PrimitiveType::PrimitiveTypeKind::String:
      output_stream << "String";
      break;
    case PrimitiveType::PrimitiveTypeKind::Char:
      output_stream << "Char";
      break;
    case PrimitiveType::PrimitiveTypeKind::Bool:
      output_stream << "Bool";
      break;
    case PrimitiveType::PrimitiveTypeKind::Void:
      output_stream << "Void";
      break;
    case PrimitiveType::PrimitiveTypeKind::I8:
      output_stream << "I8";
      break;
    case PrimitiveType::PrimitiveTypeKind::I16:
      output_stream << "I16";
      break;
    case PrimitiveType::PrimitiveTypeKind::I32:
      output_stream << "I32";
      break;
    case PrimitiveType::PrimitiveTypeKind::I64:
      output_stream << "I64";
      break;
    case PrimitiveType::PrimitiveTypeKind::U8:
      output_stream << "U8";
      break;
    case PrimitiveType::PrimitiveTypeKind::U16:
      output_stream << "U16";
      break;
    case PrimitiveType::PrimitiveTypeKind::U32:
      output_stream << "U32";
      break;
    case PrimitiveType::PrimitiveTypeKind::U64:
      output_stream << "U64";
      break;
    case PrimitiveType::PrimitiveTypeKind::F32:
      output_stream << "F32";
      break;
    case PrimitiveType::PrimitiveTypeKind::F64:
      output_stream << "F64";
      break;
    case PrimitiveType::PrimitiveTypeKind::type:
      output_stream << "type";
      break;
    }
  }

  int cur_indent = 0;
  std::ostringstream output_stream;
  bool skip_import = false;
};
