#include "LangGen.h"
#include "dialect/LangDialect.h"
#include "dialect/LangOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <memory>

using llvm::StringRef;

#define NEW_SCOPE()                                                            \
  llvm::ScopedHashTableScope<StringRef, mlir::Value> varScope(symbol_table);   \
  llvm::ScopedHashTableScope<StringRef, mlir::Type> typeScope(type_table);     \
  llvm::ScopedHashTableScope<StringRef, StructDecl *> structScope(             \
      struct_table);                                                           \
  llvm::ScopedHashTableScope<mlir::lang::StructType, StringRef>                \
      structNameScope(struct_name_table);

class LangGenImpl {
public:
  mlir::lang::FuncOp *current_function = nullptr;

  LangGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  llvm::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbol_table.count(var))
      return mlir::failure();
    symbol_table.insert(var, value);
    return mlir::success();
  }

  llvm::LogicalResult declare(llvm::StringRef var, mlir::Type type) {
    if (type_table.count(var))
      return mlir::failure();
    type_table.insert(var, type);
    return mlir::success();
  }

  mlir::ModuleOp langGen(Program *program) {
    NEW_SCOPE()
    the_module = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto &f : program->items) {
      if (f->kind() == AstNodeKind::Function) {
        auto func = langGen(dynamic_cast<Function *>(f.get()));
        if (failed(func)) {
          return nullptr;
        }
      } else if (f->kind() == AstNodeKind::TupleStructDecl) {
        auto tuple_struct = langGen(dynamic_cast<TupleStructDecl *>(f.get()));
        if (failed(tuple_struct)) {
          return nullptr;
        }
      } else if (f->kind() == AstNodeKind::StructDecl) {
        auto struct_decl = langGen(dynamic_cast<StructDecl *>(f.get()));
        if (failed(struct_decl)) {
          return nullptr;
        }
      } else if (f->kind() == AstNodeKind::ImplDecl) {
        auto impl_decl = langGen(dynamic_cast<ImplDecl *>(f.get()));
        if (failed(impl_decl)) {
          return nullptr;
        }
      } else if (f->kind() == AstNodeKind::ImportDecl) {
      } else {
        the_module.emitError("unsupported top-level item");
      }
    }

    if (failed(mlir::verify(the_module))) {
      the_module.emitError("module verification error");
      return nullptr;
    }

    return the_module;
  }

private:
  mlir::ModuleOp the_module;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<StringRef, mlir::Value> symbol_table;
  llvm::ScopedHashTable<StringRef, mlir::Type> type_table;
  llvm::ScopedHashTable<StringRef, StructDecl *> struct_table;
  llvm::ScopedHashTable<mlir::lang::StructType, StringRef> struct_name_table;
  llvm::StringMap<mlir::lang::FuncOp> function_map;

  mlir::Location loc(const TokenSpan &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr("temp.lang"),
                                     loc.line_no, loc.col_start);
  }

  mlir::LogicalResult langGen(Function *func) {
    NEW_SCOPE()
    builder.setInsertionPointToEnd(the_module.getBody());
    auto param_types = langGen(func->decl->parameters);
    if (failed(param_types)) {
      return mlir::failure();
    }
    mlir::TypeRange return_types =
        func->decl->return_type->kind() == AstNodeKind::PrimitiveType &&
                static_cast<PrimitiveType *>(func->decl->return_type.get())
                        ->type_kind == PrimitiveType::PrimitiveTypeKind::Void
            ? mlir::TypeRange()
            : mlir::TypeRange(getType(func->decl->return_type.get()).value());

    auto func_type = builder.getFunctionType(param_types.value(), return_types);
    auto func_op = builder.create<mlir::lang::FuncOp>(
        loc(func->token.span), func->decl->name, func_type);

    function_map[func->decl->name] = func_op;
    current_function = &func_op;

    auto entry_block = func_op.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);

    // Declare function parameters.
    if (failed(declare_parameters(func->decl->parameters,
                                  entry_block->getArguments()))) {
      func_op.erase();
      return emitError(loc(func->token.span), "parameter declaration error");
    }

    // Generate function body.
    if (failed(langGen(func->body.get()))) {
      func_op.erase();
      return mlir::failure();
    }

    if (func_op.getBody().back().getOperations().empty() ||
        !mlir::isa<mlir::lang::ReturnOp>(
            func_op.getBody().back().getOperations().back())) {
      builder.setInsertionPointToEnd(&func_op.getBody().back());
      builder.create<mlir::lang::ReturnOp>(loc(func->token.span));
    }
    current_function = nullptr;
    return mlir::success();
  }

  llvm::LogicalResult
  declare_parameters(std::vector<std::unique_ptr<Parameter>> &params,
                     mlir::ArrayRef<mlir::BlockArgument> args) {
    if (params.size() != args.size()) {
      the_module.emitError("parameter size mismatch");
      return mlir::failure();
    }

    for (int i = 0; i < (int)params.size(); i++) {
      // Assume identifier pattern
      auto &var_name =
          dynamic_cast<IdentifierPattern *>(params[i]->pattern.get())->name;
      if (failed(declare(var_name, args[i]))) {
        the_module.emitError("redeclaration of parameter");
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  llvm::FailureOr<llvm::SmallVector<mlir::Type, 4>>
  langGen(std::vector<std::unique_ptr<Parameter>> &params) {
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (auto &param : params) {
      auto loc = this->loc(param->token.span);
      auto type = getType(param->type.get());
      if (failed(type)) {
        return emitError(loc, "unsupported parameter type");
      }
      argTypes.push_back(type.value());
    }
    return argTypes;
  }

  llvm::FailureOr<mlir::Value> langGen(ReturnExpr *expr) {
    auto loc = this->loc(expr->token.span);
    if (!expr->value.has_value()) {
      builder.create<mlir::lang::ReturnOp>(loc);
      return mlir::success(mlir::Value());
    }
    auto value = langGen(expr->value.value().get());
    if (failed(value)) {
      emitError(loc, "unsupported return value");
      return mlir::failure();
    }

    // NOTE: Assuming only one return value for now
    auto return_type = current_function->getFunctionType().getResult(0);
    if (value.value().getType() != return_type) {
      // insert cast
      value = builder
                  .create<mlir::UnrealizedConversionCastOp>(loc, return_type,
                                                            value.value())
                  .getResult(0);
    }

    builder.create<mlir::lang::ReturnOp>(loc, value.value());
    return value;
  }

  llvm::LogicalResult langGen(TupleStructDecl *decl) {
    auto span = this->loc(decl->token.span);
    llvm::SmallVector<mlir::Type, 4> fieldTypes;
    for (auto &field : decl->fields) {
      auto type = getType(field.get());
      if (failed(type)) {
        emitError(span, "unsupported field type");
        return mlir::failure();
      }
      fieldTypes.push_back(type.value());
    }
    auto struct_type = mlir::lang::StructType::get(fieldTypes, decl->name);
    // declare struct type
    if (failed(declare(decl->name, struct_type))) {
      emitError(span, "redeclaration of struct type");
      return mlir::failure();
    }
    return mlir::success();
  }

  mlir::LogicalResult langGen(StructDecl *struct_decl) {
    llvm::SmallVector<mlir::Type, 4> element_types;
    for (auto &field : struct_decl->fields) {
      auto field_type = getType(field->type.get());
      if (failed(field_type)) {
        return mlir::failure();
      }
      element_types.push_back(field_type.value());
    }
    auto struct_type =
        mlir::lang::StructType::get(element_types, struct_decl->name);
    struct_table.insert(struct_decl->name, struct_decl);
    if (failed(declare(struct_decl->name, struct_type))) {
      return mlir::emitError(loc(struct_decl->token.span),
                             "redeclaration of struct type");
    }
    struct_name_table.insert(struct_type, struct_decl->name);
    return mlir::success();
  }

  mlir::LogicalResult langGen(ImplDecl *impl_decl) {
    auto parent_type = type_table.lookup(impl_decl->type);
    if (!parent_type) {
      return mlir::emitError(loc(impl_decl->token.span),
                             "parent type not found");
    }
    return mlir::success();
  }

  mlir::LogicalResult langGen(BlockExpression *block) {
    for (auto &stmt : block->statements) {
      if (failed(langGen(stmt.get()))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  mlir::LogicalResult langGen(Statement *stmt) {
    if (auto expr = dynamic_cast<VarDecl *>(stmt)) {
      return langGen(expr);
    } else if (auto expr = dynamic_cast<ExprStmt *>(stmt)) {
      return langGen(expr->expr.get());
    }
    return mlir::emitError(loc(stmt->token.span), "unsupported statement");
  }

  mlir::LogicalResult langGen(VarDecl *var_decl) {
    mlir::Type var_type = nullptr;
    if (var_decl->type.has_value()) {
      var_type = getType(var_decl->type.value().get()).value();
    }

    // assume the name is identifier pattern for now
    auto &var_name =
        dynamic_cast<IdentifierPattern *>(var_decl->pattern.get())->name;
    if (symbol_table.count(var_name)) {
      return mlir::emitError(loc(var_decl->token.span),
                             "a variable with name " + var_name +
                                 " already exists");
    }

    std::optional<mlir::Value> init_value = std::nullopt;
    if (var_decl->initializer.has_value()) {
      auto v = langGen(var_decl->initializer.value().get());
      if (mlir::failed(v)) {
        return mlir::failure();
      }
      init_value = v.value();
    }

    builder.create<mlir::lang::VarDeclOp>(loc(var_decl->token.span), var_name,
                                          var_type, init_value.value());
    if (failed(declare(var_name, init_value.value()))) {
      return mlir::failure();
    }
    return mlir::success();
  }

  mlir::FailureOr<mlir::Type> getType(Type *type) {
    if (type->kind() == AstNodeKind::PrimitiveType) {
      auto primitive_type = static_cast<PrimitiveType *>(type);
      if (primitive_type->type_kind == PrimitiveType::PrimitiveTypeKind::I32) {
        return builder.getIntegerType(32);
      } else if (primitive_type->type_kind ==
                 PrimitiveType::PrimitiveTypeKind::I64) {
        return builder.getIntegerType(64);
      } else if (primitive_type->type_kind ==
                 PrimitiveType::PrimitiveTypeKind::F32) {
        return builder.getF32Type();
      } else if (primitive_type->type_kind ==
                 PrimitiveType::PrimitiveTypeKind::F64) {
        return builder.getF64Type();
      } else if (primitive_type->type_kind ==
                 PrimitiveType::PrimitiveTypeKind::Bool) {
        return builder.getIntegerType(1);
      } else if (primitive_type->type_kind ==
                 PrimitiveType::PrimitiveTypeKind::Char) {
        return builder.getIntegerType(8);
      } else if (primitive_type->type_kind ==
                 PrimitiveType::PrimitiveTypeKind::Void) {
        return builder.getNoneType();
      } else {
        mlir::emitError(loc(primitive_type->token.span), "unsupported type");
      }
    } else if (type->kind() == AstNodeKind::MLIRType) {
      auto mlir_type = static_cast<MLIRType *>(type);
      llvm::StringRef type_name = mlir_type->type;
      return mlir::parseType(type_name.trim('"'), builder.getContext());
    } else if (type->kind() == AstNodeKind::IdentifierType) {
      auto identifier_type = static_cast<IdentifierType *>(type);
      auto type_name = identifier_type->name;
      mlir::Operation *op = nullptr;
      if (current_function) {
        mlir::SymbolTable symbolTable(current_function->getOperation());
        op = symbolTable.lookup(type_name);
      }
      if (!op) {
        // search in struct table
        auto struct_type = type_table.lookup(type_name);
        if (!struct_type || !mlir::isa<mlir::lang::StructType>(struct_type)) {
          return mlir::emitError(loc(identifier_type->token.span),
                                 "unknown type");
        }
        return struct_type;
      }
      return op->getResult(0).getType();
    }
    return mlir::emitError(loc(type->token.span), "unsupported type");
  }

  mlir::FailureOr<mlir::Value> langGen(Expression *expr) {
    if (auto e = dynamic_cast<LiteralExpr *>(expr)) {
      return langGen(e);
    } else if (auto e = dynamic_cast<MLIRAttribute *>(expr)) {
      return langGen(e);
    } else if (auto e = dynamic_cast<MLIRType *>(expr)) {
      return langGen(e);
    } else if (auto e = dynamic_cast<ReturnExpr *>(expr)) {
      return langGen(e);
    } else if (auto e = dynamic_cast<BlockExpression *>(expr)) {
      return langGen(e);
    } else if (auto e = dynamic_cast<VarDecl *>(expr)) {
      return langGen(e);
    } else if (auto e = dynamic_cast<CallExpr *>(expr)) {
      return langGen(e);
    } else if (auto e = dynamic_cast<FieldAccessExpr *>(expr)) {
      return langGen(e);
    } else if (auto e = dynamic_cast<IdentifierExpr *>(expr)) {
      return langGen(e);
    }
    return mlir::emitError(loc(expr->token.span),
                           "unsupported expression " + to_string(expr->kind()));
  }

  mlir::FailureOr<mlir::Value> langGen(IdentifierExpr *expr) {
    auto var = symbol_table.lookup(expr->name);
    if (!var) {
      return mlir::emitError(loc(expr->token.span),
                             "undeclared variable " + expr->name);
    }
    return var;
  }

  mlir::FailureOr<mlir::Value> langGen(CallExpr *expr) {
    // lookup callee in struct table
    auto struct_type = type_table.lookup(expr->callee);
    if (struct_type && mlir::isa<mlir::lang::StructType>(struct_type))
      return create_struct_instance(
          mlir::cast<mlir::lang::StructType>(struct_type), expr->arguments);

    // get arguments
    llvm::SmallVector<mlir::Value, 4> args;
    for (auto &arg : expr->arguments) {
      auto arg_value = langGen(arg.get());
      if (failed(arg_value)) {
        return mlir::failure();
      }
      // add an unrealized conversion cast
      arg_value = builder
                      .create<mlir::UnrealizedConversionCastOp>(
                          loc(arg->token.span), arg_value.value().getType(),
                          arg_value.value())
                      .getResult(0);
      args.push_back(arg_value.value());
    }

    auto &func_name = expr->callee;
    if (func_name == "print") {
      if (args.size() < 1)
        return mlir::emitError(loc(expr->token.span),
                               "print function expects 1 argument");
      auto arg0 = expr->arguments[0].get();
      if (arg0->kind() != AstNodeKind::LiteralExpr ||
          static_cast<LiteralExpr *>(arg0)->type !=
              LiteralExpr::LiteralType::String)
        return mlir::emitError(loc(expr->token.span),
                               "print function expects a string argument");
      auto format_str =
          std::get<std::string>(static_cast<LiteralExpr *>(arg0)->value);
      format_str = format_str.substr(1, format_str.size() - 2) + '\n';
      mlir::ValueRange rest_args = llvm::ArrayRef(args).drop_front();
      builder.create<mlir::lang::PrintOp>(loc(expr->token.span), format_str,
                                          rest_args);
      return mlir::success(mlir::Value());
    }

    if (!function_map.count(func_name)) {
      return mlir::emitError(loc(expr->token.span),
                             "function " + func_name + " not found");
    }

    auto func = function_map[func_name];
    // call function
    auto call_op = builder.create<mlir::lang::CallOp>(
        loc(expr->token.span), func, mlir::ValueRange(args));
    if (call_op.getNumResults() == 0) {
      return mlir::success(mlir::Value());
    }
    return call_op.getResult(0);
  }

  mlir::FailureOr<mlir::Value> langGen(FieldAccessExpr *expr) {
    auto base_value = langGen(expr->base.get());
    if (failed(base_value)) {
      return mlir::failure();
    }
    auto base_type = base_value.value().getType();
    if (!mlir::isa<mlir::lang::StructType>(base_type)) {
      return mlir::emitError(loc(expr->token.span),
                             "field access on non-struct type");
    }
    auto struct_type = mlir::cast<mlir::lang::StructType>(base_type);
    auto struct_name = struct_name_table.lookup(struct_type);
    auto struct_decl = struct_table.lookup(struct_name);
    if (!struct_decl) {
      return mlir::emitError(loc(expr->token.span),
                             "struct not found in struct table");
    }

    // assume field is an identifier for now
    auto &field_name =
        std::get<std::unique_ptr<IdentifierExpr>>(expr->field)->name;

    // get field index
    int field_index = -1;
    for (int i = 0; i < (int)struct_decl->fields.size(); i++) {
      if (struct_decl->fields[i]->name == field_name) {
        field_index = i;
        break;
      }
    }

    if (field_index == -1) {
      return mlir::emitError(loc(expr->token.span),
                             "field not found in struct");
    }

    return builder
        .create<mlir::lang::StructAccessOp>(loc(expr->token.span),
                                            base_value.value(), field_index)
        .getResult();
  }

  mlir::FailureOr<mlir::Value>
  create_struct_instance(mlir::lang::StructType struct_type,
                         std::vector<std::unique_ptr<Expression>> &args) {
    llvm::SmallVector<mlir::Value, 4> field_values;
    for (int i = 0; i < (int)args.size(); i++) {
      auto field_value = langGen(args[i].get());
      if (failed(field_value)) {
        return mlir::failure();
      }
      field_values.push_back(field_value.value());
    }
    return builder
        .create<mlir::lang::CreateStructOp>(loc(args[0]->token.span),
                                            struct_type, field_values)
        .getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(Type *type) {
    auto mlir_type = getType(type);
    if (failed(mlir_type)) {
      return mlir::failure();
    }
    auto type_op = builder.create<mlir::lang::TypeConstOp>(
        loc(type->token.span), mlir_type.value());
    return type_op.getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(MLIRAttribute *attr) {
    llvm::StringRef attribute = attr->attribute;
    auto mlir_attr =
        mlir::parseAttribute(attribute.trim('"'), builder.getContext());
    if (!mlir_attr) {
      // Handle parsing failure
      return mlir::emitError(loc(attr->token.span),
                             "failed to parse attribute");
    }

    auto typedAttr = mlir::cast<mlir::TypedAttr>(mlir_attr);
    auto constantOp = builder.create<mlir::lang::ConstantOp>(
        loc(attr->token.span), typedAttr.getType(), typedAttr);

    return constantOp.getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(LiteralExpr *literal) {
    if (literal->type == LiteralExpr::LiteralType::Int) {
      // Create an IntegerLiteral struct instance
      auto struct_type = type_table.lookup("IntLiteral");
      mlir::TypedAttr attr = builder.getIntegerAttr(
          builder.getIntegerType(64), std::get<int>(literal->value));
      auto field_value = builder
                             .create<mlir::lang::ConstantOp>(
                                 loc(literal->token.span), attr.getType(), attr)
                             .getResult();
      return builder
          .create<mlir::lang::CreateStructOp>(loc(literal->token.span),
                                              struct_type,
                                              mlir::ValueRange{field_value})
          .getResult();
    } else if (literal->type == LiteralExpr::LiteralType::String) {
      return builder
          .create<mlir::lang::StringConstOp>(
              loc(literal->token.span),
              mlir::lang::StringType::get(builder.getContext()),
              std::get<std::string>(literal->value))
          .getResult();
    } else if (literal->type == LiteralExpr::LiteralType::Bool) {
      mlir::TypedAttr attr = builder.getIntegerAttr(
          builder.getIntegerType(1), std::get<bool>(literal->value));
      return builder
          .create<mlir::lang::ConstantOp>(loc(literal->token.span),
                                          attr.getType(), attr)
          .getResult();

    } else if (literal->type == LiteralExpr::LiteralType::Char) {
      mlir::TypedAttr attr = builder.getIntegerAttr(
          builder.getIntegerType(8), std::get<char>(literal->value));
      return builder
          .create<mlir::lang::ConstantOp>(loc(literal->token.span),
                                          attr.getType(), attr)
          .getResult();
    } else if (literal->type == LiteralExpr::LiteralType::Float) {
      mlir::TypedAttr attr = builder.getFloatAttr(
          builder.getF64Type(), std::get<double>(literal->value));
      return builder
          .create<mlir::lang::ConstantOp>(loc(literal->token.span),
                                          attr.getType(), attr)
          .getResult();
    }
    return mlir::emitError(loc(literal->token.span), "unsupported literal");
  }
};

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> langGen(mlir::MLIRContext &context,
                                          Program *program) {
  return LangGenImpl(context).langGen(program);
}
