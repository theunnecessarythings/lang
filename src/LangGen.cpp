#include "LangGen.h"
#include "ast.hpp"
#include "compiler.hpp"
#include "dialect/LangDialect.h"
#include "dialect/LangOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <memory>

using llvm::StringRef;

#define NEW_SCOPE()                                                            \
  llvm::ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbol_table);  \
  llvm::ScopedHashTableScope<StringRef, mlir::Type> type_scope(type_table);    \
  llvm::ScopedHashTableScope<StringRef, StructDecl *> struct_scope(            \
      struct_table);                                                           \
  llvm::ScopedHashTableScope<mlir::lang::StructType, StringRef>                \
      struct_name_scope(struct_name_table);

class LangGenImpl {
public:
  mlir::lang::FuncOp *current_function = nullptr;

  LangGenImpl(mlir::MLIRContext &context, Context &ctx)
      : compiler_context(ctx), builder(&context) {}

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
  Context &compiler_context;
  mlir::ModuleOp the_module;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<StringRef, mlir::Value> symbol_table;
  llvm::ScopedHashTable<StringRef, mlir::Type> type_table;
  llvm::ScopedHashTable<StringRef, StructDecl *> struct_table;
  llvm::ScopedHashTable<mlir::lang::StructType, StringRef> struct_name_table;
  llvm::StringMap<mlir::lang::FuncOp> function_map;

  AstDumper dumper;

  mlir::Location loc(const TokenSpan &loc) {
    return mlir::FileLineColLoc::get(
        builder.getContext(),
        compiler_context.source_mgr.getBufferInfo(loc.file_id)
            .Buffer->getBufferIdentifier(),
        loc.line_no, loc.col_start);
  }

  std::string mangle(llvm::StringRef base, llvm::ArrayRef<mlir::Type> types) {
    std::string mangled_name = base.str();
    llvm::raw_string_ostream rso(mangled_name);
    if (!types.empty()) {
      rso << "_";
    }
    for (auto &param : types) {
      rso << "_";
      rso << param;
    }
    return rso.str();
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

    // if function return type is a StructType, then add it as a function
    // parameter and remove the return type
    bool has_struct_return_type =
        return_types.size() == 1 &&
        mlir::isa<mlir::lang::StructType>(return_types[0]);
    if (has_struct_return_type) {
      param_types->push_back(
          mlir::lang::PointerType::get(builder.getContext()));
      return_types = mlir::TypeRange();
    }
    auto func_name = mangle(func->decl->name, param_types.value());
    bool is_inline = func->attrs.count(Attribute::Inline);
    mlir::NamedAttrList attrs;
    if (is_inline) {
      auto inline_attr =
          builder.getNamedAttr("force_inline", builder.getBoolAttr(is_inline));
      attrs.push_back(inline_attr);
    }
    auto func_type = builder.getFunctionType(param_types.value(), return_types);
    auto func_op = builder.create<mlir::lang::FuncOp>(
        loc(func->token.span), func_name, func_type, attrs);
    function_map[func_name] = func_op;
    current_function = &func_op;

    auto entry_block = func_op.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);

    // Declare function parameters.
    if (failed(declareParameters(func->decl->parameters,
                                 entry_block->getArguments()))) {
      func_op.erase();
      return emitError(loc(func->token.span), "parameter declaration error");
    }

    // Generate function body.
    if (func->decl->extra.is_method && func->decl->name == "init") {
      auto create_self = [this, &func]() {
        // Create a new struct instance
        auto struct_type = type_table.lookup("Self");
        if (!struct_type) {
          emitError(loc(func->token.span), "Self not found");
          return;
        }
        auto struct_val = builder.create<mlir::lang::UndefOp>(
            loc(func->token.span), struct_type);
        if (failed(declare("self", struct_val))) {
          emitError(loc(func->token.span), "redeclaration of self");
        }
      };
      if (failed(langGen(func->body.get(), create_self))) {
        func_op.erase();
        return mlir::failure();
      }
    } else {

      if (failed(langGen(func->body.get()))) {
        func_op.erase();
        return mlir::failure();
      }
    }

    if (func_op.getBody().back().getOperations().empty() ||
        !mlir::isa<mlir::lang::ReturnOp>(
            func_op.getBody().back().getOperations().back())) {
      builder.setInsertionPointToEnd(&func_op.getBody().back());
      builder.create<mlir::lang::ReturnOp>(loc(func->token.span));
    }

    if (has_struct_return_type) {
      // remove the return operand and store it in the last argument
      auto return_val = func_op.getBody().back().getTerminator()->getOperand(0);
      auto struct_val = func_op.getBody().back().getArgument(
          func_op.getBody().back().getNumArguments() - 1);
      auto return_op = func_op.getBody().back().getTerminator();
      builder.create<mlir::lang::AssignOp>(
          loc(func->token.span), struct_val.getType(), struct_val, return_val);
      return_op->erase();
      builder.create<mlir::lang::ReturnOp>(loc(func->token.span));
    }

    current_function = nullptr;
    return mlir::success();
  }

  llvm::LogicalResult
  declareParameters(std::vector<std::unique_ptr<Parameter>> &params,
                    mlir::ArrayRef<mlir::BlockArgument> args) {

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
    llvm::SmallVector<mlir::Type, 4> arg_types;
    for (auto &param : params) {
      auto type = getType(param->type.get());
      if (failed(type)) {
        return mlir::failure();
      }
      arg_types.push_back(type.value());
    }
    return arg_types;
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
    // auto return_type = current_function->getFunctionType().getResult(0);
    // if (value.value().getType() != return_type) {
    //   // insert cast
    //   value = builder
    //               .create<mlir::UnrealizedConversionCastOp>(loc, return_type,
    //                                                         value.value())
    //               .getResult(0);
    // }

    builder.create<mlir::lang::ReturnOp>(loc, value.value());
    return value;
  }

  llvm::LogicalResult langGen(TupleStructDecl *decl) {
    auto span = this->loc(decl->token.span);
    llvm::SmallVector<mlir::Type, 4> field_types;
    for (auto &field : decl->fields) {
      auto type = getType(field.get());
      if (failed(type)) {
        emitError(span, "unsupported field type");
        return mlir::failure();
      }
      field_types.push_back(type.value());
    }
    auto struct_type = mlir::lang::StructType::get(field_types, decl->name);
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
    NEW_SCOPE()
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto parent_type = type_table.lookup(impl_decl->type);
    if (!parent_type) {
      // check if its an mlir type
      parent_type = mlir::parseType(impl_decl->type, builder.getContext());
      if (!parent_type) {
        return mlir::emitError(loc(impl_decl->token.span),
                               "parent type not found");
      }
      type_table.insert(impl_decl->type, parent_type);
    }
    // auto struct_type = mlir::dyn_cast<mlir::lang::StructType>(parent_type);
    if (failed(declare("Self", parent_type))) {
      return mlir::emitError(loc(impl_decl->token.span),
                             "redeclaration of Self");
    }
    builder.create<mlir::lang::ImplDeclOp>(loc(impl_decl->token.span),
                                           parent_type);
    for (auto &method : impl_decl->functions) {
      auto func = langGen(method.get());
      if (failed(func)) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult langGen(BlockExpression *block,
                              std::function<void()> callback = nullptr) {
    NEW_SCOPE()
    if (callback) {
      callback();
    }
    // iterate till the second last statement
    for (int i = 0; i < (int)block->statements.size() - 1; i++) {
      if (failed(langGen(block->statements[i].get()))) {
        return mlir::failure();
      }
    }
    // check the last statement is an expression statement or not
    if (!block->statements.empty()) {
      if (auto exprStmt =
              dynamic_cast<ExprStmt *>(block->statements.back().get())) {
        return langGen(exprStmt->expr.get());
      }
      if (failed(langGen(block->statements.back().get()))) {
        return mlir::failure();
      }
    }
    // block does not return anything, so return a void value
    return mlir::success(mlir::Value());
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

    auto op = builder.create<mlir::lang::VarDeclOp>(
        loc(var_decl->token.span),
        var_type ? mlir::TypeAttr::get(var_type) : nullptr, var_name,
        init_value.value(), var_decl->is_mut, var_decl->is_pub);
    if (failed(declare(var_name, op.getResult()))) {
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
        mlir::SymbolTable symbol_table(current_function->getOperation());
        op = symbol_table.lookup(type_name);
      }
      if (!op) {
        // search in struct table
        auto type = type_table.lookup(type_name);
        if (!type) {
          return mlir::emitError(loc(identifier_type->token.span),
                                 "unknown type");
        }
        return type;
      }
      return op->getResult(0).getType();
    } else if (type->kind() == AstNodeKind::SliceType) {
      auto slice_type = static_cast<SliceType *>(type);
      auto base_type = getType(slice_type->base.get());
      if (failed(base_type)) {
        return mlir::failure();
      }
      return mlir::lang::SliceType::get(base_type.value());
    } else if (type->kind() == AstNodeKind::ArrayType) {
      // auto array_type = static_cast<ArrayType *>(type);
      // auto base_type = getType(array_type->base.get());
      // if (failed(base_type)) {
      //   return mlir::failure();
      // }
    }
    return mlir::emitError(loc(type->token.span),
                           "unsupported type " + toString(type->kind()));
  }

  mlir::FailureOr<mlir::Value> langGen(Expression *expr,
                                       bool is_comptime = false) {
    mlir::FailureOr<mlir::Value> result;
    if (auto e = dynamic_cast<LiteralExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<MLIRAttribute *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<MLIRType *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<ReturnExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<BlockExpression *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<VarDecl *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<CallExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<FieldAccessExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<IdentifierExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<AssignExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<MLIROp *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<BinaryExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<IfExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<YieldExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<TupleExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<UnaryExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<IndexExpr *>(expr)) {
      result = langGen(e);
    } else if (auto e = dynamic_cast<ComptimeExpr *>(expr)) {
      result = langGen(e);
    } else
      return mlir::emitError(loc(expr->token.span), "unsupported expression " +
                                                        toString(expr->kind()));
    if (is_comptime) {
      result->getDefiningOp()->setAttr("comptime", builder.getBoolAttr(true));
    }
    return result;
  }

  mlir::FailureOr<mlir::Value> langGen(ComptimeExpr *expr) {
    // Wrap the expression in a comptime op
    auto value = langGen(expr->expr.get(), true);
    if (failed(value)) {
      return mlir::failure();
    }
    return value;
    // auto value_op = value->getDefiningOp();
    // auto op = builder.create<mlir::lang::ComptimeOp>(
    //     loc(expr->token.span), value->getType(),
    //     value->getDefiningOp()->getOperands());
    // auto &block = op.getBody().emplaceBlock();
    // auto locations = llvm::to_vector<4>(
    //     llvm::map_range(op.getOperands(), [](mlir::Value value) {
    //       return value.getDefiningOp()->getLoc();
    //     }));
    // block.addArguments(op.getOperandTypes(), locations);
    // value_op->moveBefore(&block, block.end());
    // // replace operands with block arguments
    // for (int i = 0; i < (int)op.getNumOperands(); i++) {
    //   value_op->setOperand(i, block.getArgument(i));
    // }
    //
    // // create a yield op that returns the value
    // mlir::OpBuilder::InsertionGuard guard(builder);
    // builder.setInsertionPointToEnd(&block);
    // builder.create<mlir::lang::YieldOp>(loc(expr->token.span),
    // value->getType(),
    //                                     *value);
    // return op.getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(IndexExpr *expr) {
    auto base = langGen(expr->base.get());
    if (failed(base)) {
      return mlir::failure();
    }
    auto index = langGen(expr->index.get());
    if (failed(index)) {
      return mlir::failure();
    }
    auto index_type = index.value().getType();
    auto base_type = base.value().getType();
    if (!mlir::isa<mlir::lang::SliceType, mlir::lang::ArrayType>(base_type)) {
      return mlir::emitError(loc(expr->token.span), "base is not a slice type");
    }
    if (index_type != builder.getIntegerType(64)) {
      return mlir::emitError(loc(expr->token.span), "index is not i64");
    }
    auto index_op = builder.create<mlir::lang::IndexAccessOp>(
        loc(expr->token.span), base_type, base.value(), index.value());
    return index_op.getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(UnaryExpr *expr) {
    auto operand = langGen(expr->operand.get());
    if (failed(operand)) {
      return mlir::failure();
    }
    std::string fn_name = "";
    llvm::raw_string_ostream stream(fn_name);
    stream << (expr->op == Operator::Sub ? "neg__" : "logical_not__")
           << operand.value().getType();
    fn_name = stream.str();

    if (!function_map.count(fn_name)) {
      return mlir::emitError(loc(expr->token.span),
                             "function " + fn_name + " not found");
    }
    auto func = function_map[fn_name];
    auto call_op = builder.create<mlir::lang::CallOp>(
        loc(expr->token.span), func, mlir::ValueRange{operand.value()});
    return call_op.getResult(0);
  }

  mlir::FailureOr<mlir::Value> langGen(TupleExpr *expr) {
    llvm::SmallVector<mlir::Value, 4> values;
    llvm::SmallVector<mlir::Type, 4> types;
    for (auto &e : expr->elements) {
      auto value = langGen(e.get());
      if (failed(value)) {
        return mlir::failure();
      }
      values.push_back(value.value());
      types.push_back(value.value().getType());
    }
    auto struct_name = mangle("anonymous_struct", types);
    auto struct_type = mlir::lang::StructType::get(types, struct_name);
    return builder
        .create<mlir::lang::TupleOp>(loc(expr->token.span), struct_type, values)
        .getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(YieldExpr *expr) {
    auto value = langGen(expr->value.get());
    if (failed(value)) {
      return mlir::failure();
    }
    return builder
        .create<mlir::lang::YieldOp>(loc(expr->token.span), value.value())
        .getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(IfExpr *expr) {
    auto cond = langGen(expr->condition.get());
    if (failed(cond))
      return mlir::failure();
    bool terminated_by_return = false;
    bool terminated_by_yield = false;
    bool non_terminating = false;
    bool can_use_scf_if_op = canUseSCFIfOp(
        expr, terminated_by_return, terminated_by_yield, non_terminating);
    auto loc = this->loc(expr->token.span);
    auto then_builder = [&](mlir::OpBuilder builder, mlir::Location loc) {
      if (failed(langGen(expr->then_block.get()))) {
        llvm::errs() << "failed then block\n";
      }
      if (isNonTerminatingBlock(expr->then_block.get())) {
        auto last_stmt = expr->then_block->statements.back().get();
        if (auto expr_stmt = dynamic_cast<ExprStmt *>(last_stmt)) {
          if (expr_stmt->expr->kind() == AstNodeKind::IfExpr) {
            auto nested_if_result = builder.getBlock()->back().getResult(0);
            builder.create<mlir::lang::YieldOp>(loc, nested_if_result);
          } else {
            builder.create<mlir::lang::YieldOp>(loc);
          }
        } else {
          builder.create<mlir::lang::YieldOp>(loc);
        }
      }
    };
    auto else_builder = [&](mlir::OpBuilder builder, mlir::Location loc) {
      if (failed(langGen(expr->else_block.value().get())))
        llvm::errs() << "failed else block\n";
      if (isNonTerminatingBlock(expr->else_block.value().get())) {
        auto last_stmt = expr->else_block.value()->statements.back().get();
        if (auto expr_stmt = dynamic_cast<ExprStmt *>(last_stmt)) {
          if (expr_stmt->expr->kind() == AstNodeKind::IfExpr) {
            auto nested_if_result = builder.getBlock()->back().getResult(0);
            builder.create<mlir::lang::YieldOp>(loc, nested_if_result);
          } else {
            builder.create<mlir::lang::YieldOp>(loc);
          }
        } else {
          builder.create<mlir::lang::YieldOp>(loc);
        }
      }
    };
    if (expr->else_block.has_value()) {
      auto if_op = builder.create<mlir::lang::IfOp>(loc, cond.value(),
                                                    then_builder, else_builder);
      if_op->setAttr("terminated_by_return",
                     builder.getBoolAttr(terminated_by_return));
      if_op->setAttr("terminated_by_yield",
                     builder.getBoolAttr(terminated_by_yield));
      if_op->setAttr("can_use_scf_if_op",
                     builder.getBoolAttr(can_use_scf_if_op));
      if_op->setAttr("non_terminating", builder.getBoolAttr(non_terminating));
      return if_op.getResult();
    }
    auto if_op =
        builder.create<mlir::lang::IfOp>(loc, cond.value(), then_builder);
    if_op->setAttr("terminated_by_return",
                   builder.getBoolAttr(terminated_by_return));
    if_op->setAttr("terminated_by_yield",
                   builder.getBoolAttr(terminated_by_yield));
    if_op->setAttr("can_use_scf_if_op", builder.getBoolAttr(can_use_scf_if_op));
    if_op->setAttr("non_terminating", builder.getBoolAttr(non_terminating));
    return if_op.getResult();
  }

  bool canUseSCFIfOp(IfExpr *expr, bool &terminated_by_return,
                     bool &terminated_by_yield, bool &non_terminating) {
    bool has_else = expr->else_block.has_value();
    // 1. Has no else, but then is a non-terminating block
    if (!has_else) {
      non_terminating = isNonTerminatingBlock(expr->then_block.get());
      return non_terminating;
    }
    // 3. Both then and else are non-terminating blocks (no return or yield)
    non_terminating = isNonTerminatingBlock(expr->then_block.get()) &&
                      isNonTerminatingBlock(expr->else_block.value().get());
    // 4. Both then and else are terminated by a return in all paths
    terminated_by_return = isTerminatedByReturn(expr->then_block.get()) &&
                           isTerminatedByReturn(expr->else_block.value().get());
    // 5. Both then and else are terminated by a yield in all paths
    // (different from above)
    terminated_by_yield = isTerminatedByYield(expr->then_block.get()) &&
                          isTerminatedByYield(expr->else_block.value().get());
    return non_terminating || terminated_by_return || terminated_by_yield;
  }

  bool isTerminatedByYield(BlockExpression *block) {
    // Check the block is terminated by a yield in all paths
    if (block->statements.empty()) {
      return false;
    }
    auto last_stmt = block->statements.back().get();
    if (last_stmt->kind() == AstNodeKind::ExprStmt) {
      auto expr = dynamic_cast<ExprStmt *>(last_stmt);
      if (expr->expr->kind() == AstNodeKind::YieldExpr) {
        return true;
      }
    } else if (auto if_expr = dynamic_cast<IfExpr *>(last_stmt)) {
      if (!if_expr->else_block.has_value()) {
        return false;
      }
      // TODO:
      return isTerminatedByYield(if_expr->then_block.get()) &&
             isTerminatedByYield(if_expr->else_block.value().get());
    } else if (auto block_expr = dynamic_cast<BlockExpression *>(last_stmt)) {
      return isTerminatedByYield(block_expr);
    }
    return false;
  }

  bool isTerminatedByReturn(BlockExpression *block) {
    // check if the block is terminated by a return in all paths
    if (block->statements.empty()) {
      return false;
    }
    for (auto &stmt : block->statements) {
      if (stmt->kind() == AstNodeKind::ExprStmt) {
        auto expr = dynamic_cast<ExprStmt *>(stmt.get());
        if (expr->expr->kind() == AstNodeKind::ReturnExpr) {
          return true;
        }
      } else if (auto if_expr = dynamic_cast<IfExpr *>(stmt.get())) {
        // check if either branches are terminated by a return
        if (isTerminatedByReturn(if_expr->then_block.get())) {
          return true;
        }
        if (isTerminatedByReturn(if_expr->else_block.value().get())) {
          return true;
        }
      } else if (auto block_expr =
                     dynamic_cast<BlockExpression *>(stmt.get())) {
        if (isTerminatedByReturn(block_expr)) {
          return true;
        }
      }
    }
    return false;
  }

  // recursively check if the block is non-terminating (i.e the last statement
  // in a block is not a return or yield)
  bool isNonTerminatingBlock(BlockExpression *block) {
    if (block->statements.empty()) {
      return true;
    }
    for (auto &stmt : block->statements) {
      if (stmt->kind() == AstNodeKind::ExprStmt) {
        auto expr = dynamic_cast<ExprStmt *>(stmt.get());
        if (expr->expr->kind() == AstNodeKind::ReturnExpr ||
            expr->expr->kind() == AstNodeKind::YieldExpr) {
          return false;
        }
      } else if (auto expr = dynamic_cast<IfExpr *>(stmt.get())) {
        // check if both branches are non-terminating
        // then_block
        if (!isNonTerminatingBlock(expr->then_block.get()))
          return false;
        // else_block
        if (expr->else_block.has_value())
          if (expr->else_block.value())
            if (!isNonTerminatingBlock(expr->else_block.value().get()))
              return false;
      } else if (auto block = dynamic_cast<BlockExpression *>(stmt.get())) {
        if (!isNonTerminatingBlock(block)) {
          return false;
        }
      }
    }
    return true;
  }

  mlir::FailureOr<mlir::Value> langGen(BinaryExpr *expr) {
    auto lhs = langGen(expr->lhs.get());
    if (failed(lhs)) {
      return mlir::failure();
    }
    auto rhs = langGen(expr->rhs.get());
    if (failed(rhs)) {
      return mlir::failure();
    }
    static std::array<llvm::StringRef, 22> op_fns = {
        "add",          "sub",           "mul",         "div",
        "mod",          "logical_and",   "logical_or",  "logical_not",
        "equal",        "not_equal",     "less_than",   "less_equal",
        "greater_than", "greater_equal", "bitwise_and", "bitwise_or",
        "bitwise_xor",  "bitwise_shl",   "bitwise_shr", "bitwise_not",
        "pow",          "invalid",
    };
    std::string fn_name = "";
    llvm::raw_string_ostream stream(fn_name);
    stream << op_fns[static_cast<int>(expr->op)] << "__"
           << lhs.value().getType() << "_" << rhs.value().getType();
    fn_name = stream.str();

    if (!function_map.count(fn_name)) {
      return mlir::emitError(loc(expr->token.span),
                             "function " + fn_name + " not found");
    }
    auto func = function_map[fn_name];

    auto call_op = builder.create<mlir::lang::CallOp>(
        loc(expr->token.span), func,
        mlir::ValueRange{lhs.value(), rhs.value()});
    return call_op.getResult(0);
  }

  mlir::FailureOr<mlir::Value> langGen(AssignExpr *expr) {
    mlir::FailureOr<mlir::Value> lhs = mlir::failure();
    if (expr->lhs->kind() == AstNodeKind::IdentifierExpr) {
      lhs = langGen(dynamic_cast<IdentifierExpr *>(expr->lhs.get()), false);
    } else {
      lhs = langGen(expr->lhs.get());
    }
    if (failed(lhs)) {
      return mlir::failure();
    }
    auto rhs = langGen(expr->rhs.get());
    if (failed(rhs)) {
      return mlir::failure();
    }
    auto lhs_type = lhs.value().getType();
    return builder
        .create<mlir::lang::AssignOp>(loc(expr->token.span), lhs_type,
                                      lhs.value(), rhs.value())
        .getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(IdentifierExpr *expr,
                                       bool get_value = true) {
    auto var = symbol_table.lookup(expr->name);
    if (!var) {
      return mlir::emitError(loc(expr->token.span),
                             "undeclared variable " + expr->name);
    }
    // for mutable values and structs, by default return the value (automatic
    // deref)
    if (get_value && mlir::isa<mlir::MemRefType>(var.getType())) {
      auto deref_op =
          builder.create<mlir::lang::DerefOp>(loc(expr->token.span), var);
      return deref_op.getResult();
    }
    return var;
  }

  mlir::FailureOr<mlir::Value> langGen(CallExpr *expr) {
    // lookup callee in struct table
    auto struct_type = type_table.lookup(expr->callee);
    if (struct_type && mlir::isa<mlir::lang::StructType>(struct_type))
      return createStructInstance(
          mlir::cast<mlir::lang::StructType>(struct_type), expr->arguments);

    // get arguments
    llvm::SmallVector<mlir::Value, 4> args;
    llvm::SmallVector<mlir::Type, 4> arg_types;
    for (auto &arg : expr->arguments) {
      auto arg_value = langGen(arg.get());
      if (failed(arg_value)) {
        return mlir::failure();
      }
      args.push_back(arg_value.value());
      arg_types.push_back(arg_value.value().getType());
    }
    auto func_name = mangle(expr->callee, arg_types);
    // if return type is a struct type, then create a struct instance
    // and pass it as an argument
    auto func_op = function_map[func_name];
    if (func_op) {
      auto func_type = func_op.getFunctionType();
      mlir::Type return_type = nullptr;
      if (func_type && func_type.getNumResults() == 1) {
        return_type = func_type.getResult(0);
      }
      if (return_type && mlir::isa<mlir::lang::StructType>(return_type)) {
        auto struct_type = mlir::cast<mlir::lang::StructType>(return_type);
        auto struct_instance = builder.create<mlir::lang::UndefOp>(
            loc(expr->token.span), struct_type);
        args.push_back(struct_instance.getResult());
      }
    }

    if (expr->callee == "print") {
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
      format_str = format_str.substr(1, format_str.size() - 2) + '\n' + '\0';
      args.front().getDefiningOp()->erase();
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

    if (!mlir::isa<mlir::lang::StructType, mlir::lang::SliceType>(base_type)) {
      return mlir::emitError(loc(expr->token.span),
                             "field access on non-struct type");
    }

    if (std::holds_alternative<std::unique_ptr<CallExpr>>(expr->field)) {
      // function/method call
      auto call_expr = std::get<std::unique_ptr<CallExpr>>(expr->field).get();
      auto struct_type = mlir::cast<mlir::lang::StructType>(base_type);
      auto struct_name = struct_name_table.lookup(struct_type);
      auto struct_decl = struct_table.lookup(struct_name);
      if (!struct_decl) {
        return mlir::emitError(loc(expr->token.span),
                               "struct not found in struct table");
      }
    }
    auto struct_type = mlir::cast<mlir::lang::StructType>(base_type);
    if (std::holds_alternative<std::unique_ptr<LiteralExpr>>(expr->field) &&
        std::get<std::unique_ptr<LiteralExpr>>(expr->field)->type ==
            LiteralExpr::LiteralType::Int) {
      auto field_index = std::get<int>(
          std::get<std::unique_ptr<LiteralExpr>>(expr->field)->value);
      return builder
          .create<mlir::lang::StructAccessOp>(loc(expr->token.span),
                                              base_value.value(), field_index)
          .getResult();
    }
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
  createStructInstance(mlir::lang::StructType struct_type,
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

    auto typed_attr = mlir::cast<mlir::TypedAttr>(mlir_attr);
    auto constant_op = builder.create<mlir::lang::ConstantOp>(
        loc(attr->token.span), typed_attr.getType(), typed_attr);

    return constant_op.getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(MLIROp *op) {
    llvm::StringRef op_name = op->op;
    op_name = op_name.trim('"');
    llvm::SmallVector<mlir::Value, 4> operands;
    llvm::SmallVector<mlir::Type, 4> operand_types;
    llvm::SmallVector<mlir::NamedAttribute, 4> named_attributes;
    llvm::SmallVector<mlir::Type, 4> result_types;

    for (auto &operand : op->operands) {
      auto operand_value = langGen(operand.get());
      if (failed(operand_value)) {
        return mlir::failure();
      }
      operands.push_back(operand_value.value());
      operand_types.push_back(operand_value.value().getType());
    }

    for (auto &attr : op->attributes) {
      llvm::StringRef name = attr.first;
      llvm::StringRef attribute = attr.second;
      auto mlir_attr =
          mlir::parseAttribute(attribute.trim('"'), builder.getContext());
      if (!mlir_attr) {
        return mlir::emitError(loc(op->token.span),
                               "failed to parse attribute");
      }
      auto named_attr =
          mlir::NamedAttribute(builder.getStringAttr(name), mlir_attr);
      named_attributes.push_back(named_attr);
    }

    for (auto &result : op->result_types) {
      llvm::StringRef type = result;
      auto result_type = mlir::parseType(type.trim('"'), builder.getContext());
      result_types.push_back(result_type);
    }
    mlir::OperationState state(loc(op->token.span), op_name);
    state.addOperands(operands);
    state.addTypes(result_types);
    state.addAttributes(named_attributes);
    auto mlir_op = mlir::Operation::create(state);
    builder.insert(mlir_op);
    return mlir_op->getResult(0);
  }

  // mlir::FailureOr<mlir::Value> langGen(MLIROp *op) {
  //   llvm::StringRef op_name = op->op;
  //   op_name = op_name.trim('"');
  //   llvm::SmallVector<mlir::Value, 4> operands;
  //   llvm::SmallVector<mlir::Type, 4> operand_types;
  //
  //   for (auto &operand : op->operands) {
  //     auto operand_value = langGen(operand.get());
  //     if (failed(operand_value)) {
  //       return mlir::failure();
  //     }
  //     operands.push_back(operand_value.value());
  //   }
  //
  //   return builder
  //       .create<mlir::lang::MlirOperationOp>(
  //           loc(op->token.span), mlir::TypeRange(builder.getNoneType()),
  //           mlir::ValueRange(operands), op_name)
  //       .getResults()[0];
  // }

  mlir::FailureOr<mlir::Value> langGen(LiteralExpr *literal) {
    if (literal->type == LiteralExpr::LiteralType::Int) {
      // Create an IntegerLiteral struct instance
      // auto struct_type = type_table.lookup("IntLiteral");
      mlir::TypedAttr attr = builder.getIntegerAttr(
          builder.getIntegerType(64), std::get<int>(literal->value));
      auto field_value = builder
                             .create<mlir::lang::ConstantOp>(
                                 loc(literal->token.span), attr.getType(), attr)
                             .getResult();
      return field_value;
      // return builder
      //     .create<mlir::lang::CreateStructOp>(loc(literal->token.span),
      //                                         struct_type,
      //                                         mlir::ValueRange{field_value})
      //     .getResult();
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
                                          Program *program, Context &ctx) {
  return LangGenImpl(context, ctx).langGen(program);
}
