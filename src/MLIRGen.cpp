#include "MLIRGen.h"
#include "ast.hpp"
#include "lexer.hpp"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <functional>
#include <string>
#include <variant>

using llvm::ArrayRef;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;

#define NEW_SCOPE()                                                            \
  ScopedHashTableScope<StringRef, mlir::Value> varScope(symbol_table);         \
  ScopedHashTableScope<StringRef, mlir::Type> typeScope(type_table);           \
  ScopedHashTableScope<StringRef, StructDecl *> structScope(struct_table);

class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp mlirGen(Program *program) {
    NEW_SCOPE()
    if (failed(builtins())) {
      emitError(loc(program->token.span), "error generating builtins");
      return nullptr;
    }
    the_module = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto &f : program->items) {
      if (dynamic_cast<Function *>(f.get())) {
        auto func = mlirGen(dynamic_cast<Function *>(f.get()));
        if (failed(func)) {
          return nullptr;
        }
      } else if (dynamic_cast<TupleStructDecl *>(f.get())) {
        auto tuple_struct = mlirGen(dynamic_cast<TupleStructDecl *>(f.get()));
        if (failed(tuple_struct)) {
          return nullptr;
        }
      } else if (dynamic_cast<StructDecl *>(f.get())) {
        auto struct_decl = mlirGen(dynamic_cast<StructDecl *>(f.get()));
        if (failed(struct_decl)) {
          return nullptr;
        }
      } else if (dynamic_cast<ImplDecl *>(f.get())) {
        auto impl_decl = mlirGen(dynamic_cast<ImplDecl *>(f.get()));
        if (failed(impl_decl)) {
          return nullptr;
        }
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
  llvm::StringMap<mlir::func::FuncOp> function_map;
  bool control_flow = false;

  struct ControlFlow {
    bool &control_flow;
    ControlFlow(bool &control_flow) : control_flow(control_flow) {
      control_flow = true;
    }
    ~ControlFlow() { control_flow = false; }
  };

  mlir::Location loc(const TokenSpan &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr("temp.lang"),
                                     loc.line_no, loc.col_start);
  }

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

  llvm::LogicalResult builtins() {
    // Generate a struct for the slice type with a pointer and length
    auto slice_type = mlir::LLVM::LLVMStructType::getIdentified(
        builder.getContext(), "Slice");
    if (failed(slice_type.setBody(
            {mlir::LLVM::LLVMPointerType::get(builder.getContext()),
             builder.getI64Type()},
            /*isPacked=*/false))) {
      return mlir::failure();
    }
    if (failed(declare("Slice", slice_type))) {
      return mlir::failure();
    }
    return mlir::success();
  }

  std::string mangle_function_name(const FunctionDecl &decl) {
    std::string mangled_name = "";
    if (decl.extra.is_method) {
      mangled_name = mangled_name + decl.extra.parent_name.value();
      if (decl.name != "init") {
        mangled_name = mangled_name + "_" + decl.name;
      }
    } else {
      mangled_name = decl.name;
    }
    return mangled_name;
  }

  std::string mangle_function_name(const CallExpr &expr,
                                   StringRef basename = "") {
    std::string mangled_name =
        basename.empty() ? expr.callee : basename.str() + "_" + expr.callee;
    return mangled_name;
  }

  llvm::FailureOr<mlir::func::FuncOp> mlirGen(Function *func) {
    NEW_SCOPE()

    builder.setInsertionPointToEnd(the_module.getBody());

    auto param_types = mlirGen(func->decl->parameters);
    mlir::TypeRange return_types =
        func->decl->return_type->kind() == AstNodeKind::PrimitiveType &&
                static_cast<PrimitiveType *>(func->decl->return_type.get())
                        ->type_kind == PrimitiveType::PrimitiveTypeKind::Void
            ? mlir::TypeRange()
            : mlir::TypeRange(mlirGen(func->decl->return_type.get()).value());

    auto mangled_name = mangle_function_name(*func->decl);
    auto func_type = builder.getFunctionType(param_types.value(), return_types);
    auto func_op = builder.create<mlir::func::FuncOp>(loc(func->token.span),
                                                      mangled_name, func_type);

    function_map[mangled_name] = func_op;

    auto entry_block = func_op.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);

    // declare function parameters
    if (failed(declare_parameters(func->decl->parameters,
                                  entry_block->getArguments()))) {
      emitError(loc(func->token.span), "parameter declaration error");
      func_op.erase();
      return mlir::failure();
    }

    if (func->decl->extra.is_method && func->decl->name == "init") {
      auto create_self = [this, &func]() {
        // Create a new struct instance
        auto struct_type = type_table.lookup("Self");
        auto struct_ptr = builder.create<mlir::LLVM::AllocaOp>(
            loc(func->token.span),
            mlir::LLVM::LLVMPointerType::get(builder.getContext()), struct_type,
            builder.create<mlir::arith::ConstantOp>(
                loc(func->token.span), builder.getI32Type(),
                builder.getI32IntegerAttr(1)));
        auto struct_val = builder.create<mlir::LLVM::LoadOp>(
            loc(func->token.span), struct_type, struct_ptr,
            /*isVolatile=*/false);
        if (failed(declare("self", struct_val))) {
          emitError(loc(func->token.span), "redeclaration of self");
        }
      };
      if (failed(mlirGen(func->body.get(), create_self))) {
        func_op.erase();
        return mlir::failure();
      }
    } else {

      if (failed(mlirGen(func->body.get()))) {
        func_op.erase();
        return mlir::failure();
      }
    }

    // Ensure that `func.return` is the last operation in the function body
    if (func_op.getBody().back().getOperations().empty() ||
        !mlir::isa<mlir::func::ReturnOp>(
            func_op.getBody().back().getOperations().back())) {
      builder.setInsertionPointToEnd(&func_op.getBody().back());
      builder.create<mlir::func::ReturnOp>(loc(func->token.span));
    }

    return func_op;
  }

  llvm::LogicalResult
  declare_parameters(std::vector<std::unique_ptr<Parameter>> &params,
                     ArrayRef<mlir::BlockArgument> args) {
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
  mlirGen(std::vector<std::unique_ptr<Parameter>> &params) {
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (auto &param : params) {
      auto loc = this->loc(param->token.span);
      auto type = mlirGen(param->type.get());
      if (failed(type)) {
        emitError(loc, "unsupported parameter type");
        return {};
      }
      argTypes.push_back(type.value());
    }
    return argTypes;
  }

  llvm::LogicalResult mlirGen(TupleStructDecl *decl) {
    auto span = this->loc(decl->token.span);
    llvm::SmallVector<mlir::Type, 4> fieldTypes;
    for (auto &field : decl->fields) {
      auto type = mlirGen(field.get());
      if (failed(type)) {
        emitError(span, "unsupported field type");
        return mlir::failure();
      }
      fieldTypes.push_back(type.value());
    }
    // auto struct_type = builder.getTupleType(fieldTypes);
    auto struct_type = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), fieldTypes);
    // declare struct type
    if (failed(declare(decl->name, struct_type))) {
      emitError(span, "redeclaration of struct type");
      return mlir::failure();
    }

    // Register a default contructor from the struct type
    if (failed(defineDefaultConstructor(fieldTypes, struct_type, decl, span))) {
      return mlir::failure();
    }
    return mlir::success();
  }

  llvm::LogicalResult mlirGen(StructDecl *decl) {
    auto span = this->loc(decl->token.span);
    llvm::SmallVector<mlir::Type, 4> fieldTypes;
    for (auto &field : decl->fields) {
      auto type = mlirGen(field.get()->type.get());
      if (failed(type)) {
        return mlir::failure();
      }
      fieldTypes.push_back(type.value());
    }
    auto struct_type = mlir::LLVM::LLVMStructType::getIdentified(
        builder.getContext(), decl->name);
    if (failed(struct_type.setBody(fieldTypes, false))) {
      emitError(span, "error in struct type definition");
      return mlir::failure();
    }

    // declare struct type
    struct_table.insert(decl->name, decl);
    if (failed(declare(decl->name, struct_type))) {
      emitError(span, "redeclaration of struct type");
      return mlir::failure();
    }
    return mlir::success();
  }

  llvm::LogicalResult mlirGen(ImplDecl *decl) {
    NEW_SCOPE()
    auto span = this->loc(decl->token.span);
    auto struct_type = type_table.lookup(decl->type);
    if (!struct_type) {
      emitError(span, "struct type not found");
      return mlir::failure();
    }
    if (failed(declare("Self", struct_type))) {
      emitError(span, "redeclaration of Self");
      return mlir::failure();
    }

    for (auto &method : decl->functions) {
      auto func = mlirGen(method.get());
      if (failed(func)) {
        return mlir::failure();
      }
    }
    // TODO: Traits
    return mlir::success();
  }

  llvm::LogicalResult
  defineDefaultConstructor(llvm::SmallVector<mlir::Type, 4> fieldTypes,
                           mlir::Type struct_type, TupleStructDecl *decl,
                           mlir::Location loc) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(the_module.getBody());

    auto constructor_type = builder.getFunctionType(fieldTypes, {struct_type});
    auto constructor_op =
        builder.create<mlir::func::FuncOp>(loc, decl->name, constructor_type);
    auto entry_block = constructor_op.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);
    auto struct_ptr = builder.create<mlir::LLVM::AllocaOp>(
        loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        struct_type,
        builder.create<mlir::arith::ConstantOp>(loc, builder.getI32Type(),
                                                builder.getI32IntegerAttr(1)));
    for (int i = 0; i < (int)fieldTypes.size(); i++) {
      mlir::Value index_val = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI32IntegerAttr(i));
      mlir::ValueRange indices = {
          builder.create<mlir::arith::ConstantOp>(loc, builder.getI32Type(),
                                                  builder.getI32IntegerAttr(0)),
          index_val};
      mlir::Value gep = builder.create<mlir::LLVM::GEPOp>(
          loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
          struct_type, struct_ptr, indices);
      builder.create<mlir::LLVM::StoreOp>(
          loc,
          builder.create<mlir::arith::ConstantOp>(
              loc, fieldTypes[i], builder.getZeroAttr(fieldTypes[i])),
          gep);
    }

    // load the struct pointer
    mlir::Value struct_val = builder.create<mlir::LLVM::LoadOp>(
        loc, struct_type, struct_ptr, /*isVolatile=*/false);

    mlir::ValueRange struct_ptr_val = {struct_val};
    builder.create<mlir::func::ReturnOp>(loc, struct_ptr_val);

    // save the constructor function
    function_map[decl->name] = constructor_op;

    return mlir::success();
  }

  llvm::LogicalResult mlirGen(BlockExpression *block,
                              std::function<void()> callback = nullptr) {
    NEW_SCOPE()
    if (callback)
      callback();

    for (auto &stmt : block->statements) {
      if (mlir::failed(mlirGen(stmt.get()))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  llvm::LogicalResult mlirGen(Statement *stmt) {
    if (auto expr = dynamic_cast<VarDecl *>(stmt)) {
      if (failed(mlirGen(expr))) {
        return mlir::failure();
      }
      return mlir::success();
    } else if (auto expr = dynamic_cast<ExprStmt *>(stmt)) {
      if (failed(mlirGen(expr->expr.get()))) {
        emitError(loc(expr->token.span), "error in expression statement");
        return mlir::failure();
      }
      return mlir::success();
    }
    the_module.emitError("unsupported statement");
    return mlir::failure();
  }

  llvm::LogicalResult mlirGen(ExprStmt *stmt) {
    return !failed(mlirGen(stmt->expr.get())) ? mlir::success()
                                              : mlir::failure();
  }

  llvm::FailureOr<mlir::Value> mlirGen(VarDecl *var_decl) {
    auto loc = this->loc(var_decl->token.span);
    // Assume identifier pattern
    auto pattern = var_decl->pattern.get();
    auto &var_name = dynamic_cast<IdentifierPattern *>(pattern)->name;
    auto init_value = mlirGen(var_decl->initializer.value().get());
    if (failed(init_value)) {
      emitError(loc, "unsupported initializer");
      return mlir::failure();
    }

    // type check
    if (var_decl->type.has_value()) {
      auto init_type = init_value->getType();
      if (!check_type(init_type, var_decl->type.value().get())) {
        if (!try_coercion(init_value.value(), var_decl->type.value().get())) {
          emitError(loc, "type mismatch in variable declaration");
          return mlir::failure();
        }
      }
    }

    if (failed(declare(var_name, init_value.value()))) {
      emitError(loc, "redeclaration of variable");
      return mlir::failure();
    }
    return init_value;
  }

  bool try_coercion(mlir::Value &value, Type *type) {
    if (auto t = dynamic_cast<SliceType *>(type)) {
      // check if type is []u8
      if (t->base->kind() != AstNodeKind::PrimitiveType ||
          dynamic_cast<PrimitiveType *>(t->base.get())->type_kind !=
              PrimitiveType::PrimitiveTypeKind::U8) {
        return false;
      }
      // target type is a slice and value is constant string  then make a
      // slice value {pointer, length} from the string literal and assign it
      // to the value
      if (auto op = value.getDefiningOp<mlir::LLVM::GEPOp>()) {
        auto str_len = mlir::cast<mlir::LLVM::LLVMArrayType>(op.getElemType())
                           .getNumElements();
        // get address of value
        auto len = builder.create<mlir::arith::ConstantOp>(
            value.getLoc(), builder.getI64Type(),
            builder.getIntegerAttr(builder.getI64Type(), str_len));
        // Create slice struct value
        auto slice_type = type_table.lookup("Slice");
        auto one_val = builder.create<mlir::arith::ConstantOp>(
            value.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(1));
        auto zero_val = builder.create<mlir::arith::ConstantOp>(
            value.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
        auto slice_ptr = builder.create<mlir::LLVM::AllocaOp>(
            value.getLoc(),
            mlir::LLVM::LLVMPointerType::get(builder.getContext()), slice_type,
            one_val);
        auto slice_val = builder.create<mlir::LLVM::LoadOp>(
            value.getLoc(), slice_type, slice_ptr, /*isVolatile=*/false);

        // store pointer and length in slice struct
        auto gep = builder.create<mlir::LLVM::GEPOp>(
            value.getLoc(),
            mlir::LLVM::LLVMPointerType::get(builder.getContext()), slice_type,
            slice_ptr, mlir::ValueRange({zero_val, zero_val}));
        builder.create<mlir::LLVM::StoreOp>(value.getLoc(), value, gep);
        gep = builder.create<mlir::LLVM::GEPOp>(
            value.getLoc(),
            mlir::LLVM::LLVMPointerType::get(builder.getContext()), slice_type,
            slice_ptr, mlir::ValueRange({zero_val, one_val}));
        builder.create<mlir::LLVM::StoreOp>(value.getLoc(), len, gep);
        value = slice_val.getResult();
        return true;
      }
    } else if (auto t = dynamic_cast<PrimitiveType *>(type)) {
      if (t->type_kind == PrimitiveType::PrimitiveTypeKind::U8) {
        // if its an integer type then coerce it to u8
        if (auto op = value.getDefiningOp<mlir::arith::ConstantOp>()) {
          auto val = mlir::cast<mlir::IntegerAttr>(op.getValue()).getValue();
          // val is APInt, so we can just truncate it
          auto trunc_val = val.trunc(8);
          auto coerced_val = builder.create<mlir::arith::ConstantOp>(
              value.getLoc(), builder.getIntegerType(8),
              builder.getIntegerAttr(builder.getIntegerType(8), trunc_val));
          value = coerced_val.getResult();
          return true;
        }
      }
    }
    return false;
  }

  llvm::FailureOr<mlir::Type> mlirGen(Type *type) {
    if (auto t = dynamic_cast<PrimitiveType *>(type)) {
      if (t->type_kind == PrimitiveType::PrimitiveTypeKind::Bool) {
        return builder.getI1Type();
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::I8) {
        return builder.getIntegerType(8);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::I16) {
        return builder.getIntegerType(16);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::I32) {
        return builder.getIntegerType(32);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::I64) {
        return builder.getIntegerType(64);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::F32) {
        return builder.getF32Type();
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::F64) {
        return builder.getF64Type();
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::Void) {
        return builder.getNoneType();
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::Char) {
        return builder.getIntegerType(8);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::U8) {
        return builder.getIntegerType(8);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::U16) {
        return builder.getIntegerType(16);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::U32) {
        return builder.getIntegerType(32);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::U64) {
        return builder.getIntegerType(64);
      } else {
        the_module.emitError("unsupported primitive type");
        return mlir::failure();
      }
    } else if (dynamic_cast<SliceType *>(type)) {
      return mlir::LLVM::LLVMStructType::getIdentified(builder.getContext(),
                                                       "Slice");
    } else if (auto t = dynamic_cast<IdentifierType *>(type)) {
      auto type = type_table.lookup(t->name);
      if (!type) {
        emitError(loc(t->token.span), "type not found");
        return mlir::failure();
      }
      return type;
    }
    the_module.emitError("unsupported type, " + to_string(type->kind()));
    return mlir::failure();
  }

  llvm::FailureOr<mlir::Value> mlirGen(Expression *expr) {
    if (auto e = dynamic_cast<LiteralExpr *>(expr)) {
      return mlirGen(e);
    } else if (auto e = dynamic_cast<BinaryExpr *>(expr)) {
      return mlirGen(e);
    } else if (auto e = dynamic_cast<IdentifierExpr *>(expr)) {
      return mlirGen(e);
    } else if (auto e = dynamic_cast<ReturnExpr *>(expr)) {
      return mlirGen(e);
    } else if (auto e = dynamic_cast<IfExpr *>(expr)) {
      return mlirGen(e);
    } else if (auto e = dynamic_cast<CallExpr *>(expr)) {
      return mlirGen(e);
    } else if (auto e = dynamic_cast<UnaryExpr *>(expr)) {
      return mlirGen(e);
    } else if (auto e = dynamic_cast<TupleExpr *>(expr)) {
      return mlirGen(e);
    } else if (auto e = dynamic_cast<AssignExpr *>(expr)) {
      return mlirGen(e);
    } else if (auto e = dynamic_cast<FieldAccessExpr *>(expr)) {
      return mlirGen(e);
    }
    emitError(loc(expr->token.span),
              "unsupported expression, " + to_string(expr->kind()));
    return mlir::failure();
  }

  llvm::FailureOr<mlir::Value> mlirGen(TupleExpr *tuple) {
    llvm::SmallVector<mlir::Value, 4> elements;
    llvm::SmallVector<mlir::Type, 4> elementTypes;
    for (auto &elem : tuple->elements) {
      auto value = mlirGen(elem.get());
      if (failed(value)) {
        return mlir::failure();
      }
      elementTypes.push_back(value.value().getType());
      elements.push_back(value.value());
    }

    auto tuple_struct_type = mlir::LLVM::LLVMStructType::getLiteral(
        builder.getContext(), elementTypes);

    mlir::Value one_val = builder.create<mlir::LLVM::ConstantOp>(
        loc(tuple->token.span), builder.getI32Type(),
        builder.getI32IntegerAttr(1));
    mlir::Value zero_val = builder.create<mlir::LLVM::ConstantOp>(
        loc(tuple->token.span), builder.getI32Type(),
        builder.getI32IntegerAttr(0));

    auto struct_ptr = builder.create<mlir::LLVM::AllocaOp>(
        loc(tuple->token.span),
        mlir::LLVM::LLVMPointerType::get(builder.getContext()),
        tuple_struct_type, one_val);

    for (int i = 0; i < (int)elements.size(); i++) {
      mlir::Value index_val = builder.create<mlir::arith::ConstantOp>(
          loc(tuple->token.span), builder.getI32IntegerAttr(i));
      mlir::ValueRange indices = {zero_val, index_val};
      mlir::Value gep = builder.create<mlir::LLVM::GEPOp>(
          loc(tuple->token.span),
          mlir::LLVM::LLVMPointerType::get(builder.getContext()),
          tuple_struct_type, struct_ptr, indices);
      builder.create<mlir::LLVM::StoreOp>(loc(tuple->token.span), elements[i],
                                          gep);
    }
    return static_cast<mlir::Value>(struct_ptr);
  }

  llvm::FailureOr<mlir::Value> mlirGen(AssignExpr *expr) {
    auto loc = this->loc(expr->token.span);
    auto value = mlirGen(expr->rhs.get());
    if (failed(value)) {
      emitError(loc, "unsupported rvalue");
      return mlir::failure();
    }

    // if lhs is not an identifier expr, index expr or field access expr
    // then it is not a valid lvalue
    auto lhs_kind = expr->lhs->kind();
    if (lhs_kind != AstNodeKind::IdentifierExpr &&
        lhs_kind != AstNodeKind::IndexExpr &&
        lhs_kind != AstNodeKind::FieldAccessExpr) {
      emitError(loc, "invalid lvalue");
      return mlir::failure();
    }

    if (lhs_kind == AstNodeKind::FieldAccessExpr) {
      auto field_access = static_cast<FieldAccessExpr *>(expr->lhs.get());
      auto lhs = mlirGen(field_access->base.get());

      if (failed(lhs)) {
        emitError(loc, "unsupported lvalue");
        return mlir::failure();
      }
      // For assign expr the field by definition must be an identifier expr
      auto field =
          std::get<std::unique_ptr<IdentifierExpr>>(field_access->field).get();

      auto baseType = lhs.value().getType();
      auto field_name = field->name;
      auto field_index = get_field_index(baseType, field_name);
      if (field_index < 0) {
        emitError(loc, "field not found");
        return mlir::failure();
      }

      // NOTE: Right now structs are returned using values, so use insertvalue
      // TODO: Fix this to return a pointer to the struct
      auto insert_value = builder.create<mlir::LLVM::InsertValueOp>(
          loc, lhs.value(), value.value(), field_index.value());

      return insert_value.getResult();
    }
    return mlir::failure();
  }

  llvm::FailureOr<mlir::Value> mlirGen(FieldAccessExpr *expr) {
    // auto loc = this->loc(expr->token.span);
    // auto base = mlirGen(expr->base.get());
    // if (failed(base)) {
    //   emitError(loc, "unsupported base");
    //   return mlir::failure();
    // }
    //
    // // NOTE: For now we only identifier field access is supported
    // auto field =
    // std::get<std::unique_ptr<IdentifierExpr>>(expr->field).get(); auto
    // field_name = field->name; auto field_index =
    // get_field_index(base.value().getType(), field_name); if (field_index <
    // 0)
    // {
    //   emitError(loc, "field not found");
    //   return mlir::failure();
    // }
    // auto baseType = base.value().getType();
    // if (mlir::isa<mlir::LLVM::LLVMPointerType>(baseType)) {
    //   auto zero = builder.create<mlir::LLVM::ConstantOp>(
    //       loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
    //   auto fieldIndexValue = builder.create<mlir::LLVM::ConstantOp>(
    //       loc, builder.getI32Type(),
    //       builder.getI32IntegerAttr(*field_index));
    //   auto fieldPtr = builder.create<mlir::LLVM::GEPOp>(
    //       loc, *base, mlir::ValueRange{zero, fieldIndexValue});
    //   return fieldPtr.getResult();
    // } else {
    //   auto extract_value = builder.create<mlir::LLVM::ExtractValueOp>(
    //       loc, base.value(), field_index.value());
    //   return extract_value.getResult();
    // }
    return mlir::failure();
  }

  llvm::FailureOr<size_t> get_field_index(mlir::Type type,
                                          llvm::StringRef field_name) {

    // if (mlir::isa<mlir::LLVM::LLVMStructType>(type)) {
    auto struct_type = mlir::cast<mlir::LLVM::LLVMStructType>(type);
    auto struct_name = struct_type.getName();
    auto struct_decl = struct_table.lookup(struct_name);
    if (!struct_decl) {
      return mlir::failure();
    }
    for (size_t i = 0; i < struct_decl->fields.size(); i++) {
      if (struct_decl->fields[i]->name == field_name) {
        return i;
      }
    }
    // }
    return mlir::failure();
  }

  llvm::FailureOr<mlir::Value> mlirGen(UnaryExpr *unary) {
    auto operand = mlirGen(unary->operand.get());
    if (failed(operand)) {
      return mlir::failure();
    }
    mlir::Value op = nullptr;
    if (unary->op == Operator::Sub) {
      if (mlir::isa<mlir::IntegerType>(operand->getType())) {
        // no negation operation for integers in MLIR
        // so we subtract from 0
        auto const_0_val = builder.create<mlir::arith::ConstantOp>(
            loc(unary->token.span), builder.getI32IntegerAttr(0));
        op = builder.create<mlir::arith::SubIOp>(loc(unary->token.span),
                                                 const_0_val, operand.value());
      } else if (mlir::isa<mlir::Float64Type>(operand->getType())) {
        op = builder.create<mlir::arith::NegFOp>(loc(unary->token.span),
                                                 operand.value());
      }
    } else if (unary->op == Operator::Not) {
      // only for boolean type or the i1 type
      if (operand->getType() != builder.getI1Type()) {
        emitError(loc(unary->token.span), "negation only for boolean type");
        return mlir::failure();
      } else {
        auto const_1_val = builder.create<mlir::arith::ConstantOp>(
            loc(unary->token.span),
            builder.getIntegerAttr(builder.getI1Type(), 1));
        op = builder.create<mlir::arith::XOrIOp>(loc(unary->token.span),
                                                 operand.value(), const_1_val);
      }
    } else {
      emitError(loc(unary->token.span), "unsupported unary operator");
      return mlir::failure();
    }
    return op;
  }

  llvm::FailureOr<mlir::Value> mlirGen(CallExpr *callExpr) {
    auto &func_name = callExpr->callee;
    if (func_name == "print") {
      if (failed(mlirGenPrintCall(callExpr))) {
        return mlir::failure();
      }
      return mlir::success(mlir::Value());
    }
    // Generate argument values
    std::vector<mlir::Value> argumentValues;
    for (auto &arg : callExpr->arguments) {
      auto argValueOrFailure = mlirGen(arg.get());
      if (failed(argValueOrFailure)) {
        emitError(loc(callExpr->token.span),
                  "Failed to generate argument for function call");
        return mlir::failure();
      }
      argumentValues.push_back(*argValueOrFailure);
    }

    // Look up the function
    auto func_symbol = function_map.find(mangle_function_name(*callExpr));
    if (func_symbol == function_map.end()) {
      emitError(loc(callExpr->token.span), "Function not found: ") << func_name;
      return mlir::failure();
    }
    auto func_op = func_symbol->second;
    auto func_type = func_op.getFunctionType();

    // Create the `func.call` operation
    mlir::Type resultType = func_type.getResult(0); // Single result
    auto callOp = builder.create<mlir::func::CallOp>(
        loc(callExpr->token.span), func_name, resultType, argumentValues);

    //  Return the result of the call
    return callOp.getResult(0); // Return the single result of the function call
  }

  llvm::FailureOr<mlir::Value> generate_unstructured_if(IfExpr *expr) {
    // Use cf dialect's cond_br for unstructured if instead of scf dialect
    auto loc = this->loc(expr->token.span);

    // Generate the condition
    auto condResult = mlirGen(expr->condition.get());
    if (failed(condResult)) {
      emitError(loc, "unsupported condition");
      return mlir::failure();
    }
    mlir::Value cond = condResult.value();

    if (cond.getType() != builder.getI1Type()) {
      emitError(loc, "condition must have a boolean type");
      return mlir::failure();
    }

    // Get the parent block and function
    auto *currentBlock = builder.getInsertionBlock();
    auto *parentRegion = currentBlock->getParent();

    // Create the blocks for the 'then', 'else', and continuation ('merge')
    // blocks
    auto *thenBlock = builder.createBlock(parentRegion);
    mlir::Block *elseBlock = nullptr;
    if (expr->else_block.has_value()) {
      elseBlock = builder.createBlock(parentRegion);
    }
    auto *mergeBlock = builder.createBlock(parentRegion);

    // Insert the conditional branch
    builder.setInsertionPointToEnd(currentBlock);
    if (elseBlock) {
      builder.create<mlir::cf::CondBranchOp>(loc, cond, thenBlock, elseBlock);
    } else {
      builder.create<mlir::cf::CondBranchOp>(loc, cond, thenBlock, mergeBlock);
    }

    // Build the 'then' block
    builder.setInsertionPointToStart(thenBlock);
    if (std::holds_alternative<std::unique_ptr<BlockExpression>>(
            expr->then_block)) {
      if (failed(mlirGen(
              std::get<std::unique_ptr<BlockExpression>>(expr->then_block)
                  .get()))) {
        emitError(loc, "error in then block");
        return mlir::failure();
      }
    } else {
      if (failed(mlirGen(
              std::get<std::unique_ptr<Expression>>(expr->then_block).get()))) {
        emitError(loc, "error in then block");
        return mlir::failure();
      }
    }

    // If 'then' block does not end with a return, branch to the merge block
    if (thenBlock->empty() ||
        !thenBlock->back().mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
      builder.setInsertionPointToEnd(thenBlock);
      builder.create<mlir::cf::BranchOp>(loc, mergeBlock);
    }

    // Build the 'else' block if it exists
    if (elseBlock) {
      builder.setInsertionPointToStart(elseBlock);
      if (std::holds_alternative<std::unique_ptr<BlockExpression>>(
              expr->else_block.value())) {
        if (failed(mlirGen(std::get<std::unique_ptr<BlockExpression>>(
                               expr->else_block.value())
                               .get()))) {
          emitError(loc, "error in else block");
          return mlir::failure();
        }
      } else {
        if (failed(mlirGen(
                std::get<std::unique_ptr<Expression>>(expr->else_block.value())
                    .get()))) {
          emitError(loc, "error in else block");
          return mlir::failure();
        }
      }

      // If 'else' block does not end with a return, branch to the merge
      // block
      if (elseBlock->empty() ||
          !elseBlock->back().mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
        builder.setInsertionPointToEnd(elseBlock);
        builder.create<mlir::cf::BranchOp>(loc, mergeBlock);
      }
    }

    // Continue building from the merge block
    builder.setInsertionPointToStart(mergeBlock);

    // Optionally, if this 'if' expression produces a value, you need to
    // handle SSA dominance and merge the values.

    // Since this function does not produce a value, return success without
    // a value
    return mlir::success(mlir::Value());
  }

  llvm::FailureOr<mlir::Value> mlirGen(IfExpr *expr) {
    if (!expr->else_block.has_value()) { // TODO: now simple check, improve
      return generate_unstructured_if(expr);
    }
    ControlFlow cf(control_flow);
    auto span = loc(expr->token.span);
    auto cond = mlirGen(expr->condition.get());
    if (failed(cond)) {
      emitError(span, "unsupported condition");
      return mlir::failure();
    }

    bool with_else_region = expr->else_block.has_value();
    // Create an scf.if operation with a condition
    auto ifOp = builder.create<mlir::scf::IfOp>(span, builder.getF32Type(),
                                                cond.value(), with_else_region);

    // Emit the "then" block
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    if (std::holds_alternative<std::unique_ptr<BlockExpression>>(
            expr->then_block)) {
      if (mlir::failed(mlirGen(
              std::get<std::unique_ptr<BlockExpression>>(expr->then_block)
                  .get()))) {
        emitError(span, "error in then block");
        return mlir::failure();
      }
    } else {
      if (failed(mlirGen(
              std::get<std::unique_ptr<Expression>>(expr->then_block).get()))) {
        emitError(span, "error in then block");
        return mlir::failure();
      }
    }

    if (with_else_region) {
      // Emit the "else" block
      builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
      if (std::holds_alternative<std::unique_ptr<BlockExpression>>(
              expr->else_block.value())) {
        if (mlir::failed(mlirGen(std::get<std::unique_ptr<BlockExpression>>(
                                     expr->else_block.value())
                                     .get()))) {
          emitError(span, "error in else block");
          return mlir::failure();
        }
      } else {
        if (failed(mlirGen(
                std::get<std::unique_ptr<Expression>>(expr->else_block.value())
                    .get()))) {
          emitError(span, "error in else block");
          return mlir::failure();
        }
      }
    }

    // Set the insertion point back to the main body after the if statement
    builder.setInsertionPointAfter(ifOp);
    if (ifOp.getNumResults() > 0) {
      return ifOp.getResult(0);
    }
    return mlir::success(mlir::Value());
  }

  llvm::FailureOr<mlir::Value> mlirGen(ReturnExpr *expr) {
    auto loc = this->loc(expr->token.span);
    auto value = mlirGen(expr->value.value().get());
    if (failed(value)) {
      emitError(loc, "unsupported return value");
      return mlir::failure();
    }

    if (!control_flow) {
      builder.create<mlir::func::ReturnOp>(loc, value.value());
    } else {
      builder.create<mlir::scf::YieldOp>(loc, value.value());
    }
    return value;
  }

  llvm::FailureOr<mlir::Value> mlirGen(LiteralExpr *literal) {
    if (literal->type == LiteralExpr::LiteralType::Int) {
      return builder
          .create<mlir::arith::ConstantOp>(
              loc(literal->token.span),
              builder.getIntegerAttr(builder.getIntegerType(32),
                                     std::get<int>(literal->value)))
          .getResult();
    } else if (literal->type == LiteralExpr::LiteralType::Float) {
      return builder
          .create<mlir::arith::ConstantOp>(
              loc(literal->token.span),
              builder.getF64FloatAttr(std::get<double>(literal->value)))
          .getResult();
    } else if (literal->type == LiteralExpr::LiteralType::String) {
      auto &str = std::get<std::string>(literal->value);
      auto str_attr = createGlobalString("str", str, loc(literal->token.span),
                                         builder, the_module);
      return getPtrToGlobalString(str_attr, builder, loc(literal->token.span),
                                  str.size());
      // auto array_type =
      //     mlir::LLVM::LLVMArrayType::get(builder.getI8Type(), str.size());
      //
      // auto str_attr = builder.getStringAttr(str);
      // return builder
      //     .create<mlir::LLVM::ConstantOp>(loc(literal->token.span),
      //     array_type,
      //                                     str_attr)
      //     .getResult();
    } else if (literal->type == LiteralExpr::LiteralType::Bool) {
      return builder
          .create<mlir::arith::ConstantOp>(
              loc(literal->token.span),
              builder.getIntegerAttr(builder.getI1Type(),
                                     std::get<bool>(literal->value)))
          .getResult();
    } else if (literal->type == LiteralExpr::LiteralType::Char) {
      return builder
          .create<mlir::arith::ConstantOp>(
              loc(literal->token.span),
              builder.getIntegerAttr(builder.getIntegerType(8),
                                     std::get<char>(literal->value)))
          .getResult();
    }

    else {
      the_module.emitError("unsupported literal");
      return mlir::failure();
    }
  }

  llvm::FailureOr<mlir::Value> mlirGen(BinaryExpr *binary) {
    auto lhs_v = mlirGen(binary->lhs.get());
    auto rhs_v = mlirGen(binary->rhs.get());
    if (failed(lhs_v) || failed(rhs_v)) {
      return mlir::failure();
    }
    auto lhs = lhs_v.value();
    auto rhs = rhs_v.value();
    mlir::Value op = nullptr;
    // if both operands are integers
    if (mlir::isa<mlir::IntegerType>(lhs.getType()) &&
        mlir::isa<mlir::IntegerType>(rhs.getType())) {
      op = integer_ops(binary, lhs, rhs).value();
    } else if (mlir::isa<mlir::Float64Type>(lhs.getType()) &&
               mlir::isa<mlir::Float64Type>(rhs.getType())) {
      op = floating_ops(binary, lhs, rhs).value();
    } else {
      return mlir::failure();
    }
    return op;
  }

  llvm::FailureOr<mlir::Value> integer_ops(BinaryExpr *binary, mlir::Value lhs,
                                           mlir::Value rhs) {
    mlir::Value op = nullptr;
    if (binary->op == Operator::Add) {
      op = builder.create<mlir::arith::AddIOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::Sub) {
      op = builder.create<mlir::arith::SubIOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::Mul) {
      op = builder.create<mlir::arith::MulIOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::Div) {
      op = builder.create<mlir::arith::DivSIOp>(loc(binary->token.span), lhs,
                                                rhs);
    } else if (binary->op == Operator::Eq) {
      op = builder.create<mlir::arith::CmpIOp>(
          loc(binary->token.span), mlir::arith::CmpIPredicate::eq, lhs, rhs);
    } else if (binary->op == Operator::Ne) {
      op = builder.create<mlir::arith::CmpIOp>(
          loc(binary->token.span), mlir::arith::CmpIPredicate::ne, lhs, rhs);
    } else if (binary->op == Operator::Lt) {
      op = builder.create<mlir::arith::CmpIOp>(
          loc(binary->token.span), mlir::arith::CmpIPredicate::slt, lhs, rhs);
    } else if (binary->op == Operator::Le) {
      op = builder.create<mlir::arith::CmpIOp>(
          loc(binary->token.span), mlir::arith::CmpIPredicate::sle, lhs, rhs);
    } else if (binary->op == Operator::Gt) {
      op = builder.create<mlir::arith::CmpIOp>(
          loc(binary->token.span), mlir::arith::CmpIPredicate::sgt, lhs, rhs);
    } else if (binary->op == Operator::Ge) {
      op = builder.create<mlir::arith::CmpIOp>(
          loc(binary->token.span), mlir::arith::CmpIPredicate::sge, lhs, rhs);
    } else if (binary->op == Operator::BitAnd) {
      op = builder.create<mlir::arith::AndIOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::BitOr) {
      op =
          builder.create<mlir::arith::OrIOp>(loc(binary->token.span), lhs, rhs);
    } else if (binary->op == Operator::BitXor) {
      op = builder.create<mlir::arith::XOrIOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::BitShl) {
      op = builder.create<mlir::arith::ShLIOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::BitShr) {
      op = builder.create<mlir::arith::ShRUIOp>(loc(binary->token.span), lhs,
                                                rhs);
    } else if (binary->op == Operator::And) {
      // Only if type is i1
      if (lhs.getType() != builder.getI1Type() ||
          rhs.getType() != builder.getI1Type()) {
        emitError(loc(binary->token.span), "logical and requires boolean type");
        return mlir::failure();
      }
      op = builder.create<mlir::arith::AndIOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::Or) {
      // Only if type is i1
      if (lhs.getType() != builder.getI1Type() ||
          rhs.getType() != builder.getI1Type()) {
        emitError(loc(binary->token.span), "logical or requires boolean type");
        return mlir::failure();
      }
      op =
          builder.create<mlir::arith::OrIOp>(loc(binary->token.span), lhs, rhs);
    } else {
      the_module.emitError("unsupported binary operator " +
                           std::to_string(static_cast<int>(binary->op)));
      return mlir::failure();
    }
    return op;
  }

  llvm::FailureOr<mlir::Value> floating_ops(BinaryExpr *binary, mlir::Value lhs,
                                            mlir::Value rhs) {
    mlir::Value op = nullptr;
    if (binary->op == Operator::Add) {
      op = builder.create<mlir::arith::AddFOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::Sub) {
      op = builder.create<mlir::arith::SubFOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::Mul) {
      op = builder.create<mlir::arith::MulFOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::Div) {
      op = builder.create<mlir::arith::DivFOp>(loc(binary->token.span), lhs,
                                               rhs);
    } else if (binary->op == Operator::Eq) {
      op = builder.create<mlir::arith::CmpIOp>(
          loc(binary->token.span), mlir::arith::CmpIPredicate::eq, lhs, rhs);
    } else if (binary->op == Operator::Ne) {
      op = builder.create<mlir::arith::CmpFOp>(
          loc(binary->token.span), mlir::arith::CmpFPredicate::ONE, lhs, rhs);
    } else if (binary->op == Operator::Lt) {
      op = builder.create<mlir::arith::CmpFOp>(
          loc(binary->token.span), mlir::arith::CmpFPredicate::OLT, lhs, rhs);
    } else if (binary->op == Operator::Le) {
      op = builder.create<mlir::arith::CmpFOp>(
          loc(binary->token.span), mlir::arith::CmpFPredicate::OLE, lhs, rhs);
    } else if (binary->op == Operator::Gt) {
      op = builder.create<mlir::arith::CmpFOp>(
          loc(binary->token.span), mlir::arith::CmpFPredicate::OGT, lhs, rhs);
    } else if (binary->op == Operator::Ge) {
      op = builder.create<mlir::arith::CmpFOp>(
          loc(binary->token.span), mlir::arith::CmpFPredicate::OGE, lhs, rhs);
    } else {
      the_module.emitError("unsupported binary operator");
      return mlir::failure();
    }
    return op;
  }

  llvm::FailureOr<mlir::Value> mlirGen(IdentifierExpr *identifier) {
    auto loc = this->loc(identifier->token.span);
    if (auto variable = symbol_table.lookup(identifier->name))
      return variable;
    emitError(loc, "undeclared variable -> " + identifier->name);
    return mlir::failure();
  }

  // check mlir type is equal to ast type
  bool check_type(mlir::Type &mlir_type, Type *ast_type) {
    if (auto t = dynamic_cast<PrimitiveType *>(ast_type)) {
      if (t->type_kind == PrimitiveType::PrimitiveTypeKind::I32) {
        return mlir::isa<mlir::IntegerType>(mlir_type) &&
               mlir_type.getIntOrFloatBitWidth() == 32;
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::F32) {
        return mlir::isa<mlir::Float32Type>(mlir_type);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::F64) {
        return mlir::isa<mlir::Float64Type>(mlir_type);
      }
    }
    return false;
  }

  void declarePrintf() {

    // Define the printf function type: (i8*, ...) -> i32
    auto i8PtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    // Create a function type with variable arguments (varargs)
    auto printfType = mlir::LLVM::LLVMFunctionType::get(
        mlir::IntegerType::get(builder.getContext(), 32), {i8PtrType},
        /*isVarArg=*/true);

    // Create the printf function declaration using LLVMFuncOp
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(the_module.getBody());

    if (!the_module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) {
      builder.create<mlir::LLVM::LLVMFuncOp>(the_module.getLoc(), "printf",
                                             printfType);
    }
  }

  mlir::LLVM::GlobalOp createGlobalString(llvm::StringRef baseName,
                                          llvm::StringRef value,
                                          mlir::Location loc,
                                          mlir::OpBuilder &builder,
                                          mlir::ModuleOp module) {
    // Generate a unique name for the global string based on its content
    std::string uniqueName =
        (baseName + "_" + std::to_string(std::hash<std::string>{}(value.str())))
            .str()
            .substr(0, 64);
    int str_length = value.size() + 1;
    auto i8Type = builder.getIntegerType(8);
    auto stringType = mlir::LLVM::LLVMArrayType::get(i8Type, str_length);

    // Create a global constant
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(the_module.getBody());

    auto global = builder.create<mlir::LLVM::GlobalOp>(
        loc, stringType,
        /*isConstant=*/true, mlir::LLVM::Linkage::Internal, uniqueName,
        builder.getStringAttr(value.str() + '\0'));

    return global;
  }

  mlir::LogicalResult mlirGenPrintCall(CallExpr *callExpr) {
    if (callExpr->arguments.size() < 1) {
      emitError(loc(callExpr->token.span),
                "print expects at least a format string argument");
      return mlir::failure();
    }

    // check calllexpr arg 0 is a literal string
    std::string formatString;
    if (auto str = dynamic_cast<LiteralExpr *>(callExpr->arguments[0].get())) {
      formatString = std::get<std::string>(str->value);
      // NOTE: Temp fix for string literal
      formatString = formatString.substr(1, formatString.size() - 2) + '\n';
      if (str->type != LiteralExpr::LiteralType::String) {
        emitError(loc(callExpr->token.span),
                  "print expects a string literal as the first argument");
        return mlir::failure();
      }
    } else {
      emitError(loc(callExpr->token.span),
                "print expects a string literal as the first argument");
      return mlir::failure();
    }
    auto formatArg =
        createGlobalString("format_string", formatString,
                           loc(callExpr->token.span), builder, the_module);

    auto formatArgPtr = getPtrToGlobalString(
        formatArg, builder, loc(callExpr->token.span), formatString.size() + 1);

    // Collect the rest of the arguments
    llvm::SmallVector<mlir::Value, 4> printfArgs;
    printfArgs.push_back(formatArgPtr);

    for (size_t i = 1; i < callExpr->arguments.size(); ++i) {
      auto argValueOrFailure = mlirGen(callExpr->arguments[i].get());
      if (failed(argValueOrFailure))
        return mlir::failure();
      printfArgs.push_back(argValueOrFailure.value());
    }

    // Declare printf if not already declared
    declarePrintf();

    // Create the call to printf
    builder.create<mlir::LLVM::CallOp>(
        loc(callExpr->token.span),
        the_module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"),
        mlir::ValueRange(printfArgs));

    return mlir::success();
  }

  mlir::Value getPtrToGlobalString(mlir::LLVM::GlobalOp global,
                                   mlir::OpBuilder &builder, mlir::Location loc,
                                   int64_t stringLength) {
    auto *context = builder.getContext();
    auto i8Type = mlir::IntegerType::get(context, 8);
    auto i8PtrType = mlir::LLVM::LLVMPointerType::get(context);
    auto arrayType = mlir::LLVM::LLVMArrayType::get(i8Type, stringLength);

    // Get the address of the global string
    auto addr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);

    // Indices to access the first element: [0, 0]
    auto zero32 = builder.create<mlir::LLVM::ConstantOp>(
        loc, mlir::IntegerType::get(context, 32), builder.getI32IntegerAttr(0));
    mlir::Value indices[] = {zero32, zero32};

    // Create the GEP operation
    auto gep = builder.create<mlir::LLVM::GEPOp>(
        loc,
        /* resultType */ i8PtrType,
        /* elementType */ arrayType,
        /* basePtr */ addr,
        /* indices */ mlir::ValueRange(indices),
        /* isInBounds */ false);

    return gep;
  }
};

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          Program *program) {
  return MLIRGenImpl(context).mlirGen(program);
}
