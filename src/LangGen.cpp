#include "LangGen.h"
#include "ast.hpp"
#include "compiler.hpp"
#include "data_structures.hpp"
#include "dialect/LangOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <memory>
#include <ranges>
#include <vector>

using llvm::StringRef;
using PatternBindings = std::vector<std::pair<llvm::StringRef, Type *>>;

#define NEW_SCOPE()                                                            \
  llvm::ScopedHashTableScope<StringRef, mlir::Value> var_scope(symbol_table);  \
  llvm::ScopedHashTableScope<StringRef, mlir::Type> type_scope(type_table);    \
  llvm::ScopedHashTableScope<StringRef, StructDecl *> struct_scope(            \
      struct_table);                                                           \
  llvm::ScopedHashTableScope<mlir::lang::StructType, StringRef>                \
      struct_name_scope(struct_name_table);                                    \
  llvm::ScopedHashTableScope<StringRef, Type *> vardecl_scope(                 \
      compiler_context.var_table);

class AnnotatingListener : public mlir::OpBuilder::Listener {
public:
  AnnotatingListener(mlir::OpBuilder &builder, mlir::StringAttr key,
                     mlir::Attribute value)
      : key(key), value(value), builder(builder) {
    builder.setListener(this);
  }

  void notifyOperationInserted(mlir::Operation *op,
                               mlir::OpBuilder::InsertPoint prev) override {
    op->setAttr(key, value);
  }

  ~AnnotatingListener() override { builder.setListener(nullptr); }

private:
  mlir::StringAttr key;
  mlir::Attribute value;
  mlir::OpBuilder &builder;
};

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
    declarePrimitiveTypes();
    for (auto &f : program->items) {
      builder.setInsertionPointToEnd(the_module.getBody());
      if (failed(langGen(f.get()))) {
        return nullptr;
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
  llvm::StringMap<Function *> function_table;
  llvm::SmallVector<mlir::Value, 16> type_values;
  AstDumper dumper;

  mlir::Location loc(const TokenSpan &loc) {
    return mlir::FileLineColLoc::get(
        builder.getContext(),
        compiler_context.source_mgr.getBufferInfo(loc.file_id)
            .Buffer->getBufferIdentifier(),
        loc.line_no, loc.col_start);
  }

  int insertType(mlir::Value value) {
    type_values.push_back(value);
    return type_values.size() - 1;
  }

  mlir::Value getTypeValue(int index) { return type_values[index]; }

  llvm::StringRef str(llvm::StringRef string) {
    return builder.getStringAttr(string);
  }

  void declarePrimitiveTypes() {
    for (int i = 1; i <= 99; ++i) {
      auto signless_type = builder.getIntegerType(i);
      type_table.insert(str("i" + std::to_string(i)), signless_type);

      auto signed_type = builder.getIntegerType(i, /*isSigned=*/true);
      type_table.insert(str("si" + std::to_string(i)), signed_type);

      auto unsigned_type = builder.getIntegerType(i, /*isSigned=*/false);
      type_table.insert(str("ui" + std::to_string(i)), unsigned_type);
    }
    type_table.insert(str("f32"), builder.getF32Type());
    type_table.insert(str("f64"), builder.getF64Type());
    type_table.insert(str("bool"), builder.getIntegerType(1));
    type_table.insert(str("char"), builder.getIntegerType(8));
    type_table.insert(str("void"), builder.getNoneType());
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

  // TODO: Implement this
  void demangle(llvm::StringRef mangled_name) {}

  mlir::LogicalResult langGen(TopLevelDecl *decl) {
    if (decl->kind() == AstNodeKind::Function) {
      auto func = decl->as<Function>();
      return langGen(func, isGenericFunction(func));
    } else if (decl->kind() == AstNodeKind::TupleStructDecl) {
      return langGen(decl->as<TupleStructDecl>());
    } else if (decl->kind() == AstNodeKind::StructDecl) {
      return langGen(decl->as<StructDecl>());
    } else if (decl->kind() == AstNodeKind::ImplDecl) {
      return langGen(decl->as<ImplDecl>());
    } else if (decl->kind() == AstNodeKind::ImportDecl) {
    }
    return the_module.emitError("unsupported top-level item");
  }

  mlir::Result<mlir::lang::FuncOp> langGen(Function *func,
                                           bool is_generic = false) {
    if (is_generic) {
      function_table[str(func->decl->name)] = func;
      return mlir::success();
    }
    NEW_SCOPE()
    auto param_types = langGen(func->decl->parameters);
    if (failed(param_types)) {
      return mlir::failure();
    }
    mlir::TypeRange return_types =
        func->decl->return_type->kind() == AstNodeKind::PrimitiveType &&
                func->decl->return_type->as<PrimitiveType>()->type_kind ==
                    PrimitiveType::PrimitiveTypeKind::Void
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
    auto func_name = func->decl->extra.is_generic
                         ? str(func->decl->name)
                         : str(mangle(func->decl->name, param_types.value()));
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
    current_function = &func_op;
    for (const auto &it : llvm::enumerate(func->decl->parameters)) {
      if (it.value()->is_comptime) {
        func_op.setArgAttr(it.index(), "lang.comptime",
                           builder.getBoolAttr(true));
      }
    }
    auto entry_block = func_op.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);

    // Declare function parameters.
    if (failed(declareParameters(func->decl->parameters,
                                 entry_block->getArguments()))) {
      func_op.erase();
      return mlir::emitError(loc(func->token.span),
                             "parameter declaration error");
    }

    // Generate function body.
    if (func->decl->extra.is_method && func->decl->name == "init") {
      auto create_self = [&]() {
        // Create a new struct instance
        auto self_type = type_table.lookup("Self");
        if (!self_type) {
          emitError(loc(func->token.span), "Self not found");
          return;
        }
        auto struct_val = builder.create<mlir::lang::UndefOp>(
            loc(func->token.span), self_type);
        if (failed(declare("self", struct_val))) {
          emitError(loc(func->token.span), "redeclaration of self");
          return;
        }
        // update func_name
        param_types->insert(param_types->begin(), self_type);
        func_name = str(mangle(func->decl->name, param_types.value()));
        func_op.setName(func_name);
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

    function_map[func_name] = func_op;
    function_table[func_name] = func;
    current_function = nullptr;
    // return mlir::success();
    return func_op;
  }

  llvm::LogicalResult
  declareParameters(std::vector<std::unique_ptr<Parameter>> &params,
                    mlir::ArrayRef<mlir::BlockArgument> args) {

    for (int i = 0; i < (int)params.size(); i++) {
      // Assume identifier pattern
      auto var_name = str(params[i]->pattern->as<IdentifierPattern>()->name);
      if (failed(declare(var_name, args[i]))) {
        return the_module.emitError("redeclaration of parameter");
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
      auto bindings =
          destructurePattern(param->pattern.get(), param->type.get());
      if (failed(bindings)) {
        return mlir::failure();
      }
      for (auto [name, t] : bindings.value()) {
        compiler_context.declareVar(name, t);
      }
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
      return emitError(loc, "unsupported return value");
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
        return emitError(span, "unsupported field type");
      }
      field_types.push_back(type.value());
    }
    auto struct_type = mlir::lang::StructType::get(builder.getContext(),
                                                   decl->name, field_types);
    // declare struct type
    if (failed(declare(decl->name, struct_type))) {
      return emitError(span, "redeclaration of struct type");
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
    auto struct_type = mlir::lang::StructType::get(
        builder.getContext(), struct_decl->name, element_types);
    struct_table.insert(struct_decl->name, struct_decl);
    if (failed(declare(struct_decl->name, struct_type))) {
      return mlir::emitError(loc(struct_decl->token.span),
                             "redeclaration of struct type");
    }
    struct_name_table.insert(struct_type, struct_decl->name);
    return mlir::success();
  }

  mlir::LogicalResult langGen(ImplDecl *impl_decl) {
    static AstDumper dumper(false, true);
    auto type = dumper.dump<Type>(impl_decl->type.get());
    NEW_SCOPE()
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto parent_type = type_table.lookup(type);
    if (!parent_type) {
      // check if its an mlir type (TODO: add unranked tensor and memref)
      // static const std::string mlir_types[] = {"complex", "function",
      // "memref",
      //                                          "opaque",  "tensor", "tuple",
      //                                          "vector"};
      // if (std::find(std::begin(mlir_types), std::end(mlir_types), type) !=
      //     std::end(mlir_types)) {
      //   return mlir::emitError(loc(impl_decl->token.span),
      //                          "mlir type not found");
      // }
      parent_type = mlir::parseType(type, builder.getContext());
      if (!parent_type) {
        return mlir::emitError(loc(impl_decl->token.span),
                               "parent type not found");
      }
      type_table.insert(type, parent_type);
    }
    // auto struct_type = mlir::dyn_cast<mlir::lang::StructType>(parent_type);
    if (failed(declare("Self", parent_type))) {
      return mlir::emitError(loc(impl_decl->token.span),
                             "redeclaration of Self");
    }
    auto decl = builder.create<mlir::lang::ImplDeclOp>(
        loc(impl_decl->token.span), parent_type);
    builder.setInsertionPointToStart(&decl.getRegion().emplaceBlock());
    for (auto &method : impl_decl->functions) {
      mlir::OpBuilder::InsertionGuard guard(builder);
      auto func = langGen(method.get());
      if (failed(func)) {
        return mlir::failure();
      }
    }
    // yield
    builder.create<mlir::lang::YieldOp>(loc(impl_decl->token.span));
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
      if (block->statements.back()->kind() == AstNodeKind::ExprStmt) {
        return langGen(block->statements.back()->as<ExprStmt>()->expr.get());
      }
      if (failed(langGen(block->statements.back().get()))) {
        return mlir::failure();
      }
    }
    // block does not return anything, so return a void value
    return mlir::success(mlir::Value());
  }

  mlir::LogicalResult langGen(Statement *stmt) {
    if (stmt->kind() == AstNodeKind::VarDecl) {
      return langGen(stmt->as<VarDecl>());
    } else if (stmt->kind() == AstNodeKind::ExprStmt) {
      return langGen(stmt->as<ExprStmt>()->expr.get());
    } else if (stmt->kind() == AstNodeKind::TopLevelDeclStmt) {
      return langGen(stmt->as<TopLevelDeclStmt>()->decl.get());
    } else if (stmt->kind() == AstNodeKind::AssignStatement) {
      return langGen(stmt->as<AssignStatement>());
    }
    return mlir::emitError(loc(stmt->token.span), "unsupported statement");
  }

  mlir::LogicalResult langGen(VarDecl *var_decl) {
    // Add the variable to the symbol table
    auto bindings =
        destructurePattern(var_decl->pattern.get(),
                           var_decl->type ? var_decl->type->get() : nullptr);
    if (llvm::failed(bindings)) {
      return mlir::failure();
    }
    for (auto [name, t] : bindings.value()) {
      compiler_context.declareVar(name, t);
    }

    mlir::Value var_type_value = nullptr;
    if (var_decl->type.has_value()) {
      if (var_decl->type.value()->kind() == AstNodeKind::ExprType) {
        auto type_value =
            langGen(var_decl->type.value()->as<ExprType>()->expr.get());
        if (failed(type_value)) {
          return mlir::failure();
        }
        var_type_value = type_value.value();
      } else {
        auto type = getType(var_decl->type.value().get());
        if (failed(type)) {
          return mlir::failure();
        }
        var_type_value = builder
                             .create<mlir::lang::TypeConstOp>(
                                 loc(var_decl->token.span),
                                 mlir::lang::TypeValueType::get(
                                     builder.getContext(), type.value()),
                                 type.value())
                             .getResult();
      }
    }

    // assume the name is identifier pattern for now
    auto &var_name = var_decl->pattern->as<IdentifierPattern>()->name;
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
    } else {
      init_value = builder.create<mlir::lang::UndefOp>(
          loc(var_decl->token.span), var_type_value.getType());
    }

    // auto op = builder.create<mlir::lang::VarDeclOp>(
    //     loc(var_decl->token.span),
    //     var_type ? mlir::TypeAttr::get(var_type) : nullptr, var_name,
    //     init_value.value(), var_decl->is_mut, var_decl->is_pub);
    auto op = builder.create<mlir::lang::VarDeclOp>(
        loc(var_decl->token.span), var_type_value, var_name, init_value.value(),
        var_decl->is_mut, var_decl->is_pub);
    if (failed(declare(var_name, op.getResult()))) {
      return mlir::failure();
    }
    return mlir::success();
  }

  mlir::FailureOr<mlir::Type> getType(Type *type) {
    if (type->kind() == AstNodeKind::PrimitiveType) {
      auto primitive_type = type->as<PrimitiveType>();
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
      } else if (primitive_type->type_kind ==
                 PrimitiveType::PrimitiveTypeKind::type) {
        return mlir::lang::LangType::get(builder.getContext());
      } else {
        mlir::emitError(loc(primitive_type->token.span), "unsupported type");
      }
    } else if (type->kind() == AstNodeKind::MLIRType) {
      auto mlir_type = type->as<MLIRType>();
      llvm::StringRef type_name = mlir_type->type;
      return mlir::parseType(type_name.trim('"'), builder.getContext());
    } else if (type->kind() == AstNodeKind::IdentifierType) {
      auto identifier_type = type->as<IdentifierType>();
      auto type_name = identifier_type->name;
      mlir::Operation *op = nullptr;
      if (current_function) {
        mlir::SymbolTable symbol_table(current_function->getOperation());
        op = symbol_table.lookup(type_name);
        if (op)
          return op->getResult(0).getType();
      }
      // search in struct table
      auto type = type_table.lookup(type_name);
      if (type)
        return type;
      auto vartype = compiler_context.var_table.lookup(type_name);
      if (vartype)
        return getType(vartype);
      return mlir::emitError(loc(identifier_type->token.span), "unknown type");
    } else if (type->kind() == AstNodeKind::SliceType) {
      auto slice_type = type->as<SliceType>();
      auto base_type = getType(slice_type->base.get());
      if (failed(base_type)) {
        return mlir::failure();
      }
      return mlir::lang::SliceType::get(builder.getContext(),
                                        base_type.value());
    } else if (type->kind() == AstNodeKind::ArrayType) {
      auto array_type = type->as<ArrayType>();
      auto base_type = getType(array_type->base.get());
      if (failed(base_type)) {
        return mlir::failure();
      }
      auto size_expr = array_type->size.get();
      AnnotatingListener listener(builder, builder.getStringAttr("comptime"),
                                  builder.getBoolAttr(true));
      auto size_value = langGen(size_expr);
      if (failed(size_value)) {
        return mlir::failure();
      }
      int64_t type_index = insertType(size_value.value());
      return mlir::lang::ArrayType::get(builder.getContext(), base_type.value(),
                                        builder.getI64IntegerAttr(type_index),
                                        builder.getBoolAttr(true));
    } else if (type->kind() == AstNodeKind::ExprType) {
      auto expr_type = type->as<ExprType>();
      auto type_value = langGen(expr_type->expr.get());
      if (failed(type_value)) {
        return mlir::failure();
      }
      return type_value.value().getType();
    }
    return mlir::emitError(loc(type->token.span),
                           "unsupported type " + toString(type->kind()));
  }

  mlir::FailureOr<mlir::Value> langGen(Expression *expr) {
    switch (expr->kind()) {
    case AstNodeKind::LiteralExpr:
      return langGen(expr->as<LiteralExpr>());
    case AstNodeKind::MLIRAttribute:
      return langGen(expr->as<MLIRAttribute>());
    case AstNodeKind::MLIRType:
      return langGen(expr->as<MLIRType>());
    case AstNodeKind::ReturnExpr:
      return langGen(expr->as<ReturnExpr>());
    case AstNodeKind::BlockExpression:
      return langGen(expr->as<BlockExpression>());
    case AstNodeKind::CallExpr:
      return langGen(expr->as<CallExpr>());
    case AstNodeKind::FieldAccessExpr:
      return langGen(expr->as<FieldAccessExpr>());
    case AstNodeKind::IdentifierExpr:
      return langGen(expr->as<IdentifierExpr>());
    case AstNodeKind::MLIROp:
      return langGen(expr->as<MLIROp>());
    case AstNodeKind::BinaryExpr:
      return langGen(expr->as<BinaryExpr>());
    case AstNodeKind::IfExpr:
      return langGen(expr->as<IfExpr>());
    case AstNodeKind::YieldExpr:
      return langGen(expr->as<YieldExpr>());
    case AstNodeKind::TupleExpr:
      return langGen(expr->as<TupleExpr>());
    case AstNodeKind::UnaryExpr:
      return langGen(expr->as<UnaryExpr>());
    case AstNodeKind::IndexExpr:
      return langGen(expr->as<IndexExpr>());
    case AstNodeKind::ComptimeExpr:
      return langGen(expr->as<ComptimeExpr>());
    case AstNodeKind::ArrayExpr:
      return langGen(expr->as<ArrayExpr>());
    default:
      return mlir::emitError(loc(expr->token.span), "unsupported expression " +
                                                        toString(expr->kind()));
    }
  }

  mlir::FailureOr<mlir::Value> langGen(ArrayExpr *expr) {
    // if const expr, then create a constant array
    if (expr->extra.is_const) {
      mlir::DenseElementsAttr attr;
      auto base_type = expr->elements[0]->as<LiteralExpr>()->type;
      auto values = expr->elements | std::views::transform([](auto &e) {
                      return e->template as<LiteralExpr>()->value;
                    });

      switch (base_type) {
      case LiteralExpr::LiteralType::Int: {
        auto int_range = values | std::views::transform([](auto v) {
                           return mlir::APInt(64, std::get<int>(v));
                         });
        std::vector<mlir::APInt> int_values(int_range.begin(), int_range.end());
        auto shape = mlir::RankedTensorType::get(
            {static_cast<long>(values.size())}, builder.getI64Type());
        attr = mlir::DenseElementsAttr::get(shape, int_values);
        break;
      }
      case LiteralExpr::LiteralType::Float: {
        auto float_range = values | std::views::transform([](auto v) {
                             return mlir::APFloat(std::get<double>(v));
                           });
        std::vector<mlir::APFloat> float_values(float_range.begin(),
                                                float_range.end());
        auto shape = mlir::RankedTensorType::get(
            {static_cast<long>(values.size())}, builder.getF64Type());
        attr = mlir::DenseElementsAttr::get(shape, float_values);
        break;
      }
        // case LiteralExpr::LiteralType::Bool: {
        //   auto bool_range = values | std::views::transform([](auto v) {
        //                       return std::get<bool>(v);
        //                     });
        //   std::vector<bool> bool_values(bool_range.begin(),
        //   bool_range.end()); auto shape = mlir::RankedTensorType::get(
        //       {static_cast<long>(values.size())}, builder.getI1Type());
        //   auto attr = mlir::DenseElementsAttr::get<bool>(shape,
        // bool_values);
        //   break;
        // }

      case LiteralExpr::LiteralType::Char: {
        auto char_range = values | std::views::transform([](auto v) {
                            return std::get<char>(v);
                          });
        std::vector<char> char_values(char_range.begin(), char_range.end());
        auto shape = mlir::RankedTensorType::get(
            {static_cast<long>(values.size())}, builder.getI8Type());
        attr = mlir::DenseElementsAttr::get<char>(shape, char_values);
        break;
      }
      default:
        return mlir::emitError(loc(expr->token.span),
                               "unsupported array element type");
      }
      return builder
          .create<mlir::lang::ConstantOp>(loc(expr->token.span), attr.getType(),
                                          attr)
          .getResult();
    }

    llvm::SmallVector<mlir::Value, 4> values;
    for (auto &e : expr->elements) {
      auto value = langGen(e.get());
      if (failed(value)) {
        return mlir::failure();
      }
      values.push_back(value.value());
    }
    auto base_type = values[0].getType();
    auto array_type = mlir::lang::ArrayType::get(
        builder.getContext(), base_type,
        builder.getI64IntegerAttr(values.size()), builder.getBoolAttr(false));
    return builder
        .create<mlir::lang::ArrayOp>(loc(expr->token.span), array_type, values)
        .getResult();
  }

  mlir::FailureOr<mlir::Value> langGen(ComptimeExpr *expr) {
    AnnotatingListener listener(builder, builder.getStringAttr("comptime"),
                                builder.getBoolAttr(true));
    auto value = langGen(expr->expr.get());
    if (failed(value)) {
      return mlir::failure();
    }
    return value;
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
    if (!mlir::isa<mlir::lang::SliceType, mlir::lang::ArrayType,
                   mlir::TensorType>(base_type)) {
      return mlir::emitError(loc(expr->token.span),
                             "base is not a slice type, got ")
             << base_type;
    }
    auto func = getSpecialMethod(loc(expr->token.span), "index",
                                 {base_type, index_type});

    auto call_op = builder.create<mlir::lang::CallOp>(
        loc(expr->token.span), func.value(),
        mlir::ValueRange{base.value(), index.value()});
    return call_op.getResult(0);
  }

  mlir::FailureOr<mlir::Value> langGen(UnaryExpr *expr) {
    auto operand = langGen(expr->operand.get());
    if (failed(operand)) {
      return mlir::failure();
    }
    auto func =
        getSpecialMethod(loc(expr->token.span),
                         expr->op == Operator::Sub ? "neg" : "logical_not",
                         {operand.value().getType()});
    if (mlir::failed(func)) {
      return mlir::failure();
    }
    auto call_op = builder.create<mlir::lang::CallOp>(
        loc(expr->token.span), func.value(), mlir::ValueRange{operand.value()});
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
    auto struct_type =
        mlir::lang::StructType::get(builder.getContext(), struct_name, types);
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
        if (last_stmt->kind() == AstNodeKind::ExprStmt) {
          auto expr_stmt = last_stmt->as<ExprStmt>();
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
        if (last_stmt->kind() == AstNodeKind::ExprStmt) {
          auto expr_stmt = last_stmt->as<ExprStmt>();
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
    terminated_by_return =
        isTerminatedBy<AstNodeKind::ReturnExpr>(expr->then_block.get()) &&
        isTerminatedBy<AstNodeKind::ReturnExpr>(expr->else_block.value().get());
    // 5. Both then and else are terminated by a yield in all paths
    // (different from above)
    terminated_by_yield =
        isTerminatedBy<AstNodeKind::YieldExpr>(expr->then_block.get()) &&
        isTerminatedBy<AstNodeKind::YieldExpr>(expr->else_block.value().get());
    return non_terminating || terminated_by_return || terminated_by_yield;
  }

  template <AstNodeKind kind> bool isTerminatedBy(BlockExpression *block) {
    // If the block is empty, it doesn't always return.
    if (block->statements.empty()) {
      return false;
    }

    for (auto &stmt : block->statements) {
      if (stmt->kind() == AstNodeKind::ExprStmt) {
        auto expr = stmt->as<ExprStmt>();
        switch (expr->expr->kind()) {
        case kind:
          // Found a guaranteed return; all paths hitting this statement return.
          // No further checks needed because any code after this is
          // unreachable.
          return true;

        case AstNodeKind::IfExpr: {
          auto if_expr = expr->expr->as<IfExpr>();
          bool thenReturns = isTerminatedBy<kind>(if_expr->then_block.get());
          bool elseReturns =
              if_expr->else_block.has_value() &&
              isTerminatedBy<kind>(if_expr->else_block.value().get());

          if (thenReturns && elseReturns) {
            // The if by itself ensures return on every path through it
            return true;
          } else {
            // This if doesn't guarantee a return on all paths.
            // Execution might continue to next statements, so keep checking.
            break;
          }
        }
        case AstNodeKind::BlockExpression: {
          auto inner_block = expr->expr->as<BlockExpression>();
          if (isTerminatedBy<kind>(inner_block)) {
            return true;
          } else {
            // Inner block doesn't guarantee return, keep checking next
            // statements.
            break;
          }
        }
        default:
          // Just a normal statement that doesn't guarantee return.
          // Keep scanning subsequent statements.
          break;
        }
      } else {
        // Non-expression statements won't guarantee return.
        continue;
      }
    }
    // If we finish scanning all statements without confirming that every path
    // returns, then the block does not always return.
    return false;
  }

  bool allPathsTerminate(BlockExpression *block) {
    // If the block is empty, there's no guaranteed return => not all paths
    // terminate.
    if (block->statements.empty()) {
      return false;
    }

    for (auto &stmt : block->statements) {
      if (stmt->kind() != AstNodeKind::ExprStmt)
        continue;

      auto expr = stmt->as<ExprStmt>();
      switch (expr->expr->kind()) {
      case AstNodeKind::ReturnExpr:
      case AstNodeKind::YieldExpr:
        // Found a guaranteed termination.
        return true;

      case AstNodeKind::IfExpr: {
        auto if_expr = expr->expr->as<IfExpr>();
        bool thenTerm = allPathsTerminate(if_expr->then_block.get());
        bool elseTerm =
            if_expr->else_block.has_value()
                ? allPathsTerminate(if_expr->else_block.value().get())
                :
                // No else block means if condition fails, we just continue
                // after the if, so no guaranteed termination on that path yet:
                false;

        // For the if to guarantee termination on all paths, both branches must
        // terminate. If not both terminate, we can’t confirm termination here,
        // we must continue checking subsequent statements.
        if (thenTerm && elseTerm) {
          return true;
        } else {
          // Not guaranteed termination yet; continue scanning later statements.
        }
        break;
      }

      case AstNodeKind::BlockExpression: {
        auto inner_block = expr->expr->as<BlockExpression>();
        if (allPathsTerminate(inner_block)) {
          // The inner block itself guarantees termination,
          // so we have a guaranteed termination here.
          return true;
        } else {
          // No guaranteed termination yet, continue scanning.
        }
        break;
      }
      default:
        // Just a normal statement; doesn't guarantee termination, continue
        // scanning.
        break;
      }
    }
    // Reached the end without finding a guaranteed terminating path => not all
    // paths terminate.
    return false;
  }

  bool isNonTerminatingBlock(BlockExpression *block) {
    return !allPathsTerminate(block);
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
    auto func = getSpecialMethod(loc(expr->token.span),
                                 op_fns[static_cast<int>(expr->op)],
                                 {lhs->getType(), rhs->getType()});
    if (failed(func)) {
      return mlir::failure();
    }

    auto call_op = builder.create<mlir::lang::CallOp>(
        loc(expr->token.span), func.value(),
        mlir::ValueRange{lhs.value(), rhs.value()});
    return call_op.getResult(0);
  }

  mlir::FailureOr<mlir::lang::FuncOp>
  getSpecialMethod(mlir::Location loc, llvm::StringRef method_name,
                   llvm::ArrayRef<mlir::Type> arg_types) {
    std::string fn_name = "";
    llvm::raw_string_ostream stream(fn_name);
    stream << method_name << "__";
    for (auto &arg_type : arg_types) {
      stream << arg_type;
      if (&arg_type != &arg_types.back())
        stream << "_";
    }
    fn_name = stream.str();

    if (!function_map.count(fn_name)) {
      auto err = mlir::emitError(loc)
                 << "operation `" << method_name << "` not implemented for ";
      for (auto &arg_type : arg_types) {
        err << arg_type;
        if (&arg_type != &arg_types.back())
          err << ", ";
        else
          err << "\n";
      }
      return err;
    }
    return function_map[fn_name];
  }

  mlir::FailureOr<mlir::Value> langGen(AssignStatement *expr) {
    mlir::FailureOr<mlir::Value> lhs = mlir::failure();
    if (expr->lhs->kind() == AstNodeKind::IdentifierExpr) {
      lhs = langGen(expr->lhs->as<IdentifierExpr>(), false);
    } else if (expr->lhs->kind() == AstNodeKind::FieldAccessExpr) {
      lhs = langGen(expr->lhs->as<FieldAccessExpr>());
    } else if (expr->lhs->kind() == AstNodeKind::IndexExpr) {
      lhs = langGen(expr->lhs->as<IndexExpr>());
    } else {
      return mlir::emitError(loc(expr->token.span),
                             "unsupported lhs expression");
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
    if (var) {
      // for mutable values and structs, by default return the value (automatic
      // deref)
      if (get_value && mlir::isa<mlir::MemRefType>(var.getType())) {
        auto deref_op =
            builder.create<mlir::lang::DerefOp>(loc(expr->token.span), var);
        return deref_op.getResult();
      }
      return var;
    }
    auto type = type_table.lookup(expr->name);
    if (type) {
      return builder
          .create<mlir::lang::TypeConstOp>(
              loc(expr->token.span),
              mlir::lang::TypeValueType::get(builder.getContext(), type), type)
          .getResult();
    }
    return mlir::emitError(loc(expr->token.span),
                           "undeclared variable " + expr->name);
  }

  bool isGenericFunction(Function *func) {
    for (auto &param : func->decl->parameters) {
      if (param->is_comptime) {
        return true;
      }
    }
    return false;
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
    bool is_generic = false;
    for (auto &arg : expr->arguments) {
      auto arg_value = langGen(arg.get());
      if (failed(arg_value)) {
        return mlir::failure();
      }
      args.push_back(arg_value.value());
      arg_types.push_back(arg_value.value().getType());
      is_generic |= mlir::isa<mlir::lang::TypeValueType>(arg_types.back());
    }

    auto func_name =
        is_generic ? str(expr->callee) : str(mangle(expr->callee, arg_types));

    // if return type is a struct type, then create a struct instance
    // and pass it as an argument
    if (function_map.lookup(func_name)) {
      auto func_type = function_map[func_name].getFunctionType();
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

    if (expr->callee == "print")
      return printCall(args, expr);

    if (is_generic) {
      auto func_decl = function_table.lookup(func_name);
      if (!func_decl) {
        return mlir::emitError(loc(expr->token.span),
                               "function not found in function table " +
                                   func_name);
      }
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(
          builder.getInsertionBlock()->getParentOp());
      auto new_func = instantiateGenericFunction(func_decl, args);
      if (failed(new_func)) {
        return mlir::emitError(loc(expr->token.span),
                               "failed to instantiate generic function");
      }
      func_name = str(mangle(func_name, new_func->getArgumentTypes()));
      if (function_map.count(func_name)) {
        // function already exists, erase the new function
        // (TODO: need to do this withouth creating the function)
        new_func->erase();
      } else {
        new_func->setName(func_name);
        function_map[func_name] = new_func.value();
      }
    }
    if (!function_map.count(func_name))
      return mlir::emitError(loc(expr->token.span),
                             "function " + func_name + " not found");
    auto func = function_map[func_name];

    // call function
    auto call_op = builder.create<mlir::lang::CallOp>(
        loc(expr->token.span), func, mlir::ValueRange(args));
    if (call_op.getNumResults() == 0) {
      return mlir::success(mlir::Value());
    }
    return call_op.getResult(0);
  }

  mlir::FailureOr<mlir::Value> printCall(mlir::ArrayRef<mlir::Value> args,
                                         CallExpr *expr) {
    if (args.size() < 1)
      return mlir::emitError(loc(expr->token.span),
                             "print function expects 1 argument");
    auto arg0 = expr->arguments[0].get();
    if (arg0->kind() != AstNodeKind::LiteralExpr ||
        arg0->as<LiteralExpr>()->type != LiteralExpr::LiteralType::String)
      return mlir::emitError(loc(expr->token.span),
                             "print function expects a string argument");
    auto format_str = std::get<std::string>(arg0->as<LiteralExpr>()->value);
    format_str = format_str.substr(1, format_str.size() - 2) + '\n' + '\0';
    args.front().getDefiningOp()->erase();
    mlir::ValueRange rest_args = llvm::ArrayRef(args).drop_front();
    builder.create<mlir::lang::PrintOp>(loc(expr->token.span), format_str,
                                        rest_args);
    return mlir::success(mlir::Value());
  }

  mlir::FailureOr<mlir::lang::FuncOp>
  instantiateGenericFunction(Function *func,
                             llvm::SmallVector<mlir::Value, 4> &args) {
    // Step 1. Identify the generic parameters and their types
    for (const auto &it : llvm::enumerate(args)) {
      if (mlir::isa<mlir::lang::TypeValueType>(it.value().getType())) {
        auto type = mlir::cast<mlir::lang::TypeValueType>(it.value().getType())
                        .getType();
        auto pattern = func->decl->parameters[it.index()]->pattern.get();
        // assume pattern is an identifier for now
        auto pattern_name = str(pattern->as<IdentifierPattern>()->name);
        // update type table
        type_table.insert(pattern_name, type);
      }
    }
    auto func_op = langGen(func);
    if (llvm::failed(func_op)) {
      return mlir::failure();
    }
    // auto inputs_type =
    //     llvm::to_vector<4>(func_op->getFunctionType().getInputs());
    // auto r_args = llvm::reverse(args);
    // remove generic args from the func_op
    // for (const auto [idx, arg] : llvm::enumerate(r_args)) {
    //   if (mlir::isa<mlir::lang::TypeValueType>(arg.getType())) {
    //     func_op->getCallableRegion()->eraseArgument(args.size() - idx - 1);
    //     args.erase(args.end() - idx - 1);
    //     inputs_type.erase(inputs_type.end() - idx - 1);
    //   }
    // }
    // auto function_type =
    //     mlir::FunctionType::get(builder.getContext(), inputs_type,
    //                             func_op->getFunctionType().getResults());
    // func_op->setType(function_type);
    return *func_op;
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
      // auto call_expr =
      // std::get<std::unique_ptr<CallExpr>>(expr->field).get();
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
        loc(type->token.span),
        mlir::lang::TypeValueType::get(builder.getContext(), mlir_type.value()),
        mlir_type.value());
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
      mlir::TypedAttr attr = builder.getIntegerAttr(
          builder.getIntegerType(64), std::get<int>(literal->value));
      auto field_value = builder
                             .create<mlir::lang::ConstantOp>(
                                 loc(literal->token.span), attr.getType(), attr)
                             .getResult();
      return field_value;
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

  llvm::FailureOr<PatternBindings> destructurePattern(Pattern *pattern,
                                                      Type *type) {
    PatternBindings result;
    if (!pattern)
      return result;

    switch (pattern->kind()) {
    case AstNodeKind::IdentifierPattern: {
      auto p = static_cast<IdentifierPattern *>(pattern);
      if (!p->name.empty()) {
        result.emplace_back(p->name, type);
      }
      break;
    }

    case AstNodeKind::WildcardPattern:
    case AstNodeKind::LiteralPattern:
    case AstNodeKind::ExprPattern:
    case AstNodeKind::RangePattern:
      // These patterns do not bind variables
      break;

    case AstNodeKind::TuplePattern: {
      auto p = static_cast<TuplePattern *>(pattern);
      auto tuple_t = type->as<TupleType>();
      if (!tuple_t) {
        return mlir::emitError(loc(pattern->token.span),
                               "Pattern and type mismatch: Expected a tuple "
                               "type");
      }
      if (p->elements.size() != tuple_t->elements.size()) {
        return mlir::emitError(loc(pattern->token.span),
                               "Tuple pattern length does not match tuple type "
                               "length");
      }
      for (size_t i = 0; i < p->elements.size(); ++i) {
        auto sub_bindings = destructurePattern(p->elements[i].get(),
                                               tuple_t->elements[i].get());
        result.insert(result.end(), sub_bindings->begin(), sub_bindings->end());
      }
      break;
    }

    case AstNodeKind::StructPattern: {
      auto p = static_cast<StructPattern *>(pattern);
      auto struct_t = type->as<StructType>();
      if (!struct_t) {
        return mlir::emitError(loc(pattern->token.span),
                               "Pattern and type mismatch: Expected a struct "
                               "type");
      }

      llvm::StringRef struct_name =
          p->name && !p->name->empty() ? *p->name : struct_t->name;
      for (auto &field_var : p->fields) {
        if (std::holds_alternative<std::unique_ptr<PatternField>>(field_var)) {
          auto &f = std::get<std::unique_ptr<PatternField>>(field_var);
          Type *field_type = getStructFieldType(struct_name, f->name);
          if (!field_type) {
            return mlir::emitError(loc(f->token.span),
                                   "Field type not found for field '" +
                                       f->name + "'");
          }
          if (f->pattern.has_value() && f->pattern.value()) {
            auto sub_bindings =
                destructurePattern(f->pattern.value().get(), field_type);
            result.insert(result.end(), sub_bindings->begin(),
                          sub_bindings->end());
          } else {
            result.emplace_back(f->name, field_type);
          }
        } else {
          // RestPattern inside a struct pattern does not produce named
          // variables by default unless specifically handled.
        }
      }
      break;
    }

    case AstNodeKind::SlicePattern: {
      auto p = static_cast<SlicePattern *>(pattern);
      auto slice_t = type->as<SliceType>();
      auto array_t = type->as<ArrayType>();
      Type *elem_type = nullptr;
      if (slice_t) {
        elem_type = slice_t->base.get();
      } else if (array_t) {
        elem_type = array_t->base.get();
      } else {
        return mlir::emitError(loc(pattern->token.span),
                               "Pattern and type mismatch: Expected a slice or "
                               "array type");
      }
      for (auto &elem_pattern : p->elements) {
        auto sub_bindings = destructurePattern(elem_pattern.get(), elem_type);
        result.insert(result.end(), sub_bindings->begin(), sub_bindings->end());
      }
      break;
    }

    case AstNodeKind::OrPattern: {
      auto p = static_cast<OrPattern *>(pattern);
      if (!p->patterns.empty()) {
        // Ideally, we should verify that all patterns produce the same
        // bindings.
        result = destructurePattern(p->patterns[0].get(), type).value();
      }
      break;
    }

    case AstNodeKind::VariantPattern: {
      auto p = static_cast<VariantPattern *>(pattern);
      auto enum_t = type->as<EnumType>();
      if (!enum_t) {
        return mlir::emitError(
            loc(pattern->token.span),
            "Pattern and type mismatch: Expected an enum type");
      }
      // Retrieve variant type info from the enum:
      auto variant_type = getEnumVariantType(enum_t->name, p->name);
      // getEnumVariantType should return either:
      // - A tuple type for tuple variants
      // - A struct type for struct variants
      // - A single-field type for single-value variants
      // - nullptr if the variant has no fields
      if (p->field.has_value()) {
        auto &var_field = p->field.value();
        if (!variant_type) {
          return mlir::emitError(loc(pattern->token.span),
                                 "Variant '" + p->name +
                                     "' does not have fields to destructure");
        }

        if (std::holds_alternative<std::unique_ptr<TuplePattern>>(var_field)) {
          auto &tp = std::get<std::unique_ptr<TuplePattern>>(var_field);
          auto variant_tuple_t = variant_type->as<TupleType>();
          if (!variant_tuple_t) {
            return mlir::emitError(loc(pattern->token.span),
                                   "Variant '" + p->name +
                                       "' is not a tuple variant");
          }
          if (tp->elements.size() != variant_tuple_t->elements.size()) {
            return mlir::emitError(loc(pattern->token.span),
                                   "Tuple pattern length does not match "
                                   "variant tuple type length");
          }
          for (size_t i = 0; i < tp->elements.size(); ++i) {
            auto sub_bindings = destructurePattern(
                tp->elements[i].get(), variant_tuple_t->elements[i].get());
            result.insert(result.end(), sub_bindings->begin(),
                          sub_bindings->end());
          }
        } else if (std::holds_alternative<std::unique_ptr<StructPattern>>(
                       var_field)) {
          auto &sp = std::get<std::unique_ptr<StructPattern>>(var_field);
          auto variant_struct_t = variant_type->as<StructType>();
          if (!variant_struct_t) {
            return mlir::emitError(loc(pattern->token.span),
                                   "Variant '" + p->name +
                                       "' is not a struct variant");
          }

          llvm::StringRef variant_name = variant_struct_t->name;
          for (auto &field_var : sp->fields) {
            if (std::holds_alternative<std::unique_ptr<PatternField>>(
                    field_var)) {
              auto &f = std::get<std::unique_ptr<PatternField>>(field_var);
              Type *field_type = getStructFieldType(variant_name, f->name);
              if (!field_type) {
                return mlir::emitError(loc(f->token.span),
                                       "Field type not found for field '" +
                                           f->name + "' in variant '" +
                                           p->name + "'");
              }
              if (f->pattern.has_value() && f->pattern.value()) {
                auto sub_bindings =
                    destructurePattern(f->pattern.value().get(), field_type);
                result.insert(result.end(), sub_bindings->begin(),
                              sub_bindings->end());

              } else {
                result.emplace_back(f->name, field_type);
              }
            } else {
              // If there's a rest pattern in the variant struct pattern,
              // handle it as in struct patterns.
            }
          }
        }
      }
      // If the variant has no fields or the field is not provided, no bindings.
      break;
    }

    case AstNodeKind::RestPattern: {
      // auto p = static_cast<RestPattern *>(pattern);
      // Rest patterns do not bind named variables by default unless you define
      // semantics for it. If p->name has a value, you could decide how to
      // handle it. Without a defined semantic, we do nothing.
      break;
    }

    default:
      // No other patterns need handling
      break;
    }

    return result;
  }

  Type *getStructFieldType(const llvm::StringRef struct_name,
                           const llvm::StringRef field_name) {
    // lookup struct table for struct_name
    auto struct_decl = struct_table.lookup(struct_name);
    if (!struct_decl) {
      return nullptr;
    }
    // lookup field in struct_decl
    for (auto &field : struct_decl->fields) {
      if (field->name == field_name) {
        return field->type.get();
      }
    }
    return nullptr;
  }

  // TODO: Implement this
  Type *getEnumVariantType(const llvm::StringRef enum_name,
                           const llvm::StringRef variant_name) {
    // // lookup enum table for enum_name
    // auto enum_decl = context->enum_table.lookup(enum_name);
    // if (!enum_decl) {
    //   return nullptr;
    // }
    // // lookup variant in enum_decl
    // for (auto &variant : enum_decl->variants) {
    //   if (variant->name == variant_name) {
    //     return variant->type.get();
    //   }
    // }
    return nullptr;
  }
};

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> langGen(mlir::MLIRContext &context,
                                          Program *program, Context &ctx) {
  return LangGenImpl(context, ctx).langGen(program);
}
