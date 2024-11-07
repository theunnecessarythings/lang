#include "MLIRGen.h"
#include "ast.hpp"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"

#include "lexer.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <variant>

using llvm::ArrayRef;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;

class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp mlirGen(Program *program) {
    the_module = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto &f : program->items) {
      if (dynamic_cast<Function *>(f.get())) {
        auto func = mlirGen(dynamic_cast<Function *>(f.get()));
        if (failed(func)) {
          the_module.emitError("error in function generation");
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

  llvm::FailureOr<mlir::func::FuncOp> mlirGen(Function *func) {
    ScopedHashTableScope<StringRef, mlir::Value> varScope(symbol_table);

    builder.setInsertionPointToEnd(the_module.getBody());

    auto param_types = mlirGen(func->decl->parameters);

    mlir::TypeRange return_types =
        func->decl->return_type->kind() == AstNodeKind::PrimitiveType &&
                static_cast<PrimitiveType *>(func->decl->return_type.get())
                        ->type_kind == PrimitiveType::PrimitiveTypeKind::Void
            ? mlir::TypeRange()
            : mlir::TypeRange(mlirGen(func->decl->return_type.get()).value());

    auto func_type = builder.getFunctionType(param_types.value(), return_types);
    auto func_op = builder.create<mlir::func::FuncOp>(
        loc(func->token.span), func->decl->name, func_type);

    function_map[func->decl->name] = func_op;

    auto entry_block = func_op.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);

    // declare function parameters
    if (failed(declare_parameters(func->decl->parameters,
                                  entry_block->getArguments()))) {
      emitError(loc(func->token.span), "parameter declaration error");
      func_op.erase();
      return mlir::failure();
    }

    if (failed(mlirGen(func->body.get()))) {
      emitError(loc(func->token.span), "function body generation error");
      func_op.erase();
      return mlir::failure();
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

  llvm::LogicalResult mlirGen(BlockExpression *block) {
    ScopedHashTableScope<StringRef, mlir::Value> varScope(symbol_table);
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
        emitError(loc, "type mismatch in variable declaration");
        return mlir::failure();
      }
    }

    if (failed(declare(var_name, init_value.value()))) {
      emitError(loc, "redeclaration of variable");
      return mlir::failure();
    }
    return init_value;
  }

  llvm::FailureOr<mlir::Type> mlirGen(Type *type) {
    if (auto t = dynamic_cast<PrimitiveType *>(type)) {
      if (t->type_kind == PrimitiveType::PrimitiveTypeKind::I32) {
        return builder.getIntegerType(32);
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::F32) {
        return builder.getF32Type();
      } else if (t->type_kind == PrimitiveType::PrimitiveTypeKind::Void) {
        return builder.getNoneType();
      } else {
        emitError(loc(t->token.span), "unsupported type");
        return mlir::failure();
      }
    }
    the_module.emitError("unsupported type");
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
    }
    emitError(loc(expr->token.span),
              "unsupported expression, " + to_string(expr->kind()));
    return mlir::failure();
  }

  llvm::FailureOr<mlir::Value> mlirGen(CallExpr *callExpr) {
    // Assume the callee is an identifier expression
    auto &func_name =
        static_cast<IdentifierExpr *>(callExpr->callee.get())->name;
    if (func_name == "print") {
      if (failed(mlirGenPrintCall(callExpr))) {
        emitError(loc(callExpr->token.span), "error in print call");
        return mlir::failure();
      }
      return mlir::success(mlir::Value());
    }
    // Step 1: Generate argument values
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

    // Step 2: Look up the function

    // auto functionSymbol = symbol_table.lookup(functionName);
    auto func_symbol = function_map.find(func_name);
    if (func_symbol == function_map.end()) {
      emitError(loc(callExpr->token.span), "Function not found: ") << func_name;
      return mlir::failure();
    }
    auto func_op = func_symbol->second;
    auto func_type = func_op.getFunctionType();

    // Step 3: Create the `func.call` operation
    mlir::Type resultType = func_type.getResult(0); // Single result
    auto callOp = builder.create<mlir::func::CallOp>(
        loc(callExpr->token.span), func_name, resultType, argumentValues);

    // Step 4: Return the result of the call
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

      // If 'else' block does not end with a return, branch to the merge block
      if (elseBlock->empty() ||
          !elseBlock->back().mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
        builder.setInsertionPointToEnd(elseBlock);
        builder.create<mlir::cf::BranchOp>(loc, mergeBlock);
      }
    }

    // Continue building from the merge block
    builder.setInsertionPointToStart(mergeBlock);

    // Optionally, if this 'if' expression produces a value, you need to handle
    // SSA dominance and merge the values.

    // Since this function does not produce a value, return success without a
    // value
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
              builder.getF32FloatAttr(std::get<double>(literal->value)))
          .getResult();
    } else if (literal->type == LiteralExpr::LiteralType::String) {
      return builder
          .create<mlir::arith::ConstantOp>(
              loc(literal->token.span),
              builder.getStringAttr(std::get<std::string>(literal->value)))
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
      emitError(loc(binary->token.span), "one of the operands is null");
      return mlir::failure();
    }
    auto lhs = lhs_v.value();
    auto rhs = rhs_v.value();
    mlir::Value op = nullptr;
    // if both operands are integers
    if (mlir::isa<mlir::IntegerType>(lhs.getType()) &&
        mlir::isa<mlir::IntegerType>(rhs.getType())) {
      op = integer_ops(binary, lhs, rhs).value();
    } else if (mlir::isa<mlir::Float32Type>(lhs.getType()) &&
               mlir::isa<mlir::Float32Type>(rhs.getType())) {
      op = floating_ops(binary, lhs, rhs).value();
    } else {
      emitError(loc(binary->token.span), "unsupported operand types");
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
    } else {
      the_module.emitError("unsupported binary operator");
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
