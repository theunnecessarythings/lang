#include "comptime_eval_pass.hpp"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitCPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include "passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"

#define DEBUG_TYPE "comptime-analysis"

namespace mlir {
#define GEN_PASS_DEF_COMPTIMEANALYSIS
#include "dialect/LangPasses.h.inc"
} // namespace mlir

//===----------------------------------------------------------------------===//
// ComptimeCodeAnalysisPass
//===----------------------------------------------------------------------===//
mlir::LogicalResult ComptimeStateAnalysis::initialize(mlir::Operation *top) {
  top->walk([&](mlir::Operation *op) {
    // Set initial result states.
    bool isConstLike = op->hasTrait<mlir::OpTrait::ConstantLike>();
    for (mlir::Value result : op->getResults()) {
      // Get or create a ComptimeState for this result.
      ComptimeState *state = getOrCreate<ComptimeState>(result);
      auto new_state = isConstLike ? ComptimeState::State::Comptime
                                   : ComptimeState::State::Uninitialized;
      mlir::ChangeResult changed =
          state->join(ComptimeState(new_state, result));
      propagateIfChanged(state, changed);
    }

    // Visit the operation to handle special cases.
    if (failed(visit(getProgramPointAfter(op)))) {
      llvm::errs() << "Failed to handle operation: " << op->getName() << "\n";
    }

    // For block arguments, we also default to runtime.
    // If you have special arguments known to be comptime, do similar logic
    // here.
  });
  return mlir::success();
}

llvm::LogicalResult ComptimeStateAnalysis::visit(mlir::ProgramPoint *point) {
  if (point->isBlockStart())
    return llvm::success();
  // llvm::errs() << "Visiting program point: " << *point << "\n";
  auto op = point->getPrevOp();
  if (!op)
    return llvm::success();
  return handleOperation(op);
}

llvm::LogicalResult
ComptimeStateAnalysis::handleOperation(mlir::Operation *op) {
  LLVM_DEBUG({
    llvm::errs() << "Handling operation: " << op->getName() << "\n";
    if (mlir::isa<mlir::func::FuncOp>(op)) {
      llvm::errs() << " Function: "
                   << mlir::cast<mlir::func::FuncOp>(op).getName() << "\n";
    }

    if (mlir::isa<mlir::func::CallOp>(op)) {
      llvm::errs() << " Call: " << *op << "\n";
    }
  });
  // Create dependencies on the operand states, so if they change, we revisit.
  mlir::SmallVector<const ComptimeState *, 4> operandStates;
  operandStates.reserve(op->getNumOperands());

  // We pick the "after" program point of this operation as the dependent, so
  // changes in operand states cause re-analysis of the operation after it.
  mlir::ProgramPoint *afterOpPoint = getProgramPointAfter(op);
  for (mlir::Value operand : op->getOperands()) {
    const ComptimeState *st =
        getOrCreateFor<ComptimeState>(afterOpPoint, operand);
    operandStates.push_back(st);
  }

  mlir::SmallVector<ComptimeState *, 4> resultStates;
  resultStates.reserve(op->getNumResults());
  for (mlir::Value result : op->getResults()) {
    ComptimeState *res = getOrCreate<ComptimeState>(result);
    resultStates.push_back(res);
  }

  // If no results, nothing to update.
  if (op->getNumResults() == 0)
    return llvm::success();

  // Check if all operands are comptime.
  bool allOperandsComptime =
      llvm::all_of(operandStates, [](const ComptimeState *st) {
        return st && st->getState() == ComptimeState::State::Comptime;
      });

  if (!allOperandsComptime) {
    // Not all operands are comptime → results must be runtime.
    setAllResultsRuntime(resultStates, op->getResults());
    return llvm::success();
  }

  // All operands are comptime. Now check if the op can be computed at compile
  // time:
  // 1. If ConstantLike, trivially comptime.
  if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
    setAllResultsComptime(resultStates, op->getResults());
    return llvm::success();
  }

  // 2. Check memory effects. If op has writes/allocs/frees, it's not pure.
  if (auto memInterface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    mlir::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
    memInterface.getEffects(effects);
    bool hasSideEffects =
        llvm::any_of(effects, [](mlir::MemoryEffects::EffectInstance &eff) {
          return llvm::isa<mlir::MemoryEffects::Write,
                           mlir::MemoryEffects::Allocate,
                           mlir::MemoryEffects::Free>(eff.getEffect());
        });
    if (hasSideEffects) {
      // Has side effects → runtime.
      setAllResultsRuntime(resultStates, op->getResults());
      return llvm::success();
    }
  }

  if (auto callOp = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
    // 3. External functions are runtime.
    return visitCallOperation(callOp, resultStates);
  }

  setAllResultsComptime(resultStates, op->getResults());
  return llvm::success();
}

// A call to a externally-defined callable has unknown predecessors.
bool isExternalCallable(mlir::Operation *op, mlir::ModuleOp module) {
  // A callable outside the analysis scope is an external callable.
  if (!module->isAncestor(op))
    return true;
  // Otherwise, check if the callable region is defined.
  if (auto callable = mlir::dyn_cast<mlir::CallableOpInterface>(op))
    return !callable.getCallableRegion();
  return false;
}

bool functionHasSideEffects(mlir::FunctionOpInterface funcOp) {
  bool hasSideEffects = false;
  funcOp.walk([&](mlir::Operation *op) {
    // Check for external calls or side effects
    if (auto callOp = llvm::dyn_cast<mlir::CallOpInterface>(op)) {
      auto module = funcOp->getParentOfType<mlir::ModuleOp>();
      auto symbol = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(
          callOp.getCallableForCallee());
      auto callee = module.lookupSymbol<mlir::FunctionOpInterface>(symbol);
      if (!callee || isExternalCallable(callee, module)) {
        hasSideEffects = true;
        return mlir::WalkResult::interrupt();
      }
    }
    // Check for memory or other side effects
    if (auto effectInterface =
            llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
      if (!effectInterface.hasNoEffect()) {
        hasSideEffects = true;
        return mlir::WalkResult::interrupt();
      }
    }
    // else if (!op->hasTrait<mlir::OpTrait::IsTerminator>()) {
    //   // Conservatively assume unknown ops have side effects
    //   hasSideEffects = true;
    //   return mlir::WalkResult::interrupt();
    // }

    return mlir::WalkResult::advance();
  });
  return hasSideEffects;
}

mlir::LogicalResult ComptimeStateAnalysis::visitCallOperation(
    mlir::CallOpInterface call_op,
    mlir::SmallVector<ComptimeState *, 4> &resultStates) {
  auto module = call_op->getParentOfType<mlir::ModuleOp>();
  if (!module) {
    return llvm::success();
  }

  // mlir::Operation *callableOp = call_op.resolveCallableInTable(&symbolTable);
  auto symbol = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(
      call_op.getCallableForCallee());
  if (!symbol) {
    setAllResultsRuntime(resultStates, call_op->getResults());
    return llvm::success();
  }
  mlir::Operation *callableOp = module.lookupSymbol(symbol);
  if (!callableOp) {
    setAllResultsRuntime(resultStates, call_op->getResults());
    return llvm::success();
  }

  if (isExternalCallable(callableOp, module)) {
    setAllResultsRuntime(resultStates, call_op->getResults());
    return llvm::success();
  }

  // 4. All the operands are comptime and if the callee is comptime, we can mark
  // the results as comptime.
  // Set block arguments to comptime and propagate changes.
  auto func_op = mlir::dyn_cast<mlir::FunctionOpInterface>(callableOp);
  if (!func_op) {
    setAllResultsRuntime(resultStates, call_op->getResults());
    return llvm::success();
  }

  mlir::ProgramPoint *calleeEntryPoint =
      getProgramPointBefore(&func_op.front());
  mlir::ProgramPoint *afterCallPoint = getProgramPointAfter(call_op);
  auto &block = func_op.getFunctionBody().front();
  auto *terminator = block.getTerminator();
  for (auto resultVal : terminator->getResults()) {
    getOrCreateFor<ComptimeState>(afterCallPoint, resultVal);
  }

  for (auto &funcArg : block.getArguments()) {
    getOrCreateFor<ComptimeState>(calleeEntryPoint, funcArg);
    auto *argState = getOrCreate<ComptimeState>(funcArg);
    mlir::ChangeResult changed =
        argState->join(ComptimeState(ComptimeState::State::Comptime, funcArg));
    propagateIfChanged(argState, changed);
  }

  bool allCalleeReturnsComptime = true;
  for (auto resultVal : terminator->getResults()) {
    if (auto *retState = getOrCreate<ComptimeState>(resultVal)) {
      if (retState->getState() != ComptimeState::State::Comptime) {
        allCalleeReturnsComptime = false;
        break;
      }
    } else {
      allCalleeReturnsComptime = false;
      break;
    }
  }

  if (functionHasSideEffects(func_op)) {
    setAllResultsRuntime(resultStates, call_op->getResults());
    return llvm::success();
  }

  if (allCalleeReturnsComptime) {
    setAllResultsComptime(resultStates, call_op->getResults());
  }

  return llvm::success();
}

/// Helper to set all result states to comptime and propagate changes.
void ComptimeStateAnalysis::setAllResultsComptime(
    mlir::ArrayRef<ComptimeState *> resultStates, mlir::ValueRange results) {
  for (auto [state, res] : llvm::zip(resultStates, results)) {
    mlir::ChangeResult changed =
        state->join(ComptimeState(ComptimeState::State::Comptime, res));
    if (changed == mlir::ChangeResult::Change)
      propagateIfChanged(state, changed);
  }
}

/// Helper to set all result states to runtime and propagate changes.
void ComptimeStateAnalysis::setAllResultsRuntime(
    mlir::ArrayRef<ComptimeState *> resultStates, mlir::ValueRange results) {
  for (auto [state, res] : llvm::zip(resultStates, results)) {
    mlir::ChangeResult changed =
        state->join(ComptimeState(ComptimeState::State::Runtime, res));
    if (changed == mlir::ChangeResult::Change)
      propagateIfChanged(state, changed);
  }
}

namespace {

struct ComptimeEvalPass
    : public mlir::PassWrapper<ComptimeEvalPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComptimeEvalPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void printOp(const mlir::Operation *op, mlir::raw_ostream &os) {
  if (auto func = mlir::dyn_cast<mlir::lang::FuncOp>(op)) {
    llvm::errs() << func.getName() << "\n";
  } else {
    llvm::errs() << *op << "\n";
  }
}

bool isComptimeOperation(mlir::Operation *op, mlir::DataFlowSolver &solver) {
  auto is_comptime = op->getAttrOfType<mlir::BoolAttr>("comptime");
  if (!is_comptime || !is_comptime.getValue())
    return false;
  auto state = solver.lookupState<ComptimeState>(op->getResult(0));
  if (!state) {
    op->emitError("Operation marked as comptime but does not have a state");
    return false;
  }
  if (state->getState() != ComptimeState::State::Comptime) {
    op->emitError("Operation marked as comptime but does not satisfy comptime "
                  "requirements");
    return false;
  }
  return true;
}

void collectComptimeOps(
    mlir::ModuleOp &module, mlir::DataFlowSolver &solver,
    mlir::SetVector<mlir::Operation *> &comptimeOps,
    llvm::MapVector<mlir::Operation *, mlir::DenseSet<mlir::Operation *>>
        &dependencyGraph) {

  std::function<void(mlir::Operation *)> collectDependencies =
      [&](mlir::Operation *op) {
        if (dependencyGraph.count(op))
          return;

        auto &deps = dependencyGraph[op];
        for (mlir::Value operand : op->getOperands()) {
          if (auto def_op = operand.getDefiningOp()) {
            deps.insert(def_op);
            // if (isComptimeOperation(defOp, solver)) {
            comptimeOps.insert(def_op);
            collectDependencies(def_op);
            // }
          }
        }

        // Handle callable operations
        if (auto call_op = mlir::dyn_cast<mlir::CallOpInterface>(op)) {
          auto symbol = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(
              call_op.getCallableForCallee());
          if (!symbol) {
            return;
          }
          auto callee = mlir::SymbolTable::lookupSymbolIn(module, symbol);
          deps.insert(callee);
          collectDependencies(callee);
        }

        // Handle callable operations
        if (auto callable_op = mlir::dyn_cast<mlir::CallableOpInterface>(op)) {
          callable_op.walk([&](mlir::CallOpInterface callOp) {
            auto symbol = llvm::dyn_cast_if_present<mlir::SymbolRefAttr>(
                callOp.getCallableForCallee());
            if (!symbol) {
              return;
            }
            auto callee = mlir::SymbolTable::lookupSymbolIn(module, symbol);
            deps.insert(callee);
            collectDependencies(callee);
          });
        }

        // Handle var decl operation
        if (auto var_decl = mlir::dyn_cast<mlir::lang::VarDeclOp>(op)) {
          if (auto init_value = var_decl.getInitValue()) {
            auto init_op = init_value.getDefiningOp();
            deps.insert(init_op);
            if (isComptimeOperation(init_op, solver)) {
              comptimeOps.insert(init_op);
              collectDependencies(init_op);
            }
          }
        }
      };

  module.walk([&](mlir::Operation *op) {
    if (isComptimeOperation(op, solver)) {
      comptimeOps.insert(op);
      collectDependencies(op);
    }
  });
}

// Generate a main function that computes all compile-time operations
void generateComptimeMainFunction(
    mlir::ModuleOp &comptime_module,
    const mlir::SmallVectorImpl<mlir::Operation *> &sorted_ops,
    mlir::IRMapping &mapping,
    mlir::DenseMap<mlir::Operation *, std::string> &result_names,
    mlir::DataFlowSolver &solver) {

  mlir::OpBuilder builder(comptime_module.getContext());
  builder.setInsertionPointToEnd(comptime_module.getBody());
  auto func_type = builder.getFunctionType({}, {});
  auto main_func = builder.create<mlir::func::FuncOp>(
      comptime_module.getLoc(), "__comptime_main", func_type);

  mlir::Block *entry_block = main_func.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  llvm::DenseMap<mlir::Operation *, mlir::Value> op_results;

  for (auto *op : sorted_ops) {
    if (mlir::isa<mlir::CallableOpInterface>(op)) {
      if (!mapping.contains(op)) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(comptime_module.getBody());
        auto cloned_func = op->clone(mapping);
        comptime_module.push_back(cloned_func);
        mapping.map(op, cloned_func);
      }
    } else {
      auto cloned_op = builder.clone(*op, mapping);
      mapping.map(op, cloned_op);

      // If this is a comptime op, store its result
      if (isComptimeOperation(op, solver)) {
        std::string result_name =
            "__result_" + std::to_string(result_names.size());
        result_names[op] = result_name;

        // Create a global variable to store the result
        auto result_type = cloned_op->getResult(0).getType();
        auto memref_type = mlir::MemRefType::get({}, result_type);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(comptime_module.getBody());
          builder.create<mlir::memref::GlobalOp>(
              op->getLoc(), result_name,
              /*sym_visibility=*/builder.getStringAttr("private"),
              /*type=*/memref_type,
              /*initial_value=*/mlir::Attribute(),
              /*constant=*/false,
              /*alignment=*/mlir::IntegerAttr());
        }

        // Store the result into the global variable
        auto ptr = builder.create<mlir::memref::GetGlobalOp>(
            op->getLoc(), memref_type, result_name);
        builder.create<mlir::memref::StoreOp>(op->getLoc(),
                                              cloned_op->getResult(0), ptr);
      }
    }
  }

  builder.create<mlir::func::ReturnOp>(comptime_module.getLoc());
}

// Lower the comptime module to LLVM dialect and execute it
llvm::LogicalResult
lowerAndExecuteComptimeModule(mlir::ModuleOp &comptime_module,
                              std::unique_ptr<mlir::ExecutionEngine> &engine) {

  // Initialize LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Set up the pass manager and lower the module
  mlir::PassManager pm(comptime_module.getContext());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::lang::createLowerToAffinePass());
  // Add passes to lower the module to LLVM dialect
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Bufferize the module
  pm.addPass(mlir::bufferization::createOneShotBufferizePass());

  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (failed(pm.run(comptime_module))) {
    return llvm::failure();
  }

  // Create the ExecutionEngine
  auto maybe_engine = mlir::ExecutionEngine::create(comptime_module);
  if (!maybe_engine) {
    return llvm::failure();
  }
  engine = std::move(maybe_engine.get());

  // Execute the main function
  auto invocation_result = engine->invokePacked("__comptime_main", {});
  if (invocation_result) {
    return llvm::failure();
  }

  return llvm::success();
}

// Retrieve the computed results by invoking getter functions
void retrieveComptimeResults(
    mlir::ExecutionEngine &engine,
    mlir::DenseMap<mlir::Operation *, std::string> &result_func_names,
    mlir::DenseMap<mlir::Operation *, int64_t> &evaluation_results) {

  for (auto &entry : result_func_names) {
    auto *op = entry.first;
    auto &func_name = entry.second;
    int64_t result;
    void *args[] = {&result};
    auto status = engine.invokePacked(func_name, args);
    if (status) {
      llvm::errs() << "Failed to execute callable: " << func_name << "\n";
      return;
    }
    evaluation_results[op] = result;
  }
}

// Create getter functions that load from global variables
void createGetterFunctions(
    mlir::ModuleOp &comptime_module,
    mlir::DenseMap<mlir::Operation *, std::string> &result_global_names,
    mlir::DenseMap<mlir::Operation *, std::string> &result_func_names) {

  mlir::OpBuilder builder(comptime_module.getContext());

  for (auto &entry : result_global_names) {
    auto *op = entry.first;
    auto &global_name = entry.second;

    std::string func_name =
        "__get_result_" + std::to_string(result_func_names.size());
    result_func_names[op] = func_name;

    // Create a function that returns the value from the global variable
    auto result_type = op->getResult(0).getType();
    auto getter_func_type = builder.getFunctionType({}, result_type);
    auto getter_func = builder.create<mlir::func::FuncOp>(
        op->getLoc(), func_name, getter_func_type);

    auto *getter_block = getter_func.addEntryBlock();
    mlir::OpBuilder getter_builder(getter_block, getter_block->end());

    // Load the value from the global variable
    auto memref_type = mlir::MemRefType::get({}, result_type);
    auto ptr = getter_builder.create<mlir::memref::GetGlobalOp>(
        op->getLoc(), memref_type, global_name);
    auto value = getter_builder.create<mlir::memref::LoadOp>(op->getLoc(),
                                                             ptr.getResult());

    getter_builder.create<mlir::func::ReturnOp>(op->getLoc(),
                                                value.getResult());

    // Add the getter function to the module
    comptime_module.push_back(getter_func);
  }
}

// Replace compile-time operations with constants in the original module
void replaceComptimeOpsWithConstants(
    mlir::ModuleOp &module,
    mlir::DenseMap<mlir::Operation *, int64_t> &evaluation_results) {

  module.walk([&](mlir::Operation *op) {
    if (auto it = evaluation_results.find(op); it != evaluation_results.end()) {
      mlir::OpBuilder op_builder(op);
      auto constant_op = op_builder.create<mlir::arith::ConstantOp>(
          op->getLoc(), op->getResult(0).getType(),
          op_builder.getIntegerAttr(op->getResult(0).getType(), it->second));
      op->replaceAllUsesWith(constant_op);
      op->erase();
    }
  });
}

void printDependencyGraph(
    const llvm::MapVector<mlir::Operation *, mlir::DenseSet<mlir::Operation *>>
        &dependency_graph) {
  for (auto &entry : dependency_graph) {
    llvm::errs() << "Operation: ";
    if (mlir::isa<mlir::func::FuncOp>(entry.first)) {
      auto func_op = mlir::cast<mlir::func::FuncOp>(entry.first);
      llvm::errs() << func_op.getSymName() << "\n";
    } else {
      llvm::errs() << *entry.first << "\n";
    }
    for (auto dep : entry.second) {
      llvm::errs() << "  -> ";
      if (mlir::isa<mlir::func::FuncOp>(dep)) {
        auto func_op = mlir::cast<mlir::func::FuncOp>(dep);
        llvm::errs() << func_op.getSymName() << "\n";
      } else {
        llvm::errs() << *dep << "\n";
      }
    }
  }
}

void printComptimeAnalysisResults(mlir::Operation *top,
                                  mlir::DataFlowSolver &solver) {
  top->walk([&](mlir::Operation *op) {
    llvm::outs() << "Operation: " << op->getName() << "\n";
    for (mlir::Value result : op->getResults()) {
      // Lookup the ComptimeState for this value
      if (auto *state = solver.lookupState<ComptimeState>(result)) {
        llvm::outs() << "  Result: ";
        // Print the value (like %0, %1, etc.)
        result.print(llvm::outs());
        llvm::outs() << " -> ";
        state->print(llvm::outs());
        llvm::outs() << "\n";
      } else {
        // If no state found, print that as well.
        llvm::outs() << "  Result: ";
        result.print(llvm::outs());
        llvm::outs() << " -> (no state)\n";
      }
    }
  });
}

void ComptimeEvalPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  mlir::DataFlowSolver solver;
  mlir::SymbolTableCollection symbol_table;

  // Load your custom compile-time analysis if needed
  solver.load<mlir::dataflow::DeadCodeAnalysis>();
  solver.load<ComptimeStateAnalysis>();

  if (failed(solver.initializeAndRun(module))) {
    llvm::errs() << "ComptimeEvalPass: Dataflow analysis failed\n";
    return signalPassFailure();
  }

  LLVM_DEBUG(printComptimeAnalysisResults(module, solver));

  // Step 1: Collect compile-time operations and dependencies
  mlir::SetVector<mlir::Operation *> comptime_ops;
  llvm::MapVector<mlir::Operation *, mlir::DenseSet<mlir::Operation *>>
      dependency_graph;
  collectComptimeOps(module, solver, comptime_ops, dependency_graph);

  LLVM_DEBUG({
    llvm::errs() << "Comptime operations: " << comptime_ops.size() << "\n";
    llvm::errs() << "Dependency graph: " << dependency_graph.size() << "\n";
    printDependencyGraph(dependency_graph);
  });

  mlir::SetVector<mlir::Operation *> sorted_ops;
  for (auto &entry : dependency_graph) {
    sorted_ops.insert(entry.first);
    for (auto dep : entry.second) {
      sorted_ops.insert(dep);
    }
  }
  auto sorted_ops_arr = sorted_ops.takeVector();
  if (!mlir::computeTopologicalSorting(sorted_ops_arr)) {
    llvm::errs() << "Failed to topologically sort the operations\n";
  }

  // Step 3: Generate the compile-time module and main function
  mlir::IRMapping mapping;
  mlir::DenseMap<mlir::Operation *, std::string> result_global_namess;
  mlir::DenseMap<mlir::Operation *, std::string> result_func_names;
  mlir::ModuleOp comptime_module = mlir::ModuleOp::create(module.getLoc());
  generateComptimeMainFunction(comptime_module, sorted_ops_arr, mapping,
                               result_global_namess, solver);

  createGetterFunctions(comptime_module, result_global_namess,
                        result_func_names);
  // Verify the comptime module
  if (failed(comptime_module.verify())) {
    comptime_module.emitError("Comptime module verification failed.");
    signalPassFailure();
    return;
  }

  comptime_module.dump();

  // Step 4: Lower the module and create ExecutionEngine
  std::unique_ptr<mlir::ExecutionEngine> engine;
  if (failed(lowerAndExecuteComptimeModule(comptime_module, engine))) {
    comptime_module.emitError("Failed to lower and execute comptime module.");
    signalPassFailure();
    return;
  }

  // Step 5: Retrieve the results
  mlir::DenseMap<mlir::Operation *, int64_t> evaluation_results;
  retrieveComptimeResults(*engine, result_func_names, evaluation_results);

  // Step 6: Replace comptime operations with constants
  replaceComptimeOpsWithConstants(module, evaluation_results);
}

std::unique_ptr<mlir::Pass> mlir::lang::createComptimeEvalPass() {
  return std::make_unique<ComptimeEvalPass>();
}
