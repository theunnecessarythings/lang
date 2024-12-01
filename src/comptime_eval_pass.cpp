#include "comptime_eval_pass.hpp"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitCPass.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
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

#define DEBUG_TYPE "comptime-propagation"

namespace mlir {
#define GEN_PASS_DEF_COMPTIMEANALYSIS
#include "dialect/LangPasses.h.inc"
} // namespace mlir

mlir::ChangeResult Comptime::setToComptime() {
  if (comptime)
    return mlir::ChangeResult::NoChange;
  comptime = true;
  return mlir::ChangeResult::Change;
}

mlir::ChangeResult Comptime::setToRuntime() {
  if (!comptime)
    return mlir::ChangeResult::NoChange;
  comptime = false;
  return mlir::ChangeResult::Change;
}

void Comptime::print(mlir::raw_ostream &os) const {
  os << (comptime ? "comptime" : "runtime");
}

void Comptime::onUpdate(mlir::DataFlowSolver *solver) const {
  AnalysisState::onUpdate(solver);

  if (mlir::ProgramPoint *pp =
          llvm::dyn_cast_if_present<mlir::ProgramPoint *>(anchor)) {
    if (pp->isBlockStart()) {
      // Re-invoke the analyses on the block itself.
      for (mlir::DataFlowAnalysis *analysis : subscribers)
        solver->enqueue({pp, analysis});
      // Re-invoke the analyses on all operations in the block.
      for (mlir::DataFlowAnalysis *analysis : subscribers)
        for (mlir::Operation &op : *pp->getBlock())
          solver->enqueue({solver->getProgramPointAfter(&op), analysis});
    }
  } else if (auto *lattice_anchor =
                 llvm::dyn_cast_if_present<mlir::GenericLatticeAnchor *>(
                     anchor)) {
    // Re-invoke the analysis on the successor block.
    if (auto *edge = mlir::dyn_cast<mlir::dataflow::CFGEdge>(lattice_anchor)) {
      for (mlir::DataFlowAnalysis *analysis : subscribers)
        solver->enqueue(
            {solver->getProgramPointBefore(edge->getTo()), analysis});
    }
  } else if (auto value = llvm::dyn_cast_if_present<mlir::Value>(anchor)) {
    // Re-invoke the analysis on the users of the value.
    for (mlir::Operation *user : value.getUsers()) {
      for (mlir::DataFlowAnalysis *analysis : subscribers)
        solver->enqueue({solver->getProgramPointAfter(user), analysis});
    }
  }
}

ComptimeCodeAnalysis::ComptimeCodeAnalysis(mlir::DataFlowSolver &solver)
    : DataFlowAnalysis(solver) {
  registerAnchorKind<mlir::dataflow::CFGEdge>();
}

llvm::LogicalResult ComptimeCodeAnalysis::initialize(mlir::Operation *top) {
  // Mark the top-level blocks as comptime.
  for (mlir::Region &region : top->getRegions()) {
    if (region.empty())
      continue;
    auto *state = getOrCreate<Comptime>(getProgramPointBefore(&region.front()));
    propagateIfChanged(state, state->setToComptime());
  }

  // Mark as overdefined the predecessors of symbol callables with potentially
  // unknown predecessors.
  initializeSymbolCallables(top);

  return initializeRecursively(top);
}

void ComptimeCodeAnalysis::initializeSymbolCallables(mlir::Operation *top) {
  analysis_scope = top;
  auto walkFn = [&](mlir::Operation *sym_table, bool all_uses_visible) {
    mlir::Region &symbol_table_region = sym_table->getRegion(0);
    mlir::Block *symbol_table_block = &symbol_table_region.front();

    bool found_symbol_callable = false;
    for (auto callable :
         symbol_table_block->getOps<mlir::CallableOpInterface>()) {
      mlir::Region *callable_region = callable.getCallableRegion();
      if (!callable_region)
        continue;
      auto symbol =
          mlir::dyn_cast<mlir::SymbolOpInterface>(callable.getOperation());
      if (!symbol)
        continue;

      // Public symbol callables or those for which we can't see all uses have
      // potentially unknown callsites.
      if (symbol.isPublic() || (!all_uses_visible && symbol.isNested())) {
        auto *state = getOrCreate<mlir::dataflow::PredecessorState>(
            getProgramPointAfter(callable));
        propagateIfChanged(state, state->setHasUnknownPredecessors());
      }
      found_symbol_callable = true;
    }

    // Exit early if no eligible symbol callables were found in the table.
    if (!found_symbol_callable)
      return;

    // Walk the symbol table to check for non-call uses of symbols.
    std::optional<mlir::SymbolTable::UseRange> uses =
        mlir::SymbolTable::getSymbolUses(&symbol_table_region);
    if (!uses) {
      // If we couldn't gather the symbol uses, conservatively assume that
      // we can't track information for any nested symbols.
      return top->walk([&](mlir::CallableOpInterface callable) {
        auto *state = getOrCreate<mlir::dataflow::PredecessorState>(
            getProgramPointAfter(callable));
        propagateIfChanged(state, state->setHasUnknownPredecessors());
      });
    }

    for (const mlir::SymbolTable::SymbolUse &use : *uses) {
      if (mlir::isa<mlir::CallOpInterface>(use.getUser()))
        continue;
      // If a callable symbol has a non-call use, then we can't be guaranteed to
      // know all callsites.
      mlir::Operation *symbol =
          symbol_table.lookupSymbolIn(top, use.getSymbolRef());
      auto *state = getOrCreate<mlir::dataflow::PredecessorState>(
          getProgramPointAfter(symbol));
      propagateIfChanged(state, state->setHasUnknownPredecessors());
    }
  };
  mlir::SymbolTable::walkSymbolTables(
      top, /*allSymUsesVisible=*/!top->getBlock(), walkFn);
}

/// Returns true if the operation is a returning terminator in region
/// control-flow or the terminator of a callable region.
static bool isRegionOrCallableReturn(mlir::Operation *op) {
  return op->getBlock() != nullptr && !op->getNumSuccessors() &&
         mlir::isa<mlir::RegionBranchOpInterface, mlir::CallableOpInterface>(
             op->getParentOp()) &&
         op->getBlock()->getTerminator() == op;
}

llvm::LogicalResult
ComptimeCodeAnalysis::initializeRecursively(mlir::Operation *op) {
  // Initialize the analysis by visiting every op with control-flow semantics.
  if (op->getNumRegions() || op->getNumSuccessors() ||
      isRegionOrCallableReturn(op) || mlir::isa<mlir::CallOpInterface>(op)) {
    // When the comptimeness of the parent block changes, make sure to re-invoke
    // the analysis on the op.
    if (op->getBlock())
      getOrCreate<Comptime>(getProgramPointBefore(op->getBlock()))
          ->blockContentSubscribe(this);
    // Visit the op.
    if (failed(visit(getProgramPointAfter(op))))
      return llvm::failure();
  }

  // Initialize the operation's operands.
  for (mlir::Value operand : op->getOperands()) {
    getOrCreate<Comptime>(operand)->blockContentSubscribe(this);
  }

  // Recurse on nested operations.
  for (mlir::Region &region : op->getRegions())
    for (mlir::Operation &op : region.getOps())
      if (failed(initializeRecursively(&op)))
        return mlir::failure();

  getOrCreate<Comptime>(getProgramPointBefore(op))->blockContentSubscribe(this);
  return llvm::success();
}

void ComptimeCodeAnalysis::markEdgeRuntime(mlir::Block *from, mlir::Block *to) {
  auto *state = getOrCreate<Comptime>(getProgramPointBefore(to));
  propagateIfChanged(state, state->setToComptime());
  auto *edge_state = getOrCreate<Comptime>(
      getLatticeAnchor<mlir::dataflow::CFGEdge>(from, to));
  propagateIfChanged(edge_state, edge_state->setToComptime());
}

void ComptimeCodeAnalysis::markEntryBlocksRuntime(mlir::Operation *op) {
  for (mlir::Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    auto *state = getOrCreate<Comptime>(getProgramPointBefore(&region.front()));
    propagateIfChanged(state, state->setToComptime());
  }
}

llvm::LogicalResult ComptimeCodeAnalysis::visit(mlir::ProgramPoint *point) {
  return llvm::success();
}

//===----------------------------------------------------------------------===//
// ComptimeCodeAnalysisPass
//===----------------------------------------------------------------------===//
struct ComptimeOpLowering
    : public mlir::OpConversionPattern<mlir::lang::ComptimeOp> {
  using mlir::OpConversionPattern<mlir::lang::ComptimeOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::lang::ComptimeOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

namespace {
struct ComptimeLoweringPass
    : public mlir::PassWrapper<ComptimeLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ComptimeLoweringPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::affine::AffineDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
  }
  void runOnOperation() final;
};

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

void ComptimeLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<mlir::lang::ComptimeOp>();

  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<mlir::lang::InlineComptimeOp>(&getContext());

  // Apply partial conversion.
  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns)))) {

    llvm::errs() << "ComptimeLoweringPass: Partial conversion failed for\n";
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> mlir::lang::createComptimeLoweringPass() {
  return std::make_unique<ComptimeLoweringPass>();
}

static void printAnalysisResults(mlir::DataFlowSolver &solver,
                                 mlir::Operation *op, mlir::raw_ostream &os) {
  op->walk([&](mlir::Operation *op) {
    os << "Operation: " << op->getName() << "\n";
    // Operand comptime states.
    os << " operands -> ";
    for (mlir::Value operand : op->getOperands()) {
      operand.printAsOperand(os, mlir::OpPrintingFlags().useLocalScope());
      os << " = ";
      auto *comptime = solver.lookupState<Comptime>(operand);
      if (comptime)
        os << *comptime;
      else
        os << "runtime";
      if (operand != op->getOperands().back())
        os << ", ";
    }
    os << "\n";

    // Region comptime states.
    for (mlir::Region &region : op->getRegions()) {
      os << " region #" << region.getRegionNumber() << "\n";
      for (mlir::Block &block : region) {
        os << "  ";
        block.printAsOperand(os);
        os << " = ";
        auto *comptime =
            solver.lookupState<Comptime>(solver.getProgramPointBefore(&block));
        if (comptime)
          os << *comptime;
        else
          os << "runtime";
        os << "\n";
        for (mlir::Block *pred : block.getPredecessors()) {
          os << "   from ";
          pred->printAsOperand(os);
          os << " = ";
          auto *comptime = solver.lookupState<Comptime>(
              solver.getLatticeAnchor<mlir::dataflow::CFGEdge>(pred, &block));
          if (comptime)
            os << *comptime;
          else
            os << "runtime";
          os << "\n";
        }
      }
      if (!region.empty()) {
        auto *preds = solver.lookupState<mlir::dataflow::PredecessorState>(
            solver.getProgramPointBefore(&region.front()));
        if (preds)
          os << "region_preds: " << *preds << "\n";
      }
    }

    auto *preds = solver.lookupState<mlir::dataflow::PredecessorState>(
        solver.getProgramPointAfter(op));
    if (preds)
      os << "op_preds: " << *preds << "\n";
  });
}

bool isComptimeOperation(mlir::Operation *op, mlir::DataFlowSolver &solver) {
  auto is_comptime = op->getAttrOfType<mlir::BoolAttr>("comptime");
  if (!is_comptime)
    return false;
  return is_comptime.getValue();
  // auto state =
  // solver.lookupState<Comptime>(solver.getProgramPointBefore(op));
  // llvm::errs() << "Is comptime: " << op->getName() << " -> "
  //              << (state ? (state->isComptime() ? "true" : "false") : "null")
  //              << "\n";
  // if (!state)
  //   return false;
  // return state->isComptime();
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
        llvm::errs() << "Collecting dependencies for: " << op->getName()
                     << "\n";
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
            llvm::errs() << "Call operation without a symbol\n";
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
              llvm::errs() << "Call operation without a symbol\n";
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
  mlir::OpPassManager &opt_pm = pm.nest<mlir::func::FuncOp>();
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::lang::createComptimeLoweringPass());
  pm.addPass(mlir::lang::createLowerToAffinePass());
  pm.addPass(mlir::lang::createUnrealizedConversionCastResolverPass());

  // Add passes to lower the module to LLVM dialect.
  opt_pm.addPass(mlir::createCSEPass());

  // Add passes to lower the module to LLVM dialect
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLowerAffinePass());
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
    llvm::errs() << "Result: " << result << "\n";
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
    if (mlir::isa<mlir::lang::FuncOp>(entry.first)) {
      auto func_op = mlir::cast<mlir::lang::FuncOp>(entry.first);
      llvm::errs() << func_op.getSymName() << "\n";
    } else {
      llvm::errs() << *entry.first << "\n";
    }
    for (auto dep : entry.second) {
      llvm::errs() << "  -> ";
      if (mlir::isa<mlir::lang::FuncOp>(dep)) {
        auto func_op = mlir::cast<mlir::lang::FuncOp>(dep);
        llvm::errs() << func_op.getSymName() << "\n";
      } else {
        llvm::errs() << *dep << "\n";
      }
    }
  }
}

void ComptimeEvalPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  mlir::DataFlowSolver solver;
  // Load your custom compile-time analysis if needed
  solver.load<ComptimeCodeAnalysis>();

  // Step 1: Collect compile-time operations and dependencies
  mlir::SetVector<mlir::Operation *> comptime_ops;
  llvm::MapVector<mlir::Operation *, mlir::DenseSet<mlir::Operation *>>
      dependency_graph;
  collectComptimeOps(module, solver, comptime_ops, dependency_graph);

  printDependencyGraph(dependency_graph);

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
    llvm::errs() << "Failed to lower and execute comptime module.\n";
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
