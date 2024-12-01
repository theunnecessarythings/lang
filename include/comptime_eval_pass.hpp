#include "dialect/LangOps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace {

/// This is a simple analysis state that represents whether the associated
/// lattice anchor (either a block or a control-flow edge) is comptime.
class Comptime : public mlir::AnalysisState {
public:
  Comptime(const Comptime &) = default;
  Comptime(Comptime &&) = default;
  Comptime &operator=(const Comptime &) = default;
  Comptime &operator=(Comptime &&) = default;
  using AnalysisState::AnalysisState;

  /// Set the state of the lattice anchor to comptime.
  mlir::ChangeResult setToComptime();

  mlir::ChangeResult setToRuntime();

  /// Get whether the lattice anchor is comptime.
  bool isComptime() const { return comptime; }

  /// Print the comptimeness.
  void print(mlir::raw_ostream &os) const override;

  /// When the state of the lattice anchor is changed to comptime, re-invoke
  /// subscribed analyses on the operations in the block and on the block
  /// itself.
  void onUpdate(mlir::DataFlowSolver *solver) const override;

  /// Subscribe an analysis to changes to the comptimeness.
  void blockContentSubscribe(mlir::DataFlowAnalysis *analysis) {
    subscribers.insert(analysis);
  }

private:
  /// Whether the lattice anchor is comptime. Optimistically assume that the
  /// lattice anchor is comptime.
  bool comptime = true;

  /// A set of analyses that should be updated when this state changes.
  mlir::SetVector<mlir::DataFlowAnalysis *,
                  mlir::SmallVector<mlir::DataFlowAnalysis *, 4>,
                  mlir::SmallPtrSet<mlir::DataFlowAnalysis *, 4>>
      subscribers;
};

//===----------------------------------------------------------------------===//
// ComptimeCodeAnalysis
// ===----------------------------------------------------------------------===//

/// This analysis uses a data-flow solver to determine which control-flow edges
/// are comptime. This is done by visiting operations with control-flow
/// semantics and deducing which of their successors are comptime.

class ComptimeCodeAnalysis : public mlir::DataFlowAnalysis {
public:
  explicit ComptimeCodeAnalysis(mlir::DataFlowSolver &solver);

  /// Initialize the analysis by visiting every operation with potential
  /// control-flow semantics.
  llvm::LogicalResult initialize(mlir::Operation *top) override;

  /// Visit an operation with control-flow semantics and deduce which of its
  /// successors are comptime.
  llvm::LogicalResult visit(mlir::ProgramPoint *point) override;

private:
  /// Find and mark symbol callables with potentially unknown callsites as
  /// having overdefined predecessors. `top` is the top-level operation that the
  /// analysis is operating on.
  void initializeSymbolCallables(mlir::Operation *top);

  /// Recursively Initialize the analysis on nested regions.
  llvm::LogicalResult initializeRecursively(mlir::Operation *op);

  /// Visit the given call operation and compute any necessary lattice state.
  void visitCallOperation(mlir::CallOpInterface call);

  /// Visit the given branch operation with successors and try to determine
  /// which are comptime from the current block.
  void visitBranchOperation(mlir::BranchOpInterface branch);

  /// Visit the given region branch operation, which defines regions, and
  /// compute any necessary lattice state. This also resolves the lattice state
  /// of both the operation results and any nested regions.
  void visitRegionBranchOperation(mlir::RegionBranchOpInterface branch);

  /// Visit the given terminator operation that exits a region under an
  /// operation with control-flow semantics. These are terminators with no CFG
  /// successors.
  void visitRegionTerminator(mlir::Operation *op,
                             mlir::RegionBranchOpInterface branch);

  /// Visit the given terminator operation that exits a callable region. These
  /// are terminators with no CFG successors.
  void visitCallableTerminator(mlir::Operation *op,
                               mlir::CallableOpInterface callable);

  /// Mark the edge between `from` and `to` as runtime.
  void markEdgeRuntime(mlir::Block *from, mlir::Block *to);

  /// Mark the entry blocks of the operation as runtime.
  void markEntryBlocksRuntime(mlir::Operation *op);

  /// Get the constant values of the operands of the operation. Returns
  /// std::nullopt if any of the operand lattices are uninitialized.
  std::optional<mlir::SmallVector<mlir::Attribute>>
  getOperandValues(mlir::Operation *op);

  /// The top-level operation the analysis is running on. This is used to detect
  /// if a callable is outside the scope of the analysis and thus must be
  /// considered an external callable.
  mlir::Operation *analysis_scope;

  /// A symbol table used for O(1) symbol lookups during simplification.
  mlir::SymbolTableCollection symbol_table;
};

} // end anonymous namespace
