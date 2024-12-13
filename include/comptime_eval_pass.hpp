#include "dialect/LangOps.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace {

class ComptimeState : public mlir::AnalysisState {
public:
  enum class State { Comptime, Runtime, Uninitialized };

  ComptimeState(mlir::LatticeAnchor anchor)
      : AnalysisState(anchor), currentState(State::Uninitialized) {}
  ComptimeState(State state, mlir::LatticeAnchor anchor)
      : AnalysisState(anchor), currentState(state) {}

  mlir::ChangeResult join(const ComptimeState &other) {
    if (currentState == State::Uninitialized) {
      currentState = other.currentState;
      return mlir::ChangeResult::Change;
    }
    if (other.currentState == State::Uninitialized) {
      return mlir::ChangeResult::NoChange;
    }

    if (currentState == State::Comptime &&
        other.currentState == State::Runtime) {
      currentState = State::Runtime;
      return mlir::ChangeResult::Change;
    }
    return mlir::ChangeResult::NoChange;
  }

  State getState() const { return currentState; }

  void setState(State s) { currentState = s; }

  void print(llvm::raw_ostream &os) const override {
    os << (currentState == State::Comptime    ? "Comptime"
           : (currentState == State::Runtime) ? "Runtime"
                                              : "Uninitialized");
  }

private:
  State currentState;
};

class ComptimeStateAnalysis : public mlir::DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  mlir::LogicalResult initialize(mlir::Operation *op) override;
  mlir::LogicalResult visit(mlir::ProgramPoint *) override;

private:
  mlir::LogicalResult handleOperation(mlir::Operation *op);
  void setAllResultsComptime(mlir::ArrayRef<ComptimeState *> resultStates,
                             mlir::ValueRange resultValues);
  void setAllResultsRuntime(mlir::ArrayRef<ComptimeState *> resultStates,
                            mlir::ValueRange resultValues);
  mlir::LogicalResult
  visitCallOperation(mlir::CallOpInterface op,
                     mlir::SmallVector<ComptimeState *, 4> &);
};

} // end anonymous namespace
