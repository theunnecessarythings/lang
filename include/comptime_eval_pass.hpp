
#include "mlir/Analysis/Interpreter/Interpreter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
namespace {
struct ComptimeEvalPass
    : public mlir::PassWrapper<ComptimeEvalPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;

  static std::unique_ptr<mlir::Pass> create();
};
} // namespace

void ComptimeEvalPass::runOnOperation() {
  auto module = getOperation();

  // Create an interpreter instance
  mlir::interpreter::InterpreterOptions options;
  mlir::interpreter::Interpreter interpreter(module, options);

  // Iterate over all functions in the module
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    func.walk([&](Operation *op) {
      if (auto callOp = dyn_cast<func::CallOp>(op)) {
        // Check if the call operation has the 'comptime' attribute
        if (!callOp->hasAttr("comptime"))
          return;

        // Proceed to evaluate the call at compile time
        if (failed(evaluateCallAtCompileTime(callOp, interpreter))) {
          callOp.emitError("Failed to evaluate call at compile time");
          signalPassFailure();
        }
      }
    });
  }
}

// Register the pass
std::unique_ptr<mlir::Pass> ComptimeEvalPass::create() {
  return std::make_unique<ComptimeEvalPass>();
}
