/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: MyDialect.td                                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mydialect {

class MyDialect : public ::mlir::Dialect {
  explicit MyDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~MyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("mydialect");
  }
};
} // namespace mydialect
MLIR_DECLARE_EXPLICIT_TYPE_ID(mydialect::MyDialect)
