#pragma once

#include "ast.hpp"
#include "dialect/LangOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <optional>

namespace mlir {

enum class ResultState { Failure, SuccessNoValue, SuccessWithValue };

template <typename T> class [[nodiscard]] Result {
public:
  Result(LogicalResult lr)
      : state(mlir::failed(lr) ? ResultState::Failure
                               : ResultState::SuccessNoValue) {}

  Result(T val) : state(ResultState::SuccessWithValue), value(std::move(val)) {}

  Result(InFlightDiagnostic &&diag) { state = ResultState::Failure; }

  static Result failure() { return Result(ResultState::Failure); }

  static Result successNoValue() { return Result(ResultState::SuccessNoValue); }

  static Result success(T val) { return Result(std::move(val)); }

  bool succeeded() const {
    return state == ResultState::SuccessNoValue ||
           state == ResultState::SuccessWithValue;
  }

  bool failed() const { return state == ResultState::Failure; }

  bool hasValue() const { return state == ResultState::SuccessWithValue; }

  const T &getValue() const {
    assert(hasValue() && "No value available");
    return *value;
  }

  T &getValue() {
    assert(hasValue() && "No value available");
    return *value;
  }

  // Convert to LogicalResult for seamless integration.
  operator LogicalResult() const {
    return succeeded() ? mlir::success() : mlir::failure();
  }

  const T *operator->() const {
    assert(hasValue() && "No value to access");
    return &*value;
  }

  T *operator->() {
    assert(hasValue() && "No value to access");
    return &*value;
  }

  const T &operator*() const {
    assert(hasValue() && "No value to access");
    return *value;
  }

  T &operator*() {
    assert(hasValue() && "No value to access");
    return *value;
  }

private:
  Result(ResultState s) : state(s) {}

  ResultState state;
  std::optional<T> value;
};

template <typename T> Result<T> toResult(FailureOr<T> fo) {
  return fo ? Result<T>::success(*fo) : Result<T>::failure();
}

} // namespace mlir

inline std::string mangle(llvm::StringRef base,
                          llvm::ArrayRef<mlir::Type> types) {
  std::string mangled_name = base.str();
  llvm::raw_string_ostream rso(mangled_name);
  if (!types.empty()) {
    rso << "_";
  }
  for (auto &type : types) {
    rso << "_";
    if (mlir::isa<mlir::lang::ArrayType>(type))
      rso << "!array";
    else if (mlir::isa<mlir::lang::SliceType>(type))
      rso << "!slice";
    else if (mlir::isa<mlir::TensorType>(type))
      rso << "tensor";
    else if (mlir::isa<mlir::TupleType>(type))
      rso << "tuple";
    // else if (mlir::isa<mlir::MemRefType>(type))
    //   rso << "memref";
    else if (mlir::isa<mlir::lang::StructType>(type))
      rso << mlir::cast<mlir::lang::StructType>(type).getName();
    else
      rso << type;
  }
  return rso.str();
}

inline std::string attrToStr(mlir::Attribute attr) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  if (mlir::isa<mlir::IntegerAttr>(attr)) {
    rso << mlir::cast<mlir::IntegerAttr>(attr).getInt();
  } else if (mlir::isa<mlir::FloatAttr>(attr)) {
    rso << mlir::cast<mlir::FloatAttr>(attr).getValueAsDouble();
  } else {
    llvm::errs() << "Unsupported attribute kind\n";
  }
  return rso.str();
}

struct SymbolTable;

static std::string sep = ":";
struct Symbol {
  enum class SymbolKind {
    Variable,
    Function,
    Struct,
    Enum,
    GenericFunction,
  };
  llvm::SmallString<64> name;  // Name of the symbol (unmangled)
  SymbolKind kind;             // Type of the symbol (e.g., Variable, Function)
  llvm::SmallString<64> scope; // Fully qualified scope (e.g., "app::math")
  llvm::SmallVector<mlir::Type, 4>
      param_types; // Parameter types for functions (empty for variables)
  llvm::SmallString<64> mangled_name;   // Mangled name of the symbol
  llvm::SmallString<64> demangled_name; // Demangled name of the symbol
  mlir::OpBuilder::InsertPoint generic_insertion_point;
  std::shared_ptr<SymbolTable> generic_scope;

  union {
    mlir::Value value;                  // Value for variables
    mlir::lang::StructType struct_type; // Type for structs
    mlir::lang::FuncOp func_op;         // Function for functions
    Function *generic_func;
  };

  // Constructor for variables
  Symbol(const llvm::SmallString<64> &name, mlir::Value value,
         SymbolKind kind = SymbolKind::Variable,
         const llvm::SmallString<64> &scope = llvm::SmallString<64>())
      : name(name), kind(kind), scope(scope), value(value) {
    mangle();
    demangle();
  }

  // Constructor for functions
  Symbol(const llvm::SmallString<64> &name,
         const llvm::SmallVector<mlir::Type, 4> param_types,
         SymbolKind kind = SymbolKind::Function,
         const llvm::SmallString<64> &scope = llvm::SmallString<64>())
      : name(name), kind(kind), scope(scope), param_types(param_types) {
    mangle();
    demangle();
  }

  void setStructType(mlir::lang::StructType struct_type) {
    assert(kind == SymbolKind::Struct && "Symbol is not a struct");
    this->struct_type = struct_type;
  }

  void setFunction(mlir::lang::FuncOp function) {
    assert(kind == SymbolKind::Function && "Symbol is not a function");
    this->func_op = function;
  }

  void setValue(mlir::Value value) {
    assert(kind == SymbolKind::Variable && "Symbol is not a variable");
    this->value = value;
  }

  void setGenericFunction(Function *generic_func,
                          mlir::OpBuilder::InsertPoint insertion_point,
                          std::shared_ptr<SymbolTable> scope) {
    assert(kind == SymbolKind::GenericFunction &&
           "Symbol is not a generic function");
    this->generic_func = generic_func;
    this->generic_insertion_point = insertion_point;
    this->generic_scope = scope;
  }

  mlir::Value getValue() const {
    assert(kind == SymbolKind::Variable && "Symbol is not a variable");
    return value;
  }

  mlir::lang::StructType getStructType() const {
    assert(kind == SymbolKind::Struct && "Symbol is not a struct");
    return struct_type;
  }

  mlir::lang::FuncOp getFuncOp() const {
    assert(kind == SymbolKind::Function && "Symbol is not a function");
    return func_op;
  }

  Function *getGenericFunction() const {
    assert(kind == SymbolKind::GenericFunction &&
           "Symbol is not a generic function");
    return generic_func;
  }

  mlir::OpBuilder::InsertPoint getGenericInsertionPoint() const {
    assert(kind == SymbolKind::GenericFunction &&
           "Symbol is not a generic function");
    return generic_insertion_point;
  }

  std::shared_ptr<SymbolTable> getGenericScope() const {
    assert(kind == SymbolKind::GenericFunction &&
           "Symbol is not a generic function");
    return generic_scope;
  }

  // Accessors
  const llvm::SmallString<64> &getName() const { return name; }
  const llvm::SmallString<64> &getScope() const { return scope; }
  const llvm::SmallVector<mlir::Type, 4> &getParamTypes() const {
    return param_types;
  }
  const llvm::SmallString<64> &getMangledName() const { return mangled_name; }
  const llvm::SmallString<64> &getDemangledName() const {
    return demangled_name;
  }

  void mangle();
  void demangle();
  std::string encodeType(const std::string &type) const;
};

struct OverloadTable {
  llvm::StringMap<std::vector<std::shared_ptr<Symbol>>> overload_table;

  void addOverload(std::shared_ptr<Symbol> symbol) {
    if (symbol->kind == Symbol::SymbolKind::Function ||
        symbol->kind == Symbol::SymbolKind::GenericFunction) {
      overload_table[symbol->getName()].push_back(symbol);
    }
  }

  std::vector<std::shared_ptr<Symbol>>
  getOverloads(const llvm::StringRef &name) const {
    auto it = overload_table.find(name);
    if (it != overload_table.end()) {
      return it->second;
    }
    return {};
  }

  void dump(int depth = 0) const {
    for (const auto &entry : overload_table) {
      llvm::errs() << std::string(depth * 2, ' ') << entry.first() << "\n";
    }
  }
};

struct SymbolTable : public std::enable_shared_from_this<SymbolTable> {
  llvm::StringMap<std::shared_ptr<Symbol>> table; // Current scope symbol table
  OverloadTable overload_table;                   // Overload table
  std::shared_ptr<SymbolTable> parent; // Parent scope for nested lookups
  llvm::SmallVector<std::shared_ptr<SymbolTable>, 4> children; // Child scopes
  int scope_id = 0; // Unique scope ID
  llvm::SmallString<64> scope_name =
      llvm::SmallString<64>(""); // Name of the current scope

  // Constructor
  SymbolTable(std::shared_ptr<SymbolTable> parent = nullptr) : parent(parent) {}

  // Add a symbol to the table
  std::shared_ptr<Symbol> addSymbol(std::shared_ptr<Symbol> symbol,
                                    bool overwrite = false);

  // Lookup a symbol by its mangled name with optional specific scope
  std::shared_ptr<Symbol> lookup(const llvm::StringRef &name) const;

  // Lookup overloads by unmangled name across scoped hierarchy
  llvm::SmallVector<std::shared_ptr<Symbol>, 4>
  lookupScopedOverloads(const llvm::StringRef &name) const;

  // Create a new child scope
  std::shared_ptr<SymbolTable>
  createChildScope(const llvm::StringRef &scope_name);

  // Helper to get the current scope name
  llvm::SmallString<64> getScopeName() const;

  void dump(int depth = 0) const;
};

class Defer {
  std::function<void()> func;

public:
  explicit Defer(std::function<void()> f) : func(std::move(f)) {}
  ~Defer() { func(); }
};
