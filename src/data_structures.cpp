#include "data_structures.hpp"

void Symbol::mangle() {
  std::string out;
  llvm::raw_string_ostream mangler(out);

  // Add name
  mangler << name;

  // Add function signature if applicable
  if (kind == SymbolKind::Function || kind == SymbolKind::GenericFunction) {
    if (parent_type)
      mangler << ":" << parent_type;
    for (const auto &param : param_types) {
      mangler << sep << param;
    }
  }

  mangled_name = mangler.str();
}

void Symbol::demangle() {
  std::string out;
  llvm::raw_string_ostream demangler(out);

  // Add name
  demangler << name;

  // Add function signature if applicable
  if (kind == SymbolKind::Function || kind == SymbolKind::GenericFunction) {
    demangler << "(";
    for (size_t i = 0; i < param_types.size(); i++) {
      demangler << param_types[i];
      if (i < param_types.size() - 1)
        demangler << ", ";
    }
    demangler << ")";
  }

  demangled_name = demangler.str();
}

// Add a symbol to the table
std::shared_ptr<Symbol> SymbolTable::addSymbol(std::shared_ptr<Symbol> symbol,
                                               bool overwrite) {
  auto &mangled_name = symbol->getMangledName();
  if (table.find(mangled_name) != table.end() && !overwrite) {
    return nullptr;
  }
  table[mangled_name] = symbol;
  if (symbol->kind == Symbol::SymbolKind::Function ||
      symbol->kind == Symbol::SymbolKind::GenericFunction)
    overload_table.addOverload(symbol);
  return symbol;
}

// Lookup a symbol by its mangled name with optional specific scope
std::shared_ptr<Symbol> SymbolTable::lookup(const llvm::StringRef &name) const {
  auto it = table.find(name);
  if (it != table.end()) {
    return it->second;
  }

  if (parent) {
    return parent->lookup(name);
  }

  return nullptr;
}

// Lookup overloads by unmangled name across scoped hierarchy
llvm::SmallVector<std::shared_ptr<Symbol>, 4>
SymbolTable::lookupScopedOverloads(const llvm::StringRef &name) const {
  llvm::SmallVector<std::shared_ptr<Symbol>, 4> all_overloads;

  // Start from the current scope and traverse up the hierarchy
  auto current = shared_from_this();
  while (current) {
    auto overloads = current->overload_table.getOverloads(name);
    all_overloads.insert(all_overloads.end(), overloads.begin(),
                         overloads.end());
    current = current->parent; // Move to the parent scope
  }

  return all_overloads;
}

// Create a new child scope
std::shared_ptr<SymbolTable>
SymbolTable::createChildScope(const llvm::StringRef &scope_name) {
  auto child = std::make_shared<SymbolTable>(shared_from_this());
  children.push_back(child);
  child->scope_id = children.size();
  child->scope_name = scope_name;
  return child;
}

// Helper to get the current scope name
llvm::SmallString<64> SymbolTable::getScopeName() const {
  if (!parent)
    return llvm::SmallString<64>("");
  auto name = parent->getScopeName() + "@" +
              (scope_name.empty() ? "S" + llvm::Twine(scope_id) : scope_name);
  return llvm::SmallString<64>(name.str());
}

void SymbolTable::dump(int depth) const {
  std::string indent(depth, ' ');
  llvm::errs() << indent << "Scope: " << getScopeName() << "\n";
  for (const auto &[key, symbol] : table) {
    llvm::errs() << indent << key << " -> " << symbol->getMangledName() << "\n";
  }
  overload_table.dump(depth + 1);
  // Print child scopes
  for (const auto &child : children) {
    llvm::errs() << indent << "Child Scope:" << "\n";
    child->dump(depth + 1);
  }
}
