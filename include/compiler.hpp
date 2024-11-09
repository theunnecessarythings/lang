#pragma once

#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"

#include "ast.hpp"
#include "lexer.hpp"
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

struct CompilerOptions {
  std::string output_file = "output";
  bool report_warnings = false;
  bool convert_warns_to_errors = false;
};

struct SourceManager {
  std::vector<std::string> source_map;
  std::unordered_set<std::string> source_set;
  std::vector<std::string> sources;
  int last_source_id = -1;

  std::optional<std::string> get_source_path(int id) {
    if (id < 0 || static_cast<size_t>(id) >= source_map.size()) {
      return std::nullopt;
    }
    return source_map[id];
  }

  bool contains_path(const std::string &path) { return source_set.count(path); }

  std::optional<std::string> get_source(int id) {
    if (id < 0 || static_cast<size_t>(id) >= sources.size()) {
      return std::nullopt;
    }
    return sources[id];
  }

  int add_path(std::string path, std::string source) {
    last_source_id++;
    source_map.push_back(path);
    sources.push_back(source);
    source_set.insert(path);
    return last_source_id;
  }

  std::optional<std::string> read_source_line(int id, int line_no) {
    auto source = get_source(id);
    if (!source) {
      return std::nullopt;
    }
    auto &src = source.value();
    int start = 0;
    int end = src.size();
    int current_line = 1;
    for (int i = 0; i < (int)src.size(); i++) {
      if (current_line == line_no) {
        start = i;
        break;
      }
      if (src[i] == '\n') {
        current_line++;
      }
    }
    for (int i = start; i < (int)src.size(); i++) {
      if (src[i] == '\n') {
        end = i;
        break;
      }
    }
    return src.substr(start, end - start);
  }
};

struct Diagnostic {
  TokenSpan span;
  std::string message;
  enum class Level {
    Error,
    Warning,
  } level;

  static std::string literal(Level &level) {
    if (level == Level::Error) {
      return "\x1b[31m\x1b[1mError\x1b[0m";
    } else {
      return "\x1b[32m\x1b[1mWarning\x1b[0m";
    }
  }
};

struct DiagnosticsManager {
  std::shared_ptr<SourceManager> source_manager;
  std::vector<std::vector<Diagnostic>> diagnostics;

  DiagnosticsManager(std::shared_ptr<SourceManager> source_manager)
      : source_manager(source_manager) {
    diagnostics.push_back({});
    diagnostics.push_back({});
  }

  void report(Diagnostic::Level level) {
    std::cout << Diagnostic::literal(level) << "\t: " << level_count(level)
              << " \t ├";
    for (int i = 0; i < 62; i++) {
      std::cout << "─";
    }
    std::cout << "┤" << std::endl;
    if (diagnostics[static_cast<int>(level)].size() > 0) {
      for (auto &diagnostic : diagnostics[static_cast<int>(level)]) {
        report_diagnostic(diagnostic);
      }
    }
  }

  int level_count(Diagnostic::Level level) {
    return diagnostics[static_cast<int>(level)].size();
  }

  void report_diagnostic(Diagnostic &diagnostic) {
    auto span = diagnostic.span;
    auto message = diagnostic.message;
    auto filename = source_manager->get_source_path(span.file_id);

    auto source_line =
        source_manager->read_source_line(span.file_id, span.line_no);
    if (!source_line.has_value()) {
      std::cout << "Failed to read file: " << filename.value_or("")
                << std::endl;
      return;
    }

    std::cout << Diagnostic::literal(diagnostic.level) << " in "
              << filename.value_or("") << ":" << span.line_no << ":"
              << span.col_start << std::endl;

    auto line_no_header = std::to_string(span.line_no) + " | ";
    std::cout << line_no_header << source_line.value_or("") << std::endl;
    int header_len = line_no_header.size();
    for (int i = 0; i < span.col_start + header_len - 1; i++) {
      std::cout << "🭻";
    }
    std::cout << "∆ " << message << std::endl;
    std::cout << std::endl;
  }

  void report_error(Token token, std::string message) {
    if (token.kind == TokenKind::Eof) {
      throw std::runtime_error("Unexpected EOF");
    }
    diagnostics[static_cast<int>(Diagnostic::Level::Error)].push_back(
        Diagnostic{token.span, message, Diagnostic::Level::Error});
  }

  void report_warning(Token token, std::string message) {
    if (token.kind == TokenKind::Eof) {
      throw std::runtime_error("Unexpected EOF");
    }
    diagnostics[static_cast<int>(Diagnostic::Level::Warning)].push_back(
        Diagnostic{token.span, message, Diagnostic::Level::Warning});
  }
};

struct Context {
  std::shared_ptr<SourceManager> source_manager;
  DiagnosticsManager diagnostics;
  CompilerOptions options;
  llvm::ScopedHashTable<llvm::StringRef, Function *> function_table;
  llvm::ScopedHashTable<llvm::StringRef, StructDecl *> struct_table;
  llvm::ScopedHashTable<llvm::StringRef, TupleStructDecl *> tuple_struct_table;
  llvm::ScopedHashTable<llvm::StringRef, EnumDecl *> enum_table;
  llvm::ScopedHashTable<llvm::StringRef, UnionDecl *> union_table;
  llvm::ScopedHashTable<llvm::StringRef, TraitDecl *> trait_table;
  llvm::ScopedHashTable<llvm::StringRef, VarDecl *> var_table;

  Context(std::shared_ptr<SourceManager> source_manager)
      : source_manager(source_manager),
        diagnostics(DiagnosticsManager(source_manager)) {}

  void declare_function(llvm::StringRef name, Function *decl) {
    if (function_table.lookup(name)) {
      diagnostics.report_error(decl->token,
                               "Function " + name.str() + " already declared");
      return;
    }
    function_table.insert(name, decl);
  }

  void declare_struct(llvm::StringRef name, StructDecl *decl) {
    if (struct_table.lookup(name)) {
      diagnostics.report_error(decl->token,
                               "Struct " + name.str() + " already declared");
      return;
    }
    struct_table.insert(name, decl);
  }

  void declare_tuple_struct(llvm::StringRef name, TupleStructDecl *decl) {
    if (tuple_struct_table.lookup(name)) {
      diagnostics.report_error(decl->token, "Tuple struct " + name.str() +
                                                " already declared");
      return;
    }
    tuple_struct_table.insert(name, decl);
  }

  void declare_enum(llvm::StringRef name, EnumDecl *decl) {
    if (enum_table.lookup(name)) {
      diagnostics.report_error(decl->token,
                               "Enum " + name.str() + " already declared");
      return;
    }
    enum_table.insert(name, decl);
  }

  void declare_union(llvm::StringRef name, UnionDecl *decl) {
    if (union_table.lookup(name)) {
      diagnostics.report_error(decl->token,
                               "Union " + name.str() + " already declared");
      return;
    }
    union_table.insert(name, decl);
  }

  void declare_trait(llvm::StringRef name, TraitDecl *decl) {
    if (trait_table.lookup(name)) {
      diagnostics.report_error(decl->token,
                               "Trait " + name.str() + " already declared");
      return;
    }
    trait_table.insert(name, decl);
  }

  void declare_var(llvm::StringRef name, VarDecl *decl) {
    if (var_table.lookup(name)) {
      diagnostics.report_error(decl->token,
                               "Variable " + name.str() + " already declared");
      return;
    }
    var_table.insert(name, decl);
  }
};

template <typename T> struct Scope {
  std::optional<std::shared_ptr<Scope>> parent;
  std::unordered_map<std::string, T> symbols;
};

template <typename T> struct SymbolTable {
  std::vector<Scope<T>> scopes;

  SymbolTable() { push_scope(); }

  void push_scope() {
    auto parent =
        scopes.size() > 0
            ? std::make_optional(std::make_shared<Scope>(scopes.back()))
            : std::nullopt;
    scopes.push_back(Scope{parent, {}});
  }

  void pop_scope() {
    if (scopes.size() > 0) {
      scopes.pop_back();
    }
  }

  void declare_symbol(std::string name, T type) {
    if (scopes.size() == 0) {
      return;
    }
    scopes.back().symbols[name] = type;
  }

  std::optional<T> get_symbol(std::string name) {
    for (int i = scopes.size() - 1; i >= 0; i--) {
      auto &scope = scopes[i];
      if (scope.symbols.count(name) > 0) {
        return scope.symbols[name];
      }
    }
    return std::nullopt;
  }

  void pretty_print() {
    for (auto &scope : scopes) {
      std::cout << "Scope: " << std::endl;
      for (auto &[name, type] : scope.symbols) {
        std::cout << name; // << " : " << type->pretty_print() << std::endl;
      }
    }
  }
};
