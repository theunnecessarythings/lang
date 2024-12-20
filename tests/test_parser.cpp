#include "ast.hpp"
#include "parser.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <fstream>
#include <iostream>
#include <string>

auto parse(const std::string &path, std::string &str, bool load_builtins = true,
           bool print_tokens = false) {

  std::shared_ptr<Context> context = std::make_shared<Context>();
  std::unique_ptr<Lexer> lexer = std::make_unique<Lexer>(str, 1);

  if (print_tokens) {
    while (true) {
      Token token = lexer->next();
      if (token.kind == TokenKind::Eof) {
        break;
      }
      std::cout << Lexer::lexeme(token.kind) << " : "
                << lexer->tokenToString(token) << std::endl;
    }
  }

  Parser parser(std::move(lexer), context);
  auto tree = parser.parseProgram(load_builtins);
  return tree;
}

class StringDiffMatcher : public Catch::Matchers::MatcherBase<std::string> {
  std::string m_expected;
  mutable std::string m_diff;

public:
  StringDiffMatcher(const std::string &expected) : m_expected(expected) {}

  bool match(const std::string &actual) const override {
    size_t minLength = std::min(m_expected.size(), actual.size());
    size_t diffPos = 0;
    while (diffPos < minLength && m_expected[diffPos] == actual[diffPos]) {
      ++diffPos;
    }

    std::ostringstream oss;
    oss << "Strings differ at position " << diffPos << ":\n";
    oss << "Expected: " << m_expected.substr(diffPos, 10) << "...\n";
    oss << "Actual  : " << actual.substr(diffPos, 10) << "...\n";
    m_diff = oss.str();

    // Return whether the strings are equal
    return m_expected == actual;
  }

  std::string describe() const override { return m_diff; }
};

inline StringDiffMatcher DiffersAt(const std::string &expected) {
  return StringDiffMatcher(expected);
}
void testRepr(const std::string &path) {
  std::cout << "Parsing " << path << std::endl;
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Could not open file " + path);
  std::string str((std::istreambuf_iterator<char>(file)),
                  std::istreambuf_iterator<char>());
  auto tree = parse(path, str, false);

  AstDumper dumper(true);
  auto repr = dumper.toString();
  std::cout << repr << std::endl;
  auto tree2 = parse(path, repr, false);
  AstDumper dumper2(true);
  dumper2.dump(tree2.get());

  REQUIRE(repr == dumper2.toString());
  // REQUIRE_THAT(repr, DiffersAt(dumper2.toString()));
}

TEST_CASE("hello", "[parser]") { testRepr("../examples/hello.lang"); }

TEST_CASE("primitives", "[parser]") { testRepr("../examples/primitives.lang"); }

TEST_CASE("literals", "[parser]") { testRepr("../examples/literals.lang"); }

TEST_CASE("tuple structs", "[parser]") {
  testRepr("../examples/tuple structs.lang");
}

TEST_CASE("import", "[parser]") { testRepr("../examples/import.lang"); }

TEST_CASE("arrays, slices", "[parser]") {
  testRepr("../examples/arrays, slices.lang");
}

TEST_CASE("structs", "[parser]") { testRepr("../examples/structs.lang"); }

TEST_CASE("enums", "[parser]") { testRepr("../examples/enums.lang"); }

TEST_CASE("c-like enums", "[parser]") {
  testRepr("../examples/c-like enums.lang");
}

TEST_CASE("variable bindings", "[parser]") {
  testRepr("../examples/variable bindings.lang");
}

TEST_CASE("mutability", "[parser]") { testRepr("../examples/mutability.lang"); }

TEST_CASE("scope", "[parser]") { testRepr("../examples/scope.lang"); }

TEST_CASE("declare first", "[parser]") {
  testRepr("../examples/declare first.lang");
}

TEST_CASE("casting", "[parser]") { testRepr("../examples/casting.lang"); }

TEST_CASE("literals 2", "[parser]") { testRepr("../examples/literals 2.lang"); }

TEST_CASE("aliasing", "[parser]") { testRepr("../examples/aliasing.lang"); }

TEST_CASE("type casting", "[parser]") {
  testRepr("../examples/type casting.lang");
}

TEST_CASE("formatter", "[parser]") { testRepr("../examples/formatter.lang"); }

TEST_CASE("block expressions", "[parser]") {
  testRepr("../examples/block expressions.lang");
}

TEST_CASE("if else", "[parser]") { testRepr("../examples/if else.lang"); }

TEST_CASE("loops", "[parser]") { testRepr("../examples/loops.lang"); }

TEST_CASE("returning from loops", "[parser]") {
  testRepr("../examples/returning from loops.lang");
}

TEST_CASE("for in", "[parser]") { testRepr("../examples/for in.lang"); }

TEST_CASE("for each", "[parser]") { testRepr("../examples/for each.lang"); }

TEST_CASE("for each mut", "[parser]") {
  testRepr("../examples/for each mut.lang");
}

TEST_CASE("match", "[parser]") { testRepr("../examples/match.lang"); }

TEST_CASE("match destructuring", "[parser]") {
  testRepr("../examples/match destructuring.lang");
}

TEST_CASE("match enums", "[parser]") {
  testRepr("../examples/match enums.lang");
}

TEST_CASE("match struct pattern", "[parser]") {
  testRepr("../examples/match struct pattern.lang");
}

TEST_CASE("functions", "[parser]") { testRepr("../examples/functions.lang"); }

TEST_CASE("methods", "[parser]") { testRepr("../examples/methods.lang"); }

TEST_CASE("method 2", "[parser]") { testRepr("../examples/method 2.lang"); }

TEST_CASE("higher order functions", "[parser]") {
  testRepr("../examples/higher order functions.lang");
}

TEST_CASE("no return, hof", "[parser]") {
  testRepr("../examples/no return, hof.lang");
}

TEST_CASE("traits", "[parser]") { testRepr("../examples/traits.lang"); }

TEST_CASE("supertraits", "[parser]") {
  testRepr("../examples/super_traits.lang");
}

TEST_CASE("basic comptime", "[parser]") {
  testRepr("../examples/basic_comptime.lang");
}

TEST_CASE("comptime trait", "[parser]") {
  testRepr("../examples/comptime_trait.lang");
}

TEST_CASE("multiple trait constraints", "[parser]") {
  testRepr("../examples/multiple_trait_constraint.lang");
}

TEST_CASE("generic struct", "[parser]") {
  testRepr("../examples/generic_struct.lang");
}

TEST_CASE("nested comptime", "[parser]") {
  testRepr("../examples/nested_comptime.lang");
}

// TEST_CASE("mlir types", "[parser]") {
// testRepr("../examples/mlir_types.lang"); }
