#include "ast.hpp"
#include "parser.hpp"
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>

auto parse(const std::string &path, std::string &str,
           bool print_tokens = false) {

  std::shared_ptr<Context> context = std::make_shared<Context>();
  std::unique_ptr<Lexer> lexer = std::make_unique<Lexer>(str, 0);

  if (print_tokens) {
    while (true) {
      Token token = lexer->next();
      if (token.kind == TokenKind::Eof) {
        break;
      }
      std::cout << Lexer::lexeme(token.kind) << " : "
                << lexer->token_to_string(token) << std::endl;
    }
  }

  Parser parser(std::move(lexer), context);
  auto tree = parser.parse_program();
  return tree;
}

void test_repr(const std::string &path) {
  std::cout << "Parsing " << path << std::endl;
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Could not open file " + path);
  std::string str((std::istreambuf_iterator<char>(file)),
                  std::istreambuf_iterator<char>());
  auto tree = parse(path, str, false);

  AstDumper dumper(true);
  dumper.dump(tree.get());
  auto repr = dumper.to_string();

  auto tree2 = parse(path, repr, false);
  AstDumper dumper2(true);
  dumper2.dump(tree2.get());

  REQUIRE(repr == dumper2.to_string());
}

TEST_CASE("hello", "[parser]") { test_repr("../examples/hello.lang"); }

TEST_CASE("primitives", "[parser]") {
  test_repr("../examples/primitives.lang");
}

TEST_CASE("literals", "[parser]") { test_repr("../examples/literals.lang"); }

TEST_CASE("tuple structs", "[parser]") {
  test_repr("../examples/tuple structs.lang");
}

TEST_CASE("import", "[parser]") { test_repr("../examples/import.lang"); }

TEST_CASE("arrays, slices", "[parser]") {
  test_repr("../examples/arrays, slices.lang");
}

TEST_CASE("structs", "[parser]") { test_repr("../examples/structs.lang"); }

TEST_CASE("enums", "[parser]") { test_repr("../examples/enums.lang"); }

TEST_CASE("c-like enums", "[parser]") {
  test_repr("../examples/c-like enums.lang");
}

TEST_CASE("variable bindings", "[parser]") {
  test_repr("../examples/variable bindings.lang");
}

TEST_CASE("mutability", "[parser]") {
  test_repr("../examples/mutability.lang");
}

TEST_CASE("scope", "[parser]") { test_repr("../examples/scope.lang"); }

TEST_CASE("declare first", "[parser]") {
  test_repr("../examples/declare first.lang");
}

TEST_CASE("casting", "[parser]") { test_repr("../examples/casting.lang"); }

TEST_CASE("literals 2", "[parser]") {
  test_repr("../examples/literals 2.lang");
}

TEST_CASE("aliasing", "[parser]") { test_repr("../examples/aliasing.lang"); }

TEST_CASE("type casting", "[parser]") {
  test_repr("../examples/type casting.lang");
}

TEST_CASE("formatter", "[parser]") { test_repr("../examples/formatter.lang"); }

TEST_CASE("block expressions", "[parser]") {
  test_repr("../examples/block expressions.lang");
}

TEST_CASE("if else", "[parser]") { test_repr("../examples/if else.lang"); }

TEST_CASE("loops", "[parser]") { test_repr("../examples/loops.lang"); }

TEST_CASE("returning from loops", "[parser]") {
  test_repr("../examples/returning from loops.lang");
}

TEST_CASE("for in", "[parser]") { test_repr("../examples/for in.lang"); }

TEST_CASE("for each", "[parser]") { test_repr("../examples/for each.lang"); }

TEST_CASE("for each mut", "[parser]") {
  test_repr("../examples/for each mut.lang");
}

TEST_CASE("match", "[parser]") { test_repr("../examples/match.lang"); }

TEST_CASE("match destructuring", "[parser]") {
  test_repr("../examples/match destructuring.lang");
}

TEST_CASE("match enums", "[parser]") {
  test_repr("../examples/match enums.lang");
}

TEST_CASE("match struct pattern", "[parser]") {
  test_repr("../examples/match struct pattern.lang");
}

TEST_CASE("functions", "[parser]") { test_repr("../examples/functions.lang"); }

TEST_CASE("methods", "[parser]") { test_repr("../examples/methods.lang"); }

TEST_CASE("method 2", "[parser]") { test_repr("../examples/method 2.lang"); }

TEST_CASE("higher order functions", "[parser]") {
  test_repr("../examples/higher order functions.lang");
}

TEST_CASE("no return, hof", "[parser]") {
  test_repr("../examples/no return, hof.lang");
}

TEST_CASE("traits", "[parser]") { test_repr("../examples/traits.lang"); }

TEST_CASE("supertraits", "[parser]") {
  test_repr("../examples/super_traits.lang");
}

TEST_CASE("basic comptime", "[parser]") {
  test_repr("../examples/basic_comptime.lang");
}

TEST_CASE("comptime trait", "[parser]") {
  test_repr("../examples/comptime_trait.lang");
}

TEST_CASE("multiple trait constraints", "[parser]") {
  test_repr("../examples/multiple_trait_constraint.lang");
}

TEST_CASE("generic struct", "[parser]") {
  test_repr("../examples/generic_struct.lang");
}

TEST_CASE("nested comptime", "[parser]") {
  test_repr("../examples/nested_comptime.lang");
}

TEST_CASE("mlir types", "[parser]") {
  test_repr("../examples/mlir_types.lang");
}
