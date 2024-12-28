#include "analyzer.hpp"
#include "compiler.hpp"
#include "parser.hpp"
#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <iostream>

auto parse(const std::string &path, std::string &str, int file_id,
           std::shared_ptr<Context> context, bool print_tokens = false) {

  std::unique_ptr<Lexer> lexer = std::make_unique<Lexer>(str, file_id);

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
  auto tree = parser.parseProgram();
  return tree;
}

void testAnalyzer(const std::string &path) {
  std::cout << "Analyzing " << path << std::endl;
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Could not open file " + path);
  std::string str((std::istreambuf_iterator<char>(file)),
                  std::istreambuf_iterator<char>());

  std::shared_ptr<Context> context = std::make_shared<Context>();
  auto tree = parse(path, str, 0, context, false);

  Analyzer analyzer(context);
  analyzer.analyze(tree.get());
}

TEST_CASE("expr", "[analyzer]") { testAnalyzer("../examples/expr.lang"); }

TEST_CASE("hello", "[analyzer]") { testAnalyzer("../examples/hello.lang"); }

TEST_CASE("primitives", "[analyzer]") {
  testAnalyzer("../examples/primitives.lang");
}

TEST_CASE("literals", "[analyzer]") {
  testAnalyzer("../examples/literals.lang");
}

TEST_CASE("tuple structs", "[analyzer]") {
  testAnalyzer("../examples/tuple structs.lang");
}

TEST_CASE("import", "[analyzer]") { testAnalyzer("../examples/import.lang"); }

TEST_CASE("arrays, slices", "[analyzer]") {
  testAnalyzer("../examples/arrays, slices.lang");
}

TEST_CASE("structs", "[analyzer]") { testAnalyzer("../examples/structs.lang"); }

TEST_CASE("enums", "[analyzer]") { testAnalyzer("../examples/enums.lang"); }

TEST_CASE("c-like enums", "[analyzer]") {
  testAnalyzer("../examples/c-like enums.lang");
}

TEST_CASE("variable bindings", "[analyzer]") {
  testAnalyzer("../examples/variable bindings.lang");
}

TEST_CASE("mutability", "[analyzer]") {
  testAnalyzer("../examples/mutability.lang");
}

TEST_CASE("scope", "[analyzer]") { testAnalyzer("../examples/scope.lang"); }

TEST_CASE("declare first", "[analyzer]") {
  testAnalyzer("../examples/declare first.lang");
}

TEST_CASE("casting", "[analyzer]") { testAnalyzer("../examples/casting.lang"); }

TEST_CASE("literals 2", "[analyzer]") {
  testAnalyzer("../examples/literals 2.lang");
}

TEST_CASE("aliasing", "[analyzer]") {
  testAnalyzer("../examples/aliasing.lang");
}

TEST_CASE("type casting", "[analyzer]") {
  testAnalyzer("../examples/type casting.lang");
}

TEST_CASE("formatter", "[analyzer]") {
  testAnalyzer("../examples/formatter.lang");
}

TEST_CASE("block expressions", "[analyzer]") {
  testAnalyzer("../examples/block expressions.lang");
}

TEST_CASE("if else", "[analyzer]") { testAnalyzer("../examples/if else.lang"); }

TEST_CASE("loops", "[analyzer]") { testAnalyzer("../examples/loops.lang"); }

TEST_CASE("returning from loops", "[analyzer]") {
  testAnalyzer("../examples/returning from loops.lang");
}

TEST_CASE("for in", "[analyzer]") { testAnalyzer("../examples/for in.lang"); }

TEST_CASE("for each", "[analyzer]") {
  testAnalyzer("../examples/for each.lang");
}

TEST_CASE("for each mut", "[analyzer]") {
  testAnalyzer("../examples/for each mut.lang");
}

TEST_CASE("match", "[analyzer]") { testAnalyzer("../examples/match.lang"); }

TEST_CASE("match destructuring", "[analyzer]") {
  testAnalyzer("../examples/match destructuring.lang");
}

TEST_CASE("match enums", "[analyzer]") {
  testAnalyzer("../examples/match enums.lang");
}

TEST_CASE("match struct pattern", "[analyzer]") {
  testAnalyzer("../examples/match struct pattern.lang");
}

TEST_CASE("functions", "[analyzer]") {
  testAnalyzer("../examples/functions.lang");
}

TEST_CASE("methods", "[analyzer]") { testAnalyzer("../examples/methods.lang"); }

TEST_CASE("method 2", "[analyzer]") {
  testAnalyzer("../examples/method 2.lang");
}

TEST_CASE("higher order functions", "[analyzer]") {
  testAnalyzer("../examples/higher order functions.lang");
}

TEST_CASE("no return, hof", "[analyzer]") {
  testAnalyzer("../examples/no return, hof.lang");
}

TEST_CASE("traits", "[analyzer]") { testAnalyzer("../examples/traits.lang"); }

TEST_CASE("supertraits", "[analyzer]") {
  testAnalyzer("../examples/super_traits.lang");
}

TEST_CASE("basic comptime", "[analyzer]") {
  testAnalyzer("../examples/basic_comptime.lang");
}

TEST_CASE("comptime trait", "[analyzer]") {
  testAnalyzer("../examples/comptime_trait.lang");
}

TEST_CASE("multiple trait constraints", "[analyzer]") {
  testAnalyzer("../examples/multiple_trait_constraint.lang");
}

TEST_CASE("generic struct", "[analyzer]") {
  testAnalyzer("../examples/generic_struct.lang");
}

TEST_CASE("nested comptime", "[analyzer]") {
  testAnalyzer("../examples/nested_comptime.lang");
}
