#include "analyzer.hpp"
#include "compiler.hpp"
#include "parser.hpp"
#include <catch2/catch_test_macros.hpp>
#include <fstream>

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
                << lexer->token_to_string(token) << std::endl;
    }
  }

  Parser parser(std::move(lexer), context);
  auto tree = parser.parse_program();
  if (context->diagnostics.level_count(Diagnostic::Level::Error) > 0) {
    context->diagnostics.report(Diagnostic::Level::Error);
    REQUIRE(false);
  }
  if (context->diagnostics.level_count(Diagnostic::Level::Warning) > 0) {
    context->diagnostics.report(Diagnostic::Level::Warning);
    REQUIRE(false);
  }
  return tree;
}

void test_analyzer(const std::string &path) {
  std::cout << "Parsing " << path << std::endl;
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Could not open file " + path);
  std::string str((std::istreambuf_iterator<char>(file)),
                  std::istreambuf_iterator<char>());

  std::shared_ptr<SourceManager> source_manager =
      std::make_shared<SourceManager>();
  std::shared_ptr<Context> context = std::make_shared<Context>(source_manager);
  int file_id = context->source_manager->add_path(path, str);

  auto tree = parse(path, str, file_id, context, false);

  Analyzer analyzer(context);
  analyzer.analyze(tree.get());

  if (context->diagnostics.level_count(Diagnostic::Level::Error) > 0) {
    context->diagnostics.report(Diagnostic::Level::Error);
    REQUIRE(false);
  }
}

TEST_CASE("expr", "[analyzer]") { test_analyzer("../examples/expr.lang"); }

// TEST_CASE("hello", "[analyzer]") { test_analyzer("../examples/hello.lang"); }
//
// TEST_CASE("primitives", "[analyzer]") {
//   test_analyzer("../examples/primitives.lang");
// }
//
// TEST_CASE("literals", "[analyzer]") {
//   test_analyzer("../examples/literals.lang");
// }
//
// TEST_CASE("tuple structs", "[analyzer]") {
//   test_analyzer("../examples/tuple structs.lang");
// }
//
// TEST_CASE("import", "[analyzer]") { test_analyzer("../examples/import.lang");
// }
//
// TEST_CASE("arrays, slices", "[analyzer]") {
//   test_analyzer("../examples/arrays, slices.lang");
// }
//
// TEST_CASE("structs", "[analyzer]") {
//   test_analyzer("../examples/structs.lang");
// }
//
// TEST_CASE("enums", "[analyzer]") { test_analyzer("../examples/enums.lang"); }
//
// TEST_CASE("c-like enums", "[analyzer]") {
//   test_analyzer("../examples/c-like enums.lang");
// }
//
// TEST_CASE("variable bindings", "[analyzer]") {
//   test_analyzer("../examples/variable bindings.lang");
// }
//
// TEST_CASE("mutability", "[analyzer]") {
//   test_analyzer("../examples/mutability.lang");
// }
//
// TEST_CASE("scope", "[analyzer]") { test_analyzer("../examples/scope.lang"); }
//
// TEST_CASE("declare first", "[analyzer]") {
//   test_analyzer("../examples/declare first.lang");
// }
//
// TEST_CASE("casting", "[analyzer]") {
//   test_analyzer("../examples/casting.lang");
// }
//
// TEST_CASE("literals 2", "[analyzer]") {
//   test_analyzer("../examples/literals 2.lang");
// }
//
// TEST_CASE("aliasing", "[analyzer]") {
//   test_analyzer("../examples/aliasing.lang");
// }
//
// TEST_CASE("type casting", "[analyzer]") {
//   test_analyzer("../examples/type casting.lang");
// }
//
// TEST_CASE("formatter", "[analyzer]") {
//   test_analyzer("../examples/formatter.lang");
// }
//
// TEST_CASE("block expressions", "[analyzer]") {
//   test_analyzer("../examples/block expressions.lang");
// }
//
// TEST_CASE("if else", "[analyzer]") {
//   test_analyzer("../examples/if else.lang");
// }
//
// TEST_CASE("loops", "[analyzer]") { test_analyzer("../examples/loops.lang"); }
//
// TEST_CASE("returning from loops", "[analyzer]") {
//   test_analyzer("../examples/returning from loops.lang");
// }
//
// TEST_CASE("for in", "[analyzer]") { test_analyzer("../examples/for in.lang");
// }
//
// TEST_CASE("for each", "[analyzer]") {
//   test_analyzer("../examples/for each.lang");
// }
//
// TEST_CASE("for each mut", "[analyzer]") {
//   test_analyzer("../examples/for each mut.lang");
// }
//
// TEST_CASE("match", "[analyzer]") { test_analyzer("../examples/match.lang"); }
//
// TEST_CASE("match destructuring", "[analyzer]") {
//   test_analyzer("../examples/match destructuring.lang");
// }
//
// TEST_CASE("match enums", "[analyzer]") {
//   test_analyzer("../examples/match enums.lang");
// }
//
// TEST_CASE("match struct pattern", "[analyzer]") {
//   test_analyzer("../examples/match struct pattern.lang");
// }
//
// TEST_CASE("functions", "[analyzer]") {
//   test_analyzer("../examples/functions.lang");
// }
//
// TEST_CASE("methods", "[analyzer]") {
//   test_analyzer("../examples/methods.lang");
// }
//
// TEST_CASE("method 2", "[analyzer]") {
//   test_analyzer("../examples/method 2.lang");
// }
//
// TEST_CASE("higher order functions", "[analyzer]") {
//   test_analyzer("../examples/higher order functions.lang");
// }
//
// TEST_CASE("no return, hof", "[analyzer]") {
//   test_analyzer("../examples/no return, hof.lang");
// }
//
// TEST_CASE("traits", "[analyzer]") { test_analyzer("../examples/traits.lang");
// }
//
// TEST_CASE("supertraits", "[analyzer]") {
//   test_analyzer("../examples/super_traits.lang");
// }
//
// TEST_CASE("basic comptime", "[analyzer]") {
//   test_analyzer("../examples/basic_comptime.lang");
// }
//
// TEST_CASE("comptime trait", "[analyzer]") {
//   test_analyzer("../examples/comptime_trait.lang");
// }
//
// TEST_CASE("multiple trait constraints", "[analyzer]") {
//   test_analyzer("../examples/multiple_trait_constraint.lang");
// }
//
// TEST_CASE("generic struct", "[analyzer]") {
//   test_analyzer("../examples/generic_struct.lang");
// }
//
// TEST_CASE("nested comptime", "[analyzer]") {
//   test_analyzer("../examples/nested_comptime.lang");
// }
