#include "compiler.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

void testBehaviour(const std::string &path, std::string_view expected = "") {
  std::cout << "Compiling " << path << std::endl;

  char buffer[1024];
  memset(buffer, 0, sizeof(buffer));

  FILE *originalStdout = stdout;
  FILE *pipe = tmpfile();
  REQUIRE(pipe != nullptr);
  stdout = pipe;

  Compiler::dumpMLIRLang(InputType::Lang, path, true);

  fflush(stdout);
  rewind(pipe);
  fread(buffer, 1, sizeof(buffer), pipe);
  stdout = originalStdout;
  fclose(pipe);

  REQUIRE(std::string(buffer) == expected);
}

// TEST_CASE("hello", "[behaviour]") {
//   testBehaviour("../examples/hello.lang", "Hello, World!\n");
// }

// TEST_CASE("primitives", "[behaviour]") {
//   testBehaviour("../examples/primitives.lang",
//                 "(logical: 1, a_float: 1.000000, an_integer: 5,
//                 default_float: " "3.000000, default_integer: 7" "before,
//                 mutable: 12" "after, mutable: 21");
// }
