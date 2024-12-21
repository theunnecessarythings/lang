#include "compiler.hpp"
#include <catch2/catch_test_macros.hpp>
#include <iostream>

Compiler compiler;

void testBehaviour(const std::string &path, std::string_view expected = "") {
  if (!compiler.initialized)
    compiler.init();
  std::cout << "Compiling " << path << std::endl;
  char buffer[1024];
  memset(buffer, 0, sizeof(buffer));

  FILE *original_stdout = stdout;
  FILE *pipe = tmpfile();
  REQUIRE(pipe != nullptr);
  stdout = pipe;

  compiler.dumpMLIRLang(InputType::Lang, path, true, false);

  fflush(stdout);
  rewind(pipe);
  fread(buffer, 1, sizeof(buffer), pipe);
  stdout = original_stdout;
  fclose(pipe);

  REQUIRE(std::string(buffer) == expected);
}

TEST_CASE("hello", "[behaviour]") {
  testBehaviour("../examples/hello.lang", "Hello, World!\n");
}

TEST_CASE("primitives", "[behaviour]") {
  testBehaviour("../examples/primitives.lang",
                "logical: 1, a_float: 1.000000, an_integer: 5, default_float: "
                "3.000000, default_integer: 7\n"
                "before, mutable: 12\n"
                "after, mutable: 21\n");
}

TEST_CASE("literals", "[behaviour]") {
  testBehaviour("../examples/literals.lang",
                "1 + 2 = 3\n"
                "1 - 2 = -1\n"
                "1e4 is 1.000000e+04, -2.5e-3 is -2.500000e-03\n"
                "true AND false is 0\n"
                "true OR false is 1\n"
                "NOT true is 0\n"
                "0011 AND 0101 is 1\n"
                "0011 OR 0101 is 7\n"
                "0011 XOR 0101 is 6\n"
                "1 << 5 is 32\n"
                "0x80 >> 2 is 32\n"
                "One million is written as 1000000\n");
}

TEST_CASE("tuple structs", "[behaviour]") {
  testBehaviour("../examples/tuple structs.lang",
                "tuple of tuples: 4\n"
                "Pair is (1, 0)\n"
                "One element tuple: 6\n"
                "Just an integer: 5\n"
                "Matrix: 1.100000, 1.200000, 2.100000, 2.200000\n");
}
