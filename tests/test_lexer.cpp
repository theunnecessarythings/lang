#include "lexer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <string>

void test_lexer(std::basic_string<char> source,
                std::vector<TokenKind> expected) {
  Lexer lexer(source, 0);
  for (auto &expected_kind : expected) {
    Token token = lexer.next();
    REQUIRE(expected_kind == token.kind);
  }
  Token last_token = lexer.next();
  REQUIRE(TokenKind::Eof == last_token.kind);
  REQUIRE((int)source.size() == last_token.span.start);
  REQUIRE((int)source.size() == last_token.span.end);
}

TEST_CASE("keywords", "[lexer]") {
  test_lexer("match is mut impl as and break const continue else "
             "enum fn for if import "
             "in not or pub return struct var module trait union comptime",
             {
                 TokenKind::KeywordMatch,    TokenKind::KeywordIs,
                 TokenKind::KeywordMut,      TokenKind::KeywordImpl,
                 TokenKind::KeywordAs,       TokenKind::KeywordAnd,
                 TokenKind::KeywordBreak,    TokenKind::KeywordConst,
                 TokenKind::KeywordContinue, TokenKind::KeywordElse,
                 TokenKind::KeywordEnum,     TokenKind::KeywordFn,
                 TokenKind::KeywordFor,      TokenKind::KeywordIf,
                 TokenKind::KeywordImport,   TokenKind::KeywordIn,
                 TokenKind::KeywordNot,      TokenKind::KeywordOr,
                 TokenKind::KeywordPub,      TokenKind::KeywordReturn,
                 TokenKind::KeywordStruct,   TokenKind::KeywordVar,
                 TokenKind::KeywordModule,   TokenKind::KeywordTrait,
                 TokenKind::KeywordUnion,    TokenKind::KeywordComptime,
             });
}

TEST_CASE("newline in string literal", "[lexer]") {
  test_lexer("\"\n\"", {TokenKind::Invalid, TokenKind::Invalid});
}

TEST_CASE("float literal e exponent", "[lexer]") {
  test_lexer("a = 4.94065645841246544177e-324;\n",
             {TokenKind::Identifier, TokenKind::Equal, TokenKind::NumberLiteral,
              TokenKind::Semicolon});
}

TEST_CASE("float literal p exponent", "[lexer]") {
  test_lexer("a = 0x1.a827999fcef32p+1022;\n",
             {TokenKind::Identifier, TokenKind::Equal, TokenKind::NumberLiteral,
              TokenKind::Semicolon});
}

TEST_CASE("pipe and then Invalid", "[lexer]") {
  test_lexer("||=", {TokenKind::PipePipe, TokenKind::Equal});
}

TEST_CASE("line comment", "[lexer]") { test_lexer("//", {}); }

TEST_CASE("line comment followed by identifier", "[lexer]") {
  test_lexer("//\nAnother", {TokenKind::Identifier});
}

TEST_CASE("number literals decimal", "[lexer]") {
  test_lexer("0", {TokenKind::NumberLiteral});
  test_lexer("1", {TokenKind::NumberLiteral});
  test_lexer("2", {TokenKind::NumberLiteral});
  test_lexer("3", {TokenKind::NumberLiteral});
  test_lexer("4", {TokenKind::NumberLiteral});
  test_lexer("5", {TokenKind::NumberLiteral});
  test_lexer("6", {TokenKind::NumberLiteral});
  test_lexer("7", {TokenKind::NumberLiteral});
  test_lexer("8", {TokenKind::NumberLiteral});
  test_lexer("9", {TokenKind::NumberLiteral});
  test_lexer("1..", {TokenKind::NumberLiteral, TokenKind::DotDot});
  test_lexer("0a", {TokenKind::NumberLiteral});
  test_lexer("9b", {TokenKind::NumberLiteral});
  test_lexer("1z", {TokenKind::NumberLiteral});
  test_lexer("1z_1", {TokenKind::NumberLiteral});
  test_lexer("9z3", {TokenKind::NumberLiteral});

  test_lexer("0_0", {TokenKind::NumberLiteral});
  test_lexer("0001", {TokenKind::NumberLiteral});
  test_lexer("01234567890", {TokenKind::NumberLiteral});
  test_lexer("012_345_6789_0", {TokenKind::NumberLiteral});
  test_lexer("0_1_2_3_4_5_6_7_8_9_0", {TokenKind::NumberLiteral});

  test_lexer("00_", {TokenKind::NumberLiteral});
  test_lexer("0_0_", {TokenKind::NumberLiteral});
  test_lexer("0__0", {TokenKind::NumberLiteral});
  test_lexer("0_0f", {TokenKind::NumberLiteral});
  test_lexer("0_0_f", {TokenKind::NumberLiteral});
  test_lexer("0_0_f_00", {TokenKind::NumberLiteral});
  test_lexer("1_,", {TokenKind::NumberLiteral, TokenKind::Comma});

  test_lexer("0.0", {TokenKind::NumberLiteral});
  test_lexer("1.0", {TokenKind::NumberLiteral});
  test_lexer("10.0", {TokenKind::NumberLiteral});
  test_lexer("0e0", {TokenKind::NumberLiteral});
  test_lexer("1e0", {TokenKind::NumberLiteral});
  test_lexer("1e100", {TokenKind::NumberLiteral});
  test_lexer("1.0e100", {TokenKind::NumberLiteral});
  test_lexer("1.0e+100", {TokenKind::NumberLiteral});
  test_lexer("1.0e-100", {TokenKind::NumberLiteral});
  test_lexer("1_0_0_0.0_0_0_0_0_1e1_0_0_0", {TokenKind::NumberLiteral});

  test_lexer("1.", {TokenKind::NumberLiteral, TokenKind::Dot});
  test_lexer("1e", {TokenKind::NumberLiteral});
  test_lexer("1.e100", {TokenKind::NumberLiteral});
  test_lexer("1.0e1f0", {TokenKind::NumberLiteral});
  test_lexer("1.0p100", {TokenKind::NumberLiteral});
  test_lexer("1.0p-100", {TokenKind::NumberLiteral});
  test_lexer("1.0p1f0", {TokenKind::NumberLiteral});
  test_lexer("1._+", {TokenKind::NumberLiteral, TokenKind::Plus});
  test_lexer("1._e", {TokenKind::NumberLiteral});
  test_lexer("1.0e", {TokenKind::NumberLiteral});
  test_lexer("1.0e,", {TokenKind::NumberLiteral, TokenKind::Comma});
  test_lexer("1.0e_", {TokenKind::NumberLiteral});
  test_lexer("1.0e+_", {TokenKind::NumberLiteral});
  test_lexer("1.0e-_", {TokenKind::NumberLiteral});
  test_lexer("1.0e0_+", {TokenKind::NumberLiteral, TokenKind::Plus});
}

TEST_CASE("number literal binary", "[lexer]") {
  test_lexer("0b0", {TokenKind::NumberLiteral});
  test_lexer("0b1", {TokenKind::NumberLiteral});
  test_lexer("0b2", {TokenKind::NumberLiteral});
  test_lexer("0b3", {TokenKind::NumberLiteral});
  test_lexer("0b4", {TokenKind::NumberLiteral});
  test_lexer("0b5", {TokenKind::NumberLiteral});
  test_lexer("0b6", {TokenKind::NumberLiteral});
  test_lexer("0b7", {TokenKind::NumberLiteral});
  test_lexer("0b8", {TokenKind::NumberLiteral});
  test_lexer("0b9", {TokenKind::NumberLiteral});
  test_lexer("0ba", {TokenKind::NumberLiteral});
  test_lexer("0bb", {TokenKind::NumberLiteral});
  test_lexer("0bc", {TokenKind::NumberLiteral});
  test_lexer("0bd", {TokenKind::NumberLiteral});
  test_lexer("0be", {TokenKind::NumberLiteral});
  test_lexer("0bf", {TokenKind::NumberLiteral});
  test_lexer("0bz", {TokenKind::NumberLiteral});

  test_lexer("0b0000_0000", {TokenKind::NumberLiteral});
  test_lexer("0b1111_1111", {TokenKind::NumberLiteral});
  test_lexer("0b10_10_10_10", {TokenKind::NumberLiteral});
  test_lexer("0b0_1_0_1_0_1_0_1", {TokenKind::NumberLiteral});
  test_lexer("0b1.", {TokenKind::NumberLiteral, TokenKind::Dot});
  test_lexer("0b1.0", {TokenKind::NumberLiteral});

  test_lexer("0B0", {TokenKind::NumberLiteral});
  test_lexer("0b_", {TokenKind::NumberLiteral});
  test_lexer("0b_0", {TokenKind::NumberLiteral});
  test_lexer("0b1_", {TokenKind::NumberLiteral});
  test_lexer("0b0__1", {TokenKind::NumberLiteral});
  test_lexer("0b0_1_", {TokenKind::NumberLiteral});
  test_lexer("0b1e", {TokenKind::NumberLiteral});
  test_lexer("0b1p", {TokenKind::NumberLiteral});
  test_lexer("0b1e0", {TokenKind::NumberLiteral});
  test_lexer("0b1p0", {TokenKind::NumberLiteral});
  test_lexer("0b_,", {TokenKind::NumberLiteral, TokenKind::Comma});
}

TEST_CASE("number literal octal", "[lexer]") {
  test_lexer("0o0", {TokenKind::NumberLiteral});
  test_lexer("0o1", {TokenKind::NumberLiteral});
  test_lexer("0o2", {TokenKind::NumberLiteral});
  test_lexer("0o3", {TokenKind::NumberLiteral});
  test_lexer("0o4", {TokenKind::NumberLiteral});
  test_lexer("0o5", {TokenKind::NumberLiteral});
  test_lexer("0o6", {TokenKind::NumberLiteral});
  test_lexer("0o7", {TokenKind::NumberLiteral});
  test_lexer("0o8", {TokenKind::NumberLiteral});
  test_lexer("0o9", {TokenKind::NumberLiteral});
  test_lexer("0oa", {TokenKind::NumberLiteral});
  test_lexer("0ob", {TokenKind::NumberLiteral});
  test_lexer("0oc", {TokenKind::NumberLiteral});
  test_lexer("0od", {TokenKind::NumberLiteral});
  test_lexer("0oe", {TokenKind::NumberLiteral});
  test_lexer("0of", {TokenKind::NumberLiteral});
  test_lexer("0oz", {TokenKind::NumberLiteral});

  test_lexer("0o01234567", {TokenKind::NumberLiteral});
  test_lexer("0o0123_4567", {TokenKind::NumberLiteral});
  test_lexer("0o01_23_45_67", {TokenKind::NumberLiteral});
  test_lexer("0o0_1_2_3_4_5_6_7", {TokenKind::NumberLiteral});

  test_lexer("0O0", {TokenKind::NumberLiteral});
  test_lexer("0o_", {TokenKind::NumberLiteral});
  test_lexer("0o_0", {TokenKind::NumberLiteral});
  test_lexer("0o1_", {TokenKind::NumberLiteral});
  test_lexer("0o0__1", {TokenKind::NumberLiteral});
  test_lexer("0o0_1_", {TokenKind::NumberLiteral});
  test_lexer("0o1e", {TokenKind::NumberLiteral});
  test_lexer("0o1p", {TokenKind::NumberLiteral});
  test_lexer("0o1e0", {TokenKind::NumberLiteral});
  test_lexer("0o1p0", {TokenKind::NumberLiteral});
  test_lexer("0o_,", {TokenKind::NumberLiteral, TokenKind::Comma});
}

TEST_CASE("number literals hexadecimal", "[lexer]") {
  test_lexer("0x0", {TokenKind::NumberLiteral});
  test_lexer("0x1", {TokenKind::NumberLiteral});
  test_lexer("0x2", {TokenKind::NumberLiteral});
  test_lexer("0x3", {TokenKind::NumberLiteral});
  test_lexer("0x4", {TokenKind::NumberLiteral});
  test_lexer("0x5", {TokenKind::NumberLiteral});
  test_lexer("0x6", {TokenKind::NumberLiteral});
  test_lexer("0x7", {TokenKind::NumberLiteral});
  test_lexer("0x8", {TokenKind::NumberLiteral});
  test_lexer("0x9", {TokenKind::NumberLiteral});
  test_lexer("0xa", {TokenKind::NumberLiteral});
  test_lexer("0xb", {TokenKind::NumberLiteral});
  test_lexer("0xc", {TokenKind::NumberLiteral});
  test_lexer("0xd", {TokenKind::NumberLiteral});
  test_lexer("0xe", {TokenKind::NumberLiteral});
  test_lexer("0xf", {TokenKind::NumberLiteral});
  test_lexer("0xA", {TokenKind::NumberLiteral});
  test_lexer("0xB", {TokenKind::NumberLiteral});
  test_lexer("0xC", {TokenKind::NumberLiteral});
  test_lexer("0xD", {TokenKind::NumberLiteral});
  test_lexer("0xE", {TokenKind::NumberLiteral});
  test_lexer("0xF", {TokenKind::NumberLiteral});
  test_lexer("0x0z", {TokenKind::NumberLiteral});
  test_lexer("0xz", {TokenKind::NumberLiteral});

  test_lexer("0x0123456789ABCDEF", {TokenKind::NumberLiteral});
  test_lexer("0x0123_4567_89AB_CDEF", {TokenKind::NumberLiteral});
  test_lexer("0x01_23_45_67_89AB_CDE_F", {TokenKind::NumberLiteral});
  test_lexer("0x0_1_2_3_4_5_6_7_8_9_A_B_C_D_E_F", {TokenKind::NumberLiteral});

  test_lexer("0X0", {TokenKind::NumberLiteral});
  test_lexer("0x_", {TokenKind::NumberLiteral});
  test_lexer("0x_1", {TokenKind::NumberLiteral});
  test_lexer("0x1_", {TokenKind::NumberLiteral});
  test_lexer("0x0__1", {TokenKind::NumberLiteral});
  test_lexer("0x0_1_", {TokenKind::NumberLiteral});
  test_lexer("0x_,", {TokenKind::NumberLiteral, TokenKind::Comma});

  test_lexer("0x1.0", {TokenKind::NumberLiteral});
  test_lexer("0xF.0", {TokenKind::NumberLiteral});
  test_lexer("0xF.F", {TokenKind::NumberLiteral});
  test_lexer("0xF.Fp0", {TokenKind::NumberLiteral});
  test_lexer("0xF.FP0", {TokenKind::NumberLiteral});
  test_lexer("0x1p0", {TokenKind::NumberLiteral});
  test_lexer("0xfp0", {TokenKind::NumberLiteral});
  test_lexer("0x1.0+0xF.0", {TokenKind::NumberLiteral, TokenKind::Plus,
                             TokenKind::NumberLiteral});

  test_lexer("0x1.", {TokenKind::NumberLiteral, TokenKind::Dot});
  test_lexer("0xF.", {TokenKind::NumberLiteral, TokenKind::Dot});
  test_lexer("0x1.+0xF.",
             {TokenKind::NumberLiteral, TokenKind::Dot, TokenKind::Plus,
              TokenKind::NumberLiteral, TokenKind::Dot});
  test_lexer("0xff.p10", {TokenKind::NumberLiteral});

  test_lexer("0x0123456.789ABCDEF", {TokenKind::NumberLiteral});
  test_lexer("0x0_123_456.789_ABC_DEF", {TokenKind::NumberLiteral});
  test_lexer("0x0_1_2_3_4_5_6.7_8_9_A_B_C_D_E_F", {TokenKind::NumberLiteral});
  test_lexer("0x0p0", {TokenKind::NumberLiteral});
  test_lexer("0x0.0p0", {TokenKind::NumberLiteral});
  test_lexer("0xff.ffp10", {TokenKind::NumberLiteral});
  test_lexer("0xff.ffP10", {TokenKind::NumberLiteral});
  test_lexer("0xffp10", {TokenKind::NumberLiteral});
  test_lexer("0xff_ff.ff_ffp1_0_0_0", {TokenKind::NumberLiteral});
  test_lexer("0xf_f_f_f.f_f_f_fp+1_000", {TokenKind::NumberLiteral});
  test_lexer("0xf_f_f_f.f_f_f_fp-1_00_0", {TokenKind::NumberLiteral});

  test_lexer("0x1e", {TokenKind::NumberLiteral});
  test_lexer("0x1e0", {TokenKind::NumberLiteral});
  test_lexer("0x1p", {TokenKind::NumberLiteral});
  test_lexer("0xfp0z1", {TokenKind::NumberLiteral});
  test_lexer("0xff.ffpff", {TokenKind::NumberLiteral});
  test_lexer("0x0.p", {TokenKind::NumberLiteral});
  test_lexer("0x0.z", {TokenKind::NumberLiteral});
  test_lexer("0x0._", {TokenKind::NumberLiteral});
  test_lexer("0x0_.0", {TokenKind::NumberLiteral});
  test_lexer("0x0_.0.0", {TokenKind::NumberLiteral, TokenKind::Dot,
                          TokenKind::NumberLiteral});
  test_lexer("0x0._0", {TokenKind::NumberLiteral});
  test_lexer("0x0.0_", {TokenKind::NumberLiteral});
  test_lexer("0x0_p0", {TokenKind::NumberLiteral});
  test_lexer("0x0_.p0", {TokenKind::NumberLiteral});
  test_lexer("0x0._p0", {TokenKind::NumberLiteral});
  test_lexer("0x0.0_p0", {TokenKind::NumberLiteral});
  test_lexer("0x0._0p0", {TokenKind::NumberLiteral});
  test_lexer("0x0.0p_0", {TokenKind::NumberLiteral});
  test_lexer("0x0.0p+_0", {TokenKind::NumberLiteral});
  test_lexer("0x0.0p-_0", {TokenKind::NumberLiteral});
  test_lexer("0x0.0p0_", {TokenKind::NumberLiteral});
}

TEST_CASE("code point literal with hex escape", "[lexer]") {
  test_lexer({0x27, 0x5c, 0x78, 0x31, 0x62, 0x27}, {TokenKind::CharLiteral});
  test_lexer({0x27, 0x5c, 0x78, 0x31, 0x27},
             {TokenKind::Invalid, TokenKind::Invalid});
}
TEST_CASE("newline in char literal", "[lexer]") {
  test_lexer("'\n'", {TokenKind::Invalid, TokenKind::Invalid});
}

// TEST_CASE("code point literal with unicode escapes", "[lexer]") {
//   test_lexer("'\u{3}'", {TokenKind::CharLiteral});
//   test_lexer("'\u{01}'", {TokenKind::CharLiteral});
//   test_lexer("'\u{2a}'", {TokenKind::CharLiteral});
//   test_lexer("'\u{3f9}'", {TokenKind::CharLiteral});
//   test_lexer("\"\u{440}\"", {TokenKind::StringLiteral});
//   // test_lexer("'\u{6E09aBc1523}'", {TokenKind::CharLiteral});
//   // test_lexer("'\u'", {TokenKind::Invalid});
//   // test_lexer("'\u{{'", {TokenKind::Invalid, TokenKind::Invalid});
//   // test_lexer("'\u{}'", {TokenKind::CharLiteral});
//   // test_lexer("'\u{s}'", {TokenKind::Invalid, TokenKind::Invalid});
//   // test_lexer("'\u{2z}'", {TokenKind::Invalid, TokenKind::Invalid});
//   // test_lexer("'\u{4a'", {TokenKind::Invalid});
//   // test_lexer("'\u0333'", {TokenKind::Invalid, TokenKind::Invalid});
//   // test_lexer("'\U0333'", {TokenKind::Invalid, TokenKind::NumberLiteral,
//   //                         TokenKind::Invalid});
// }

TEST_CASE("chars", "[lexer]") { test_lexer("'c'", {TokenKind::CharLiteral}); }

TEST_CASE("Invalid token characters", "[lexer]") {
  test_lexer("#", {TokenKind::Invalid});
  test_lexer("`", {TokenKind::Invalid});
  test_lexer("'c", {TokenKind::Invalid});
  test_lexer("'", {TokenKind::Invalid});
  test_lexer("''", {TokenKind::Invalid, TokenKind::Invalid});
}

TEST_CASE("Invalid literal/comment characters", "[lexer]") {
  test_lexer({0x22, 0x00, 0x22},
             {TokenKind::StringLiteral, TokenKind::Invalid});
  test_lexer({0x2f, 0x2f, 0x00}, {TokenKind::Invalid});
  test_lexer("//\x1f", {TokenKind::Invalid});
  test_lexer("//\x7f", {TokenKind::Invalid});
}

TEST_CASE("utf8", "[lexer]") {
  test_lexer("//\xc2\x80", {});
  test_lexer("//\xf4\x8f\xbf\xbf", {});
}

TEST_CASE("Invalid utf8", "[lexer]") {
  test_lexer("//\x80", {TokenKind::Invalid});
  test_lexer("//\xbf", {TokenKind::Invalid});
  test_lexer("//\xf8", {TokenKind::Invalid});
  test_lexer("//\xff", {TokenKind::Invalid});
  test_lexer("//\xc2\xc0", {TokenKind::Invalid});
  test_lexer("//\xe0", {TokenKind::Invalid});
  test_lexer("//\xf0", {TokenKind::Invalid});
  test_lexer("//\xf0\x90\x80\xc0", {TokenKind::Invalid});
}

TEST_CASE("illegal unicode codepoints", "[lexer]") {
  test_lexer("//\xc2\x84", {});
  test_lexer("//\xc2\x85", {TokenKind::Invalid});
  test_lexer("//\xc2\x86", {});
  test_lexer("//\xe2\x80\xa7", {});
  test_lexer("//\xe2\x80\xa8", {TokenKind::Invalid});
  test_lexer("//\xe2\x80\xa9", {TokenKind::Invalid});
  test_lexer("//\xe2\x80\xaa", {});
}

TEST_CASE("shift operator", "[lexer]") {
  test_lexer("<<", {TokenKind::LessLess});
  test_lexer(">>", {TokenKind::GreaterGreater});
}

TEST_CASE("code point literal with unicode code point", "[lexer]") {
  test_lexer("'💩'", {TokenKind::CharLiteral});
}

TEST_CASE("@", "[lexer]") {
  test_lexer("@mlir(\"i32\")",
             {TokenKind::At, TokenKind::Identifier, TokenKind::LParen,
              TokenKind::StringLiteral, TokenKind::RParen});
}
