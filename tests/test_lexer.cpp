#include "lexer.hpp"
#include <catch2/catch_test_macros.hpp>
#include <string>

void testLexer(std::basic_string<char> source,
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
  testLexer("match is mut impl as and break const continue else "
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
  testLexer("\"\n\"", {TokenKind::Invalid, TokenKind::Invalid});
}

TEST_CASE("float literal e exponent", "[lexer]") {
  testLexer("a = 4.94065645841246544177e-324;\n",
            {TokenKind::Identifier, TokenKind::Equal, TokenKind::NumberLiteral,
             TokenKind::Semicolon});
}

TEST_CASE("float literal p exponent", "[lexer]") {
  testLexer("a = 0x1.a827999fcef32p+1022;\n",
            {TokenKind::Identifier, TokenKind::Equal, TokenKind::NumberLiteral,
             TokenKind::Semicolon});
}

TEST_CASE("pipe and then Invalid", "[lexer]") {
  testLexer("||=", {TokenKind::PipePipe, TokenKind::Equal});
}

TEST_CASE("line comment", "[lexer]") { testLexer("//", {}); }

TEST_CASE("line comment followed by identifier", "[lexer]") {
  testLexer("//\nAnother", {TokenKind::Identifier});
}

TEST_CASE("number literals decimal", "[lexer]") {
  testLexer("0", {TokenKind::NumberLiteral});
  testLexer("1", {TokenKind::NumberLiteral});
  testLexer("2", {TokenKind::NumberLiteral});
  testLexer("3", {TokenKind::NumberLiteral});
  testLexer("4", {TokenKind::NumberLiteral});
  testLexer("5", {TokenKind::NumberLiteral});
  testLexer("6", {TokenKind::NumberLiteral});
  testLexer("7", {TokenKind::NumberLiteral});
  testLexer("8", {TokenKind::NumberLiteral});
  testLexer("9", {TokenKind::NumberLiteral});
  testLexer("1..", {TokenKind::NumberLiteral, TokenKind::DotDot});
  testLexer("0a", {TokenKind::NumberLiteral});
  testLexer("9b", {TokenKind::NumberLiteral});
  testLexer("1z", {TokenKind::NumberLiteral});
  testLexer("1z_1", {TokenKind::NumberLiteral});
  testLexer("9z3", {TokenKind::NumberLiteral});

  testLexer("0_0", {TokenKind::NumberLiteral});
  testLexer("0001", {TokenKind::NumberLiteral});
  testLexer("01234567890", {TokenKind::NumberLiteral});
  testLexer("012_345_6789_0", {TokenKind::NumberLiteral});
  testLexer("0_1_2_3_4_5_6_7_8_9_0", {TokenKind::NumberLiteral});

  testLexer("00_", {TokenKind::NumberLiteral});
  testLexer("0_0_", {TokenKind::NumberLiteral});
  testLexer("0__0", {TokenKind::NumberLiteral});
  testLexer("0_0f", {TokenKind::NumberLiteral});
  testLexer("0_0_f", {TokenKind::NumberLiteral});
  testLexer("0_0_f_00", {TokenKind::NumberLiteral});
  testLexer("1_,", {TokenKind::NumberLiteral, TokenKind::Comma});

  testLexer("0.0", {TokenKind::NumberLiteral});
  testLexer("1.0", {TokenKind::NumberLiteral});
  testLexer("10.0", {TokenKind::NumberLiteral});
  testLexer("0e0", {TokenKind::NumberLiteral});
  testLexer("1e0", {TokenKind::NumberLiteral});
  testLexer("1e100", {TokenKind::NumberLiteral});
  testLexer("1.0e100", {TokenKind::NumberLiteral});
  testLexer("1.0e+100", {TokenKind::NumberLiteral});
  testLexer("1.0e-100", {TokenKind::NumberLiteral});
  testLexer("1_0_0_0.0_0_0_0_0_1e1_0_0_0", {TokenKind::NumberLiteral});

  testLexer("1.", {TokenKind::NumberLiteral, TokenKind::Dot});
  testLexer("1e", {TokenKind::NumberLiteral});
  testLexer("1.e100", {TokenKind::NumberLiteral});
  testLexer("1.0e1f0", {TokenKind::NumberLiteral});
  testLexer("1.0p100", {TokenKind::NumberLiteral});
  testLexer("1.0p-100", {TokenKind::NumberLiteral});
  testLexer("1.0p1f0", {TokenKind::NumberLiteral});
  testLexer("1._+", {TokenKind::NumberLiteral, TokenKind::Plus});
  testLexer("1._e", {TokenKind::NumberLiteral});
  testLexer("1.0e", {TokenKind::NumberLiteral});
  testLexer("1.0e,", {TokenKind::NumberLiteral, TokenKind::Comma});
  testLexer("1.0e_", {TokenKind::NumberLiteral});
  testLexer("1.0e+_", {TokenKind::NumberLiteral});
  testLexer("1.0e-_", {TokenKind::NumberLiteral});
  testLexer("1.0e0_+", {TokenKind::NumberLiteral, TokenKind::Plus});
}

TEST_CASE("number literal binary", "[lexer]") {
  testLexer("0b0", {TokenKind::NumberLiteral});
  testLexer("0b1", {TokenKind::NumberLiteral});
  testLexer("0b2", {TokenKind::NumberLiteral});
  testLexer("0b3", {TokenKind::NumberLiteral});
  testLexer("0b4", {TokenKind::NumberLiteral});
  testLexer("0b5", {TokenKind::NumberLiteral});
  testLexer("0b6", {TokenKind::NumberLiteral});
  testLexer("0b7", {TokenKind::NumberLiteral});
  testLexer("0b8", {TokenKind::NumberLiteral});
  testLexer("0b9", {TokenKind::NumberLiteral});
  testLexer("0ba", {TokenKind::NumberLiteral});
  testLexer("0bb", {TokenKind::NumberLiteral});
  testLexer("0bc", {TokenKind::NumberLiteral});
  testLexer("0bd", {TokenKind::NumberLiteral});
  testLexer("0be", {TokenKind::NumberLiteral});
  testLexer("0bf", {TokenKind::NumberLiteral});
  testLexer("0bz", {TokenKind::NumberLiteral});

  testLexer("0b0000_0000", {TokenKind::NumberLiteral});
  testLexer("0b1111_1111", {TokenKind::NumberLiteral});
  testLexer("0b10_10_10_10", {TokenKind::NumberLiteral});
  testLexer("0b0_1_0_1_0_1_0_1", {TokenKind::NumberLiteral});
  testLexer("0b1.", {TokenKind::NumberLiteral, TokenKind::Dot});
  testLexer("0b1.0", {TokenKind::NumberLiteral});

  testLexer("0B0", {TokenKind::NumberLiteral});
  testLexer("0b_", {TokenKind::NumberLiteral});
  testLexer("0b_0", {TokenKind::NumberLiteral});
  testLexer("0b1_", {TokenKind::NumberLiteral});
  testLexer("0b0__1", {TokenKind::NumberLiteral});
  testLexer("0b0_1_", {TokenKind::NumberLiteral});
  testLexer("0b1e", {TokenKind::NumberLiteral});
  testLexer("0b1p", {TokenKind::NumberLiteral});
  testLexer("0b1e0", {TokenKind::NumberLiteral});
  testLexer("0b1p0", {TokenKind::NumberLiteral});
  testLexer("0b_,", {TokenKind::NumberLiteral, TokenKind::Comma});
}

TEST_CASE("number literal octal", "[lexer]") {
  testLexer("0o0", {TokenKind::NumberLiteral});
  testLexer("0o1", {TokenKind::NumberLiteral});
  testLexer("0o2", {TokenKind::NumberLiteral});
  testLexer("0o3", {TokenKind::NumberLiteral});
  testLexer("0o4", {TokenKind::NumberLiteral});
  testLexer("0o5", {TokenKind::NumberLiteral});
  testLexer("0o6", {TokenKind::NumberLiteral});
  testLexer("0o7", {TokenKind::NumberLiteral});
  testLexer("0o8", {TokenKind::NumberLiteral});
  testLexer("0o9", {TokenKind::NumberLiteral});
  testLexer("0oa", {TokenKind::NumberLiteral});
  testLexer("0ob", {TokenKind::NumberLiteral});
  testLexer("0oc", {TokenKind::NumberLiteral});
  testLexer("0od", {TokenKind::NumberLiteral});
  testLexer("0oe", {TokenKind::NumberLiteral});
  testLexer("0of", {TokenKind::NumberLiteral});
  testLexer("0oz", {TokenKind::NumberLiteral});

  testLexer("0o01234567", {TokenKind::NumberLiteral});
  testLexer("0o0123_4567", {TokenKind::NumberLiteral});
  testLexer("0o01_23_45_67", {TokenKind::NumberLiteral});
  testLexer("0o0_1_2_3_4_5_6_7", {TokenKind::NumberLiteral});

  testLexer("0O0", {TokenKind::NumberLiteral});
  testLexer("0o_", {TokenKind::NumberLiteral});
  testLexer("0o_0", {TokenKind::NumberLiteral});
  testLexer("0o1_", {TokenKind::NumberLiteral});
  testLexer("0o0__1", {TokenKind::NumberLiteral});
  testLexer("0o0_1_", {TokenKind::NumberLiteral});
  testLexer("0o1e", {TokenKind::NumberLiteral});
  testLexer("0o1p", {TokenKind::NumberLiteral});
  testLexer("0o1e0", {TokenKind::NumberLiteral});
  testLexer("0o1p0", {TokenKind::NumberLiteral});
  testLexer("0o_,", {TokenKind::NumberLiteral, TokenKind::Comma});
}

TEST_CASE("number literals hexadecimal", "[lexer]") {
  testLexer("0x0", {TokenKind::NumberLiteral});
  testLexer("0x1", {TokenKind::NumberLiteral});
  testLexer("0x2", {TokenKind::NumberLiteral});
  testLexer("0x3", {TokenKind::NumberLiteral});
  testLexer("0x4", {TokenKind::NumberLiteral});
  testLexer("0x5", {TokenKind::NumberLiteral});
  testLexer("0x6", {TokenKind::NumberLiteral});
  testLexer("0x7", {TokenKind::NumberLiteral});
  testLexer("0x8", {TokenKind::NumberLiteral});
  testLexer("0x9", {TokenKind::NumberLiteral});
  testLexer("0xa", {TokenKind::NumberLiteral});
  testLexer("0xb", {TokenKind::NumberLiteral});
  testLexer("0xc", {TokenKind::NumberLiteral});
  testLexer("0xd", {TokenKind::NumberLiteral});
  testLexer("0xe", {TokenKind::NumberLiteral});
  testLexer("0xf", {TokenKind::NumberLiteral});
  testLexer("0xA", {TokenKind::NumberLiteral});
  testLexer("0xB", {TokenKind::NumberLiteral});
  testLexer("0xC", {TokenKind::NumberLiteral});
  testLexer("0xD", {TokenKind::NumberLiteral});
  testLexer("0xE", {TokenKind::NumberLiteral});
  testLexer("0xF", {TokenKind::NumberLiteral});
  testLexer("0x0z", {TokenKind::NumberLiteral});
  testLexer("0xz", {TokenKind::NumberLiteral});

  testLexer("0x0123456789ABCDEF", {TokenKind::NumberLiteral});
  testLexer("0x0123_4567_89AB_CDEF", {TokenKind::NumberLiteral});
  testLexer("0x01_23_45_67_89AB_CDE_F", {TokenKind::NumberLiteral});
  testLexer("0x0_1_2_3_4_5_6_7_8_9_A_B_C_D_E_F", {TokenKind::NumberLiteral});

  testLexer("0X0", {TokenKind::NumberLiteral});
  testLexer("0x_", {TokenKind::NumberLiteral});
  testLexer("0x_1", {TokenKind::NumberLiteral});
  testLexer("0x1_", {TokenKind::NumberLiteral});
  testLexer("0x0__1", {TokenKind::NumberLiteral});
  testLexer("0x0_1_", {TokenKind::NumberLiteral});
  testLexer("0x_,", {TokenKind::NumberLiteral, TokenKind::Comma});

  testLexer("0x1.0", {TokenKind::NumberLiteral});
  testLexer("0xF.0", {TokenKind::NumberLiteral});
  testLexer("0xF.F", {TokenKind::NumberLiteral});
  testLexer("0xF.Fp0", {TokenKind::NumberLiteral});
  testLexer("0xF.FP0", {TokenKind::NumberLiteral});
  testLexer("0x1p0", {TokenKind::NumberLiteral});
  testLexer("0xfp0", {TokenKind::NumberLiteral});
  testLexer("0x1.0+0xF.0", {TokenKind::NumberLiteral, TokenKind::Plus,
                            TokenKind::NumberLiteral});

  testLexer("0x1.", {TokenKind::NumberLiteral, TokenKind::Dot});
  testLexer("0xF.", {TokenKind::NumberLiteral, TokenKind::Dot});
  testLexer("0x1.+0xF.",
            {TokenKind::NumberLiteral, TokenKind::Dot, TokenKind::Plus,
             TokenKind::NumberLiteral, TokenKind::Dot});
  testLexer("0xff.p10", {TokenKind::NumberLiteral});

  testLexer("0x0123456.789ABCDEF", {TokenKind::NumberLiteral});
  testLexer("0x0_123_456.789_ABC_DEF", {TokenKind::NumberLiteral});
  testLexer("0x0_1_2_3_4_5_6.7_8_9_A_B_C_D_E_F", {TokenKind::NumberLiteral});
  testLexer("0x0p0", {TokenKind::NumberLiteral});
  testLexer("0x0.0p0", {TokenKind::NumberLiteral});
  testLexer("0xff.ffp10", {TokenKind::NumberLiteral});
  testLexer("0xff.ffP10", {TokenKind::NumberLiteral});
  testLexer("0xffp10", {TokenKind::NumberLiteral});
  testLexer("0xff_ff.ff_ffp1_0_0_0", {TokenKind::NumberLiteral});
  testLexer("0xf_f_f_f.f_f_f_fp+1_000", {TokenKind::NumberLiteral});
  testLexer("0xf_f_f_f.f_f_f_fp-1_00_0", {TokenKind::NumberLiteral});

  testLexer("0x1e", {TokenKind::NumberLiteral});
  testLexer("0x1e0", {TokenKind::NumberLiteral});
  testLexer("0x1p", {TokenKind::NumberLiteral});
  testLexer("0xfp0z1", {TokenKind::NumberLiteral});
  testLexer("0xff.ffpff", {TokenKind::NumberLiteral});
  testLexer("0x0.p", {TokenKind::NumberLiteral});
  testLexer("0x0.z", {TokenKind::NumberLiteral});
  testLexer("0x0._", {TokenKind::NumberLiteral});
  testLexer("0x0_.0", {TokenKind::NumberLiteral});
  testLexer("0x0_.0.0", {TokenKind::NumberLiteral, TokenKind::Dot,
                         TokenKind::NumberLiteral});
  testLexer("0x0._0", {TokenKind::NumberLiteral});
  testLexer("0x0.0_", {TokenKind::NumberLiteral});
  testLexer("0x0_p0", {TokenKind::NumberLiteral});
  testLexer("0x0_.p0", {TokenKind::NumberLiteral});
  testLexer("0x0._p0", {TokenKind::NumberLiteral});
  testLexer("0x0.0_p0", {TokenKind::NumberLiteral});
  testLexer("0x0._0p0", {TokenKind::NumberLiteral});
  testLexer("0x0.0p_0", {TokenKind::NumberLiteral});
  testLexer("0x0.0p+_0", {TokenKind::NumberLiteral});
  testLexer("0x0.0p-_0", {TokenKind::NumberLiteral});
  testLexer("0x0.0p0_", {TokenKind::NumberLiteral});
}

TEST_CASE("code point literal with hex escape", "[lexer]") {
  testLexer({0x27, 0x5c, 0x78, 0x31, 0x62, 0x27}, {TokenKind::CharLiteral});
  testLexer({0x27, 0x5c, 0x78, 0x31, 0x27},
            {TokenKind::Invalid, TokenKind::Invalid});
}
TEST_CASE("newline in char literal", "[lexer]") {
  testLexer("'\n'", {TokenKind::Invalid, TokenKind::Invalid});
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

TEST_CASE("chars", "[lexer]") { testLexer("'c'", {TokenKind::CharLiteral}); }

TEST_CASE("Invalid token characters", "[lexer]") {
  testLexer("#", {TokenKind::Invalid});
  testLexer("`", {TokenKind::Invalid});
  testLexer("'c", {TokenKind::Invalid});
  testLexer("'", {TokenKind::Invalid});
  testLexer("''", {TokenKind::Invalid, TokenKind::Invalid});
}

TEST_CASE("Invalid literal/comment characters", "[lexer]") {
  testLexer({0x22, 0x00, 0x22}, {TokenKind::StringLiteral, TokenKind::Invalid});
  testLexer({0x2f, 0x2f, 0x00}, {TokenKind::Invalid});
  testLexer("//\x1f", {TokenKind::Invalid});
  testLexer("//\x7f", {TokenKind::Invalid});
}

TEST_CASE("utf8", "[lexer]") {
  testLexer("//\xc2\x80", {});
  testLexer("//\xf4\x8f\xbf\xbf", {});
}

TEST_CASE("Invalid utf8", "[lexer]") {
  testLexer("//\x80", {TokenKind::Invalid});
  testLexer("//\xbf", {TokenKind::Invalid});
  testLexer("//\xf8", {TokenKind::Invalid});
  testLexer("//\xff", {TokenKind::Invalid});
  testLexer("//\xc2\xc0", {TokenKind::Invalid});
  testLexer("//\xe0", {TokenKind::Invalid});
  testLexer("//\xf0", {TokenKind::Invalid});
  testLexer("//\xf0\x90\x80\xc0", {TokenKind::Invalid});
}

TEST_CASE("illegal unicode codepoints", "[lexer]") {
  testLexer("//\xc2\x84", {});
  testLexer("//\xc2\x85", {TokenKind::Invalid});
  testLexer("//\xc2\x86", {});
  testLexer("//\xe2\x80\xa7", {});
  testLexer("//\xe2\x80\xa8", {TokenKind::Invalid});
  testLexer("//\xe2\x80\xa9", {TokenKind::Invalid});
  testLexer("//\xe2\x80\xaa", {});
}

TEST_CASE("shift operator", "[lexer]") {
  testLexer("<<", {TokenKind::LessLess});
  testLexer(">>", {TokenKind::GreaterGreater});
}

TEST_CASE("code point literal with unicode code point", "[lexer]") {
  testLexer("'💩'", {TokenKind::CharLiteral});
}

TEST_CASE("@", "[lexer]") {
  testLexer("@mlir(\"i32\")",
            {TokenKind::At, TokenKind::Identifier, TokenKind::LParen,
             TokenKind::StringLiteral, TokenKind::RParen});
}
