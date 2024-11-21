#pragma once

#include <cassert>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>

enum class TokenKind {
  Dummy,

  StringLiteral,
  CharLiteral,
  NumberLiteral,
  Identifier,

  Bang,
  BangEqual,
  Pipe,
  PipePipe,
  PipeEqual,
  Equal,
  EqualEqual,
  Caret,
  CaretEqual,
  Plus,
  PlusEqual,
  PlusPlus,
  Minus,
  MinusEqual,
  Star,
  StarEqual,
  StarStar,
  Percent,
  PercentEqual,
  Slash,
  SlashEqual,
  Ampersand,
  AmpersandEqual,
  At,

  Tilde,
  Less,
  LessEqual,
  LessLess,
  LessLessEqual,
  Greater,
  GreaterEqual,
  GreaterGreater,
  GreaterGreaterEqual,

  Question,
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  Comma,
  Dot,
  DotDot,
  DotDotEqual,
  Colon,
  Semicolon,
  EqualGreater,
  MinusGreater,

  // Keywords
  KeywordMut,
  KeywordImpl,
  KeywordUnion,
  KeywordTrait,
  KeywordMatch,
  KeywordIs,
  KeywordAs,
  KeywordAnd,
  KeywordBreak,
  KeywordConst,
  KeywordContinue,
  KeywordElse,
  KeywordEnum,
  KeywordFn,
  KeywordFor,
  KeywordWhile,
  KeywordIf,
  KeywordImport,
  KeywordIn,
  KeywordNot,
  KeywordOr,
  KeywordPub,
  KeywordReturn,
  KeywordStruct,
  KeywordVar,
  KeywordComptime,
  KeywordModule,
  KeywordYield,

  Eof,
  Invalid,
};

struct TokenSpan {
  int file_id;
  int line_no;
  int col_start;
  int start;
  int end;
};

struct Token {
  TokenKind kind;
  TokenSpan span;
};

struct Lexer {
  std::basic_string<char> source;
  int index;
  std::optional<Token> pending_invalid_token;
  int file_id;
  int line_no;
  int col_start;

  enum class State {
    Start,
    Identifier,
    StringLiteral,
    StringLiteralBackslash,
    CharLiteral,
    CharLiteralBackslash,
    CharLiteralHexEscape,
    CharLiteralUnicodeEscapeSawU,
    CharLiteralUnicodeEscape,
    CharLiteralUnicodeInvalid,
    CharLiteralUnicode,
    CharLiteralEnd,
    Equal,
    Bang,
    Pipe,
    Minus,
    Star,
    Slash,
    LineComment,
    LineCommentStart,
    Int,
    IntExponent,
    IntPeriod,
    Float,
    FloatExponent,
    Ampersand,
    Caret,
    Percent,
    Plus,
    Less,
    LessLess,
    Greater,
    GreaterGreater,
    Dot,
    DotDot,
  };

  std::unordered_map<std::string, TokenKind> keywords = {
      {"mut", TokenKind::KeywordMut},
      {"impl", TokenKind::KeywordImpl},
      {"match", TokenKind::KeywordMatch},
      {"union", TokenKind::KeywordUnion},
      {"trait", TokenKind::KeywordTrait},
      {"is", TokenKind::KeywordIs},
      {"as", TokenKind::KeywordAs},
      {"and", TokenKind::KeywordAnd},
      {"break", TokenKind::KeywordBreak},
      {"const", TokenKind::KeywordConst},
      {"continue", TokenKind::KeywordContinue},
      {"else", TokenKind::KeywordElse},
      {"enum", TokenKind::KeywordEnum},
      {"fn", TokenKind::KeywordFn},
      {"for", TokenKind::KeywordFor},
      {"while", TokenKind::KeywordWhile},
      {"if", TokenKind::KeywordIf},
      {"import", TokenKind::KeywordImport},
      {"in", TokenKind::KeywordIn},
      {"not", TokenKind::KeywordNot},
      {"or", TokenKind::KeywordOr},
      {"pub", TokenKind::KeywordPub},
      {"return", TokenKind::KeywordReturn},
      {"struct", TokenKind::KeywordStruct},
      {"var", TokenKind::KeywordVar},
      {"comptime", TokenKind::KeywordComptime},
      {"module", TokenKind::KeywordModule},
      {"yield", TokenKind::KeywordYield},
  };

  Lexer(std::basic_string<char> source, int file_id)
      : source(source), file_id(file_id), line_no(1), col_start(1) {
    if (source.size() >= 3 && source[0] == '\xEF' && source[1] == '\xBB' &&
        source[2] == '\xBF') {
      index = 3;
    } else {
      index = 0;
    }
  }

  int get_source_file_id() { return file_id; }

  std::string token_to_string(Token token) {
    return source.substr(token.span.start, token.span.end - token.span.start);
  }

  Token next() {
    if (pending_invalid_token.has_value()) {
      Token token = pending_invalid_token.value();
      pending_invalid_token.reset();
      col_start += (token.span.end - token.span.start);
      return token;
    }

    State state = State::Start;
    Token result =
        Token{TokenKind::Eof, TokenSpan{file_id, line_no, col_start, index, 0}};

    int seen_escape_digits;
    int remaining_code_units;

    bool loop_break = false;
    while (true) {
      unsigned char ch = source[index];
      switch (state) {
      case State::Start:
        switch (ch) {
        case 0:
          if (index != (int)source.size()) {
            result.kind = TokenKind::Invalid;
            result.span.start = index;
            index += 1;
            result.span.end = index;
            result.span.col_start = col_start;
            result.span.line_no = line_no;
            col_start += 1;
            return result;
          }
          loop_break = true;
          break;
        case ' ':
        case '\t':
        case '\r':
          result.span.start = index + 1;
          col_start += 1;
          break;
        case '\n':
          line_no += 1;
          col_start = 1;
          result.span.start = index + 1;
          break;
        case '"':
          state = State::StringLiteral;
          result.kind = TokenKind::StringLiteral;
          break;
        case '\'':
          state = State::CharLiteral;
          break;
        case 'a' ... 'z':
        case 'A' ... 'Z':
        case '_':
          state = State::Identifier;
          result.kind = TokenKind::Identifier;
          break;
        case '(':
          result.kind = TokenKind::LParen;
          index += 1;
          loop_break = true;
          break;
        case ')':
          result.kind = TokenKind::RParen;
          index += 1;
          loop_break = true;
          break;
        case '{':
          result.kind = TokenKind::LBrace;
          index += 1;
          loop_break = true;
          break;
        case '}':
          result.kind = TokenKind::RBrace;
          index += 1;
          loop_break = true;
          break;
        case '[':
          result.kind = TokenKind::LBracket;
          index += 1;
          loop_break = true;
          break;
        case ']':
          result.kind = TokenKind::RBracket;
          index += 1;
          loop_break = true;
          break;
        case ',':
          result.kind = TokenKind::Comma;
          index += 1;
          loop_break = true;
          break;
        case ';':
          result.kind = TokenKind::Semicolon;
          index += 1;
          loop_break = true;
          break;
        case '?':
          result.kind = TokenKind::Question;
          index += 1;
          loop_break = true;
          break;
        case ':':
          result.kind = TokenKind::Colon;
          index += 1;
          loop_break = true;
          break;
        case '@':
          result.kind = TokenKind::At;
          index += 1;
          loop_break = true;
          break;

        case '~':
          result.kind = TokenKind::Tilde;
          index += 1;
          loop_break = true;
          break;
        case '=':
          state = State::Equal;
          break;
        case '!':
          state = State::Bang;
          break;
        case '|':
          state = State::Pipe;
          break;
        case '-':
          state = State::Minus;
          break;
        case '*':
          state = State::Star;
          break;
        case '/':
          state = State::Slash;
          break;
        case '&':
          state = State::Ampersand;
          break;
        case '^':
          state = State::Caret;
          break;
        case '%':
          state = State::Percent;
          break;
        case '+':
          state = State::Plus;
          break;
        case '<':
          state = State::Less;
          break;
        case '>':
          state = State::Greater;
          break;
        case '.':
          state = State::Dot;
          break;

        case '0' ... '9':
          state = State::Int;
          result.kind = TokenKind::NumberLiteral;
          break;
        default:
          result.kind = TokenKind::Invalid;
          result.span.end = index;
          index += 1;
          result.span.col_start = col_start;
          result.span.line_no = line_no;
          col_start += result.span.end - result.span.start;
          return result;
        }
        break;
      case State::Ampersand:
        switch (ch) {
        case '=':
          result.kind = TokenKind::AmpersandEqual;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Ampersand;
          loop_break = true;
          break;
        }
        break;
      case State::Star:
        switch (ch) {
        case '=':
          result.kind = TokenKind::StarEqual;
          index += 1;
          loop_break = true;
          break;
        case '*':
          result.kind = TokenKind::StarStar;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Star;
          loop_break = true;
          break;
        }
        break;
      case State::Percent:
        switch (ch) {
        case '=':
          result.kind = TokenKind::PercentEqual;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Percent;
          loop_break = true;
          break;
        }
        break;
      case State::Plus:
        switch (ch) {
        case '=':
          result.kind = TokenKind::PlusEqual;
          index += 1;
          loop_break = true;
          break;
        case '+':
          result.kind = TokenKind::PlusPlus;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Plus;
          loop_break = true;
          break;
        }
        break;
      case State::Caret:
        switch (ch) {
        case '=':
          result.kind = TokenKind::CaretEqual;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Caret;
          loop_break = true;
          break;
        }
        break;
      case State::Identifier:
        switch (ch) {
        case 'a' ... 'z':
        case 'A' ... 'Z':
        case '0' ... '9':
        case '_':
          break;
        default:
          // get substring, index is not length its last index
          std::string id =
              source.substr(result.span.start, index - result.span.start);
          if (keywords.find(id) != keywords.end()) {
            result.kind = keywords[id];
          }
          loop_break = true;
          break;
        }
        break;
      case State::StringLiteral:
        switch (ch) {
        case '\\':
          state = State::StringLiteralBackslash;
          break;
        case '"':
          index += 1;
          loop_break = true;
          break;
        case 0:
          if (index == (int)source.size()) {
            result.kind = TokenKind::Invalid;
            loop_break = true;
            break;
          } else {
            check_literal_character();
          }
          break;
        case '\n':
          line_no += 1;
          col_start = 1;
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        default:
          check_literal_character();
          break;
        }
        break;
      case State::StringLiteralBackslash:
        switch (ch) {
        case 0:
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        case '\n':
          line_no += 1;
          col_start = 1;
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        default:
          state = State::StringLiteral;
          break;
        }
        break;
      case State::CharLiteral:
        switch (ch) {
        case 0:
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        case '\\':
          state = State::CharLiteralBackslash;
          break;
        case '\'':
        case 0x80 ... 0xbf:
        case 0xf8 ... 0xff:
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        case 0xc0 ... 0xdf:
          remaining_code_units = 1;
          state = State::CharLiteralUnicode;
          break;
        case 0xe0 ... 0xef:
          remaining_code_units = 2;
          state = State::CharLiteralUnicode;
          break;
        case 0xf0 ... 0xf7:
          remaining_code_units = 3;
          state = State::CharLiteralUnicode;
          break;
        case '\n':
          line_no += 1;
          col_start = 1;
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        default:
          state = State::CharLiteralEnd;
          break;
        }
        break;
      case State::CharLiteralBackslash:
        switch (ch) {
        case 0:
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        case '\n':
          line_no += 1;
          col_start = 1;
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        case 'x':
          state = State::CharLiteralHexEscape;
          seen_escape_digits = 0;
          break;
        case 'u':
          state = State::CharLiteralUnicodeEscapeSawU;
          break;
        default:
          state = State::CharLiteralEnd;
          break;
        }
        break;
      case State::CharLiteralHexEscape:
        switch (ch) {
        case '0' ... '9':
        case 'a' ... 'f':
        case 'A' ... 'F':
          seen_escape_digits += 1;
          if (seen_escape_digits == 2) {
            state = State::CharLiteralEnd;
          }
          break;
        default:
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        }
        break;
      case State::CharLiteralUnicodeEscapeSawU:
        switch (ch) {
        case 0:
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        case '{':
          state = State::CharLiteralUnicodeEscape;
          break;
        default:
          result.kind = TokenKind::Invalid;
          state = State::CharLiteralUnicodeInvalid;
          break;
        }
        break;
      case State::CharLiteralUnicodeEscape:
        switch (ch) {
        case 0:
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        case '0' ... '9':
        case 'a' ... 'f':
        case 'A' ... 'F':
          break;
        case '}':
          state = State::CharLiteralEnd;
          break;
        default:
          result.kind = TokenKind::Invalid;
          state = State::CharLiteralUnicodeInvalid;
          break;
        }
        break;
      case State::CharLiteralUnicodeInvalid:
        switch (ch) {
        case '0' ... '9':
        case 'a' ... 'z':
        case 'A' ... 'Z':
        case '}':
          break;
        default:
          loop_break = true;
          break;
        }
        break;
      case State::CharLiteralEnd:
        switch (ch) {
        case '\'':
          result.kind = TokenKind::CharLiteral;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        }
        break;
      case State::CharLiteralUnicode:
        switch (ch) {
        case 0x80 ... 0xbf:
          remaining_code_units -= 1;
          if (remaining_code_units == 0) {
            state = State::CharLiteralEnd;
          }
          break;
        default:
          result.kind = TokenKind::Invalid;
          loop_break = true;
          break;
        }
        break;
      case State::Bang:
        switch (ch) {
        case '=':
          result.kind = TokenKind::BangEqual;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Bang;
          loop_break = true;
          break;
        }
        break;
      case State::Pipe:
        switch (ch) {
        case '|':
          result.kind = TokenKind::PipePipe;
          index += 1;
          loop_break = true;
          break;
        case '=':
          result.kind = TokenKind::PipeEqual;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Pipe;
          loop_break = true;
          break;
        }
        break;
      case State::Equal:
        switch (ch) {
        case '=':
          result.kind = TokenKind::EqualEqual;
          index += 1;
          loop_break = true;
          break;
        case '>':
          result.kind = TokenKind::EqualGreater;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Equal;
          loop_break = true;
          break;
        }
        break;
      case State::Minus:
        switch (ch) {
        case '=':
          result.kind = TokenKind::MinusEqual;
          index += 1;
          loop_break = true;
          break;
        case '>':
          result.kind = TokenKind::MinusGreater;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Minus;
          loop_break = true;
          break;
        }
        break;
      case State::Less:
        switch (ch) {
        case '=':
          result.kind = TokenKind::LessEqual;
          index += 1;
          loop_break = true;
          break;
        case '<':
          state = State::LessLess;
          break;
        default:
          result.kind = TokenKind::Less;
          loop_break = true;
          break;
        }
        break;
      case State::LessLess:
        switch (ch) {
        case '=':
          result.kind = TokenKind::LessLessEqual;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::LessLess;
          loop_break = true;
          break;
        }
        break;
      case State::Greater:
        switch (ch) {
        case '=':
          result.kind = TokenKind::GreaterEqual;
          index += 1;
          loop_break = true;
          break;
        case '>':
          state = State::GreaterGreater;
          break;
        default:
          result.kind = TokenKind::Greater;
          loop_break = true;
          break;
        }
        break;
      case State::GreaterGreater:
        switch (ch) {
        case '=':
          result.kind = TokenKind::GreaterGreaterEqual;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::GreaterGreater;
          loop_break = true;
          break;
        }
        break;
      case State::Dot:
        switch (ch) {
        case '.':
          state = State::DotDot;
          break;
        default:
          result.kind = TokenKind::Dot;
          loop_break = true;
          break;
        }
        break;
      case State::DotDot:
        switch (ch) {
        case '=':
          result.kind = TokenKind::DotDotEqual;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::DotDot;
          loop_break = true;
          break;
        }
        break;
      case State::Slash:
        switch (ch) {
        case '/':
          state = State::LineCommentStart;
          break;
        case '=':
          result.kind = TokenKind::SlashEqual;
          index += 1;
          loop_break = true;
          break;
        default:
          result.kind = TokenKind::Slash;
          loop_break = true;
          break;
        }
        break;
      case State::LineCommentStart:
        switch (ch) {
        case 0:
          if (index != (int)source.size()) {
            result.kind = TokenKind::Invalid;
            index += 1;
          }
          loop_break = true;
          break;
        case '\n':
          state = State::Start;
          result.span.start = index + 1;
          line_no += 1;
          col_start = 1;
          break;
        case '\t':
          state = State::LineComment;
          break;
        default:
          state = State::LineComment;
          check_literal_character();
          break;
        }
        break;
      case State::LineComment:
        switch (ch) {
        case 0:
          if (index != (int)source.size()) {
            result.kind = TokenKind::Invalid;
            index += 1;
          }
          loop_break = true;
          break;
        case '\n':
          state = State::Start;
          result.span.start = index + 1;
          line_no += 1;
          col_start = 1;
          break;
        case '\t':
          break;
        default:
          check_literal_character();
          break;
        }
        break;
      case State::Int:
        switch (ch) {
        case '.':
          state = State::IntPeriod;
          break;
        case '_':
        case 'a' ... 'd':
        case 'f' ... 'o':
        case 'q' ... 'z':
        case 'A' ... 'D':
        case 'F' ... 'O':
        case 'Q' ... 'Z':
        case '0' ... '9':
          break;
        case 'e':
        case 'E':
        case 'p':
        case 'P':
          state = State::IntExponent;
          break;
        default:
          loop_break = true;
          break;
        }
        break;
      case State::IntExponent:
        switch (ch) {
        case '+':
        case '-':
          state = State::Float;
          break;
        default:
          index -= 1;
          state = State::Int;
          break;
        }
        break;
      case State::IntPeriod:
        switch (ch) {
        case '_':
        case 'a' ... 'd':
        case 'f' ... 'o':
        case 'q' ... 'z':
        case 'A' ... 'D':
        case 'F' ... 'O':
        case 'Q' ... 'Z':
        case '0' ... '9':
          state = State::Float;
          break;
        case 'e':
        case 'E':
        case 'p':
        case 'P':
          state = State::FloatExponent;
          break;
        default:
          index -= 1;
          loop_break = true;
          break;
        }
        break;
      case State::Float:
        switch (ch) {
        case '_':
        case 'a' ... 'd':
        case 'f' ... 'o':
        case 'q' ... 'z':
        case 'A' ... 'D':
        case 'F' ... 'O':
        case 'Q' ... 'Z':
        case '0' ... '9':
          break;
        case 'e':
        case 'E':
        case 'p':
        case 'P':
          state = State::FloatExponent;
          break;
        default:
          loop_break = true;
          break;
        }
        break;
      case State::FloatExponent:
        switch (ch) {
        case '+':
        case '-':
          state = State::Float;
          break;
        default:
          index -= 1;
          state = State::Float;
          break;
        }
        break;
      }
      if (loop_break) {
        break;
      }
      index += 1;
    }

    if (result.kind == TokenKind::Eof) {
      if (pending_invalid_token.has_value()) {
        Token token = pending_invalid_token.value();
        pending_invalid_token.reset();
        result.span.col_start = col_start;
        col_start += result.span.end - result.span.start;
        return token;
      }
      result.span.start = index;
    }

    result.span.end = index;
    result.span.col_start = col_start;
    result.span.line_no = line_no;
    col_start += result.span.end - result.span.start;
    return result;
  }

  void check_literal_character() {
    if (pending_invalid_token.has_value()) {
      return;
    }
    int invalid_length = get_invalid_character_length();
    if (invalid_length == 0)
      return;
    pending_invalid_token =
        Token{TokenKind::Invalid, TokenSpan{file_id, line_no, col_start, index,
                                            index + invalid_length}};
  }

  uint8_t get_invalid_character_length() {
    unsigned char c0 = source[index];
    // if (c0 < 128) { // is_ascii
    if (isascii(c0)) {
      if (c0 == '\r') {
        if (index + 1 < (int)source.size() && source[index + 1] == '\n')
          return 0;
        else
          return 1;
      } else if (c0 <= 0x1f || c0 == 0x7f) { // is_control
        return 1;
      }
      return 0;
    } else {
      int length = utf8_byte_sequence_length(c0);
      if (length == -1)
        return 1;
      if (index + length > (int)source.size())
        return source.size() - index;

      const char *bytes = source.c_str() + index;
      switch (length) {
      case 2: {
        uint32_t value = utf8_decode2(bytes);
        if (value == UINT32_MAX)
          return length; // Decoding error
        if (value == 0x85)
          return length; // U+0085 (NEL)
        break;
      }
      case 3: {
        uint32_t value = utf8_decode3(bytes);
        if (value == UINT32_MAX)
          return length; // Decoding error
        if (value == 0x2028)
          return length; // U+2028 (LS)
        if (value == 0x2029)
          return length; // U+2029 (PS)
        break;
      }
      case 4: {
        uint32_t value = utf8_decode4(bytes);
        if (value == UINT32_MAX)
          return length; // Decoding error
        break;
      }
      default:
        std::cout << "Invalid length: " << length << std::endl;
        assert(false); // Unreachable
      }
      index += length - 1;
      return 0;
    }
  }

  uint8_t utf8_byte_sequence_length(unsigned char first_byte) {
    switch (first_byte) {
    case 0b0000'0000 ... 0b0111'1111:
      return 1;
    case 0b1100'0000 ... 0b1101'1111:
      return 2;
    case 0b1110'0000 ... 0b1110'1111:
      return 3;
    case 0b1111'0000 ... 0b1111'0111:
      return 4;
    default:
      return -1;
    }
  }

  uint32_t utf8_decode2(const char *bytes) {
    unsigned char c0 = bytes[0];
    unsigned char c1 = bytes[1];
    if ((c1 & 0xC0) != 0x80)
      return UINT32_MAX; // Invalid continuation byte
    uint32_t codepoint = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    return codepoint;
  }

  uint32_t utf8_decode3(const char *bytes) {
    unsigned char c0 = bytes[0];
    unsigned char c1 = bytes[1];
    unsigned char c2 = bytes[2];
    if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80)
      return UINT32_MAX; // Invalid continuation bytes
    uint32_t codepoint = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    return codepoint;
  }

  uint32_t utf8_decode4(const char *bytes) {
    unsigned char c0 = bytes[0];
    unsigned char c1 = bytes[1];
    unsigned char c2 = bytes[2];
    unsigned char c3 = bytes[3];
    if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80)
      return UINT32_MAX; // Invalid continuation bytes
    uint32_t codepoint = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) |
                         ((c2 & 0x3F) << 6) | (c3 & 0x3F);
    return codepoint;
  }

  static std::string lexeme(TokenKind kind) {
    switch (kind) {
    case TokenKind::Dummy:
      return "dummy";

    case TokenKind::StringLiteral:
      return "a string literal";
    case TokenKind::CharLiteral:
      return "a character literal";
    case TokenKind::NumberLiteral:
      return "a number literal";
    case TokenKind::Identifier:
      return "an identifier";
    case TokenKind::Eof:
      return "end of file";
    case TokenKind::Invalid:
      return "an invalid token";

    case TokenKind::At:
      return "@";
    case TokenKind::Bang:
      return "!";
    case TokenKind::BangEqual:
      return "!=";
    case TokenKind::Pipe:
      return "|";
    case TokenKind::PipePipe:
      return "||";
    case TokenKind::PipeEqual:
      return "|=";
    case TokenKind::Equal:
      return "=";
    case TokenKind::EqualEqual:
      return "==";
    case TokenKind::Caret:
      return "^";
    case TokenKind::CaretEqual:
      return "^=";
    case TokenKind::Plus:
      return "+";
    case TokenKind::PlusEqual:
      return "+=";
    case TokenKind::PlusPlus:
      return "++";
    case TokenKind::Minus:
      return "-";
    case TokenKind::MinusEqual:
      return "-=";
    case TokenKind::Star:
      return "*";
    case TokenKind::StarEqual:
      return "*=";
    case TokenKind::StarStar:
      return "**";
    case TokenKind::Percent:
      return "%";
    case TokenKind::PercentEqual:
      return "%=";
    case TokenKind::Slash:
      return "/";
    case TokenKind::SlashEqual:
      return "/=";
    case TokenKind::Ampersand:
      return "&";
    case TokenKind::AmpersandEqual:
      return "&=";
    case TokenKind::Tilde:
      return "~";
    case TokenKind::Less:
      return "<";
    case TokenKind::LessEqual:
      return "<=";
    case TokenKind::LessLess:
      return "<<";
    case TokenKind::LessLessEqual:
      return "<<=";
    case TokenKind::Greater:
      return ">";
    case TokenKind::GreaterEqual:
      return ">=";
    case TokenKind::GreaterGreater:
      return ">>";
    case TokenKind::GreaterGreaterEqual:
      return ">>=";
    case TokenKind::Question:
      return "?";
    case TokenKind::LParen:
      return "(";
    case TokenKind::RParen:
      return ")";
    case TokenKind::LBrace:
      return "{";
    case TokenKind::RBrace:
      return "}";
    case TokenKind::LBracket:
      return "[";
    case TokenKind::RBracket:
      return "]";
    case TokenKind::Comma:
      return ",";
    case TokenKind::Dot:
      return ".";
    case TokenKind::DotDot:
      return "..";
    case TokenKind::DotDotEqual:
      return "..=";
    case TokenKind::Colon:
      return ":";
    case TokenKind::Semicolon:
      return ";";
    case TokenKind::EqualGreater:
      return "=>";

    case TokenKind::MinusGreater:
      return "->";
    case TokenKind::KeywordMut:
      return "mut";
    case TokenKind::KeywordImpl:
      return "impl";
    case TokenKind::KeywordUnion:
      return "union";
    case TokenKind::KeywordTrait:
      return "trait";
    case TokenKind::KeywordMatch:
      return "match";
    case TokenKind::KeywordIs:
      return "is";
    case TokenKind::KeywordAs:
      return "as";
    case TokenKind::KeywordAnd:
      return "and";
    case TokenKind::KeywordBreak:
      return "break";
    case TokenKind::KeywordConst:
      return "const";
    case TokenKind::KeywordContinue:
      return "continue";
    case TokenKind::KeywordElse:
      return "else";
    case TokenKind::KeywordEnum:
      return "enum";
    case TokenKind::KeywordFn:
      return "fn";
    case TokenKind::KeywordFor:
      return "for";
    case TokenKind::KeywordWhile:
      return "while";
    case TokenKind::KeywordIf:
      return "if";
    case TokenKind::KeywordImport:
      return "import";
    case TokenKind::KeywordIn:
      return "in";
    case TokenKind::KeywordNot:
      return "not";
    case TokenKind::KeywordOr:
      return "or";
    case TokenKind::KeywordPub:
      return "pub";
    case TokenKind::KeywordReturn:
      return "return";
    case TokenKind::KeywordStruct:
      return "struct";
    case TokenKind::KeywordVar:
      return "var";
    case TokenKind::KeywordComptime:
      return "comptime";
    case TokenKind::KeywordModule:
      return "module";
    case TokenKind::KeywordYield:
      return "yield";
    }
    return "invalid token";
  }
};
