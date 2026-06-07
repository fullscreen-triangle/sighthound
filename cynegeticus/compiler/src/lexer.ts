/**
 * Cynegeticus Language - Lexer
 * Tokenizes source code into a stream of tokens
 */

import { Token, TokenType } from "./types";

export class Lexer {
  private source: string;
  private position: number = 0;
  private line: number = 1;
  private column: number = 1;
  private tokens: Token[] = [];

  private keywords: Map<string, TokenType> = new Map([
    ["declare", TokenType.DECLARE],
    ["const", TokenType.CONST],
    ["partition", TokenType.PARTITION],
    ["satellite", TokenType.SATELLITE],
    ["constellation", TokenType.CONSTELLATION],
    ["measure", TokenType.MEASURE],
    ["atmosphere", TokenType.ATMOSPHERE],
    ["entropy", TokenType.ENTROPY],
    ["resolve", TokenType.RESOLVE],
    ["position", TokenType.POSITION],
    ["triangulate", TokenType.TRIANGULATE],
    ["validate", TokenType.VALIDATE],
    ["circular", TokenType.CIRCULAR],
    ["closure", TokenType.CLOSURE],
    ["against", TokenType.AGAINST],
    ["known", TokenType.KNOWN],
    ["accuracy", TokenType.ACCURACY],
    ["check", TokenType.CHECK],
    ["target", TokenType.TARGET],
    ["emit", TokenType.EMIT],
    ["log", TokenType.LOG],
    ["output", TokenType.OUTPUT],
    ["show", TokenType.SHOW],
    ["from", TokenType.FROM],
    ["at", TokenType.AT],
    ["store", TokenType.STORE],
    ["into", TokenType.INTO],
    ["with", TokenType.WITH],
    ["of", TokenType.OF],
    ["count", TokenType.COUNT],
    ["altitude", TokenType.ALTITUDE],
    ["rmse", TokenType.RMSE],
    ["here", TokenType.HERE],
    ["in", TokenType.IN],
    ["action", TokenType.ACTION],
    ["do", TokenType.DO],
    ["if", TokenType.IF],
    ["else", TokenType.ELSE],
    ["for", TokenType.FOR],
    ["while", TokenType.WHILE],
    ["vibrational", TokenType.VIBRATIONAL],
    ["rotational", TokenType.ROTATIONAL],
    ["translational", TokenType.TRANSLATIONAL],
    ["collisional", TokenType.COLLISIONAL],
    ["energy", TokenType.ENERGY],
    ["coord", TokenType.COORD],
    ["measurement", TokenType.MEASUREMENT],
    ["array", TokenType.ARRAY],
    ["cm", TokenType.CM],
    ["mm", TokenType.MM],
    ["m", TokenType.M],
    ["km", TokenType.KM],
    ["Hz", TokenType.HZ],
    ["THz", TokenType.THZ],
    ["K", TokenType.K],
    ["Pa", TokenType.PA],
    ["hPa", TokenType.HPA],
  ]);

  constructor(source: string) {
    this.source = source;
  }

  /**
   * Main tokenization method
   */
  tokenize(): Token[] {
    this.tokens = [];

    while (this.position < this.source.length) {
      this.skipWhitespaceAndComments();

      if (this.position >= this.source.length) break;

      const token = this.nextToken();
      if (token) {
        this.tokens.push(token);
      }
    }

    this.tokens.push({
      type: TokenType.EOF,
      value: "",
      line: this.line,
      column: this.column,
      length: 0,
    });

    return this.tokens;
  }

  /**
   * Get the next token
   */
  private nextToken(): Token | null {
    const startLine = this.line;
    const startColumn = this.column;
    const startPos = this.position;

    const char = this.peek();

    // Single character tokens
    if (char === "(") {
      this.advance();
      return { type: TokenType.LPAREN, value: "(", line: startLine, column: startColumn, length: 1 };
    }
    if (char === ")") {
      this.advance();
      return { type: TokenType.RPAREN, value: ")", line: startLine, column: startColumn, length: 1 };
    }
    if (char === "{") {
      this.advance();
      return { type: TokenType.LBRACE, value: "{", line: startLine, column: startColumn, length: 1 };
    }
    if (char === "}") {
      this.advance();
      return { type: TokenType.RBRACE, value: "}", line: startLine, column: startColumn, length: 1 };
    }
    if (char === "[") {
      this.advance();
      return { type: TokenType.LBRACKET, value: "[", line: startLine, column: startColumn, length: 1 };
    }
    if (char === "]") {
      this.advance();
      return { type: TokenType.RBRACKET, value: "]", line: startLine, column: startColumn, length: 1 };
    }
    if (char === ",") {
      this.advance();
      return { type: TokenType.COMMA, value: ",", line: startLine, column: startColumn, length: 1 };
    }
    if (char === ".") {
      this.advance();
      return { type: TokenType.DOT, value: ".", line: startLine, column: startColumn, length: 1 };
    }
    if (char === ":") {
      this.advance();
      return { type: TokenType.COLON, value: ":", line: startLine, column: startColumn, length: 1 };
    }
    if (char === "+") {
      this.advance();
      return { type: TokenType.PLUS, value: "+", line: startLine, column: startColumn, length: 1 };
    }
    if (char === "-") {
      this.advance();
      return { type: TokenType.MINUS, value: "-", line: startLine, column: startColumn, length: 1 };
    }
    if (char === "*") {
      this.advance();
      return { type: TokenType.STAR, value: "*", line: startLine, column: startColumn, length: 1 };
    }
    if (char === "/") {
      this.advance();
      return { type: TokenType.SLASH, value: "/", line: startLine, column: startColumn, length: 1 };
    }
    if (char === "%") {
      this.advance();
      return { type: TokenType.PERCENT, value: "%", line: startLine, column: startColumn, length: 1 };
    }

    // Multi-character operators
    if (char === "=") {
      this.advance();
      if (this.peek() === "=") {
        this.advance();
        return { type: TokenType.EQ, value: "==", line: startLine, column: startColumn, length: 2 };
      }
      return { type: TokenType.EQUALS, value: "=", line: startLine, column: startColumn, length: 1 };
    }

    if (char === "!") {
      this.advance();
      if (this.peek() === "=") {
        this.advance();
        return { type: TokenType.NEQ, value: "!=", line: startLine, column: startColumn, length: 2 };
      }
      return { type: TokenType.NOT, value: "!", line: startLine, column: startColumn, length: 1 };
    }

    if (char === "<") {
      this.advance();
      if (this.peek() === "=") {
        this.advance();
        return { type: TokenType.LTE, value: "<=", line: startLine, column: startColumn, length: 2 };
      }
      return { type: TokenType.LT, value: "<", line: startLine, column: startColumn, length: 1 };
    }

    if (char === ">") {
      this.advance();
      if (this.peek() === "=") {
        this.advance();
        return { type: TokenType.GTE, value: ">=", line: startLine, column: startColumn, length: 2 };
      }
      return { type: TokenType.GT, value: ">", line: startLine, column: startColumn, length: 1 };
    }

    if (char === "&" && this.peekAhead() === "&") {
      this.advance();
      this.advance();
      return { type: TokenType.AND, value: "&&", line: startLine, column: startColumn, length: 2 };
    }

    if (char === "|" && this.peekAhead() === "|") {
      this.advance();
      this.advance();
      return { type: TokenType.OR, value: "||", line: startLine, column: startColumn, length: 2 };
    }

    // Strings
    if (char === '"' || char === "'") {
      return this.readString(startLine, startColumn);
    }

    // Numbers (including scientific notation)
    if (this.isDigit(char)) {
      return this.readNumber(startLine, startColumn);
    }

    // Identifiers and keywords (including S for S-entropy)
    if (this.isIdentifierStart(char)) {
      return this.readIdentifierOrKeyword(startLine, startColumn);
    }

    // Unknown character
    this.advance();
    return null;
  }

  /**
   * Read a string literal
   */
  private readString(line: number, column: number): Token {
    const quote = this.peek();
    this.advance(); // consume opening quote

    let value = "";
    const startPos = this.position;

    while (this.position < this.source.length && this.peek() !== quote) {
      if (this.peek() === "\\") {
        this.advance();
        if (this.position < this.source.length) {
          const escaped = this.peek();
          switch (escaped) {
            case "n":
              value += "\n";
              break;
            case "t":
              value += "\t";
              break;
            case "r":
              value += "\r";
              break;
            case "\\":
              value += "\\";
              break;
            case '"':
              value += '"';
              break;
            case "'":
              value += "'";
              break;
            default:
              value += escaped;
          }
          this.advance();
        }
      } else {
        value += this.peek();
        this.advance();
      }
    }

    if (this.peek() === quote) {
      this.advance(); // consume closing quote
    }

    return {
      type: TokenType.STRING,
      value,
      line,
      column,
      length: this.position - startPos + 2,
    };
  }

  /**
   * Read a number (integer or float, with optional unit)
   */
  private readNumber(line: number, column: number): Token {
    const startPos = this.position;
    let value = "";

    // Read integer part
    while (this.isDigit(this.peek())) {
      value += this.peek();
      this.advance();
    }

    // Read decimal part
    if (this.peek() === "." && this.isDigit(this.peekAhead())) {
      value += this.peek();
      this.advance();
      while (this.isDigit(this.peek())) {
        value += this.peek();
        this.advance();
      }
    }

    // Read exponent (scientific notation)
    if (this.peek() === "e" || this.peek() === "E") {
      value += this.peek();
      this.advance();
      if (this.peek() === "+" || this.peek() === "-") {
        value += this.peek();
        this.advance();
      }
      while (this.isDigit(this.peek())) {
        value += this.peek();
        this.advance();
      }
    }

    return {
      type: TokenType.NUMBER,
      value: parseFloat(value),
      line,
      column,
      length: this.position - startPos,
    };
  }

  /**
   * Read identifier or keyword
   */
  private readIdentifierOrKeyword(line: number, column: number): Token {
    const startPos = this.position;
    let value = "";

    while (this.isIdentifierChar(this.peek())) {
      value += this.peek();
      this.advance();
    }

    // Check if it's a keyword
    const lowerValue = value.toLowerCase();
    const keywordType = this.keywords.get(lowerValue);

    if (keywordType === TokenType.S) {
      return { type: TokenType.S, value, line, column, length: this.position - startPos };
    }

    if (keywordType) {
      return {
        type: keywordType,
        value,
        line,
        column,
        length: this.position - startPos,
      };
    }

    // Regular identifier
    return {
      type: TokenType.IDENTIFIER,
      value,
      line,
      column,
      length: this.position - startPos,
    };
  }

  /**
   * Skip whitespace and comments
   */
  private skipWhitespaceAndComments(): void {
    while (this.position < this.source.length) {
      const char = this.peek();

      if (char === " " || char === "\t" || char === "\r") {
        this.advance();
      } else if (char === "\n") {
        this.advance();
        this.line++;
        this.column = 1;
      } else if (char === "#") {
        // Skip comment until end of line
        while (this.position < this.source.length && this.peek() !== "\n") {
          this.advance();
        }
      } else {
        break;
      }
    }
  }

  /**
   * Helper methods
   */
  private peek(): string {
    return this.source[this.position] || "";
  }

  private peekAhead(): string {
    return this.source[this.position + 1] || "";
  }

  private advance(): void {
    if (this.position < this.source.length) {
      this.position++;
      this.column++;
    }
  }

  private isDigit(char: string): boolean {
    return char >= "0" && char <= "9";
  }

  private isIdentifierStart(char: string): boolean {
    return (
      (char >= "a" && char <= "z") ||
      (char >= "A" && char <= "Z") ||
      char === "_" ||
      char === "S"
    );
  }

  private isIdentifierChar(char: string): boolean {
    return this.isIdentifierStart(char) || this.isDigit(char);
  }
}

export function tokenize(source: string): Token[] {
  const lexer = new Lexer(source);
  return lexer.tokenize();
}
