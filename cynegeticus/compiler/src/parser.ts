/**
 * Cynegeticus Language - Parser
 * Recursive descent parser that builds Abstract Syntax Tree (AST)
 */

import {
  Token,
  TokenType,
  Program,
  Declaration,
  Statement,
  Expression,
  CoordinateDecl,
  ConstDecl,
  PartitionDecl,
  OrbitDecl,
  SCoord,
  MeasurementStmt,
  PositionStmt,
  ValidationStmt,
  EmitStmt,
  AssignmentStmt,
  ForLoopStmt,
  IfStmt,
  NumberLiteral,
  StringLiteral,
  VariableRef,
  BinaryOp,
  CallExpr,
  Modality,
  Unit,
  CompilationError,
} from "./types";

export class Parser {
  private tokens: Token[];
  private current: number = 0;
  private errors: CompilationError[] = [];

  constructor(tokens: Token[]) {
    this.tokens = tokens;
  }

  /**
   * Main parse method
   */
  parse(): { program: Program | null; errors: CompilationError[] } {
    try {
      const declarations: Declaration[] = [];
      const statements: Statement[] = [];

      // Parse declarations first
      while (!this.isAtEnd() && this.check(TokenType.DECLARE, TokenType.CONST, TokenType.PARTITION, TokenType.SATELLITE)) {
        const decl = this.parseDeclaration();
        if (decl) declarations.push(decl);
      }

      // Then parse statements
      while (!this.isAtEnd() && this.peek().type !== TokenType.EOF) {
        const stmt = this.parseStatement();
        if (stmt) statements.push(stmt);
      }

      const program: Program = {
        declarations,
        statements,
        line: 1,
        column: 1,
      };

      return { program, errors: this.errors };
    } catch (error) {
      this.errors.push({
        type: "syntax",
        message: `Parse error: ${error instanceof Error ? error.message : String(error)}`,
        line: this.peek().line,
        column: this.peek().column,
        length: this.peek().length,
      });
      return { program: null, errors: this.errors };
    }
  }

  /**
   * Parse a declaration
   */
  private parseDeclaration(): Declaration | null {
    const token = this.peek();

    if (this.match(TokenType.DECLARE)) {
      if (this.match(TokenType.COORD)) {
        return this.parseCoordinateDecl(token);
      }
    }

    if (this.match(TokenType.CONST)) {
      return this.parseConstDecl(token);
    }

    if (this.match(TokenType.PARTITION)) {
      return this.parsePartitionDecl(token);
    }

    if (this.match(TokenType.SATELLITE)) {
      return this.parseOrbitDecl(token);
    }

    return null;
  }

  private parseCoordinateDecl(startToken: Token): CoordinateDecl {
    const name = this.consumeIdentifier("Expected coordinate name");
    this.consume(TokenType.EQUALS, "Expected '=' after coordinate name");
    const value = this.parseSCoord();

    return {
      type: "CoordinateDecl",
      name,
      value,
      line: startToken.line,
      column: startToken.column,
    };
  }

  private parseConstDecl(startToken: Token): ConstDecl {
    const name = this.consumeIdentifier("Expected constant name");
    this.consume(TokenType.COLON, "Expected ':' after constant name");

    let valueType: "coord" | "number" | "string" | "measurement";
    if (this.match(TokenType.COORD)) {
      valueType = "coord";
    } else if (this.match(TokenType.NUMBER)) {
      valueType = "number";
    } else if (this.match(TokenType.STRING)) {
      valueType = "string";
    } else if (this.match(TokenType.MEASUREMENT)) {
      valueType = "measurement";
    } else {
      this.error("Expected type");
      valueType = "number";
    }

    this.consume(TokenType.EQUALS, "Expected '=' after type");
    const value = this.parseExpression();

    return {
      type: "ConstDecl",
      name,
      valueType,
      value,
      line: startToken.line,
      column: startToken.column,
    };
  }

  private parsePartitionDecl(startToken: Token): PartitionDecl {
    const name = this.consumeIdentifier("Expected partition name");
    this.consume(TokenType.LBRACE, "Expected '{'");

    const regions = [];
    while (!this.check(TokenType.RBRACE) && !this.isAtEnd()) {
      const regionName = this.consumeIdentifier("Expected region name");
      this.consume(TokenType.IN, "Expected 'in'");

      const skMin = this.parseFloat();
      this.consume(TokenType.COLON, "Expected ':'");
      const skMax = this.parseFloat();
      this.consume(TokenType.COMMA, "Expected ','");

      const stMin = this.parseFloat();
      this.consume(TokenType.COLON, "Expected ':'");
      const stMax = this.parseFloat();
      this.consume(TokenType.COMMA, "Expected ','");

      const seMin = this.parseFloat();
      this.consume(TokenType.COLON, "Expected ':'");
      const seMax = this.parseFloat();

      regions.push({
        name: regionName,
        skBounds: [skMin, skMax],
        stBounds: [stMin, stMax],
        seBounds: [seMin, seMax],
        line: startToken.line,
        column: startToken.column,
      });

      if (!this.check(TokenType.RBRACE)) {
        this.match(TokenType.COMMA);
      }
    }

    this.consume(TokenType.RBRACE, "Expected '}'");

    return {
      type: "PartitionDecl",
      name,
      regions,
      line: startToken.line,
      column: startToken.column,
    };
  }

  private parseOrbitDecl(startToken: Token): OrbitDecl {
    this.consume(TokenType.CONSTELLATION, "Expected 'constellation'");
    const name = this.consumeIdentifier("Expected constellation name");

    let constellationType: "GPS" | "GALILEO" | "GLONASS" | "CUSTOM" = "CUSTOM";
    // Simple parsing for constellation type (can be inferred from name or explicit)

    this.consume(TokenType.COUNT, "Expected 'count'");
    this.consume(TokenType.EQUALS, "Expected '='");
    const count = this.parseExpression();

    this.consume(TokenType.ALTITUDE, "Expected 'altitude'");
    this.consume(TokenType.EQUALS, "Expected '='");
    const altitude = this.parseExpression();

    return {
      type: "OrbitDecl",
      name,
      constellationType,
      count,
      altitude,
      line: startToken.line,
      column: startToken.column,
    };
  }

  /**
   * Parse a statement
   */
  private parseStatement(): Statement | null {
    const token = this.peek();

    if (this.check(TokenType.MEASURE, TokenType.ATMOSPHERE, TokenType.ENTROPY)) {
      return this.parseMeasurementStmt(token);
    }

    if (this.check(TokenType.RESOLVE, TokenType.TRIANGULATE, TokenType.POSITION, TokenType.ACCURACY)) {
      return this.parsePositionStmt(token);
    }

    if (this.check(TokenType.VALIDATE)) {
      return this.parseValidationStmt(token);
    }

    if (this.check(TokenType.EMIT, TokenType.LOG, TokenType.OUTPUT)) {
      return this.parseEmitStmt(token);
    }

    if (this.check(TokenType.IF)) {
      return this.parseIfStmt(token);
    }

    if (this.check(TokenType.FOR)) {
      return this.parseForLoopStmt(token);
    }

    if (this.check(TokenType.IDENTIFIER)) {
      const next = this.peekAhead();
      if (next.type === TokenType.EQUALS) {
        return this.parseAssignmentStmt(token);
      }
    }

    // Try to skip unknown tokens
    if (!this.isAtEnd()) {
      this.advance();
    }

    return null;
  }

  private parseMeasurementStmt(startToken: Token): MeasurementStmt {
    let subtype: "measure" | "atmosphere" | "entropy" = "measure";

    if (this.match(TokenType.MEASURE)) {
      subtype = "measure";
    } else if (this.match(TokenType.ATMOSPHERE)) {
      subtype = "atmosphere";
    } else if (this.match(TokenType.ENTROPY)) {
      subtype = "entropy";
    }

    let modality: Modality = "vibrational";
    if (
      this.match(
        TokenType.VIBRATIONAL,
        TokenType.ROTATIONAL,
        TokenType.TRANSLATIONAL,
        TokenType.COLLISIONAL,
        TokenType.ENERGY
      )
    ) {
      modality = this.previous().value.toString().toLowerCase() as Modality;
    }

    let location: SCoord | string | undefined;
    let param: number | undefined;
    let storeAs: string | undefined;
    let intoVar: string | undefined;

    if (this.match(TokenType.AT)) {
      if (this.peek().type === TokenType.S) {
        location = this.parseSCoord();
      } else if (this.match(TokenType.HERE)) {
        location = "here";
      } else {
        location = this.consumeIdentifier("Expected location");
      }
    }

    if (subtype === "atmosphere") {
      this.consume(TokenType.LPAREN, "Expected '('");
      param = this.parseFloat();
      this.consume(TokenType.RPAREN, "Expected ')'");
    }

    if (this.match(TokenType.STORE)) {
      storeAs = this.consumeIdentifier("Expected variable name");
    }

    if (this.match(TokenType.INTO)) {
      intoVar = this.consumeIdentifier("Expected variable name");
    }

    return {
      type: "MeasurementStmt",
      subtype,
      modality,
      location,
      param,
      storeAs,
      intoVar,
      line: startToken.line,
      column: startToken.column,
    };
  }

  private parsePositionStmt(startToken: Token): PositionStmt {
    let action: "resolve" | "triangulate" | "show" | "accuracy_check" | "show_satellites" = "show";

    if (this.match(TokenType.RESOLVE)) {
      this.consume(TokenType.POSITION, "Expected 'position'");
      action = "resolve";
      this.consume(TokenType.FROM, "Expected 'from'");
      const sCoord = this.parseSCoord();
      return {
        type: "PositionStmt",
        action,
        sCoord,
        line: startToken.line,
        column: startToken.column,
      };
    }

    if (this.match(TokenType.TRIANGULATE)) {
      action = "triangulate";
      this.consume(TokenType.WITH, "Expected 'with'");
      const count = this.parseExpression();
      this.consume(TokenType.SATELLITE, "Expected 'satellite' or 'satellites'") ||
        this.consume(TokenType.IDENTIFIER, "");
      return {
        type: "PositionStmt",
        action,
        satelliteCount: count,
        line: startToken.line,
        column: startToken.column,
      };
    }

    if (this.match(TokenType.POSITION)) {
      if (this.match(TokenType.SHOW)) {
        action = "show";
      }
      return {
        type: "PositionStmt",
        action,
        line: startToken.line,
        column: startToken.column,
      };
    }

    if (this.match(TokenType.ACCURACY)) {
      this.consume(TokenType.CHECK, "Expected 'check'");
      action = "accuracy_check";
      this.consume(TokenType.TARGET, "Expected 'target'");
      this.consume(TokenType.EQUALS, "Expected '='");
      const target = this.parseFloat();
      const unit = this.parseUnit();
      return {
        type: "PositionStmt",
        action,
        targetAccuracy: target,
        unit,
        line: startToken.line,
        column: startToken.column,
      };
    }

    return {
      type: "PositionStmt",
      action: "show",
      line: startToken.line,
      column: startToken.column,
    };
  }

  private parseValidationStmt(startToken: Token): ValidationStmt {
    this.consume(TokenType.VALIDATE, "Expected 'validate'");

    let validationType: "circular_closure" | "position_against_known" = "circular_closure";

    if (this.match(TokenType.CIRCULAR)) {
      this.consume(TokenType.CLOSURE, "Expected 'closure'");
      validationType = "circular_closure";
      this.consume(TokenType.RMSE, "Expected 'rmse'");
      this.consume(TokenType.LT, "Expected '<'");
      const threshold = this.parseFloat();
      const unit = this.parseUnit();

      return {
        type: "ValidationStmt",
        validationType,
        threshold,
        thresholdUnit: unit,
        line: startToken.line,
        column: startToken.column,
      };
    }

    if (this.match(TokenType.POSITION)) {
      this.consume(TokenType.AGAINST, "Expected 'against'");
      this.consume(TokenType.KNOWN, "Expected 'known'");
      validationType = "position_against_known";

      return {
        type: "ValidationStmt",
        validationType,
        line: startToken.line,
        column: startToken.column,
      };
    }

    return {
      type: "ValidationStmt",
      validationType,
      line: startToken.line,
      column: startToken.column,
    };
  }

  private parseEmitStmt(startToken: Token): EmitStmt {
    let emitType: "emit" | "log" | "output" = "emit";

    if (this.match(TokenType.EMIT)) {
      emitType = "emit";
      const identifier = this.consumeIdentifier("Expected identifier");
      return {
        type: "EmitStmt",
        emitType,
        identifier,
        line: startToken.line,
        column: startToken.column,
      };
    }

    if (this.match(TokenType.LOG)) {
      emitType = "log";
      const message = this.consumeString("Expected string");
      return {
        type: "EmitStmt",
        emitType,
        message,
        line: startToken.line,
        column: startToken.column,
      };
    }

    if (this.match(TokenType.OUTPUT)) {
      emitType = "output";
      const value = this.parseExpression();
      return {
        type: "EmitStmt",
        emitType,
        value,
        line: startToken.line,
        column: startToken.column,
      };
    }

    return {
      type: "EmitStmt",
      emitType,
      line: startToken.line,
      column: startToken.column,
    };
  }

  private parseAssignmentStmt(startToken: Token): AssignmentStmt {
    const target = this.consumeIdentifier("Expected variable name");
    this.consume(TokenType.EQUALS, "Expected '='");
    const value = this.parseExpression();

    return {
      type: "AssignmentStmt",
      target,
      value,
      line: startToken.line,
      column: startToken.column,
    };
  }

  private parseIfStmt(startToken: Token): IfStmt {
    this.consume(TokenType.IF, "Expected 'if'");
    const condition = this.parseExpression();
    this.consume(TokenType.DO, "Expected 'do'");

    const thenBranch: Statement[] = [];
    while (!this.check(TokenType.ELSE) && !this.isAtEnd()) {
      const stmt = this.parseStatement();
      if (stmt) thenBranch.push(stmt);
    }

    let elseBranch: Statement[] | undefined;
    if (this.match(TokenType.ELSE)) {
      elseBranch = [];
      while (!this.isAtEnd()) {
        const stmt = this.parseStatement();
        if (stmt) elseBranch.push(stmt);
      }
    }

    return {
      type: "IfStmt",
      condition,
      thenBranch,
      elseBranch,
      line: startToken.line,
      column: startToken.column,
    };
  }

  private parseForLoopStmt(startToken: Token): ForLoopStmt {
    this.consume(TokenType.FOR, "Expected 'for'");
    const variable = this.consumeIdentifier("Expected variable name");
    this.consume(TokenType.EQUALS, "Expected '='");
    const start = this.parseExpression();
    this.consume(TokenType.TO, "Expected 'to'") || this.match(TokenType.IDENTIFIER);
    const end = this.parseExpression();

    let step: Expression | undefined;
    if (this.match(TokenType.STEP)) {
      step = this.parseExpression();
    }

    this.consume(TokenType.DO, "Expected 'do'");

    const body: Statement[] = [];
    while (!this.isAtEnd()) {
      const stmt = this.parseStatement();
      if (stmt) body.push(stmt);
    }

    return {
      type: "ForLoopStmt",
      variable,
      start,
      end,
      step,
      body,
      line: startToken.line,
      column: startToken.column,
    };
  }

  /**
   * Parse expressions
   */
  private parseExpression(): Expression {
    return this.parseLogicalOr();
  }

  private parseLogicalOr(): Expression {
    let expr = this.parseLogicalAnd();

    while (this.match(TokenType.OR)) {
      const operator = this.previous().value.toString();
      const right = this.parseLogicalAnd();
      expr = {
        type: "BinaryOp",
        operator,
        left: expr,
        right,
        line: expr.line,
        column: expr.column,
      };
    }

    return expr;
  }

  private parseLogicalAnd(): Expression {
    let expr = this.parseEquality();

    while (this.match(TokenType.AND)) {
      const operator = this.previous().value.toString();
      const right = this.parseEquality();
      expr = {
        type: "BinaryOp",
        operator,
        left: expr,
        right,
        line: expr.line,
        column: expr.column,
      };
    }

    return expr;
  }

  private parseEquality(): Expression {
    let expr = this.parseComparison();

    while (this.match(TokenType.EQ, TokenType.NEQ)) {
      const operator = this.previous().value.toString();
      const right = this.parseComparison();
      expr = {
        type: "BinaryOp",
        operator,
        left: expr,
        right,
        line: expr.line,
        column: expr.column,
      };
    }

    return expr;
  }

  private parseComparison(): Expression {
    let expr = this.parseAddition();

    while (this.match(TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE)) {
      const operator = this.previous().value.toString();
      const right = this.parseAddition();
      expr = {
        type: "BinaryOp",
        operator,
        left: expr,
        right,
        line: expr.line,
        column: expr.column,
      };
    }

    return expr;
  }

  private parseAddition(): Expression {
    let expr = this.parseMultiplication();

    while (this.match(TokenType.PLUS, TokenType.MINUS)) {
      const operator = this.previous().value.toString();
      const right = this.parseMultiplication();
      expr = {
        type: "BinaryOp",
        operator,
        left: expr,
        right,
        line: expr.line,
        column: expr.column,
      };
    }

    return expr;
  }

  private parseMultiplication(): Expression {
    let expr = this.parseUnary();

    while (this.match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT)) {
      const operator = this.previous().value.toString();
      const right = this.parseUnary();
      expr = {
        type: "BinaryOp",
        operator,
        left: expr,
        right,
        line: expr.line,
        column: expr.column,
      };
    }

    return expr;
  }

  private parseUnary(): Expression {
    if (this.match(TokenType.NOT, TokenType.MINUS)) {
      const operator = this.previous().value.toString();
      const operand = this.parseUnary();
      return {
        type: "UnaryOp",
        operator,
        operand,
        line: this.previous().line,
        column: this.previous().column,
      };
    }

    return this.parsePrimary();
  }

  private parsePrimary(): Expression {
    const token = this.peek();

    if (this.match(TokenType.NUMBER)) {
      const unit = this.parseUnit();
      return {
        type: "NumberLiteral",
        value: this.previous().value as number,
        unit,
        line: token.line,
        column: token.column,
      };
    }

    if (this.match(TokenType.STRING)) {
      return {
        type: "StringLiteral",
        value: this.previous().value as string,
        line: token.line,
        column: token.column,
      };
    }

    if (this.match(TokenType.S)) {
      return this.parseSCoord();
    }

    if (this.match(TokenType.LPAREN)) {
      const expr = this.parseExpression();
      this.consume(TokenType.RPAREN, "Expected ')'");
      return expr;
    }

    if (this.match(TokenType.IDENTIFIER)) {
      const name = this.previous().value as string;

      // Check for function call
      if (this.match(TokenType.LPAREN)) {
        const args: Expression[] = [];
        if (!this.check(TokenType.RPAREN)) {
          do {
            args.push(this.parseExpression());
          } while (this.match(TokenType.COMMA));
        }
        this.consume(TokenType.RPAREN, "Expected ')'");

        return {
          type: "CallExpr",
          function: name,
          arguments: args,
          line: token.line,
          column: token.column,
        };
      }

      return {
        type: "VariableRef",
        name,
        line: token.line,
        column: token.column,
      };
    }

    this.error(`Unexpected token: ${token.type}`);
    return {
      type: "NumberLiteral",
      value: 0,
      line: token.line,
      column: token.column,
    };
  }

  /**
   * Parse S-entropy coordinate
   */
  private parseSCoord(): SCoord {
    const token = this.peek();
    this.consume(TokenType.S, "Expected 'S'");
    this.consume(TokenType.LPAREN, "Expected '('");

    const sk = this.parseExpression();
    this.consume(TokenType.COMMA, "Expected ','");

    const st = this.parseExpression();
    this.consume(TokenType.COMMA, "Expected ','");

    const se = this.parseExpression();
    this.consume(TokenType.RPAREN, "Expected ')'");

    return {
      type: "SCoord",
      sk,
      st,
      se,
      line: token.line,
      column: token.column,
    };
  }

  /**
   * Parse a unit (cm, m, km, Hz, etc.)
   */
  private parseUnit(): Unit | undefined {
    if (this.match(
      TokenType.CM,
      TokenType.MM,
      TokenType.M,
      TokenType.KM,
      TokenType.HZ,
      TokenType.THZ,
      TokenType.K,
      TokenType.PA,
      TokenType.HPA
    )) {
      return this.previous().value.toString().toLowerCase() as Unit;
    }
    return undefined;
  }

  /**
   * Helper methods for parsing
   */
  private parseFloat(): number {
    if (this.match(TokenType.NUMBER)) {
      return this.previous().value as number;
    }
    if (this.match(TokenType.MINUS)) {
      const value = this.parseFloat();
      return -value;
    }
    this.error("Expected number");
    return 0;
  }

  private consumeIdentifier(message: string): string {
    if (this.match(TokenType.IDENTIFIER)) {
      return this.previous().value as string;
    }
    this.error(message);
    return "";
  }

  private consumeString(message: string): string {
    if (this.match(TokenType.STRING)) {
      return this.previous().value as string;
    }
    this.error(message);
    return "";
  }

  private match(...types: TokenType[]): boolean {
    for (const type of types) {
      if (this.check(type)) {
        this.advance();
        return true;
      }
    }
    return false;
  }

  private check(...types: TokenType[]): boolean {
    if (this.isAtEnd()) return false;
    return types.includes(this.peek().type);
  }

  private advance(): Token {
    if (!this.isAtEnd()) this.current++;
    return this.previous();
  }

  private isAtEnd(): boolean {
    return this.peek().type === TokenType.EOF;
  }

  private peek(): Token {
    return this.tokens[this.current];
  }

  private peekAhead(): Token {
    if (this.current + 1 >= this.tokens.length) {
      return this.tokens[this.tokens.length - 1];
    }
    return this.tokens[this.current + 1];
  }

  private previous(): Token {
    return this.tokens[this.current - 1];
  }

  private consume(type: TokenType, message: string): boolean {
    if (this.check(type)) {
      this.advance();
      return true;
    }
    this.error(message);
    return false;
  }

  private error(message: string): void {
    const token = this.peek();
    this.errors.push({
      type: "syntax",
      message,
      line: token.line,
      column: token.column,
      length: token.length,
    });
  }
}

export function parse(tokens: Token[]): { program: Program | null; errors: CompilationError[] } {
  const parser = new Parser(tokens);
  return parser.parse();
}
