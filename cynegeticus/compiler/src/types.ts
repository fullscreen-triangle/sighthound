/**
 * Cynegeticus Language - Type Definitions
 * Complete AST and type system for position-focused GPS-free geolocation language
 */

// ============================================================================
// TOKENS AND LEXICAL TYPES
// ============================================================================

export enum TokenType {
  // Literals
  NUMBER = "NUMBER",
  STRING = "STRING",
  IDENTIFIER = "IDENTIFIER",

  // Keywords
  DECLARE = "DECLARE",
  CONST = "CONST",
  PARTITION = "PARTITION",
  SATELLITE = "SATELLITE",
  CONSTELLATION = "CONSTELLATION",
  MEASURE = "MEASURE",
  ATMOSPHERE = "ATMOSPHERE",
  ENTROPY = "ENTROPY",
  RESOLVE = "RESOLVE",
  POSITION = "POSITION",
  TRIANGULATE = "TRIANGULATE",
  VALIDATE = "VALIDATE",
  CIRCULAR = "CIRCULAR",
  CLOSURE = "CLOSURE",
  AGAINST = "AGAINST",
  KNOWN = "KNOWN",
  ACCURACY = "ACCURACY",
  CHECK = "CHECK",
  TARGET = "TARGET",
  EMIT = "EMIT",
  LOG = "LOG",
  OUTPUT = "OUTPUT",
  SHOW = "SHOW",
  FROM = "FROM",
  AT = "AT",
  STORE = "STORE",
  INTO = "INTO",
  WITH = "WITH",
  OF = "OF",
  COUNT = "COUNT",
  ALTITUDE = "ALTITUDE",
  RMSE = "RMSE",
  HERE = "HERE",
  IN = "IN",
  ACTION = "ACTION",
  DO = "DO",
  IF = "IF",
  ELSE = "ELSE",
  FOR = "FOR",
  WHILE = "WHILE",

  // Modalities
  VIBRATIONAL = "VIBRATIONAL",
  ROTATIONAL = "ROTATIONAL",
  TRANSLATIONAL = "TRANSLATIONAL",
  COLLISIONAL = "COLLISIONAL",
  ENERGY = "ENERGY",

  // Types
  COORD = "COORD",
  MEASUREMENT = "MEASUREMENT",
  ARRAY = "ARRAY",

  // Units
  CM = "CM",
  MM = "MM",
  M = "M",
  KM = "KM",
  HZ = "HZ",
  THZ = "THZ",
  K = "K",
  PA = "PA",
  HPA = "HPA",

  // Operators
  EQUALS = "EQUALS",
  COLON = "COLON",
  LPAREN = "LPAREN",
  RPAREN = "RPAREN",
  LBRACE = "LBRACE",
  RBRACE = "RBRACE",
  LBRACKET = "LBRACKET",
  RBRACKET = "RBRACKET",
  COMMA = "COMMA",
  DOT = "DOT",
  PLUS = "PLUS",
  MINUS = "MINUS",
  STAR = "STAR",
  SLASH = "SLASH",
  PERCENT = "PERCENT",
  LT = "LT",
  GT = "GT",
  LTE = "LTE",
  GTE = "GTE",
  EQ = "EQ",
  NEQ = "NEQ",
  AND = "AND",
  OR = "OR",
  NOT = "NOT",

  // Special
  S = "S",
  EOF = "EOF",
  NEWLINE = "NEWLINE",
}

export interface Token {
  type: TokenType;
  value: string | number;
  line: number;
  column: number;
  length: number;
}

// ============================================================================
// AST NODE TYPES
// ============================================================================

export interface ASTNode {
  line: number;
  column: number;
}

export interface Program extends ASTNode {
  declarations: Declaration[];
  statements: Statement[];
}

export type Declaration = CoordinateDecl | PartitionDecl | OrbitDecl | ConstDecl;

export interface CoordinateDecl extends ASTNode {
  type: "CoordinateDecl";
  name: string;
  value: SCoord;
}

export interface ConstDecl extends ASTNode {
  type: "ConstDecl";
  name: string;
  valueType: PrimitiveType;
  value: Expression;
}

export interface PartitionDecl extends ASTNode {
  type: "PartitionDecl";
  name: string;
  regions: RegionDef[];
}

export interface RegionDef extends ASTNode {
  name: string;
  skBounds: [number, number];
  stBounds: [number, number];
  seBounds: [number, number];
}

export interface OrbitDecl extends ASTNode {
  type: "OrbitDecl";
  name: string;
  constellationType: "GPS" | "GALILEO" | "GLONASS" | "CUSTOM";
  count: Expression;
  altitude: Expression;
}

export type Statement =
  | MeasurementStmt
  | PositionStmt
  | ValidationStmt
  | EmitStmt
  | AssignmentStmt
  | ForLoopStmt
  | IfStmt;

export interface MeasurementStmt extends ASTNode {
  type: "MeasurementStmt";
  subtype: "measure" | "atmosphere" | "entropy";
  modality: Modality;
  location?: SCoord | string;
  param?: number;
  storeAs?: string;
  intoVar?: string;
}

export interface PositionStmt extends ASTNode {
  type: "PositionStmt";
  action:
    | "resolve"
    | "triangulate"
    | "show"
    | "accuracy_check"
    | "show_satellites";
  sCoord?: SCoord;
  satelliteCount?: Expression;
  targetAccuracy?: number;
  unit?: Unit;
}

export interface ValidationStmt extends ASTNode {
  type: "ValidationStmt";
  validationType: "circular_closure" | "position_against_known";
  threshold?: number;
  thresholdUnit?: Unit;
  knownPosition?: SCoord;
}

export interface EmitStmt extends ASTNode {
  type: "EmitStmt";
  emitType: "emit" | "log" | "output";
  identifier?: string;
  message?: string;
  value?: Expression;
}

export interface AssignmentStmt extends ASTNode {
  type: "AssignmentStmt";
  target: string;
  value: Expression;
}

export interface ForLoopStmt extends ASTNode {
  type: "ForLoopStmt";
  variable: string;
  start: Expression;
  end: Expression;
  step?: Expression;
  body: Statement[];
}

export interface IfStmt extends ASTNode {
  type: "IfStmt";
  condition: Expression;
  thenBranch: Statement[];
  elseBranch?: Statement[];
}

// ============================================================================
// EXPRESSIONS AND COORDINATES
// ============================================================================

export interface SCoord extends ASTNode {
  type: "SCoord";
  sk: Expression; // [0, 1]
  st: Expression; // [0, 1]
  se: Expression; // [0, 1]
}

export type Expression =
  | NumberLiteral
  | StringLiteral
  | VariableRef
  | BinaryOp
  | UnaryOp
  | CallExpr
  | SCoord;

export interface NumberLiteral extends ASTNode {
  type: "NumberLiteral";
  value: number;
  unit?: Unit;
}

export interface StringLiteral extends ASTNode {
  type: "StringLiteral";
  value: string;
}

export interface VariableRef extends ASTNode {
  type: "VariableRef";
  name: string;
}

export interface BinaryOp extends ASTNode {
  type: "BinaryOp";
  operator: string;
  left: Expression;
  right: Expression;
}

export interface UnaryOp extends ASTNode {
  type: "UnaryOp";
  operator: string;
  operand: Expression;
}

export interface CallExpr extends ASTNode {
  type: "CallExpr";
  function: string;
  arguments: Expression[];
}

// ============================================================================
// TYPE SYSTEM
// ============================================================================

export type PrimitiveType = "coord" | "number" | "string" | "measurement";

export interface CompositeType {
  kind: "array";
  elementType: PrimitiveType;
}

export type VariableType = PrimitiveType | CompositeType;

export interface TypedVariable {
  name: string;
  type: VariableType;
  value?: any;
  mutable: boolean;
}

// ============================================================================
// SEMANTIC TYPES
// ============================================================================

export type Modality =
  | "vibrational"
  | "rotational"
  | "translational"
  | "collisional"
  | "energy";

export type Unit =
  | "cm"
  | "mm"
  | "m"
  | "km"
  | "Hz"
  | "THz"
  | "K"
  | "Pa"
  | "hPa";

// ============================================================================
// RUNTIME STATE
// ============================================================================

export interface Coord {
  sk: number;
  st: number;
  se: number;
}

export interface Measurement {
  modality: Modality;
  value: number;
  unit: Unit;
  timestamp: number;
}

export interface Satellite {
  id: number;
  lat: number;
  lon: number;
  altitude: number;
}

export interface Position {
  lat: number;
  lon: number;
  altitude: number;
  timestamp: number;
}

export interface Validation {
  type: string;
  passed: boolean;
  value: number;
  threshold: number;
  details: string;
}

export interface ProgramState {
  coordinates: Map<string, Coord>;
  measurements: Map<string, Measurement>;
  variables: Map<string, any>;
  satellites: Satellite[];
  position: Position | null;
  accuracy: number | null;
  validations: Validation[];
  executionTrace: ExecutionFrame[];
}

export interface ExecutionFrame {
  statement: Statement;
  stateBefore: Partial<ProgramState>;
  stateAfter: Partial<ProgramState>;
  line: number;
  timestamp: number;
}

// ============================================================================
// ERRORS
// ============================================================================

export interface CompilationError {
  type: "syntax" | "semantic" | "runtime" | "validation";
  message: string;
  line: number;
  column: number;
  length: number;
}

export interface CompilationResult {
  success: boolean;
  ast?: Program;
  code?: string;
  errors: CompilationError[];
  warnings: CompilationError[];
}

// ============================================================================
// VISITOR PATTERN FOR AST TRAVERSAL
// ============================================================================

export interface ASTVisitor {
  visitProgram?(node: Program): void;
  visitCoordinateDecl?(node: CoordinateDecl): void;
  visitPartitionDecl?(node: PartitionDecl): void;
  visitOrbitDecl?(node: OrbitDecl): void;
  visitMeasurementStmt?(node: MeasurementStmt): void;
  visitPositionStmt?(node: PositionStmt): void;
  visitValidationStmt?(node: ValidationStmt): void;
  visitSCoord?(node: SCoord): void;
  visitNumberLiteral?(node: NumberLiteral): void;
  visitStringLiteral?(node: StringLiteral): void;
  visitVariableRef?(node: VariableRef): void;
}
