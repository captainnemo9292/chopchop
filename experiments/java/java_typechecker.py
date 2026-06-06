"""
Concrete AST type checker for Java.

Type rules follow JLS SE8:
  §5.1.2  Widening primitive conversion
  §5.1.7  Boxing conversion
  §5.1.8  Unboxing conversion
  §5.2    Assignment contexts (incl. constant-expression narrowing)
  §5.3    Invocation contexts
  §5.6    Numeric promotion (unary and binary)
  §14.9   if / if-else  (boolean or Boolean condition)
  §14.10  assert
  §14.11  switch  (char/byte/short/int/String/enum only)
  §14.12  while
  §14.13  do-while
  §14.14  for / enhanced-for
  §14.17  return
  §14.18  throw  (must be Throwable)
  §14.19  synchronized  (reference type required)
  §14.20  try-catch  (catch param must be Throwable)
  §15.14  postfix ++/--
  §15.15  prefix ++/--, unary +/-/~/!
  §15.17  multiplicative  * / %
  §15.18  additive  + -  (incl. String concatenation)
  §15.19  shift  << >> >>>
  §15.20  relational  < > <= >= instanceof
  §15.21  equality  == !=  (numeric / boolean / reference — mixed is error)
  §15.22  bitwise & ^ |  (integer and boolean variants)
  §15.23  && (boolean only)
  §15.24  || (boolean only)
  §15.25  ternary  ? :
  §15.26  assignment / compound assignment  (implicit cast for op=)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import tree_sitter_java as tsj
from tree_sitter import Language, Parser, Node

JAVA_LANGUAGE = Language(tsj.language())
_parser = Parser(JAVA_LANGUAGE)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Type:
    def __eq__(self, other): return type(self) == type(other)
    def __hash__(self): return hash(type(self))
    def __repr__(self): return type(self).__name__

@dataclass(frozen=True)
class ByteType(Type):    pass
@dataclass(frozen=True)
class ShortType(Type):   pass
@dataclass(frozen=True)
class IntType(Type):     pass
@dataclass(frozen=True)
class LongType(Type):    pass
@dataclass(frozen=True)
class FloatType(Type):   pass
@dataclass(frozen=True)
class DoubleType(Type):  pass
@dataclass(frozen=True)
class CharType(Type):    pass
@dataclass(frozen=True)
class BooleanType(Type): pass
@dataclass(frozen=True)
class StringType(Type):  pass
@dataclass(frozen=True)
class VoidType(Type):    pass
@dataclass(frozen=True)
class NullType(Type):    pass
@dataclass(frozen=True)
class TopType(Type):     pass   # unknown / unconstrained
@dataclass(frozen=True)
class EmptyType(Type):   pass   # no valid type (error sentinel)

@dataclass(frozen=True)
class ArrayType(Type):
    element: Type
    def __repr__(self): return f"{self.element}[]"

@dataclass(frozen=True)
class ClassType(Type):
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class FuncType(Type):
    params: tuple[Type, ...]
    return_type: Type
    extensible: bool = False
    def __repr__(self):
        return f"({', '.join(map(repr, self.params))}) -> {self.return_type!r}"

# Singletons
BYTE    = ByteType()
SHORT   = ShortType()
INT     = IntType()
LONG    = LongType()
FLOAT   = FloatType()
DOUBLE  = DoubleType()
CHAR    = CharType()
BOOLEAN = BooleanType()
STRING  = StringType()
VOID    = VoidType()
NULL    = NullType()
TOP     = TopType()
EMPTY   = EmptyType()

# Boxed wrapper types (§5.1.7)
_BOXED: dict[Type, ClassType] = {
    BOOLEAN: ClassType("Boolean"),
    BYTE:    ClassType("Byte"),
    SHORT:   ClassType("Short"),
    CHAR:    ClassType("Character"),
    INT:     ClassType("Integer"),
    LONG:    ClassType("Long"),
    FLOAT:   ClassType("Float"),
    DOUBLE:  ClassType("Double"),
}
_UNBOXED: dict[ClassType, Type] = {v: k for k, v in _BOXED.items()}

# §5.1.2 Widening primitive conversions
_WIDENING: dict[Type, set[Type]] = {
    BYTE:  {SHORT, INT, LONG, FLOAT, DOUBLE},
    SHORT: {INT, LONG, FLOAT, DOUBLE},
    CHAR:  {INT, LONG, FLOAT, DOUBLE},
    INT:   {LONG, FLOAT, DOUBLE},
    LONG:  {FLOAT, DOUBLE},
    FLOAT: {DOUBLE},
}

# Types valid as switch expression (§14.11)
_SWITCH_TYPES: set[Type] = {
    CHAR, BYTE, SHORT, INT,
    ClassType("Character"), ClassType("Byte"),
    ClassType("Short"), ClassType("Integer"),
    STRING,
    # enums handled separately via ClassType
}

def is_numeric(t: Type) -> bool:
    return t in (BYTE, SHORT, INT, LONG, FLOAT, DOUBLE, CHAR)

def is_integral(t: Type) -> bool:
    return t in (BYTE, SHORT, INT, LONG, CHAR)

def is_reference_type(t: Type) -> bool:
    return isinstance(t, (ClassType, ArrayType, StringType)) or t == NULL

def is_primitive(t: Type) -> bool:
    return t in (BYTE, SHORT, INT, LONG, FLOAT, DOUBLE, CHAR, BOOLEAN)

# §5.6.1 Unary numeric promotion
def unary_promote(t: Type) -> Type:
    if t in (BYTE, SHORT, CHAR): return INT
    return t

# §5.6.2 Binary numeric promotion
def binary_promote(a: Type, b: Type) -> Type:
    for wide in (DOUBLE, FLOAT, LONG):
        if a == wide or b == wide:
            return wide
    return INT


# ---------------------------------------------------------------------------
# Class hierarchy
# ---------------------------------------------------------------------------

@dataclass
class ClassHierarchy:
    _parents: dict[str, set[str]] = field(default_factory=dict)

    # Throwable hierarchy pre-loaded
    def __post_init__(self):
        for cls in ("Exception", "RuntimeException", "Error",
                    "IllegalArgumentException", "NullPointerException",
                    "IndexOutOfBoundsException", "ArrayIndexOutOfBoundsException",
                    "ClassCastException", "ArithmeticException",
                    "UnsupportedOperationException", "IllegalStateException",
                    "IOException", "FileNotFoundException"):
            self._parents.setdefault(cls, set())
        self._parents["Exception"].add("Throwable")
        self._parents["Error"].add("Throwable")
        self._parents["RuntimeException"].add("Exception")
        for sub in ("IllegalArgumentException", "NullPointerException",
                    "IndexOutOfBoundsException", "ClassCastException",
                    "ArithmeticException", "UnsupportedOperationException",
                    "IllegalStateException"):
            self._parents[sub].add("RuntimeException")
        self._parents["ArrayIndexOutOfBoundsException"].add("IndexOutOfBoundsException")
        self._parents["FileNotFoundException"].add("IOException")
        self._parents["IOException"].add("Exception")

    def add(self, child: str, *parents: str) -> ClassHierarchy:
        self._parents.setdefault(child, set()).update(parents)
        return self

    def is_throwable(self, t: Type) -> bool:
        if isinstance(t, ClassType):
            return t.name == "Throwable" or self._class_is_subtype(t.name, "Throwable")
        return False

    def is_subtype(self, sub: Type, sup: Type) -> bool:
        """Check sub <: sup per §4.10, including boxing/unboxing and widening."""
        if sub == sup or sup == TOP:
            return True
        # null assignable to any reference type (§5.2)
        if sub == NULL and is_reference_type(sup):
            return True
        # widening primitive (§5.1.2)
        if sub in _WIDENING and sup in _WIDENING.get(sub, set()):
            return True
        # boxing: int <: Integer (§5.1.7)
        if sub in _BOXED and _BOXED[sub] == sup:
            return True
        # boxing + widening reference: int <: Number (§5.2)
        if sub in _BOXED:
            boxed = _BOXED[sub]
            if isinstance(sup, ClassType):
                return self._class_is_subtype(boxed.name, sup.name)
        # unboxing: Integer <: int (§5.1.8)
        if isinstance(sub, ClassType) and sub in _UNBOXED:
            unboxed = _UNBOXED[sub]
            if sup == unboxed:
                return True
            # unboxing + widening: Integer <: long
            if unboxed in _WIDENING and sup in _WIDENING[unboxed]:
                return True
        # class hierarchy (§5.1.5)
        if isinstance(sub, ClassType) and isinstance(sup, ClassType):
            return self._class_is_subtype(sub.name, sup.name)
        # array covariance
        if isinstance(sub, ArrayType) and isinstance(sup, ArrayType):
            return self.is_subtype(sub.element, sup.element)
        # String <: Object
        if sub == STRING and isinstance(sup, ClassType) and sup.name in ("Object", "Comparable", "Serializable"):
            return True
        return False

    def _class_is_subtype(self, sub: str, sup: str) -> bool:
        if sub == sup:
            return True
        visited: set[str] = set()
        stack = list(self._parents.get(sub, []))
        while stack:
            current = stack.pop()
            if current == sup:
                return True
            if current not in visited:
                visited.add(current)
                stack.extend(self._parents.get(current, []))
        return False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class Env:
    _bindings: dict[str, list[tuple[Type, bool]]] = field(default_factory=dict)

    def lookup(self, name: str) -> Optional[Type]:
        entries = self._bindings.get(name, [])
        if not entries:      return None
        if len(entries) == 1: return entries[0][0]
        return TOP  # overloaded

    def lookup_all(self, name: str) -> list[Type]:
        return [t for t, _ in self._bindings.get(name, [])]

    def is_mutable(self, name: str) -> bool:
        entries = self._bindings.get(name, [])
        return entries[0][1] if entries else False

    def add(self, name: str, typ: Type, mutable: bool = True) -> Env:
        new = Env({k: list(v) for k, v in self._bindings.items()})
        new._bindings.setdefault(name, []).append((typ, mutable))
        return new


# ---------------------------------------------------------------------------
# Default environment
# ---------------------------------------------------------------------------

def make_default_env(hierarchy: ClassHierarchy) -> Env:
    env = Env()
    stdlib = {
        "Math.sqrt":  (FuncType((DOUBLE,), DOUBLE), False),
        "Math.pow":   (FuncType((DOUBLE, DOUBLE), DOUBLE), False),
        "Math.abs":   (FuncType((DOUBLE,), DOUBLE), False),
        "Math.floor": (FuncType((DOUBLE,), DOUBLE), False),
        "Math.ceil":  (FuncType((DOUBLE,), DOUBLE), False),
        "Math.round": (FuncType((DOUBLE,), LONG), False),
        "Math.min":   (FuncType((DOUBLE, DOUBLE), DOUBLE), False),
        "Math.max":   (FuncType((DOUBLE, DOUBLE), DOUBLE), False),
        "Math.PI":    (DOUBLE, False),
        "System.out.println": (FuncType((TOP,), VOID), False),
        "System.out.print":   (FuncType((TOP,), VOID), False),
    }
    for name, (typ, mutable) in stdlib.items():
        env = env.add(name, typ, mutable)
    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text(node: Node) -> str:
    return node.text.decode()

def named_children(node: Node) -> list[Node]:
    return [c for c in node.children if c.is_named]

def child_by_type(node: Node, *types: str) -> Optional[Node]:
    for c in node.children:
        if c.type in types:
            return c
    return None

def parse_java_type(node: Node) -> Type:
    match node.type:
        case "integral_type":
            match text(node):
                case "int":   return INT
                case "long":  return LONG
                case "short": return SHORT
                case "byte":  return BYTE
                case "char":  return CHAR
        case "floating_point_type":
            match text(node):
                case "float":  return FLOAT
                case "double": return DOUBLE
        case "boolean_type":  return BOOLEAN
        case "void_type":     return VOID
        case "type_identifier":
            name = text(node)
            if name == "String": return STRING
            if name == "var":    return TOP
            return ClassType(name)
        case "array_type":
            elem_node = named_children(node)[0]
            return ArrayType(parse_java_type(elem_node))
        case _:
            return TOP

def _is_int_constant(node: Node) -> Optional[int]:
    """Return integer value if node is a compile-time integer constant, else None. §15.28"""
    match node.type:
        case "decimal_integer_literal":
            s = text(node).rstrip("lL")
            try: return int(s)
            except ValueError: return None
        case "hex_integer_literal":
            s = text(node).rstrip("lL")
            try: return int(s, 16)
            except ValueError: return None
        case "binary_integer_literal":
            s = text(node).rstrip("lL").replace("0b", "").replace("0B", "")
            try: return int(s, 2)
            except ValueError: return None
        case "octal_integer_literal":
            s = text(node).rstrip("lL")
            try: return int(s, 8)
            except ValueError: return None
        case _:
            return None

# Ranges for narrowing constant expressions (§5.2)
_NARROW_RANGE: dict[Type, tuple[int, int]] = {
    BYTE:  (-128, 127),
    SHORT: (-32768, 32767),
    CHAR:  (0, 65535),
}


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------

@dataclass
class TypeError_:
    node: Node
    message: str
    def __str__(self):
        line = self.node.start_point[0] + 1
        col  = self.node.start_point[1] + 1
        return f"Line {line}:{col}: {self.message}"


# ---------------------------------------------------------------------------
# Type checker
# ---------------------------------------------------------------------------

class JavaTypeChecker:
    def __init__(self, env: Env, hierarchy: ClassHierarchy):
        self.env = env
        self.hierarchy = hierarchy
        self.errors: list[TypeError_] = []

    def error(self, node: Node, msg: str):
        self.errors.append(TypeError_(node, msg))

    def assignable(self, target: Type, source: Type) -> bool:
        """Assignment context compatibility (§5.2)."""
        return self.hierarchy.is_subtype(source, target)

    def assignable_with_constant(self, target: Type, source: Type,
                                  rhs_node: Optional[Node]) -> bool:
        """§5.2: also allows constant int narrowing to byte/short/char."""
        if self.assignable(target, source):
            return True
        # constant expression narrowing: int constant → byte/short/char
        if source == INT and target in _NARROW_RANGE and rhs_node is not None:
            val = _is_int_constant(rhs_node)
            if val is not None:
                lo, hi = _NARROW_RANGE[target]
                return lo <= val <= hi
        return False

    def _boolean_compatible(self, t: Type) -> bool:
        """§14.9: condition may be boolean or Boolean (unboxed)."""
        return t == BOOLEAN or t == ClassType("Boolean") or t == TOP

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def check_program(self, source: str) -> list[TypeError_]:
        tree = _parser.parse(bytes(source, "utf8"))
        self._check_block(tree.root_node, self.env, VOID)
        return self.errors

    # -----------------------------------------------------------------------
    # Statement sequence — threads Γ left to right
    # -----------------------------------------------------------------------

    def _check_block(self, block: Node, env: Env, return_type: Type) -> Env:
        stmts = [c for c in block.children
                 if c.is_named and c.type not in
                 ("comment", "line_comment", "block_comment")]
        for stmt in stmts:
            env = self._check_statement(stmt, env, return_type)
        return env

    # -----------------------------------------------------------------------
    # Statement dispatch
    # -----------------------------------------------------------------------

    def _check_statement(self, node: Node, env: Env, return_type: Type) -> Env:
        match node.type:

            case "local_variable_declaration":
                return self._check_local_decl(node, env)

            case "expression_statement":
                inner = named_children(node)
                expr = inner[0] if inner else node
                match expr.type:
                    case "assignment_expression":
                        op_node = child_by_type(expr, "=", "+=", "-=", "*=", "/=",
                                                "&=", "|=", "^=", "<<=", ">>=", ">>>=")
                        op = text(op_node) if op_node else "="
                        if op == "=":
                            self._check_assignment(expr, env)
                        else:
                            self._check_augmented_assignment(expr, env)
                    case "update_expression":
                        self._check_update(expr, env)
                    case _:
                        self._infer(expr, env)
                return env

            case "return_statement":
                self._check_return(node, env, return_type)
                return env

            case "method_declaration":
                return self._check_method_decl(node, env)

            case "for_statement":
                self._check_for_loop(node, env, return_type)
                return env

            case "enhanced_for_statement":
                self._check_enhanced_for(node, env, return_type)
                return env

            case "while_statement":
                self._check_while(node, env, return_type)
                return env

            # §14.13 do-while
            case "do_statement":
                self._check_do_while(node, env, return_type)
                return env

            case "if_statement":
                self._check_if(node, env, return_type)
                return env

            # §14.11 switch
            case "switch_statement":
                self._check_switch(node, env, return_type)
                return env

            # §14.18 throw
            case "throw_statement":
                self._check_throw(node, env)
                return env

            # §14.19 synchronized
            case "synchronized_statement":
                self._check_synchronized(node, env, return_type)
                return env

            # §14.20 try-catch-finally
            case "try_statement":
                self._check_try(node, env, return_type)
                return env

            # §14.10 assert
            case "assert_statement":
                self._check_assert(node, env)
                return env

            case "block":
                self._check_block(node, env, return_type)
                return env

            case _:
                return env

    # -----------------------------------------------------------------------
    # §5.2 / §14.x: local variable declaration
    # -----------------------------------------------------------------------

    def _check_local_decl(self, node: Node, env: Env) -> Env:
        type_node = named_children(node)[0]
        declared_type = parse_java_type(type_node)
        is_final = any(c.type == "modifiers" and "final" in text(c)
                       for c in node.children)
        mutable = not is_final

        for declarator in node.children:
            if declarator.type != "variable_declarator":
                continue
            children = named_children(declarator)
            name = text(children[0])
            rhs = children[1] if len(children) > 1 else None

            if declared_type == TOP:
                # var — infer type from rhs
                inferred = self._infer(rhs, env) if rhs else TOP
                env = env.add(name, inferred, mutable)
            else:
                if rhs:
                    rhs_type = self._infer(rhs, env)
                    # §5.2: try normal assignability first, then constant narrowing
                    if not self.assignable_with_constant(declared_type, rhs_type, rhs):
                        self.error(rhs,
                            f"Cannot assign {rhs_type!r} to {declared_type!r} "
                            f"(variable '{name}')")
                env = env.add(name, declared_type, mutable)
        return env

    # -----------------------------------------------------------------------
    # §15.26.1: simple assignment  x = expr
    # -----------------------------------------------------------------------

    def _check_assignment(self, node: Node, env: Env):
        children = named_children(node)
        lhs, rhs = children[0], children[1]
        name = text(lhs)
        if not env.is_mutable(name):
            self.error(lhs, f"Cannot assign to final variable '{name}'")
            return
        lhs_type = env.lookup(name)
        if lhs_type is None:
            self.error(lhs, f"Undeclared variable '{name}'"); return
        rhs_type = self._infer(rhs, env)
        if not self.assignable_with_constant(lhs_type, rhs_type, rhs):
            self.error(rhs,
                f"Cannot assign {rhs_type!r} to {lhs_type!r} (variable '{name}')")

    # -----------------------------------------------------------------------
    # §15.26.2: compound assignment  x op= expr
    # Semantics: v op= e  ≡  v = (T)((v) op (e))
    # The implicit cast means the result is always narrowed back to lhs type,
    # so the rhs just needs to be numeric-compatible with the underlying op.
    # -----------------------------------------------------------------------

    def _check_augmented_assignment(self, node: Node, env: Env):
        children = named_children(node)
        lhs, rhs = children[0], children[1]
        name = text(lhs)
        if not env.is_mutable(name):
            self.error(lhs, f"Cannot use compound assignment on final '{name}'"); return
        lhs_type = env.lookup(name)
        if lhs_type is None:
            self.error(lhs, f"Undeclared variable '{name}'"); return

        op_node = child_by_type(node, "+=", "-=", "*=", "/=", "%=",
                                 "&=", "|=", "^=", "<<=", ">>=", ">>>=")
        op = text(op_node) if op_node else "+="

        # String += anything is always valid (§15.18.1)
        if op == "+=" and lhs_type == STRING:
            self._infer(rhs, env)  # still evaluate rhs for its own errors
            return

        # Bitwise compound assignments require integral types
        if op in ("&=", "|=", "^=", "<<=", ">>=", ">>>="):
            if not is_integral(lhs_type) and lhs_type not in (
                ClassType("Integer"), ClassType("Long")):
                self.error(lhs, f"'{op}' requires integral lhs, got {lhs_type!r}")
            rhs_type = self._infer(rhs, env)
            if not is_numeric(rhs_type) and rhs_type not in _UNBOXED:
                self.error(rhs, f"'{op}' requires numeric rhs, got {rhs_type!r}")
            return

        # Arithmetic compound: lhs must be numeric (or boxed numeric)
        lhs_unboxed = _UNBOXED.get(lhs_type, lhs_type) if isinstance(lhs_type, ClassType) else lhs_type
        if not is_numeric(lhs_unboxed):
            self.error(lhs, f"'{op}' requires numeric lhs, got {lhs_type!r}")
        # rhs can be any numeric — result is implicitly cast to lhs type
        rhs_type = self._infer(rhs, env)
        if not is_numeric(rhs_type) and rhs_type not in _UNBOXED:
            self.error(rhs, f"'{op}' requires numeric rhs, got {rhs_type!r}")

    # -----------------------------------------------------------------------
    # §15.14/§15.15: increment / decrement  x++  ++x  x--  --x
    # -----------------------------------------------------------------------

    def _check_update(self, node: Node, env: Env):
        operand = named_children(node)[0] if named_children(node) else None
        if operand is None: return
        name = text(operand)
        typ = env.lookup(name)
        if not env.is_mutable(name):
            self.error(operand, f"Cannot increment/decrement final '{name}'")
        if typ:
            unboxed = _UNBOXED.get(typ, typ) if isinstance(typ, ClassType) else typ
            if not is_numeric(unboxed):
                self.error(operand, f"++/-- requires numeric type, got {typ!r}")

    # -----------------------------------------------------------------------
    # §14.17: return
    # -----------------------------------------------------------------------

    def _check_return(self, node: Node, env: Env, return_type: Type):
        children = named_children(node)
        if not children:
            if return_type != VOID:
                self.error(node, f"Expected return of {return_type!r}, got void")
            return
        expr_type = self._infer(children[0], env)
        if not self.assignable(return_type, expr_type):
            self.error(children[0],
                f"Return type mismatch: expected {return_type!r}, got {expr_type!r}")

    # -----------------------------------------------------------------------
    # §14.x: method declaration
    # -----------------------------------------------------------------------

    def _check_method_decl(self, node: Node, env: Env) -> Env:
        ret_node    = named_children(node)[0]
        name_node   = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        body        = node.child_by_field_name("body")

        ret_type = parse_java_type(ret_node)
        name     = text(name_node) if name_node else "_"

        inner_env = env
        param_types: list[Type] = []
        if params_node:
            for p in named_children(params_node):
                if p.type == "formal_parameter":
                    ptype_node = named_children(p)[0]
                    pname_node = named_children(p)[1]
                    ptype = parse_java_type(ptype_node)
                    pname = text(pname_node)
                    param_types.append(ptype)
                    inner_env = inner_env.add(pname, ptype, mutable=True)

        func_type = FuncType(tuple(param_types), ret_type)
        inner_env = inner_env.add(name, func_type, mutable=False)
        env       = env.add(name, func_type, mutable=False)

        if body:
            self._check_block(body, inner_env, ret_type)
        return env

    # -----------------------------------------------------------------------
    # §14.14.1: for loop
    # -----------------------------------------------------------------------

    def _check_for_loop(self, node: Node, env: Env, return_type: Type):
        init   = node.child_by_field_name("init")
        cond   = node.child_by_field_name("condition")
        update = node.child_by_field_name("update")
        body   = node.child_by_field_name("body")

        inner_env = env
        if init:
            if init.type == "local_variable_declaration":
                inner_env = self._check_local_decl(init, inner_env)
            else:
                self._check_statement(init, inner_env, VOID)

        # §14.14.1: condition must be boolean or Boolean
        if cond:
            cond_type = self._infer(cond, inner_env)
            if not self._boolean_compatible(cond_type):
                self.error(cond, f"For condition must be boolean, got {cond_type!r}")

        if update:
            for u in named_children(update):
                self._infer(u, inner_env)

        if body:
            self._check_statement(body, inner_env, return_type)

    # -----------------------------------------------------------------------
    # §14.14.2: enhanced for  for (T x : expr)
    # -----------------------------------------------------------------------

    def _check_enhanced_for(self, node: Node, env: Env, return_type: Type):
        type_node = child_by_type(node,
            "integral_type", "floating_point_type", "boolean_type",
            "type_identifier", "array_type")
        name_node = node.child_by_field_name("name")
        iter_node = node.child_by_field_name("value")
        body      = node.child_by_field_name("body")

        declared_type = parse_java_type(type_node) if type_node else TOP
        iter_type     = self._infer(iter_node, env) if iter_node else TOP

        if isinstance(iter_type, ArrayType):
            elem_type = iter_type.element
            if not self.assignable(declared_type, elem_type):
                self.error(iter_node,
                    f"Enhanced for: declared type {declared_type!r} "
                    f"incompatible with element type {elem_type!r}")
        elif iter_type != TOP and not isinstance(iter_type, ClassType):
            self.error(iter_node,
                f"Enhanced for requires array or Iterable, got {iter_type!r}")

        var_name  = text(name_node) if name_node else "_"
        inner_env = env.add(var_name, declared_type, mutable=True)
        if body:
            self._check_statement(body, inner_env, return_type)

    # -----------------------------------------------------------------------
    # §14.12: while
    # -----------------------------------------------------------------------

    def _check_while(self, node: Node, env: Env, return_type: Type):
        cond = node.child_by_field_name("condition")
        body = node.child_by_field_name("body")
        if cond:
            cond_type = self._infer(cond, env)
            if not self._boolean_compatible(cond_type):
                self.error(cond, f"While condition must be boolean, got {cond_type!r}")
        if body:
            self._check_statement(body, env, return_type)

    # -----------------------------------------------------------------------
    # §14.13: do-while
    # -----------------------------------------------------------------------

    def _check_do_while(self, node: Node, env: Env, return_type: Type):
        body = node.child_by_field_name("body")
        cond = node.child_by_field_name("condition")
        if body:
            self._check_statement(body, env, return_type)
        if cond:
            cond_type = self._infer(cond, env)
            if not self._boolean_compatible(cond_type):
                self.error(cond, f"Do-while condition must be boolean, got {cond_type!r}")

    # -----------------------------------------------------------------------
    # §14.9: if / if-else
    # -----------------------------------------------------------------------

    def _check_if(self, node: Node, env: Env, return_type: Type):
        cond  = node.child_by_field_name("condition")
        then  = node.child_by_field_name("consequence")
        else_ = node.child_by_field_name("alternative")
        if cond:
            cond_type = self._infer(cond, env)
            if not self._boolean_compatible(cond_type):
                self.error(cond, f"If condition must be boolean, got {cond_type!r}")
        if then:
            self._check_statement(then, env, return_type)
        if else_:
            self._check_statement(else_, env, return_type)

    # -----------------------------------------------------------------------
    # §14.11: switch
    # Expression must be char/byte/short/int (or boxed), String, or enum.
    # -----------------------------------------------------------------------

    def _check_switch(self, node: Node, env: Env, return_type: Type):
        # tree-sitter: switch_statement -> parenthesized_expression, switch_block
        paren = child_by_type(node, "parenthesized_expression")
        if paren:
            expr_nodes = named_children(paren)
            if expr_nodes:
                expr_type = self._infer(expr_nodes[0], env)
                if not self._switch_compatible(expr_type):
                    self.error(paren,
                        f"Switch expression must be char/byte/short/int/String/enum, "
                        f"got {expr_type!r}")

        switch_block = child_by_type(node, "switch_block")
        if switch_block:
            for child in named_children(switch_block):
                if child.type in ("switch_block_statement_group",):
                    for stmt in named_children(child):
                        if stmt.type not in ("switch_label",):
                            self._check_statement(stmt, env, return_type)

    def _switch_compatible(self, t: Type) -> bool:
        return (t in _SWITCH_TYPES or
                (isinstance(t, ClassType) and t not in
                 (ClassType("Boolean"), ClassType("Double"),
                  ClassType("Float"), ClassType("Long"))) or
                t == TOP)

    # -----------------------------------------------------------------------
    # §14.18: throw — expression must be assignable to Throwable
    # -----------------------------------------------------------------------

    def _check_throw(self, node: Node, env: Env):
        children = named_children(node)
        if not children: return
        expr_type = self._infer(children[0], env)
        if expr_type != TOP and expr_type != NULL:
            if not self.hierarchy.is_throwable(expr_type):
                self.error(children[0],
                    f"throw requires Throwable subtype, got {expr_type!r}")

    # -----------------------------------------------------------------------
    # §14.19: synchronized — expression must be reference type
    # -----------------------------------------------------------------------

    def _check_synchronized(self, node: Node, env: Env, return_type: Type):
        paren = child_by_type(node, "parenthesized_expression")
        body  = child_by_type(node, "block")
        if paren:
            expr_nodes = named_children(paren)
            if expr_nodes:
                expr_type = self._infer(expr_nodes[0], env)
                if is_primitive(expr_type):
                    self.error(paren,
                        f"synchronized requires reference type, got {expr_type!r}")
        if body:
            self._check_block(body, env, return_type)

    # -----------------------------------------------------------------------
    # §14.20: try-catch-finally
    # catch parameter must be Throwable; multi-catch types must not be subtypes
    # -----------------------------------------------------------------------

    def _check_try(self, node: Node, env: Env, return_type: Type):
        for child in node.children:
            match child.type:
                case "block":
                    self._check_block(child, env, return_type)
                case "catch_clause":
                    self._check_catch(child, env, return_type)
                case "finally_clause":
                    block = child_by_type(child, "block")
                    if block:
                        self._check_block(block, env, return_type)

    def _check_catch(self, node: Node, env: Env, return_type: Type):
        # catch_clause -> catch_formal_parameter -> catch_type, identifier
        param = child_by_type(node, "catch_formal_parameter")
        block = child_by_type(node, "block")

        inner_env = env
        if param:
            # catch_type may be a union: IOException | SQLException
            catch_type_node = child_by_type(param, "catch_type")
            name_node = child_by_type(param, "identifier")
            name = text(name_node) if name_node else "_"

            caught_types: list[Type] = []
            if catch_type_node:
                for t in named_children(catch_type_node):
                    caught = parse_java_type(t)
                    caught_types.append(caught)
                    if not self.hierarchy.is_throwable(caught) and caught != TOP:
                        self.error(t,
                            f"Catch parameter must be Throwable subtype, got {caught!r}")

            # Multi-catch: no type may be a subtype of another (§14.20)
            for i, ti in enumerate(caught_types):
                for j, tj in enumerate(caught_types):
                    if i != j and self.hierarchy.is_subtype(ti, tj):
                        self.error(param,
                            f"Multi-catch: {ti!r} is a subtype of {tj!r}")

            # Bind exception variable to union of caught types (use first for simplicity)
            exc_type = caught_types[0] if len(caught_types) == 1 else TOP
            inner_env = env.add(name, exc_type, mutable=False)  # catch param is effectively final

        if block:
            self._check_block(block, inner_env, return_type)

    # -----------------------------------------------------------------------
    # §14.10: assert
    # Condition must be boolean; detail expression must not be void
    # -----------------------------------------------------------------------

    def _check_assert(self, node: Node, env: Env):
        children = named_children(node)
        if not children: return
        cond_type = self._infer(children[0], env)
        if not self._boolean_compatible(cond_type):
            self.error(children[0],
                f"Assert condition must be boolean, got {cond_type!r}")
        if len(children) > 1:
            detail_type = self._infer(children[1], env)
            if detail_type == VOID:
                self.error(children[1], "Assert detail expression must not be void")

    # -----------------------------------------------------------------------
    # Expression inference
    # -----------------------------------------------------------------------

    def _infer(self, node: Node, env: Env) -> Type:
        match node.type:
            # §15.8.1 literals
            case "decimal_integer_literal" | "hex_integer_literal" | \
                 "octal_integer_literal"   | "binary_integer_literal":
                return LONG if text(node).endswith(("l", "L")) else INT

            case "decimal_floating_point_literal":
                return FLOAT if text(node).endswith(("f", "F")) else DOUBLE

            case "hex_floating_point_literal":
                return DOUBLE

            case "true" | "false":
                return BOOLEAN

            case "character_literal":
                return CHAR

            case "string_literal":
                return STRING

            case "null_literal":
                return NULL

            case "identifier":
                name = text(node)
                typ = env.lookup(name)
                if typ is None:
                    self.error(node, f"Undeclared variable '{name}'")
                    return EMPTY
                return typ

            case "field_access":
                return self._infer_field_access(node, env)

            case "method_invocation":
                return self._infer_method_invocation(node, env)

            case "binary_expression":
                return self._infer_binary(node, env)

            case "unary_expression":
                return self._infer_unary(node, env)

            # §15.14 postfix ++ --
            case "update_expression":
                return self._infer_update_expr(node, env)

            case "ternary_expression":
                return self._infer_ternary(node, env)

            case "cast_expression":
                return self._infer_cast(node, env)

            case "instanceof_expression":
                return self._infer_instanceof(node, env)

            case "array_access":
                return self._infer_array_access(node, env)

            case "array_creation_expression":
                return self._infer_array_creation(node, env)

            case "object_creation_expression":
                return self._infer_object_creation(node, env)

            case "parenthesized_expression":
                inner = named_children(node)
                return self._infer(inner[0], env) if inner else TOP

            # §15.8.2 class literal  String.class → Class<String>
            case "class_literal":
                return ClassType("Class")

            # §15.8.3 this
            case "this":
                return TOP  # would need enclosing class context

            case _:
                return TOP

    # -----------------------------------------------------------------------
    # Field access
    # -----------------------------------------------------------------------

    def _infer_field_access(self, node: Node, env: Env) -> Type:
        obj = node.child_by_field_name("object")
        fld = node.child_by_field_name("field")
        full = f"{text(obj)}.{text(fld)}" if obj and fld else text(node)
        typ = env.lookup(full)
        if typ is None:
            self.error(node, f"Unknown field '{full}'")
            return EMPTY
        return typ

    # -----------------------------------------------------------------------
    # §15.12: method invocation
    # -----------------------------------------------------------------------

    def _infer_method_invocation(self, node: Node, env: Env) -> Type:
        obj_node    = node.child_by_field_name("object")
        method_node = node.child_by_field_name("name")
        args_node   = node.child_by_field_name("arguments")

        full_name = (f"{text(obj_node)}.{text(method_node)}"
                     if obj_node else
                     text(method_node) if method_node else "")

        candidates = env.lookup_all(full_name)
        if not candidates:
            self.error(node, f"Unknown method '{full_name}'")
            return EMPTY

        args = ([c for c in named_children(args_node)
                 if c.type not in ("(", ")", ",")]
                if args_node else [])
        arg_types = [self._infer(a, env) for a in args]
        return self._resolve_overload(node, full_name, candidates, args, arg_types)

    def _resolve_overload(self, call_node: Node, name: str,
                          candidates: list[Type],
                          arg_nodes: list[Node],
                          arg_types: list[Type]) -> Type:
        func_types = [c for c in candidates if isinstance(c, FuncType)]
        if not func_types:
            self.error(call_node, f"'{name}' is not a method"); return EMPTY

        # §15.12.2 phase 1: strict (widening only), phase 2: loose (boxing ok)
        matches = []
        for ft in func_types:
            n = len(ft.params)
            if ft.extensible:
                fits = (len(arg_types) >= n and
                        all(self.assignable(p, a)
                            for p, a in zip(ft.params, arg_types[:n])))
            else:
                fits = (len(arg_types) == n and
                        all(self.assignable(p, a)
                            for p, a in zip(ft.params, arg_types)))
            if fits:
                matches.append(ft)

        if not matches:
            self.error(call_node,
                f"No matching overload for '{name}' "
                f"with arg types ({', '.join(repr(t) for t in arg_types)})")
            return EMPTY

        chosen = matches[0]
        for i, (anode, atype) in enumerate(zip(arg_nodes, arg_types)):
            if i < len(chosen.params) and not self.assignable(chosen.params[i], atype):
                self.error(anode,
                    f"Argument {i+1} of '{name}': "
                    f"expected {chosen.params[i]!r}, got {atype!r}")
        return chosen.return_type

    # -----------------------------------------------------------------------
    # §15.17-§15.24: binary expressions
    # -----------------------------------------------------------------------

    def _infer_binary(self, node: Node, env: Env) -> Type:
        lhs     = node.child_by_field_name("left")
        rhs     = node.child_by_field_name("right")
        op_node = child_by_type(node,
            "+", "-", "*", "/", "%",
            ">", "<", ">=", "<=", "==", "!=",
            "&&", "||", "&", "|", "^",
            "<<", ">>", ">>>")
        op = text(op_node) if op_node else "?"

        lhs_type = self._infer(lhs, env)
        rhs_type = self._infer(rhs, env)

        # §15.18.1: String + anything → String
        if op == "+" and (lhs_type == STRING or rhs_type == STRING):
            return STRING

        # §15.17/§15.18.2: arithmetic  * / % + -
        if op in ("+", "-", "*", "/", "%"):
            if not is_numeric(lhs_type):
                self.error(lhs, f"'{op}' requires numeric lhs, got {lhs_type!r}")
            if not is_numeric(rhs_type):
                self.error(rhs, f"'{op}' requires numeric rhs, got {rhs_type!r}")
            return binary_promote(lhs_type, rhs_type)

        # §15.20.1: relational
        if op in (">", "<", ">=", "<="):
            if not is_numeric(lhs_type):
                self.error(lhs, f"'{op}' requires numeric lhs, got {lhs_type!r}")
            if not is_numeric(rhs_type):
                self.error(rhs, f"'{op}' requires numeric rhs, got {rhs_type!r}")
            return BOOLEAN

        # §15.21: equality — operands must be same kind (numeric/boolean/reference)
        if op in ("==", "!="):
            lhs_num = is_numeric(lhs_type)
            rhs_num = is_numeric(rhs_type)
            lhs_bool = lhs_type == BOOLEAN or lhs_type == ClassType("Boolean")
            rhs_bool = rhs_type == BOOLEAN or rhs_type == ClassType("Boolean")
            lhs_ref  = is_reference_type(lhs_type)
            rhs_ref  = is_reference_type(rhs_type)
            if lhs_num and rhs_bool or lhs_bool and rhs_num:
                self.error(node,
                    f"'{op}' cannot mix numeric and boolean: "
                    f"{lhs_type!r} vs {rhs_type!r}")
            elif lhs_num and rhs_ref or lhs_ref and rhs_num:
                self.error(node,
                    f"'{op}' cannot mix numeric and reference: "
                    f"{lhs_type!r} vs {rhs_type!r}")
            elif lhs_bool and rhs_ref or lhs_ref and rhs_bool:
                self.error(node,
                    f"'{op}' cannot mix boolean and reference: "
                    f"{lhs_type!r} vs {rhs_type!r}")
            return BOOLEAN

        # §15.23/§15.24: short-circuit boolean ops
        if op in ("&&", "||"):
            if not self._boolean_compatible(lhs_type):
                self.error(lhs, f"'{op}' requires boolean lhs, got {lhs_type!r}")
            if not self._boolean_compatible(rhs_type):
                self.error(rhs, f"'{op}' requires boolean rhs, got {rhs_type!r}")
            return BOOLEAN

        # §15.22: bitwise/logical  & ^ |
        if op in ("&", "|", "^"):
            lhs_bool = self._boolean_compatible(lhs_type)
            rhs_bool = self._boolean_compatible(rhs_type)
            lhs_int  = is_integral(lhs_type)
            rhs_int  = is_integral(rhs_type)
            if lhs_bool and rhs_bool:
                return BOOLEAN   # §15.22.2 boolean logical
            if lhs_int and rhs_int:
                return binary_promote(lhs_type, rhs_type)  # §15.22.1 integer bitwise
            self.error(node,
                f"'{op}' requires both operands to be boolean or both integral: "
                f"{lhs_type!r} vs {rhs_type!r}")
            return TOP

        # §15.19: shift  << >> >>>
        if op in ("<<", ">>", ">>>"):
            if not is_integral(lhs_type):
                self.error(lhs, f"Shift '{op}' requires integral lhs, got {lhs_type!r}")
            if not is_integral(rhs_type):
                self.error(rhs, f"Shift '{op}' requires integral rhs, got {rhs_type!r}")
            # result type = unary promotion of left operand only
            return unary_promote(lhs_type)

        return TOP

    # -----------------------------------------------------------------------
    # §15.15: unary expressions  + - ~ !
    # -----------------------------------------------------------------------

    def _infer_unary(self, node: Node, env: Env) -> Type:
        op_node = child_by_type(node, "-", "+", "!", "~")
        operand = named_children(node)[0] if named_children(node) else None
        if operand is None: return TOP
        op = text(op_node) if op_node else "?"
        operand_type = self._infer(operand, env)

        if op in ("-", "+"):
            if not is_numeric(operand_type):
                self.error(operand, f"Unary '{op}' requires numeric, got {operand_type!r}")
            return unary_promote(operand_type)  # §5.6.1

        if op == "~":
            if not is_integral(operand_type):
                self.error(operand, f"'~' requires integral type, got {operand_type!r}")
            return unary_promote(operand_type)

        if op == "!":
            if not self._boolean_compatible(operand_type):
                self.error(operand, f"'!' requires boolean, got {operand_type!r}")
            return BOOLEAN

        return TOP

    # -----------------------------------------------------------------------
    # §15.14: postfix update expressions (when used as expression, not statement)
    # -----------------------------------------------------------------------

    def _infer_update_expr(self, node: Node, env: Env) -> Type:
        operand = named_children(node)[0] if named_children(node) else None
        if operand is None: return TOP
        t = self._infer(operand, env)
        unboxed = _UNBOXED.get(t, t) if isinstance(t, ClassType) else t
        if not is_numeric(unboxed):
            self.error(operand, f"++/-- requires numeric type, got {t!r}")
        return unary_promote(unboxed)

    # -----------------------------------------------------------------------
    # §15.25: ternary  cond ? then : else
    # -----------------------------------------------------------------------

    def _infer_ternary(self, node: Node, env: Env) -> Type:
        cond  = node.child_by_field_name("condition")
        then  = node.child_by_field_name("consequence")
        else_ = node.child_by_field_name("alternative")

        if cond:
            cond_type = self._infer(cond, env)
            if not self._boolean_compatible(cond_type):
                self.error(cond, f"Ternary condition must be boolean, got {cond_type!r}")

        then_type = self._infer(then, env)  if then  else TOP
        else_type = self._infer(else_, env) if else_ else TOP

        # §15.25.1: both boolean → boolean
        if then_type == BOOLEAN and else_type == BOOLEAN:
            return BOOLEAN
        # §15.25.2: both numeric → binary promoted
        if is_numeric(then_type) and is_numeric(else_type):
            return binary_promote(then_type, else_type)
        # §15.25.3: reference conditional
        if not (self.assignable(then_type, else_type) or
                self.assignable(else_type, then_type)):
            self.error(node,
                f"Ternary branches incompatible: {then_type!r} vs {else_type!r}")
        return then_type if self.assignable(then_type, else_type) else else_type

    # -----------------------------------------------------------------------
    # §15.16: cast  (T) expr
    # -----------------------------------------------------------------------

    def _infer_cast(self, node: Node, env: Env) -> Type:
        type_node = named_children(node)[0] if named_children(node) else None
        expr_node = named_children(node)[1] if len(named_children(node)) > 1 else None
        target = parse_java_type(type_node) if type_node else TOP
        if expr_node:
            src = self._infer(expr_node, env)
            # boolean ↔ numeric casts are always illegal
            if ((target == BOOLEAN or target == ClassType("Boolean")) and is_numeric(src)) or \
               (is_numeric(target) and (src == BOOLEAN or src == ClassType("Boolean"))):
                self.error(node, f"Invalid cast from {src!r} to {target!r}")
        return target

    # -----------------------------------------------------------------------
    # §15.20.2: instanceof  expr instanceof Type → boolean
    # Left operand must be a reference type (not any primitive)
    # -----------------------------------------------------------------------

    def _infer_instanceof(self, node: Node, env: Env) -> Type:
        expr_node = named_children(node)[0] if named_children(node) else None
        if expr_node:
            expr_type = self._infer(expr_node, env)
            # All primitives are invalid — not just numeric
            if is_primitive(expr_type):
                self.error(expr_node,
                    f"'instanceof' left operand must be reference type, "
                    f"got {expr_type!r}")
        return BOOLEAN

    # -----------------------------------------------------------------------
    # §15.10.3: array access  arr[i]
    # -----------------------------------------------------------------------

    def _infer_array_access(self, node: Node, env: Env) -> Type:
        arr_node   = node.child_by_field_name("array")
        index_node = node.child_by_field_name("index")
        arr_type   = self._infer(arr_node, env)   if arr_node   else TOP
        index_type = self._infer(index_node, env) if index_node else TOP

        if index_type != TOP and not is_integral(index_type):
            self.error(index_node,
                f"Array index must be integral type, got {index_type!r}")

        if isinstance(arr_type, ArrayType):
            return arr_type.element
        if arr_type not in (TOP, EMPTY):
            self.error(arr_node, f"Cannot index non-array type {arr_type!r}")
        return TOP

    # -----------------------------------------------------------------------
    # §15.10.1: array creation  new int[5]
    # -----------------------------------------------------------------------

    def _infer_array_creation(self, node: Node, env: Env) -> Type:
        # check dimension expressions are integral (§15.10.1)
        for child in node.children:
            if child.type == "dimensions_expr":
                for dim in named_children(child):
                    dim_type = self._infer(dim, env)
                    if not is_integral(dim_type) and dim_type != TOP:
                        self.error(dim,
                            f"Array dimension must be integral, got {dim_type!r}")
        type_node = named_children(node)[0] if named_children(node) else None
        elem_type = parse_java_type(type_node) if type_node else TOP
        return ArrayType(elem_type)

    # -----------------------------------------------------------------------
    # object creation  new Foo(args)
    # -----------------------------------------------------------------------

    def _infer_object_creation(self, node: Node, env: Env) -> Type:
        type_node = node.child_by_field_name("type")
        if type_node:
            return parse_java_type(type_node)
        return TOP


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def typecheck(source: str,
              extra_env: Optional[dict[str, tuple[Type, bool]]] = None,
              hierarchy: Optional[ClassHierarchy] = None) -> list[TypeError_]:
    h   = hierarchy or ClassHierarchy()
    env = make_default_env(h)
    if extra_env:
        for name, (typ, mut) in extra_env.items():
            env = env.add(name, typ, mut)
    checker = JavaTypeChecker(env, h)
    return checker.check_program(source)
