"""
Concrete AST type checker for Java.

Extends the TypeScript concrete_typechecker approach with Java-specific rules:
  - Multiple numeric primitives + widening conversions
  - String + concatenation
  - null assignability to reference types
  - Subtype polymorphism via ClassHierarchy
  - Method overloading resolution
  - Enhanced for loop  (for T x : collection)
  - Cast expressions
  - Array types
  - instanceof operator
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import tree_sitter_java as tsj
from tree_sitter import Language, Parser, Node

JAVA_LANGUAGE = Language(tsj.language())
_parser = Parser(JAVA_LANGUAGE)


# ---------------------------------------------------------------------------
# Type definitions
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
class NullType(Type):    pass   # type of the literal `null`
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
    extensible: bool = False   # True for varargs
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

# Numeric types ordered for widening
_NUMERIC_PRIMITIVES = (BYTE, SHORT, INT, LONG, FLOAT, DOUBLE)

# Rule 1: Numeric widening order
# byte → short → int → long → float → double
# char → int → ...
_WIDENING: dict[Type, set[Type]] = {
    BYTE:  {SHORT, INT, LONG, FLOAT, DOUBLE},
    SHORT: {INT, LONG, FLOAT, DOUBLE},
    CHAR:  {INT, LONG, FLOAT, DOUBLE},
    INT:   {LONG, FLOAT, DOUBLE},
    LONG:  {FLOAT, DOUBLE},
    FLOAT: {DOUBLE},
}

def is_numeric(t: Type) -> bool:
    return t in _NUMERIC_PRIMITIVES or t == CHAR

def is_reference_type(t: Type) -> bool:
    return isinstance(t, (ClassType, ArrayType, StringType))

def numeric_result(a: Type, b: Type) -> Type:
    """Binary numeric promotion: wider of the two types."""
    for wide in (DOUBLE, FLOAT, LONG):
        if a == wide or b == wide:
            return wide
    return INT   # byte/short/int/char all promote to int


# ---------------------------------------------------------------------------
# Class hierarchy (subtype polymorphism)
# ---------------------------------------------------------------------------

@dataclass
class ClassHierarchy:
    """Maps class name -> set of direct parent names."""
    _parents: dict[str, set[str]] = field(default_factory=dict)

    def add(self, child: str, *parents: str) -> ClassHierarchy:
        self._parents.setdefault(child, set()).update(parents)
        return self

    def is_subtype(self, sub: Type, sup: Type) -> bool:
        """Check sub <: sup, including transitivity."""
        if sub == sup or sup == TOP:
            return True
        # null is assignable to any reference type
        if sub == NULL and is_reference_type(sup):
            return True
        # widening for numeric primitives
        if sub in _WIDENING and sup in _WIDENING.get(sub, set()):
            return True
        # class hierarchy traversal
        if isinstance(sub, ClassType) and isinstance(sup, ClassType):
            return self._class_is_subtype(sub.name, sup.name)
        # array covariance: Dog[] <: Animal[] iff Dog <: Animal
        if isinstance(sub, ArrayType) and isinstance(sup, ArrayType):
            return self.is_subtype(sub.element, sup.element)
        # string is a class type
        if sub == STRING and isinstance(sup, ClassType) and sup.name == "Object":
            return True
        return False

    def _class_is_subtype(self, sub: str, sup: str) -> bool:
        if sub == sup:
            return True
        visited = set()
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
    """Maps name -> list[(Type, is_mutable)] to support overloads."""
    _bindings: dict[str, list[tuple[Type, bool]]] = field(default_factory=dict)

    def lookup(self, name: str) -> Optional[Type]:
        """Return type if exactly one binding; None if absent."""
        entries = self._bindings.get(name, [])
        if not entries:
            return None
        if len(entries) == 1:
            return entries[0][0]
        return TOP   # overloaded — use resolve_overload instead

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
# Default environment — Java standard library subset
# ---------------------------------------------------------------------------

def make_default_env(hierarchy: ClassHierarchy) -> Env:
    env = Env()
    stdlib: dict[str, tuple[Type, bool]] = {
        # Math methods
        "Math.sqrt":  (FuncType((DOUBLE,), DOUBLE), False),
        "Math.pow":   (FuncType((DOUBLE, DOUBLE), DOUBLE), False),
        "Math.abs":   (FuncType((DOUBLE,), DOUBLE), False),
        "Math.floor": (FuncType((DOUBLE,), DOUBLE), False),
        "Math.ceil":  (FuncType((DOUBLE,), DOUBLE), False),
        "Math.round": (FuncType((DOUBLE,), LONG), False),
        "Math.min":   (FuncType((DOUBLE, DOUBLE), DOUBLE), False),
        "Math.max":   (FuncType((DOUBLE, DOUBLE), DOUBLE), False),
        "Math.PI":    (DOUBLE, False),
        # System.out
        "System.out.println": (FuncType((TOP,), VOID, extensible=False), False),
        "System.out.print":   (FuncType((TOP,), VOID, extensible=False), False),
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

def has_error(node: Node) -> bool:
    if node.type == "ERROR" or node.is_missing:
        return True
    return any(has_error(c) for c in node.children)

def parse_java_type(node: Node) -> Type:
    """Convert a tree-sitter type node to a Type."""
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
            if name == "var":    return TOP   # Java 10 type inference
            return ClassType(name)
        case "array_type":
            elem_node = named_children(node)[0]
            return ArrayType(parse_java_type(elem_node))
        case _:
            return TOP


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
        return self.hierarchy.is_subtype(source, target)

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def check_program(self, source: str) -> list[TypeError_]:
        tree = _parser.parse(bytes(source, "utf8"))
        self._check_block(tree.root_node, self.env, VOID)
        return self.errors

    # -----------------------------------------------------------------------
    # Rule 21: statement sequence — threads Γ left to right
    # -----------------------------------------------------------------------

    def _check_block(self, block: Node, env: Env, return_type: Type) -> Env:
        stmts = [c for c in block.children
                 if c.is_named and c.type not in ("comment", "line_comment", "block_comment")]
        for stmt in stmts:
            env = self._check_statement(stmt, env, return_type)
        return env

    # -----------------------------------------------------------------------
    # Statement dispatch
    # -----------------------------------------------------------------------

    def _check_statement(self, node: Node, env: Env, return_type: Type) -> Env:
        match node.type:

            # Rule 11/12: typed and var declarations
            case "local_variable_declaration":
                return self._check_local_decl(node, env)

            # Rule 13/14/15: assignment, +=, ++
            case "expression_statement":
                inner = named_children(node)
                expr = inner[0] if inner else node
                match expr.type:
                    case "assignment_expression":
                        op_node = child_by_type(expr, "=", "+=", "-=", "*=", "/=")
                        op = text(op_node) if op_node else "="
                        if op == "=":
                            self._check_assignment(expr, env)       # Rule 13
                        else:
                            self._check_augmented_assignment(expr, env)  # Rule 14
                    case "update_expression":
                        self._check_update(expr, env)               # Rule 15
                    case _:
                        self._infer(expr, env)
                return env

            # Rule 16: return
            case "return_statement":
                self._check_return(node, env, return_type)
                return env

            # Rule 17: method declaration
            case "method_declaration":
                return self._check_method_decl(node, env)

            # Rule 18: regular for loop
            case "for_statement":
                self._check_for_loop(node, env, return_type)
                return env

            # Rule 18b: enhanced for  (for T x : collection)
            case "enhanced_for_statement":
                self._check_enhanced_for(node, env, return_type)
                return env

            # Rule 19: while loop
            case "while_statement":
                self._check_while(node, env, return_type)
                return env

            # Rule 20: if / if-else
            case "if_statement":
                self._check_if(node, env, return_type)
                return env

            case "block":
                self._check_block(node, env, return_type)
                return env

            case _:
                return env

    # -----------------------------------------------------------------------
    # Rule 11/12: local variable declaration
    # int x = 5;   /   var x = 5;   /   String s = null;
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
            rhs_nodes = [c for c in children[1:]]
            rhs = rhs_nodes[0] if rhs_nodes else None

            if declared_type == TOP:
                # Rule 12: var — infer from rhs
                inferred = self._infer(rhs, env) if rhs else TOP
                env = env.add(name, inferred, mutable)
            else:
                # Rule 11: typed declaration
                if rhs:
                    rhs_type = self._infer(rhs, env)
                    if not self.assignable(declared_type, rhs_type):
                        self.error(rhs,
                            f"Cannot assign {rhs_type!r} to {declared_type!r} "
                            f"(variable '{name}')")
                env = env.add(name, declared_type, mutable)
        return env

    # -----------------------------------------------------------------------
    # Rule 13: assignment  x = expr
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
            self.error(lhs, f"Undeclared variable '{name}'")
            return
        rhs_type = self._infer(rhs, env)
        if not self.assignable(lhs_type, rhs_type):
            self.error(rhs,
                f"Cannot assign {rhs_type!r} to {lhs_type!r} (variable '{name}')")

    # -----------------------------------------------------------------------
    # Rule 14: augmented assignment  x += expr
    # -----------------------------------------------------------------------

    def _check_augmented_assignment(self, node: Node, env: Env):
        children = named_children(node)
        lhs, rhs = children[0], children[1]
        name = text(lhs)

        if not env.is_mutable(name):
            self.error(lhs, f"Cannot use compound assignment on final '{name}'")
            return
        lhs_type = env.lookup(name)
        if lhs_type is None:
            self.error(lhs, f"Undeclared variable '{name}'")
            return

        op_node = child_by_type(node, "+=", "-=", "*=", "/=")
        op = text(op_node) if op_node else "+="

        if op == "+=" and lhs_type == STRING:
            pass   # string concatenation via += is always ok
        elif not is_numeric(lhs_type):
            self.error(lhs, f"'{op}' requires numeric type, got {lhs_type!r}")
        else:
            rhs_type = self._infer(rhs, env)
            if not is_numeric(rhs_type):
                self.error(rhs, f"'{op}' requires numeric rhs, got {rhs_type!r}")

    # -----------------------------------------------------------------------
    # Rule 15: x++ / ++x
    # -----------------------------------------------------------------------

    def _check_update(self, node: Node, env: Env):
        operand = named_children(node)[0] if named_children(node) else None
        if operand is None:
            return
        name = text(operand)
        typ = env.lookup(name)
        if not env.is_mutable(name):
            self.error(operand, f"Cannot increment final variable '{name}'")
        if typ and not is_numeric(typ):
            self.error(operand, f"++ / -- requires numeric type, got {typ!r}")

    # -----------------------------------------------------------------------
    # Rule 16: return
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
    # Rule 17: method declaration
    # int add(int x, int y) { return x + y; }
    # -----------------------------------------------------------------------

    def _check_method_decl(self, node: Node, env: Env) -> Env:
        ret_node   = named_children(node)[0]
        name_node  = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        body       = node.child_by_field_name("body")

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
        # Add method to env (enables recursion); overloads accumulate
        inner_env = inner_env.add(name, func_type, mutable=False)
        env       = env.add(name, func_type, mutable=False)

        if body:
            self._check_block(body, inner_env, ret_type)

        return env

    # -----------------------------------------------------------------------
    # Rule 18: for loop
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

        if cond:
            cond_type = self._infer(cond, inner_env)
            if not self.assignable(BOOLEAN, cond_type):
                self.error(cond, f"For condition must be boolean, got {cond_type!r}")

        if update:
            for u in named_children(update):
                self._infer(u, inner_env)

        if body:
            self._check_statement(body, inner_env, return_type)

    # -----------------------------------------------------------------------
    # Rule 18b: enhanced for  for (int x : arr)
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

        # iter must be T[] or Iterable<T>; we only handle arrays for now
        if isinstance(iter_type, ArrayType):
            elem_type = iter_type.element
            if not self.assignable(declared_type, elem_type):
                self.error(iter_node,
                    f"Enhanced for: declared type {declared_type!r} "
                    f"incompatible with element type {elem_type!r}")
        elif iter_type != TOP:
            self.error(iter_node,
                f"Enhanced for requires array or Iterable, got {iter_type!r}")

        var_name  = text(name_node) if name_node else "_"
        inner_env = env.add(var_name, declared_type, mutable=True)

        if body:
            self._check_statement(body, inner_env, return_type)

    # -----------------------------------------------------------------------
    # Rule 19: while
    # -----------------------------------------------------------------------

    def _check_while(self, node: Node, env: Env, return_type: Type):
        cond = node.child_by_field_name("condition")
        body = node.child_by_field_name("body")

        if cond:
            cond_type = self._infer(cond, env)
            if not self.assignable(BOOLEAN, cond_type):
                self.error(cond, f"While condition must be boolean, got {cond_type!r}")
        if body:
            self._check_statement(body, env, return_type)

    # -----------------------------------------------------------------------
    # Rule 20: if / if-else
    # -----------------------------------------------------------------------

    def _check_if(self, node: Node, env: Env, return_type: Type):
        cond  = node.child_by_field_name("condition")
        then  = node.child_by_field_name("consequence")
        else_ = node.child_by_field_name("alternative")

        if cond:
            cond_type = self._infer(cond, env)
            if not self.assignable(BOOLEAN, cond_type):
                self.error(cond, f"If condition must be boolean, got {cond_type!r}")
        if then:
            self._check_statement(then, env, return_type)
        if else_:
            self._check_statement(else_, env, return_type)

    # -----------------------------------------------------------------------
    # Expression inference (bottom-up)
    # -----------------------------------------------------------------------

    def _infer(self, node: Node, env: Env) -> Type:
        match node.type:

            # Rule 1: numeric literals
            case "decimal_integer_literal" | "hex_integer_literal" | \
                 "octal_integer_literal" | "binary_integer_literal":
                return LONG if text(node).endswith(("l", "L")) else INT

            case "decimal_floating_point_literal":
                return FLOAT if text(node).endswith(("f", "F")) else DOUBLE

            case "hex_floating_point_literal":
                return DOUBLE

            # Rule 2: boolean literals
            case "true" | "false":
                return BOOLEAN

            # character literal  'a'
            case "character_literal":
                return CHAR

            # string literals
            case "string_literal":
                return STRING

            # null literal  (Rule 3 — null assignable to any reference type)
            case "null_literal":
                return NULL

            # Rule 3: variable reference
            case "identifier":
                name = text(node)
                typ = env.lookup(name)
                if typ is None:
                    self.error(node, f"Undeclared variable '{name}'")
                    return EMPTY
                return typ

            # member access: Math.PI, Math.sqrt, obj.field
            case "field_access":
                return self._infer_field_access(node, env)

            # Rule 4/5: method call
            case "method_invocation":
                return self._infer_method_invocation(node, env)

            # Rules 6/7/8: binary expressions
            case "binary_expression":
                return self._infer_binary(node, env)

            # Rule 9: unary minus, logical not
            case "unary_expression":
                return self._infer_unary(node, env)

            # Rule 10: ternary
            case "ternary_expression":
                return self._infer_ternary(node, env)

            # Cast expression  (int) expr
            case "cast_expression":
                return self._infer_cast(node, env)

            # instanceof  obj instanceof String → boolean
            case "instanceof_expression":
                return self._infer_instanceof(node, env)

            # Array access  arr[i]
            case "array_access":
                return self._infer_array_access(node, env)

            # Array creation  new int[5]
            case "array_creation_expression":
                return self._infer_array_creation(node, env)

            # Parenthesized
            case "parenthesized_expression":
                inner = named_children(node)
                return self._infer(inner[0], env) if inner else TOP

            case _:
                return TOP

    # -----------------------------------------------------------------------
    # Field access  Math.PI / System.out.println
    # -----------------------------------------------------------------------

    def _infer_field_access(self, node: Node, env: Env) -> Type:
        obj  = node.child_by_field_name("object")
        fld  = node.child_by_field_name("field")
        full = f"{text(obj)}.{text(fld)}" if obj and fld else text(node)
        typ  = env.lookup(full)
        if typ is None:
            self.error(node, f"Unknown field '{full}'")
            return EMPTY
        return typ

    # -----------------------------------------------------------------------
    # Rule 4/5: method invocation
    # Handles: sqrt(x), Math.sqrt(x), obj.method(x)
    # -----------------------------------------------------------------------

    def _infer_method_invocation(self, node: Node, env: Env) -> Type:
        # Resolve method name
        obj_node    = node.child_by_field_name("object")
        method_node = node.child_by_field_name("name")
        args_node   = node.child_by_field_name("arguments")

        if obj_node:
            full_name = f"{text(obj_node)}.{text(method_node)}"
        else:
            full_name = text(method_node) if method_node else ""

        candidates = env.lookup_all(full_name)
        if not candidates:
            self.error(node, f"Unknown method '{full_name}'")
            return EMPTY

        # Collect argument types
        args = [c for c in named_children(args_node)
                if args_node and c.type not in ("(", ")", ",")] if args_node else []
        arg_types = [self._infer(a, env) for a in args]

        # Rule 6 (overloading): pick best matching overload
        return self._resolve_overload(node, full_name, candidates, args, arg_types)

    def _resolve_overload(
        self,
        call_node: Node,
        name: str,
        candidates: list[Type],
        arg_nodes: list[Node],
        arg_types: list[Type],
    ) -> Type:
        func_types = [c for c in candidates if isinstance(c, FuncType)]
        if not func_types:
            self.error(call_node, f"'{name}' is not a method")
            return EMPTY

        matches = []
        for ft in func_types:
            if ft.extensible:
                fits = len(arg_types) >= len(ft.params) and all(
                    self.assignable(p, a)
                    for p, a in zip(ft.params, arg_types[:len(ft.params)])
                )
            else:
                fits = len(arg_types) == len(ft.params) and all(
                    self.assignable(p, a) for p, a in zip(ft.params, arg_types)
                )
            if fits:
                matches.append(ft)

        if not matches:
            self.error(call_node,
                f"No matching overload for '{name}' "
                f"with arg types ({', '.join(repr(t) for t in arg_types)})")
            return EMPTY

        # Most specific: fewest widening conversions — pick first match for now
        chosen = matches[0]

        # Report individual arg errors for the chosen overload (best effort)
        for i, (arg_node, arg_type) in enumerate(zip(arg_nodes, arg_types)):
            if i < len(chosen.params):
                if not self.assignable(chosen.params[i], arg_type):
                    self.error(arg_node,
                        f"Argument {i+1} of '{name}': "
                        f"expected {chosen.params[i]!r}, got {arg_type!r}")

        return chosen.return_type

    # -----------------------------------------------------------------------
    # Rules 6/7/8: binary expression
    # -----------------------------------------------------------------------

    def _infer_binary(self, node: Node, env: Env) -> Type:
        lhs      = node.child_by_field_name("left")
        rhs      = node.child_by_field_name("right")
        op_node  = child_by_type(node,
            "+", "-", "*", "/", "%",
            ">", "<", ">=", "<=", "==", "!=",
            "&&", "||", "&", "|", "^",
            "<<", ">>", ">>>")
        op = text(op_node) if op_node else "?"

        lhs_type = self._infer(lhs, env)
        rhs_type = self._infer(rhs, env)

        # Rule 2 (String): String + anything → String
        if op == "+" and (lhs_type == STRING or rhs_type == STRING):
            return STRING

        # Rule 6: arithmetic
        if op in ("+", "-", "*", "/", "%"):
            if not is_numeric(lhs_type):
                self.error(lhs, f"'{op}' requires numeric lhs, got {lhs_type!r}")
            if not is_numeric(rhs_type):
                self.error(rhs, f"'{op}' requires numeric rhs, got {rhs_type!r}")
            return numeric_result(lhs_type, rhs_type)

        # Rule 7: comparison → boolean
        if op in (">", "<", ">=", "<="):
            if not is_numeric(lhs_type):
                self.error(lhs, f"'{op}' requires numeric lhs, got {lhs_type!r}")
            if not is_numeric(rhs_type):
                self.error(rhs, f"'{op}' requires numeric rhs, got {rhs_type!r}")
            return BOOLEAN

        if op in ("==", "!="):
            return BOOLEAN   # works for any types (reference equality)

        # Rule 8: boolean ops
        if op in ("&&", "||"):
            if not self.assignable(BOOLEAN, lhs_type):
                self.error(lhs, f"'{op}' requires boolean lhs, got {lhs_type!r}")
            if not self.assignable(BOOLEAN, rhs_type):
                self.error(rhs, f"'{op}' requires boolean rhs, got {rhs_type!r}")
            return BOOLEAN

        # Bitwise ops: require integral types
        if op in ("&", "|", "^", "<<", ">>", ">>>"):
            if not is_numeric(lhs_type):
                self.error(lhs, f"Bitwise '{op}' requires integral lhs, got {lhs_type!r}")
            return numeric_result(lhs_type, rhs_type)

        return TOP

    # -----------------------------------------------------------------------
    # Rule 9: unary expression  -x  !b  ~n
    # -----------------------------------------------------------------------

    def _infer_unary(self, node: Node, env: Env) -> Type:
        op_node = child_by_type(node, "-", "+", "!", "~")
        operand = named_children(node)[0] if named_children(node) else None
        if operand is None:
            return TOP
        op = text(op_node) if op_node else "?"
        operand_type = self._infer(operand, env)

        if op in ("-", "+", "~"):
            if not is_numeric(operand_type):
                self.error(operand, f"Unary '{op}' requires numeric, got {operand_type!r}")
            return INT if operand_type in (BYTE, SHORT, CHAR) else operand_type

        if op == "!":
            if not self.assignable(BOOLEAN, operand_type):
                self.error(operand, f"'!' requires boolean, got {operand_type!r}")
            return BOOLEAN

        return TOP

    # -----------------------------------------------------------------------
    # Rule 10: ternary  cond ? then : else
    # -----------------------------------------------------------------------

    def _infer_ternary(self, node: Node, env: Env) -> Type:
        cond  = node.child_by_field_name("condition")
        then  = node.child_by_field_name("consequence")
        else_ = node.child_by_field_name("alternative")

        if cond:
            cond_type = self._infer(cond, env)
            if not self.assignable(BOOLEAN, cond_type):
                self.error(cond, f"Ternary condition must be boolean, got {cond_type!r}")

        then_type = self._infer(then, env) if then else TOP
        else_type = self._infer(else_, env) if else_ else TOP

        # Both branches must be assignable to a common type
        if not (self.assignable(then_type, else_type) or
                self.assignable(else_type, then_type)):
            self.error(node,
                f"Ternary branch types incompatible: "
                f"{then_type!r} vs {else_type!r}")

        # Return the wider of the two types
        if self.assignable(then_type, else_type):
            return then_type
        return else_type

    # -----------------------------------------------------------------------
    # Cast expression  (int) expr
    # -----------------------------------------------------------------------

    def _infer_cast(self, node: Node, env: Env) -> Type:
        type_node = named_children(node)[0] if named_children(node) else None
        expr_node = named_children(node)[1] if len(named_children(node)) > 1 else None

        target = parse_java_type(type_node) if type_node else TOP
        if expr_node:
            src = self._infer(expr_node, env)
            # Warn on clearly invalid casts (boolean ↔ numeric)
            if (self.assignable(BOOLEAN, target) and is_numeric(src)) or \
               (is_numeric(target) and self.assignable(BOOLEAN, src)):
                self.error(node, f"Invalid cast from {src!r} to {target!r}")
        return target

    # -----------------------------------------------------------------------
    # instanceof  obj instanceof ClassName → boolean
    # -----------------------------------------------------------------------

    def _infer_instanceof(self, node: Node, env: Env) -> Type:
        expr_node = named_children(node)[0] if named_children(node) else None
        if expr_node:
            expr_type = self._infer(expr_node, env)
            if is_numeric(expr_type) or expr_type == BOOLEAN:
                self.error(node,
                    f"'instanceof' requires a reference type, got {expr_type!r}")
        return BOOLEAN

    # -----------------------------------------------------------------------
    # Array access  arr[i]
    # -----------------------------------------------------------------------

    def _infer_array_access(self, node: Node, env: Env) -> Type:
        arr_node   = node.child_by_field_name("array")
        index_node = node.child_by_field_name("index")

        arr_type   = self._infer(arr_node, env) if arr_node else TOP
        index_type = self._infer(index_node, env) if index_node else TOP

        if not is_numeric(index_type) and index_type != TOP:
            self.error(index_node, f"Array index must be int, got {index_type!r}")

        if isinstance(arr_type, ArrayType):
            return arr_type.element
        if arr_type != TOP:
            self.error(arr_node, f"Cannot index non-array type {arr_type!r}")
        return TOP

    # -----------------------------------------------------------------------
    # Array creation  new int[5]
    # -----------------------------------------------------------------------

    def _infer_array_creation(self, node: Node, env: Env) -> Type:
        type_node = named_children(node)[0] if named_children(node) else None
        elem_type = parse_java_type(type_node) if type_node else TOP
        return ArrayType(elem_type)


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
