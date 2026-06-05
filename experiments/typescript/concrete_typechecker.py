"""
Concrete AST type checker for TypeScript.

Mirrors typescript_typechecker.py but operates on concrete tree-sitter AST nodes
instead of the symbolic TreeGrammar program space.

Each function either:
  - infer_type(node, env) -> Type        (bottom-up: what type does this produce?)
  - check_type(node, env, target) -> bool (top-down: does this satisfy the target?)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser, Node

from .types import (
    Type, NUMBERTYPE, BOOLEANTYPE, STRINGTYPE, VOIDTYPE,
    FuncType, ProdType, UnionType, TopType, EmptyType, contains
)

TS_LANGUAGE = Language(tsts.language_typescript())
_parser = Parser(TS_LANGUAGE)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class Env:
    """Simple map from name -> (Type, is_mutable)."""
    _bindings: dict[str, tuple[Type, bool]] = field(default_factory=dict)

    def lookup(self, name: str) -> Optional[Type]:
        entry = self._bindings.get(name)
        return entry[0] if entry else None

    def is_mutable(self, name: str) -> bool:
        entry = self._bindings.get(name)
        return entry[1] if entry else False

    def add(self, name: str, typ: Type, mutable: bool = True) -> Env:
        new = Env(dict(self._bindings))
        new._bindings[name] = (typ, mutable)
        return new

    def extend(self, bindings: dict[str, tuple[Type, bool]]) -> Env:
        new = Env(dict(self._bindings))
        new._bindings.update(bindings)
        return new


# Default global environment (mirrors typescript_typechecker.py default_env)
def make_default_env() -> Env:
    env = Env()
    for name, typ in {
        "Math.PI":    NUMBERTYPE,
        "Math.pow":   FuncType.of(ProdType.of(NUMBERTYPE, NUMBERTYPE), NUMBERTYPE),
        "Math.log2":  FuncType.of(ProdType.of(NUMBERTYPE), NUMBERTYPE),
        "Math.sqrt":  FuncType.of(ProdType.of(NUMBERTYPE), NUMBERTYPE),
        "Math.floor": FuncType.of(ProdType.of(NUMBERTYPE), NUMBERTYPE),
        "Math.round": FuncType.of(ProdType.of(NUMBERTYPE), NUMBERTYPE),
        "Math.ceil":  FuncType.of(ProdType.of(NUMBERTYPE), NUMBERTYPE),
        "Math.clz32": FuncType.of(ProdType.of(NUMBERTYPE), NUMBERTYPE),
        "Math.min":   FuncType.of(ProdType.of(NUMBERTYPE, extensible=True), NUMBERTYPE),
        "Math.max":   FuncType.of(ProdType.of(NUMBERTYPE, extensible=True), NUMBERTYPE),
    }.items():
        env = env.add(name, typ, mutable=False)
    return env


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text(node: Node) -> str:
    return node.text.decode()

def child_by_type(node: Node, *types: str) -> Optional[Node]:
    for c in node.children:
        if c.type in types:
            return c
    return None

def named_children(node: Node) -> list[Node]:
    return [c for c in node.children if c.is_named]

def has_error(node: Node) -> bool:
    if node.type == "ERROR" or node.is_missing:
        return True
    return any(has_error(c) for c in node.children)

def parse_type_annotation(node: Node) -> Type:
    """Convert a tree-sitter type_annotation node to a Type."""
    # unwrap ':' wrapper if present
    inner = node
    if node.type == "type_annotation":
        inner = named_children(node)[0]
    match inner.type:
        case "predefined_type":
            match text(inner):
                case "number":  return NUMBERTYPE
                case "boolean": return BOOLEANTYPE
                case "string":  return STRINGTYPE
                case _:         return TopType()
        case _:
            return TopType()


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------

@dataclass
class TypeError_:
    node: Node
    message: str
    def __str__(self):
        line = self.node.start_point[0] + 1
        return f"Line {line}: {self.message}"


class TypeChecker:
    def __init__(self, env: Env):
        self.env = env
        self.errors: list[TypeError_] = []

    def error(self, node: Node, msg: str):
        self.errors.append(TypeError_(node, msg))

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def check_program(self, source: str) -> list[TypeError_]:
        tree = _parser.parse(bytes(source, "utf8"))
        self._check_block(tree.root_node, self.env, VOIDTYPE)
        return self.errors

    # -----------------------------------------------------------------------
    # Rule 21: Statement sequence
    # Mirrors typeprune_return_seqs / CommandSeq
    # Each complete statement updates env for subsequent ones.
    # -----------------------------------------------------------------------

    def _check_block(self, block: Node, env: Env, return_type: Type) -> Env:
        stmts = [c for c in block.children if c.is_named and c.type != "comment"]
        for stmt in stmts:
            env = self._check_statement(stmt, env, return_type)
        return env

    # -----------------------------------------------------------------------
    # Statement dispatch
    # -----------------------------------------------------------------------

    def _check_statement(self, node: Node, env: Env, return_type: Type) -> Env:
        match node.type:
            # Rule 11: typed declaration  (const x: number = ...)
            case "lexical_declaration":
                return self._check_lexical_decl(node, env, return_type)

            # Rule 13: assignment  (x = ...)
            case "expression_statement":
                inner = named_children(node)[0] if named_children(node) else node
                if inner.type == "assignment_expression":
                    self._check_assignment(inner, env)
                elif inner.type == "augmented_assignment_expression":
                    # Rule 14: += assignment
                    self._check_augmented_assignment(inner, env)
                elif inner.type in ("update_expression",):
                    # Rule 15: x++ / ++x
                    self._check_update(inner, env)
                else:
                    # Rule: expression statement — any type ok
                    self._infer(inner, env)
                return env

            # Rule 16: return statement
            case "return_statement":
                self._check_return(node, env, return_type)
                return env

            # Rule 17: function declaration
            case "function_declaration":
                return self._check_func_decl(node, env)

            # Rule 18: for loop
            case "for_statement":
                self._check_for_loop(node, env, return_type)
                return env

            # Rule 19: while loop
            case "while_statement":
                self._check_while(node, env, return_type)
                return env

            # Rule 20: if/else
            case "if_statement":
                self._check_if(node, env, return_type)
                return env

            case "statement_block":
                self._check_block(node, env, return_type)
                return env

            case _:
                return env  # unknown statement — skip

    # -----------------------------------------------------------------------
    # Rule 11 & 12: variable declarations
    # const x: number = expr   /   const x = expr
    # -----------------------------------------------------------------------

    def _check_lexical_decl(self, node: Node, env: Env, return_type: Type) -> Env:
        mutable = text(node.children[0]) == "let"
        for declarator in node.children:
            if declarator.type != "variable_declarator":
                continue
            name_node = named_children(declarator)[0]
            name = text(name_node)

            # Find type annotation if present
            type_ann = child_by_type(declarator, "type_annotation")
            rhs_nodes = [c for c in named_children(declarator)
                         if c != name_node and c.type != "type_annotation"]
            rhs = rhs_nodes[0] if rhs_nodes else None

            if type_ann:
                # Rule 11: typed declaration
                declared_type = parse_type_annotation(type_ann)
                if rhs:
                    rhs_type = self._infer(rhs, env)
                    if not contains(declared_type, rhs_type):
                        self.error(rhs,
                            f"Cannot assign {rhs_type} to {declared_type} "
                            f"(variable '{name}')")
                env = env.add(name, declared_type, mutable)
            else:
                # Rule 12: untyped declaration — infer type from rhs
                if rhs:
                    inferred = self._infer(rhs, env)
                    env = env.add(name, inferred, mutable)
                else:
                    env = env.add(name, TopType(), mutable)
        return env

    # -----------------------------------------------------------------------
    # Rule 13: assignment  x = expr
    # -----------------------------------------------------------------------

    def _check_assignment(self, node: Node, env: Env):
        lhs = named_children(node)[0]
        rhs = named_children(node)[1]
        lhs_name = text(lhs)

        if not env.is_mutable(lhs_name):
            self.error(lhs, f"Cannot assign to const '{lhs_name}'")
            return

        lhs_type = env.lookup(lhs_name)
        if lhs_type is None:
            self.error(lhs, f"Undeclared variable '{lhs_name}'")
            return

        rhs_type = self._infer(rhs, env)
        if not contains(lhs_type, rhs_type):
            self.error(rhs,
                f"Cannot assign {rhs_type} to {lhs_type} (variable '{lhs_name}')")

    # -----------------------------------------------------------------------
    # Rule 14: augmented assignment  x += expr
    # -----------------------------------------------------------------------

    def _check_augmented_assignment(self, node: Node, env: Env):
        lhs = named_children(node)[0]
        rhs = named_children(node)[1]
        lhs_name = text(lhs)

        if not env.is_mutable(lhs_name):
            self.error(lhs, f"Cannot use += on const '{lhs_name}'")
            return

        lhs_type = env.lookup(lhs_name)
        if not contains(NUMBERTYPE, lhs_type):
            self.error(lhs, f"+= requires number, but '{lhs_name}' is {lhs_type}")

        rhs_type = self._infer(rhs, env)
        if not contains(NUMBERTYPE, rhs_type):
            self.error(rhs, f"+= requires number rhs, got {rhs_type}")

    # -----------------------------------------------------------------------
    # Rule 15: x++ / ++x
    # -----------------------------------------------------------------------

    def _check_update(self, node: Node, env: Env):
        operand = named_children(node)[0]
        name = text(operand)
        typ = env.lookup(name)
        if not env.is_mutable(name):
            self.error(operand, f"Cannot increment const '{name}'")
        if typ and not contains(NUMBERTYPE, typ):
            self.error(operand, f"++ requires number, got {typ}")

    # -----------------------------------------------------------------------
    # Rule 16: return statement
    # -----------------------------------------------------------------------

    def _check_return(self, node: Node, env: Env, return_type: Type):
        children = named_children(node)
        if not children:
            # return; — void return
            if not contains(return_type, VOIDTYPE):
                self.error(node, f"Expected return type {return_type}, got void")
            return
        expr = children[0]
        expr_type = self._infer(expr, env)
        if not contains(return_type, expr_type):
            self.error(expr,
                f"Return type mismatch: expected {return_type}, got {expr_type}")

    # -----------------------------------------------------------------------
    # Rule 17: function declaration
    # function f(x: number, y: number): number { ... }
    # -----------------------------------------------------------------------

    def _check_func_decl(self, node: Node, env: Env) -> Env:
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        ret_ann = node.child_by_field_name("return_type")
        body = node.child_by_field_name("body")

        name = text(name_node) if name_node else None
        ret_type = parse_type_annotation(ret_ann) if ret_ann else VOIDTYPE

        # Build param types and inner env
        inner_env = env
        param_types = []
        if params_node:
            for p in named_children(params_node):
                if p.type == "required_parameter":
                    pname_node = named_children(p)[0]
                    ptype_ann = child_by_type(p, "type_annotation")
                    pname = text(pname_node)
                    ptype = parse_type_annotation(ptype_ann) if ptype_ann else TopType()
                    param_types.append(ptype)
                    inner_env = inner_env.add(pname, ptype, mutable=True)

        # Build the function's own type and add to env (enables recursion)
        func_type = FuncType.of(ProdType.of(*param_types), ret_type)
        if name:
            inner_env = inner_env.add(name, func_type, mutable=False)
            env = env.add(name, func_type, mutable=False)

        # Check the body against the declared return type
        if body:
            self._check_block(body, inner_env, ret_type)

        return env

    # -----------------------------------------------------------------------
    # Rule 18: for loop
    # for (let i: number = 0; i < 10; i++) { ... }
    # -----------------------------------------------------------------------

    def _check_for_loop(self, node: Node, env: Env, return_type: Type):
        init = node.child_by_field_name("initializer")
        cond = node.child_by_field_name("condition")
        update = node.child_by_field_name("increment")
        body = node.child_by_field_name("body")

        inner_env = env
        if init and init.type == "lexical_declaration":
            inner_env = self._check_lexical_decl(init, inner_env, VOIDTYPE)
        elif init:
            self._check_statement(init, inner_env, VOIDTYPE)

        # Rule 18: condition must be boolean
        if cond:
            cond_type = self._infer(cond, inner_env)
            if not contains(BOOLEANTYPE, cond_type):
                self.error(cond, f"For condition must be boolean, got {cond_type}")

        if update:
            self._check_statement(update, inner_env, VOIDTYPE)

        if body:
            self._check_block(body, inner_env, return_type)

    # -----------------------------------------------------------------------
    # Rule 19: while loop
    # -----------------------------------------------------------------------

    def _check_while(self, node: Node, env: Env, return_type: Type):
        cond = node.child_by_field_name("condition")
        body = node.child_by_field_name("body")

        if cond:
            cond_type = self._infer(cond, env)
            if not contains(BOOLEANTYPE, cond_type):
                self.error(cond, f"While condition must be boolean, got {cond_type}")

        if body:
            self._check_block(body, env, return_type)

    # -----------------------------------------------------------------------
    # Rule 20: if / if-else
    # -----------------------------------------------------------------------

    def _check_if(self, node: Node, env: Env, return_type: Type):
        cond = node.child_by_field_name("condition")
        then = node.child_by_field_name("consequence")
        else_ = node.child_by_field_name("alternative")

        # Condition must be boolean
        if cond:
            cond_type = self._infer(cond, env)
            if not contains(BOOLEANTYPE, cond_type):
                self.error(cond, f"If condition must be boolean, got {cond_type}")

        if then:
            self._check_statement(then, env, return_type)
        if else_:
            self._check_statement(else_, env, return_type)

    # -----------------------------------------------------------------------
    # Type inference for expressions (bottom-up)
    # -----------------------------------------------------------------------

    def _infer(self, node: Node, env: Env) -> Type:
        match node.type:
            # Rule 1: integer literal
            case "number":
                return NUMBERTYPE

            # Rule 2: boolean literal
            case "true" | "false":
                return BOOLEANTYPE

            # string literal
            case "string":
                return STRINGTYPE

            # Rule 3: variable reference
            case "identifier":
                name = text(node)
                typ = env.lookup(name)
                if typ is None:
                    self.error(node, f"Undeclared variable '{name}'")
                    return EmptyType()
                return typ

            # Member expression: Math.sqrt, x.toString, etc.
            case "member_expression":
                obj = node.child_by_field_name("object")
                prop = node.child_by_field_name("property")
                full_name = f"{text(obj)}.{text(prop)}"
                typ = env.lookup(full_name)
                if typ is None:
                    self.error(node, f"Unknown member '{full_name}'")
                    return EmptyType()
                return typ

            # Rule 4 & 5: function call
            case "call_expression":
                return self._infer_call(node, env)

            # Rules 6, 7, 8: binary expressions
            case "binary_expression":
                return self._infer_binary(node, env)

            # Rule 9: unary minus
            case "unary_expression":
                return self._infer_unary(node, env)

            # Rule 10: ternary expression
            case "ternary_expression":
                return self._infer_ternary(node, env)

            # Parenthesized expression — propagate through
            case "parenthesized_expression":
                inner = named_children(node)[0]
                return self._infer(inner, env)

            case _:
                return TopType()

    # -----------------------------------------------------------------------
    # Rule 4 & 5: function / method call
    # -----------------------------------------------------------------------

    def _infer_call(self, node: Node, env: Env) -> Type:
        func = node.child_by_field_name("function")
        args_node = node.child_by_field_name("arguments")
        args = [c for c in named_children(args_node)
                if c.type not in ("(", ")", ",")] if args_node else []

        func_type = self._infer(func, env)

        if not isinstance(func_type, FuncType):
            self.error(func, f"'{text(func)}' is not a function (type: {func_type})")
            return EmptyType()

        # Rule 5: check each argument against the declared parameter types
        param_types = func_type.params.types if isinstance(func_type.params, ProdType) else ()
        for i, arg in enumerate(args):
            arg_type = self._infer(arg, env)
            if i < len(param_types):
                expected = param_types[i]
                if not contains(expected, arg_type):
                    self.error(arg,
                        f"Argument {i+1} of '{text(func)}': "
                        f"expected {expected}, got {arg_type}")
            elif not func_type.params.extensible:
                self.error(arg,
                    f"Too many arguments to '{text(func)}'")

        return func_type.return_type

    # -----------------------------------------------------------------------
    # Rules 6, 7, 8: binary expressions
    # -----------------------------------------------------------------------

    def _infer_binary(self, node: Node, env: Env) -> Type:
        lhs = node.child_by_field_name("left")
        op_node = child_by_type(node, "+", "-", "*", "/", "%",
                                 ">", "<", ">=", "<=", "==", "!=", "===", "!==",
                                 "&&", "||")
        rhs = node.child_by_field_name("right")

        op = text(op_node) if op_node else "?"
        lhs_type = self._infer(lhs, env)
        rhs_type = self._infer(rhs, env)

        # Rule 6: arithmetic  + - * / %  → both number, result number
        if op in ("+", "-", "*", "/", "%"):
            if not contains(NUMBERTYPE, lhs_type):
                self.error(lhs, f"Arithmetic op '{op}' requires number lhs, got {lhs_type}")
            if not contains(NUMBERTYPE, rhs_type):
                self.error(rhs, f"Arithmetic op '{op}' requires number rhs, got {rhs_type}")
            return NUMBERTYPE

        # Rule 7: comparison  > < >= <= == != === !==  → both number, result boolean
        if op in (">", "<", ">=", "<=", "==", "!=", "===", "!=="):
            if not contains(NUMBERTYPE, lhs_type):
                self.error(lhs, f"Comparison '{op}' requires number lhs, got {lhs_type}")
            if not contains(NUMBERTYPE, rhs_type):
                self.error(rhs, f"Comparison '{op}' requires number rhs, got {rhs_type}")
            return BOOLEANTYPE

        # Rule 8: boolean ops  && ||  → both boolean, result boolean
        if op in ("&&", "||"):
            if not contains(BOOLEANTYPE, lhs_type):
                self.error(lhs, f"Boolean op '{op}' requires boolean lhs, got {lhs_type}")
            if not contains(BOOLEANTYPE, rhs_type):
                self.error(rhs, f"Boolean op '{op}' requires boolean rhs, got {rhs_type}")
            return BOOLEANTYPE

        return TopType()

    # -----------------------------------------------------------------------
    # Rule 9: unary minus
    # -----------------------------------------------------------------------

    def _infer_unary(self, node: Node, env: Env) -> Type:
        op_node = child_by_type(node, "-", "!")
        operand = named_children(node)[0] if named_children(node) else None
        if operand is None:
            return EmptyType()

        op = text(op_node) if op_node else "?"
        operand_type = self._infer(operand, env)

        if op == "-":
            if not contains(NUMBERTYPE, operand_type):
                self.error(operand, f"Unary minus requires number, got {operand_type}")
            return NUMBERTYPE

        if op == "!":
            if not contains(BOOLEANTYPE, operand_type):
                self.error(operand, f"'!' requires boolean, got {operand_type}")
            return BOOLEANTYPE

        return TopType()

    # -----------------------------------------------------------------------
    # Rule 10: ternary expression  cond ? then : else
    # -----------------------------------------------------------------------

    def _infer_ternary(self, node: Node, env: Env) -> Type:
        cond = node.child_by_field_name("condition")
        then = node.child_by_field_name("consequence")
        else_ = node.child_by_field_name("alternative")

        # Condition must be boolean
        cond_type = self._infer(cond, env)
        if not contains(BOOLEANTYPE, cond_type):
            self.error(cond, f"Ternary condition must be boolean, got {cond_type}")

        then_type = self._infer(then, env)
        else_type = self._infer(else_, env)

        # Both branches must have the same type
        if then_type != else_type:
            self.error(else_,
                f"Ternary branches must have same type: "
                f"then={then_type}, else={else_type}")

        return then_type


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def typecheck(source: str) -> list[TypeError_]:
    checker = TypeChecker(make_default_env())
    return checker.check_program(source)
