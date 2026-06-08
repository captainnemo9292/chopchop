"""
LSP-backed symbol resolver for the Java type checker.

Instead of a hand-written class member table, member access, method
invocation, and constructor signatures are resolved by querying a real
Java language server (eclipse.jdt.ls) through multilspy.

The contract (mirroring the workspace notebook):

  * Place the cursor at the *end position* of a symbol name.
  * Request completions there with ``allow_incomplete=True``.
  * Keep only completions whose ``completionText`` *exactly* (case
    sensitive) matches the symbol.
  * The completion ``detail`` carries the full signature, from which we
    read the receiver type, parameter types, and return type:

        field        kind 5   ``Demo.count : int``
        method       kind 2   ``String.concat(String arg0) : String``
        constructor  kind 4   ``java.util.ArrayList.ArrayList(int arg0)``

This module is intentionally free of any dependency on the checker's
``Type`` system — it deals only in raw strings.  The checker converts
those strings into ``Type`` objects (see ``java_type_from_string``).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

from multilspy.multilspy_types import Position


# CompletionItemKind values we care about (LSP spec).
KIND_METHOD = 2
KIND_FUNCTION = 3
KIND_CONSTRUCTOR = 4
KIND_FIELD = 5
KIND_VARIABLE = 6
KIND_CLASS = 7
KIND_INTERFACE = 8
KIND_ENUM_MEMBER = 20

_CALLABLE_KINDS = {KIND_METHOD, KIND_FUNCTION, KIND_CONSTRUCTOR}
_CLASS_LIKE_KINDS = {KIND_CLASS, KIND_INTERFACE, 9, 22}  # class/interface/module/struct

_TYPE_QUALIFIERS = {"final", "static", "transient", "volatile",
                    "abstract", "synchronized", "default", "public",
                    "protected", "private", "native", "strictfp"}

_CLOSE = {"{": "}", "(": ")", "[": "]"}


# ---------------------------------------------------------------------------
# detail-string parsing
# ---------------------------------------------------------------------------

def _strip_qualifiers(type_str: str) -> str:
    parts = type_str.split()
    cleaned = [p for p in parts if p.lower() not in _TYPE_QUALIFIERS]
    return " ".join(cleaned).strip() if cleaned else type_str.strip()


def _strip_generics(type_str: str) -> str:
    depth, out = 0, []
    for ch in type_str:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()


def normalize_type_str(type_str: Optional[str]) -> Optional[str]:
    """Drop modifiers and generic arguments: ``final List<String>`` -> ``List``."""
    if type_str is None:
        return None
    return _strip_generics(_strip_qualifiers(type_str)).strip() or None


def _return_type_str(detail: str) -> Optional[str]:
    """Type after the last top-level ``:`` (``a(b) : int`` -> ``int``)."""
    # Only a ':' that is outside any bracket separates the return type.
    depth = 0
    idx = -1
    for i, ch in enumerate(detail):
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        elif ch == ":" and depth == 0:
            idx = i
    if idx == -1:
        return None
    return normalize_type_str(detail[idx + 1:].strip())


def _param_type_strs(detail: str) -> tuple[list[str], bool]:
    """
    Parameter types from the first top-level ``( ... )`` group.

    Returns ``(types, is_varargs)``; param *names* are dropped, generics and
    qualifiers normalized.  ``foo(String arg0, int... rest)`` ->
    ``(["String", "int[]"], True)``.
    """
    try:
        start = detail.index("(")
    except ValueError:
        return [], False
    depth, end = 0, None
    for i in range(start, len(detail)):
        if detail[i] == "(":
            depth += 1
        elif detail[i] == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return [], False
    inner = detail[start + 1:end].strip()
    if not inner:
        return [], False

    raw_params: list[str] = []
    depth, cur = 0, ""
    for ch in inner:
        if ch in "<([":
            depth += 1
            cur += ch
        elif ch in ">)]":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            raw_params.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        raw_params.append(cur)

    types: list[str] = []
    varargs = False
    for p in raw_params:
        tokens = p.strip().split()
        # type is everything except the trailing parameter name
        type_part = " ".join(tokens[:-1]) if len(tokens) >= 2 else (tokens[0] if tokens else "")
        if "..." in type_part or "..." in p:
            varargs = True
            type_part = type_part.replace("...", "").strip() + "[]"
        norm = normalize_type_str(type_part)
        types.append(norm if norm else type_part.strip())
    return types, varargs


@dataclass(frozen=True)
class CompletionInfo:
    name: str
    kind: int
    detail: str

    @property
    def return_type_str(self) -> Optional[str]:
        return _return_type_str(self.detail)

    @property
    def param_type_strs(self) -> list[str]:
        return _param_type_strs(self.detail)[0]

    @property
    def is_varargs(self) -> bool:
        return _param_type_strs(self.detail)[1]

    @property
    def is_callable(self) -> bool:
        return self.kind in _CALLABLE_KINDS

    @property
    def is_constructor(self) -> bool:
        return self.kind == KIND_CONSTRUCTOR

    @property
    def is_class_like(self) -> bool:
        return self.kind in _CLASS_LIKE_KINDS


# ---------------------------------------------------------------------------
# resolver
# ---------------------------------------------------------------------------

class LspResolver:
    """
    Wraps an *already started* multilspy ``SyncLanguageServer`` (i.e. used
    inside ``with lsp.start_server(): with lsp.open_file(rel): ...``).

    The source being type-checked is treated as text inserted into ``rel``
    at ``(base_line, base_col)``.  Tree-sitter node positions (0-indexed
    within that source) are mapped to absolute file positions so completions
    are requested at exactly the symbol's end.
    """

    def __init__(self, lsp, relative_file_path: str,
                 base_line: int = 0, base_col: int = 0):
        self.lsp = lsp
        self.rel = relative_file_path
        self.base_line = base_line
        self.base_col = base_col
        self.source = ""

    def set_source(self, source: str) -> None:
        self.source = source

    # -- position mapping ----------------------------------------------------

    def _insert_extent(self, text: str) -> tuple[int, int]:
        lines = text.split("\n")
        if len(lines) == 1:
            return self.base_line, self.base_col + len(lines[0])
        return self.base_line + len(lines) - 1, len(lines[-1])

    def node_end_cursor(self, node) -> tuple[int, int]:
        """File (line, col) at the end of a tree-sitter node."""
        end_line, end_col = node.end_point
        if end_line == 0:
            return self.base_line, self.base_col + end_col
        return self.base_line + end_line, end_col

    @staticmethod
    def _unclosed_openers(text: str) -> list[str]:
        stack: list[str] = []
        for ch in text:
            if ch in "({[":
                stack.append(ch)
            elif ch in ")}]":
                if stack and _CLOSE[stack[-1]] == ch:
                    stack.pop()
        return stack

    # -- completions ---------------------------------------------------------

    @staticmethod
    def _mk(c) -> CompletionInfo:
        name = c.get("completionText", "") or ""
        if name.startswith("★ "):
            name = name[2:]
        return CompletionInfo(name=name, kind=c.get("kind", 0),
                              detail=c.get("detail") or "")

    def _completions_at(self, cur_line: int, cur_col: int) -> List[CompletionInfo]:
        """Insert source, query completions, balance brackets, clean up."""
        text = self.source
        end_line, end_col = self._insert_extent(text)
        self.lsp.insert_text_at_position(self.rel, self.base_line, self.base_col, text)

        results: list[CompletionInfo] = []
        try:
            raw = self.lsp.request_completions(self.rel, cur_line, cur_col,
                                               allow_incomplete=True)
            results = [self._mk(c) for c in raw]
        except Exception:
            pass

        openers = self._unclosed_openers(text)
        if openers:
            closers = "".join(_CLOSE[b] for b in reversed(openers))
            self.lsp.insert_text_at_position(self.rel, cur_line, cur_col, closers)
            try:
                raw2 = self.lsp.request_completions(self.rel, cur_line, cur_col,
                                                    allow_incomplete=True)
                seen = {(r.name, r.detail) for r in results}
                for c in raw2:
                    ci = self._mk(c)
                    if (ci.name, ci.detail) not in seen:
                        seen.add((ci.name, ci.detail))
                        results.append(ci)
            except Exception:
                pass
            self.lsp.delete_text_between_positions(
                self.rel,
                Position(line=cur_line, character=cur_col),
                Position(line=cur_line, character=cur_col + len(closers)),
            )

        self.lsp.delete_text_between_positions(
            self.rel,
            Position(line=self.base_line, character=self.base_col),
            Position(line=end_line, character=end_col),
        )
        return results

    def match(self, name: str, node) -> List[CompletionInfo]:
        """Exact case-sensitive matches for ``name`` at the node's end."""
        cur_line, cur_col = self.node_end_cursor(node)
        return [c for c in self._completions_at(cur_line, cur_col) if c.name == name]
