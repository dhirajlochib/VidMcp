"""Safe expression evaluator for run_ops conditions and recipe `when` clauses.

Supports: literals, dotted names resolved in a context dict, comparisons,
and/or/not, + - * /, `in`, unary minus. No calls, no attributes on objects,
no subscripts, no comprehensions.
"""

from __future__ import annotations

import ast
import operator as op
from typing import Any

_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod}
_CMP = {
    ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le,
    ast.Gt: op.gt, ast.GtE: op.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}


class ExprError(ValueError):
    pass


def _resolve(path: str, ctx: dict[str, Any]) -> Any:
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _eval(node: ast.expr, ctx: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = _dotted_name(node)
        if name is None:
            raise ExprError("unsupported attribute expression")
        if name in ("true", "True"):
            return True
        if name in ("false", "False"):
            return False
        if name in ("none", "None", "null"):
            return None
        return _resolve(name, ctx)
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, ctx)
        if isinstance(node.op, ast.Not):
            return not v
        if isinstance(node.op, ast.USub):
            return -v
        raise ExprError("unsupported unary op")
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, ctx) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        return _BIN[type(node.op)](_eval(node.left, ctx), _eval(node.right, ctx))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, ctx)
        for cmp_op, right_node in zip(node.ops, node.comparators):
            right = _eval(right_node, ctx)
            fn = _CMP.get(type(cmp_op))
            if fn is None:
                raise ExprError("unsupported comparison")
            try:
                if left is None or right is None:
                    # None comparisons: only ==/!= meaningful, others false
                    if isinstance(cmp_op, (ast.Eq, ast.NotEq)):
                        if not fn(left, right):
                            return False
                    else:
                        return False
                elif not fn(left, right):
                    return False
            except TypeError:
                return False
            left = right
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, ctx) for e in node.elts]
    raise ExprError(f"unsupported expression node: {type(node).__name__}")


def evaluate(expr: str, ctx: dict[str, Any] | None = None) -> Any:
    """Evaluate a restricted expression against a context dict. Missing names → None."""
    if not expr or not str(expr).strip():
        return True
    tree = ast.parse(str(expr), mode="eval")
    return _eval(tree.body, ctx or {})
