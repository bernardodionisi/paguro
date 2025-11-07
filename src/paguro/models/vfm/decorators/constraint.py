from __future__ import annotations

import inspect
import types

import polars as pl

from paguro.models.vfm.utils import _clean_doc, _return_is_pl_expr


class _ConstraintMethod:
    """Descriptor that behaves like a classmethod and marks 'constraint' methods."""

    def __init__(self, fn: types.FunctionType):
        if not isinstance(fn, types.FunctionType):
            raise TypeError("@constraint must decorate a plain function")
        self.fn = fn
        self.__name__ = fn.__name__

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(self, obj, owner):
        return types.MethodType(self.fn, owner)


def constraint(fn: types.FunctionType) -> _ConstraintMethod:
    """
    Mark a method as a constraint. Users write only @constraint.
    """
    return _ConstraintMethod(fn)


def _collect_constraints_from_namespace(
        ns: dict[str, object],
        *,
        owner_cls: type,
) -> tuple[dict[str, pl.Expr], dict[str, str]]:
    """
    Returns constraints defined in THIS class body.

    (constraint_exprs, constraint_docs)
    """
    constraint_exprs: dict[str, pl.Expr] = {}
    constraint_docs: dict[str, str] = {}

    for name, obj in ns.items():
        if isinstance(obj, _ConstraintMethod):
            sig = inspect.signature(obj.fn)
            if len(sig.parameters) != 1:
                raise TypeError(
                    f"{owner_cls.__name__}.{name}: "
                    f"a @constraint must accept only 'cls'."
                )
            result = getattr(owner_cls, name)()  # bound to cls
            if not isinstance(result, pl.Expr):
                raise TypeError(
                    f"{owner_cls.__name__}.{name}: "
                    f"a @constraint must return pl.Expr, got {type(result)!r}."
                )
            constraint_exprs[name] = result
            doc = _clean_doc(obj.fn)
            if doc:
                constraint_docs[name] = doc
            continue

        # Broader path: @classmethod with -> pl.Expr annotation
        if isinstance(obj, classmethod):
            fn = obj.__func__
            if isinstance(fn, types.FunctionType):
                sig = inspect.signature(fn)
                if len(sig.parameters) == 1 and _return_is_pl_expr(fn):
                    res = fn(owner_cls)
                    if not isinstance(res, pl.Expr):
                        raise TypeError(
                            f"{owner_cls.__name__}.{name}: annotated return pl.Expr but returned {type(res)!r}"
                        )
                    constraint_exprs[name] = res
                    doc = _clean_doc(fn)
                    if doc:
                        constraint_docs[name] = doc

    return constraint_exprs, constraint_docs
