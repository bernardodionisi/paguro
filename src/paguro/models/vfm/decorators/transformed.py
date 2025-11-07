from __future__ import annotations

import copy
import types
import typing
from typing import TYPE_CHECKING
import polars as pl

from paguro.defer import LazyFrameExpr
from paguro.models.vfm.utils import _clean_doc
from paguro.validation.valid_frame.valid_frame import ValidFrame

if TYPE_CHECKING:
    from paguro.models.vfm.vfmodel import VFrameModel
    from paguro.typing import IntoValidators


class _TransformedMethod:
    """
    Descriptor that behaves like a classmethod and carries:
      - the target model (VFrameModel subclass) whose _valid_frame will be deep-copied
        and annotated with a _transform (DeferredLazyFrame), and
      - optional validator expressions to attach to a *local* ValidFrame that
        also carries the same _transform and gets appended to this class's frames.
    """

    def __init__(
            self,
            fn: types.FunctionType,
            model: type[VFrameModel],
            validators: tuple[IntoValidators | typing.Collection[IntoValidators], ...],
    ) -> None:

        if not isinstance(fn, types.FunctionType):
            msg = "@transformed must decorate a plain function"
            raise TypeError(msg)

        from paguro.models.vfm.vfmodel import VFrameModel
        if not (isinstance(model, type) and issubclass(model, VFrameModel)):
            msg = "@transformed(model=...) must receive a VFrameModel subclass"
            raise TypeError(msg)

        self.fn = fn
        self.model = model
        self.validators = validators
        self.__name__ = fn.__name__

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(self, obj, owner):
        return types.MethodType(self.fn, owner)


def transformed(
        *validators: IntoValidators | typing.Collection[IntoValidators],
        model: type[VFrameModel],
):
    """
    Parameterized decorator.

    Usage:
        class M(VCModel):
            @transformed(model=OtherModel)
            def project_prices(cls):
                \"\"\"Multiply prices by 1.2 for scenario A.\"\"\"
                return (pl.col("price") * 1.2).alias("price")

        # With validators additionally appended as a local frame:
        class N(VCModel):
            @transformed(pl.col("x") >= 0, pl.col("y") < 10, model=OtherModel)
            def scenario_a(cls):
                return (pl.col("price") * 1.1).alias("price")
    """

    def _wrap(fn: types.FunctionType) -> _TransformedMethod:
        return _TransformedMethod(fn, model, validators)

    return _wrap


def _collect_transforms_from_namespace(
        ns: dict[str, object],
        *,
        owner_cls: type,
) -> tuple[ValidFrame, ...]:
    out: list[ValidFrame] = []

    for name, obj in ns.items():
        if not isinstance(obj, _TransformedMethod):
            continue

        # Call the bound method: returns DeferredLazyFrame or pl.Expr
        result = getattr(owner_cls, name)()

        # Normalize to DeferredLazyFrame
        if isinstance(result, pl.Expr):
            deferred = LazyFrameExpr(name).select(result)
        else:
            if not isinstance(result, LazyFrameExpr):
                raise TypeError(
                    f"{owner_cls.__name__}.{name}: a @transformed must return a polars.Expr "
                    f"or a paguro.DeferredLazyFrame object; got {type(result)!r}."
                )
            deferred = result

        # (A) Attach transform to the MODEL PASSED to @transformed
        target_model = obj.model
        try:
            target_valid_frame = target_model._valid_frame
        except Exception as e:
            msg = f"{owner_cls.__name__}.{name}: provided model has no _valid_frame."
            raise TypeError(msg) from e

        valid_frame_copy = copy.deepcopy(target_valid_frame)
        valid_frame_copy._transform = deferred

        # Only set the transform description; do not override title/description/constraints here.
        tdoc = _clean_doc(obj.fn)
        if tdoc:
            valid_frame_copy = valid_frame_copy.with_info(transform_description=tdoc)

        target_model._valid_frame = valid_frame_copy  # rebind on the model
        out.append(valid_frame_copy)

        # (B) If validators were provided, also create a *local* ValidFrame that
        # carries the same transform and those validators, and append it.
        if obj.validators:
            local_vf = ValidFrame(
                *obj.validators,
                transform=deferred
            )
            out.append(local_vf)

    return tuple(out)
