from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Iterable

from paguro.shared.serialize import fingerprint
from paguro.shared.serialize.encoder import CustomJSONEncoder
from paguro.utils.dependencies import copy, json

from paguro.validation.valid_column.valid_column import ValidColumn
from paguro.validation.valid_frame.valid_frame import ValidFrame

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable, Iterator

    import polars as pl
    from polars._typing import JoinStrategy

    from paguro.typing import IntoKeepColumns, ValidationMode

    if sys.version_info >= (3, 11):
        pass
    else:
        pass

from typing import Generic, TypeVar

T = TypeVar("T", ValidColumn, ValidFrame)


class _ValidListBase(Generic[T]):
    def __init__(self, valid_list: Iterable[T]) -> None:
        self._valid_list: list[T] = list(valid_list)

    def _extend(self, valid_list: list[T]) -> None:
        _type = valid_list[0].__class__.__name__

        existing = [j._name for j in self._valid_list]

        for i in valid_list:
            if i._name in existing:
                warnings.warn(
                    f"\n{_type} named '{i._name}', already exists, "
                    f"replacing it with the new value.",
                    stacklevel=2,
                )
                self._valid_list[existing.index(i._name)] = i
            else:
                self._valid_list.append(i)
                existing.append(i._name)

    def __bool__(self) -> bool:
        return bool(self._valid_list)

    # def __str__(self) -> str:
    #     content = ",\n ".join(
    #         f"\t{f.__str__()}" for f in self._valid_list
    #     )
    #     return f"{self.__class__.__qualname__}(\n{content}\n)"

    def __str__(self) -> str:
        return self._repr_or_str(string=True)

    def __repr__(self) -> str:
        return self._repr_or_str(string=False)

    def _repr_or_str(self, string: bool) -> str:
        if string:
            content_list: list[str] = [
                vc.__str__() for vc in self._valid_list
            ]
        else:
            content_list = [vc.__repr__() for vc in self._valid_list]

        content: str = ",\n".join(content_list)
        content = content.replace("\n", "\n\t")
        return f"{self.__class__.__qualname__}(\n\t{content}\n)"

    def __iter__(self) -> Iterator[T]:
        return iter(self._valid_list)

    def __getitem__(self, key: int | str) -> T:
        if isinstance(key, str):
            key = [n._name for n in self._valid_list].index(key)

        return self._valid_list[key]

    def __contains__(self, name: str) -> bool:
        return name in [n._name for n in self._valid_list]

    def __delitem__(self, key: int | str) -> None:
        if isinstance(key, str):
            key = [n._name for n in self._valid_list].index(key)

        del self._valid_list[key]

    def _to_list_of_dicts(self) -> list[dict[str, Any]]:
        return [
            f._to_dict(_fingerprint=False, include_info=True)
            for f in self._valid_list
        ]

    def serialize(self) -> str:
        return json.dumps(self._to_list_of_dicts(), cls=CustomJSONEncoder)

    def _fingerprint(
            self, *, as_bytes: bool, include_info: bool
    ) -> bytes | str:
        digest: list = [
            f._fingerprint(as_bytes=True, include_info=include_info)
            for f in self._valid_list
        ]
        digest_bytes: bytes = fingerprint.combine_unordered(digest)
        if as_bytes:
            return digest_bytes
        return digest_bytes.hex()

    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        return [v._name for v in self._valid_list]

    # ------------------------------------------------------------------

    def _gather_errors(
            self,
            frame: pl.LazyFrame,
            mode: ValidationMode,
            *,
            keep_columns: IntoKeepColumns,
            with_row_index: bool | str,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            cast: bool,
            _root_down: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}

        for v in self._valid_list:
            errors: Any = v._gather_errors(
                frame=frame,
                mode=mode,
                keep_columns=keep_columns,
                with_row_index=with_row_index,
                get_expr=get_expr,
                cast=cast,
                _root_down=_root_down,
            )

            if errors:
                out[v._name] = errors

        return out

    # ------------------------------------------------------------------

    def _gather_predicates(
            self,
            schema: pl.Schema | None,
            *,
            get_expr: Callable[[str, Any, str | pl.Expr | None], pl.Expr],
            _root_down: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}

        for v in self._valid_list:
            predicates: Any = v._gather_predicates(
                schema=schema,
                get_expr=get_expr,
                _root_down=_root_down,
            )

            if predicates:
                out[v._name] = predicates

        return out

    # ------------------------------------------------------------------

    def _join(
            self,
            other: _ValidListBase[T],
            how: JoinStrategy,
            *,
            suffix: str | None,
            join_constraints: bool,
    ) -> list[T]:
        # right (left)
        # self (other) is primary:
        #   - keep all self (other) items and append
        #   any other (self) items whose _name is not
        # in self (other_).

        if how == "right":
            joined_list = _join_valid_lists(
                left=other,
                right=self,
                join_constraints=join_constraints,
                suffix=suffix,
            )
        elif how in {"anti", "semi"}:
            # only left columns are kept
            joined_list = copy.deepcopy(self._valid_list)

        else:  # {'inner', 'left', 'full', 'cross'}
            joined_list = _join_valid_lists(
                left=self,
                right=other,
                join_constraints=join_constraints,
                suffix=suffix,
            )

        return joined_list


def _join_valid_lists(
        left: _ValidListBase[T],
        right: _ValidListBase[T],
        *,
        suffix: str | None,
        join_constraints: bool,
) -> list[T]:
    """
    Joins two ValidColumnLists or two ValidFrameLists.

    If the lists contain elements with the same names:
        - if join_constraints=True, we join the constraints of the right into the constraints of the left

    """
    if join_constraints and suffix is not None:
        raise TypeError(
            "'suffix' can only be used if 'join_constraints'=False"
        )
    left = copy.deepcopy(left)
    right = copy.deepcopy(right)

    out: list[T] = []

    # TODO: improve: we are currently checking selectors with string repr.
    existing_names: list[str | None] = []

    for lf in left._valid_list:
        out.append(lf)
        # for named ideally use set but we need the index of the name to modify
        existing_names.append(
            str(lf._name) if lf._name is not None else None
        )

    for ri in right._valid_list:
        if ri._name is None and ri._name not in existing_names:
            out.append(ri)
        elif ri._name is not None and str(ri._name) not in existing_names:
            out.append(ri)

        else:  # the name is already in the list
            name = f"{ri._name!r}" if ri._name is not None else None

            if join_constraints:  # the name is in both lists
                idx = existing_names.index(
                    str(ri._name) if ri._name is not None else None
                )

                # update the right constraints inplace
                ri._constraints.update(out[idx]._constraints)
                out[idx]._constraints = ri._constraints
            else:
                if suffix is not None:
                    if not isinstance(ri._name, str):
                        raise TypeError(
                            f"Unable to add suffix to {ri.__class__.__name__} named: {name}"
                        )

                    ri._name = f"{str(ri._name)}{suffix}"
                    if ri._name not in existing_names:
                        out.append(ri)
                warnings.warn(
                    f"\nSkipping {ri.__class__.__name__} named: {name}. "
                    f"{name} already exists in left list",
                    DuplicateNameWarning,
                    stacklevel=2,
                )
                continue

    return out


class DuplicateNameWarning(UserWarning):
    pass

# ----------------------------------------------------------------------
