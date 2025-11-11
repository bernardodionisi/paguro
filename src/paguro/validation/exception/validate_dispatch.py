from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from paguro.utils.dependencies import re, shlex
from paguro.validation.exception.errors.base_validation_error import (
    _BaseValidationError,
)
from paguro.validation.exception.errors.validation_error import (
    ValidationError,
)
from paguro.validation.shared.utils import data_to_frame_like
from paguro.shared._typing import typed_dicts

if TYPE_CHECKING:
    from paguro.typing import IntoValidation, OnFailureExtra, OnSuccess, FrameLike, \
        CollectionLike, CollectConfig


def _validate_dispatch_base(
        errors: dict,
        *,
        collect: bool | CollectConfig,
        on_success: OnSuccess,
        on_failure: OnFailureExtra,
        validation_error_type: _BaseValidationError | None = None,
        data: FrameLike | CollectionLike,  # not IntoValidation anymore
        with_row_index: bool | str,  # None used for Collection?
):  # returns None | [input] data | some child of BaseValidationError
    # -----------
    if not errors:  # empty dictionary: success
        return _return_on_success(on_success=on_success, data=data)
    # -----------

    # -----------  what validation error type should we use
    if validation_error_type is None:
        # base validation error has no filtering capabilities
        validation_error_type = _to_base_validation_error(
            errors=errors,
            data=data,
            with_row_index=with_row_index,
        )

    # -----------
    if collect:
        # inplace collect all errors
        validation_error_type._collect_and_replace(
            key="maybe_errors",
            collect=collect,
        )
        if validation_error_type._count_errors() == 0:
            return _return_on_success(on_success=on_success, data=data)
    # -----------

    if on_failure == "return_error":
        return validation_error_type

    elif on_failure == "raise":
        raise validation_error_type

    else:  # return data
        raise NotImplementedError


def _validate_dispatch(
        errors: typed_dicts.ValidationErrors,  # remove dict[str, Any]
        *,
        collect: bool | CollectConfig,
        on_success: OnSuccess,
        on_failure: OnFailureExtra,
        data: FrameLike | CollectionLike,  # not IntoValidation anymore
        with_row_index: bool | str,
) -> FrameLike | ValidationError | None:  # returns None | [input] data | some child of BaseValidationError

    # -----------
    if not errors:  # empty dictionary: success
        return _return_on_success(on_success=on_success, data=data)
    # -----------

    validation_error: ValidationError = _to_validation_error(
        errors=errors,
        data=data,
        with_row_index=with_row_index,
    )

    # -----------
    if collect:
        # inplace collect all errors
        validation_error._collect_and_replace(
            key="maybe_errors",
            collect=collect,
        )

        # if validation_error._count_errors() == 0:
        #     return _return_on_success(on_success=on_success, data=data)

    if validation_error._count_errors() == 0:
        return _return_on_success(on_success=on_success, data=data)

    # -----

    if on_failure == "return_error":
        return validation_error

    elif on_failure == "raise":
        raise validation_error

    else:  # return data
        if on_failure.startswith("warn-"):
            validation_error._warn(stacklevel=3)

        if on_failure.endswith("return_data"):
            return validation_error._data

        elif "return_valid_data" in on_failure:
            indices = parse_trailing_index(on_failure)

            return validation_error._filter(
                how="using_predicates",
                return_valid=True,
                collect=collect,
                row_index=indices,
            )
        elif "return_invalid_data" in on_failure:
            indices = parse_trailing_index(on_failure)

            return validation_error._filter(
                how="using_predicates",
                return_valid=False,
                collect=collect,
                row_index=indices,
            )
        else:
            msg = f"Unknown on_failure={on_failure!r}"
            raise ValueError(msg)


def _to_base_validation_error(
        errors: dict,
        *,
        data: FrameLike | CollectionLike,
        with_row_index: bool | str,
) -> _BaseValidationError:
    ve = _BaseValidationError(mapping=errors)

    ve._set_data_and_row_index(
        data=data,  # could be anything here for now
        with_row_index=with_row_index,
    )
    return ve


def _to_validation_error(
        errors: typed_dicts.ValidationErrors,
        *,
        data: IntoValidation,
        with_row_index: bool | str,
) -> ValidationError:
    ve = ValidationError(mapping=errors)

    ve._set_data_and_row_index(
        data=data_to_frame_like(data=data),
        with_row_index=with_row_index,
    )
    return ve


def _return_on_success(
        on_success: Literal["return_data", "return_none"],
        data: IntoValidation,
):
    return None if on_success == "return_none" else data


# ----------------------------------------------------------------------


def parse_trailing_index(s: str) -> list[str] | None:
    """
    Extracts and parses a list of strings from the end of the given string.

    Examples:
        "some text [a, b, c]" -> ['a', 'b', 'c']
        "example [\"x, y\", z]" -> ['x, y', 'z']
        "no brackets here" -> None
    """
    # Match trailing [...] at the end of the string
    _match = re.search(r'\[([^\]]*)\]\s*$', s)
    if not _match:
        return None

    content = _match.group(1).strip()
    if not content:
        return []

    # Use shlex to split like shell arguments,
    # so it handles quotes and commas inside quotes
    lexer = shlex.shlex(content, posix=True)
    lexer.whitespace = ','
    lexer.whitespace_split = True
    lexer.commenters = ''

    # Return list of stripped strings
    return [token.strip() for token in lexer if token.strip()]
