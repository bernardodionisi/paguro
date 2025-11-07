from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Collection

if TYPE_CHECKING:
    from collections.abc import Sequence


class ColumnsPolicy:
    """
    Validator for column names with configurable ordering, extra-column, and missing-column policies.

    Parameters
    ----------
    expected_column_names
        Expected column names as any iterable of strings. Duplicate names are rejected.
    ordered
        Whether column order must match.
        - If True and allow_extra is True: the expected columns must appear
          as a consecutive subsequence within the actual columns (starting anywhere).
        - If True and allow_extra is False: the actual columns must match the expected
          columns exactly (subject to allow_missing; see below).
    allow_extra
        Whether extra columns beyond those specified are allowed.
        - If False, any column not in `expected_column_names`/override is an error.
    allow_missing
        Whether missing expected columns are allowed.
        - If False, any expected column absent from `actual` is an error.
        - If True, ordering checks (when `ordered=True`) are performed on the
          subset of expected columns that are present in `actual`.

    Notes on ordering with allow_missing=True
    -----------------------------------------
    - With ordered=True and allow_extra=True:
        The present expected columns must appear consecutively and in order within `actual`.
    - With ordered=True and allow_extra=False:
        The `actual` sequence must equal the present expected columns in order
        (i.e., `actual` == [c for c in expected_column_names if c in actual]).
    """

    __slots__ = ("_allow_extra", "_allow_missing", "_expected_column_names", "_ordered")

    def __init__(
            self,
            *,
            ordered: bool = False,
            allow_extra: bool = True,
            allow_missing: bool = True,
            expected_column_names: Collection[str] | None = None,
    ) -> None:
        self._ordered = ordered
        self._allow_extra = allow_extra
        self._allow_missing = allow_missing

        if expected_column_names is None:
            self._expected_column_names = None
        else:
            self._check_expected_column_names(columns=expected_column_names)
            self._expected_column_names = list(expected_column_names)

    # ---------- internal helpers ----------

    def _check_expected_column_names(self, *, columns: Collection[str]) -> None:
        if not all(isinstance(x, str) for x in columns):
            raise TypeError("All column names must be strings.")
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate column names are not allowed.")

    def _gather_errors(
            self,
            *,
            actual: Sequence[str],
            expected_column_names_override: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """
        Validate actual columns against the expected specification.

        Parameters
        ----------
        actual
            The actual column names to validate.
        expected_column_names_override
            Optional override for the expected column names (used instead of the instance's `expected_column_names`).

        Returns
        -------
        dict[str, Any]
            Empty dict if no errors, otherwise:
            {
                "expected_column_names": <original policy expected list or None>,
                "errors": {
                    "missing": [...],  # omitted when allow_missing=True or none missing
                    "extra":   [...],  # present only when allow_extra=False and extras found
                    "order":   {"expected": [...], "actual": [...]},  # when ordered check fails
                }
            }
        """
        if expected_column_names_override is None:
            if self._expected_column_names is None:
                warnings.warn(
                    "Attempted validating column policy but no expected column "
                    "names were provided to the policy validator.",
                    stacklevel=2,
                )
                return {}
            expected = self._expected_column_names
            expected_set = set(self._expected_column_names)
        else:
            expected = list(expected_column_names_override)
            expected_set = set(expected)

        actual_list = list(actual)
        actual_set = set(actual_list)

        missing = expected_set - actual_set
        extra = (actual_set - expected_set) if not self._allow_extra else set()

        errors: dict[str, dict] = {}
        # todo: fix representation of lists since it misalignes the terminal printing
        # after that we can remove the str() from errors

        if missing and not self._allow_missing:
            msg = f"Missing columns: {str(sorted(missing))}"
            errors["missing"] = {"errors": msg}
        if extra:
            msg = f"Extra columns are not allowed.\nExtra columns: {str(sorted(extra))}"
            errors["extra"] = {"errors": msg}

        # Ordering checks only proceed if no other errors
        if self._ordered:
            if errors:
                errors["order"] = {
                    "warning": (
                        "Ordering is validated only after there "
                        "are no errors from the missing/extra policies."
                    )
                }

            else:  # not errors
                # Determine which expected columns participate in the order check
                if self._allow_missing:
                    expected_for_order = [c for c in expected if c in actual_set]
                else:
                    expected_for_order = expected

                # If nothing to check (e.g., all expected missing and allowed), treat as OK
                if expected_for_order:
                    if self._allow_extra:
                        # Expected must appear as a consecutive subsequence in actual
                        if not self._is_partial_order_match(
                                expected=expected_for_order,
                                actual=actual_list,
                        ):
                            errors["order"] = {
                                "expected": str(expected),
                                "errors": str(actual_list),
                            }
                    else:
                        # No extras allowed. Actual must equal expected_for_order exactly.
                        if actual_list != expected_for_order:
                            errors["order"] = {
                                "expected": str(expected),
                                "errors": str(actual_list),
                            }

        return errors

    def _is_partial_order_match(
            self,
            *,
            expected: list[str],
            actual: list[str],
    ) -> bool:
        """
        True if `expected` appears as a consecutive subsequence of `actual`.
        """
        if not expected:
            return True

        try:
            start_idx = actual.index(expected[0])
        except ValueError:
            return False

        end_idx = start_idx + len(expected)
        return actual[start_idx:end_idx] == expected

    # ---------- (de)serialization ----------

    def _to_dict(self) -> dict[str, object]:
        """
        Convert to a dictionary representation.

        Returns
        -------
        dict[str, object]
            Dictionary with 'expected_column_names', 'ordered', 'allow_extra', 'allow_missing'.
        """
        return {
            "ordered": self._ordered,
            "allow_extra": self._allow_extra,
            "allow_missing": self._allow_missing,
            "expected_column_names": (
                self._expected_column_names.copy()
                if self._expected_column_names is not None
                else None
            ),
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ColumnsPolicy:
        """
        Create from a dictionary representation.

        Parameters
        ----------
        data : dict[str, object]
            Dictionary with 'expected_column_names', 'ordered', 'allow_extra', 'allow_missing'.

        Returns
        -------
        ColumnsPolicy
            New instance created from the dictionary.
        """
        return cls(
            ordered=data.get("ordered", False),
            allow_extra=data.get("allow_extra", True),
            allow_missing=data.get("allow_missing", False),
            expected_column_names=data.get("expected_column_names", None),
        )

    # ---------- dunder methods ----------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}("
            f"ordered={self._ordered}, "
            f"allow_extra={self._allow_extra}, "
            f"allow_missing={self._allow_missing}, "
            f"expected_column_names={self._expected_column_names}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColumnsPolicy):
            return NotImplemented
        return (
                self._expected_column_names == other._expected_column_names
                and self._ordered == other._ordered
                and self._allow_extra == other._allow_extra
                and self._allow_missing == other._allow_missing
        )
