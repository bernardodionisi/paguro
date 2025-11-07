from __future__ import annotations

import warnings
import re
import pytest

from paguro.validation.valid_frame.utils.columns_policy import ColumnsPolicy


def _assert_errors_has_items(result: dict, key: str, items: list[str]) -> None:
    """Assert that result[key]['errors'] mentions all `items`, regardless of
    whether errors is a list or a human-readable string."""
    assert key in result, f"Expected '{key}' in result, got {result!r}"
    errs = result[key].get("errors")
    assert errs is not None, f"Expected '{key}' to contain 'errors', got {result[key]!r}"
    if isinstance(errs, str):
        for it in items:
            assert it in errs, f"'{it}' not mentioned in string errors: {errs!r}"
    else:
        # Assume it's an iterable of items
        assert set(items) <= set(errs), f"Missing expected items {items} in {errs!r}"


def _assert_order_error(
        result: dict,
        expected: list[str],
        actual: list[str],
) -> None:
    assert "order" in result, f"No 'order' key found in {result!r}"
    od = result["order"]

    exp = od.get("expected")
    got = od.get("errors")

    if "warning" in od and (exp is None and got is None):
        return

    assert exp is not None and got is not None, f"Order input malformed: {od!r}"

    if isinstance(exp, str):
        # Ensure all expected members are mentioned
        for e in expected:
            assert e in exp, f"Expected member {e!r} not referenced in {exp!r}"
    else:
        assert list(exp) == list(
            expected), f"Expected sequence mismatch: {exp!r} != {expected!r}"

    if isinstance(got, str):
        # Ensure all actual members are mentioned (weakly)
        for a in actual:
            assert a in got, f"Actual member {a!r} not referenced in {got!r}"
    else:
        assert list(got) == list(
            actual), f"Actual sequence mismatch: {got!r} != {actual!r}"


def test_init_rejects_non_string_expected():
    with pytest.raises(
            TypeError,
            match=re.compile(r"All column names must be strings", re.I),
    ):
        ColumnsPolicy(expected_column_names=["a", 1, "c"])


def test_init_rejects_duplicate_expected():
    with pytest.raises(
            ValueError,
            match=re.compile(r"Duplicate column names", re.I),
    ):
        ColumnsPolicy(expected_column_names=["a", "b", "a"])


def test_gather_errors_warns_and_returns_empty_when_no_expected_and_no_override():
    policy = ColumnsPolicy(expected_column_names=None)
    with pytest.warns(
            UserWarning,
            match=re.compile(r"no expected column.*provided", re.I), ):
        result = policy._gather_errors(actual=["a", "b"])
    assert result == {}  # returns empty dict when no config present


def test_gather_errors_with_override_emits_no_warning_and_no_errors_when_ok():
    policy = ColumnsPolicy(expected_column_names=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # fail if any warning is emitted
        result = policy._gather_errors(actual=["x", "a", "y"],
                                       expected_column_names_override=["a"])
    assert result == {}  # allow_missing=True by default, and ordering not enforced by default


def test_missing_is_omitted_when_allow_missing_true():
    policy = ColumnsPolicy(expected_column_names=["a", "b"], allow_missing=True,
                           allow_extra=True)
    result = policy._gather_errors(actual=["a"])  # b is missing but allowed
    assert result == {}  # no errors because missing is allowed and extras allowed


def test_missing_is_reported_when_allow_missing_false():
    policy = ColumnsPolicy(expected_column_names=["a", "b"], allow_missing=False,
                           allow_extra=True)
    result = policy._gather_errors(actual=["a"])
    _assert_errors_has_items(result, "missing", ["b"])


def test_extra_is_reported_when_allow_extra_false():
    policy = ColumnsPolicy(
        expected_column_names=["a", "b"],
        allow_missing=True,
        allow_extra=False,
    )
    result = policy._gather_errors(actual=["a", "b", "x"])
    _assert_errors_has_items(result, "extra", ["x"])


def test_missing_and_extra_both_when_disallowing_missing_and_extra():
    policy = ColumnsPolicy(expected_column_names=["a", "b"], allow_missing=False,
                           allow_extra=False)
    result = policy._gather_errors(actual=["a", "x"])
    _assert_errors_has_items(result, "missing", ["b"])
    _assert_errors_has_items(result, "extra", ["x"])


def test_order_error_allow_extra_true_not_consecutive_with_allow_missing_true():
    # ordered=True + allow_extra=True => expected (present subset if allow_missing) must be consecutive
    policy = ColumnsPolicy(
        expected_column_names=["a", "b", "c"], ordered=True, allow_extra=True,
        allow_missing=True
    )
    actual = ["x", "a", "b", "x",
              "c"]  # present expected subset is ["a","b","c"] but not consecutive
    result = policy._gather_errors(actual=actual)
    _assert_order_error(result, expected=["a", "b", "c"], actual=actual)


def test_order_ok_allow_extra_true_consecutive_subsequence_with_allow_missing_true():
    policy = ColumnsPolicy(
        expected_column_names=["a", "b", "c"], ordered=True, allow_extra=True,
        allow_missing=True
    )
    result = policy._gather_errors(actual=["y", "a", "b", "c", "z"])
    assert result == {}  # consecutive subsequence satisfied


def test_order_error_allow_extra_false_not_exact_match_when_allow_missing_false():
    # With allow_extra=False and ordered=True,
    # actual must equal expected exactly when allow_missing=False.
    policy = ColumnsPolicy(
        expected_column_names=["a", "b", "c"], ordered=True, allow_extra=False,
        allow_missing=False
    )
    actual = ["a", "c", "b"]
    result = policy._gather_errors(actual=actual)
    _assert_order_error(result, expected=["a", "b", "c"], actual=actual)


def test_order_check_skipped_when_other_errors_present():
    # If there's an 'extra' error, ordering check is deferred;
    # implementation attaches an 'order' warning.
    policy = ColumnsPolicy(
        expected_column_names=["a", "b"], ordered=True, allow_extra=False,
        allow_missing=True,
    )
    result = policy._gather_errors(actual=["a", "b", "x"])  # extra present
    _assert_errors_has_items(result, "extra", ["x"])
    # Implementation adds an informational 'order' warning;
    # allow it but don't require exact text.
    if "order" in result:
        assert "warning" in result["order"]


def test_order_allow_extra_false_with_allow_missing_true_requires_exact_subset():
    # allow_missing=True, allow_extra=False:
    # actual must equal the present expected subset exactly.
    policy = ColumnsPolicy(
        expected_column_names=["a", "b", "c"], ordered=True, allow_extra=False,
        allow_missing=True
    )
    # Present expected subset is ["a","c"];
    # actual has an extra "x" and wrong order
    actual = ["a", "x", "c"]
    result = policy._gather_errors(actual=actual)
    _assert_order_error(result, expected=["a", "b", "c"], actual=actual)

    # Now the exact subset in order, with no extras:
    ok = policy._gather_errors(actual=["a", "c"])
    assert ok == {}


@pytest.mark.parametrize(
    "expected, actual, ok",
    [
        ([], ["anything"], True),
        (["a"], ["a"], True),
        (["a"], ["x", "a", "y"], True),
        (["a", "b"], ["x", "a", "b", "y"], True),  # consecutive subsequence
        (["a", "b", "c"], ["a", "x", "b", "c"], False),
        (["a", "b"], ["a", "x", "b"], False),
        (["a", "b", "c"], ["a", "b"], False),
        (["z"], ["a", "b", "c"], False),
    ],
)
def test_is_partial_order_match_cases(expected, actual, ok):
    policy = ColumnsPolicy(expected_column_names=["irrelevant"])
    assert policy._is_partial_order_match(expected=list(expected),
                                          actual=list(actual)) is ok


# -----------------------------
# (De)serialization helpers
# -----------------------------

def test_to_dict_returns_copy_and_is_immutable_outside():
    policy = ColumnsPolicy(
        expected_column_names=["a", "b"], ordered=True, allow_extra=False,
        allow_missing=False
    )
    d = policy._to_dict()
    assert d == {
        "ordered": True,
        "allow_extra": False,
        "allow_missing": False,
        "expected_column_names": ["a", "b"],
    }
    # Mutating the returned list must not affect the policy's internal state
    d["expected_column_names"].append("c")
    assert policy._to_dict()["expected_column_names"] == ["a", "b"]


def test_from_dict_round_trip_and_defaults():
    data = {
        "expected_column_names": ["a"],
        "ordered": True,
        "allow_extra": False,
        "allow_missing": True,
    }
    inst = ColumnsPolicy._from_dict(data)
    assert inst == ColumnsPolicy(
        expected_column_names=["a"], ordered=True, allow_extra=False, allow_missing=True
    )

    # Defaults in _from_dict:
    # ordered defaults to False, allow_extra to True, allow_missing to False, expected_column_names to None
    inst_defaulted = ColumnsPolicy._from_dict({})
    assert inst_defaulted == ColumnsPolicy(
        expected_column_names=None, ordered=False, allow_extra=True, allow_missing=False
    )


def test_repr_contains_key_fields():
    policy = ColumnsPolicy(
        expected_column_names=["a", "b"], ordered=True, allow_extra=False,
        allow_missing=False
    )
    r = repr(policy)
    assert "ordered=True" in r
    assert "allow_extra=False" in r
    assert "allow_missing=False" in r
    assert "expected_column_names=['a', 'b']" in r


def test_equality_and_inequality():
    a = ColumnsPolicy(
        expected_column_names=["a", "b"], ordered=True, allow_extra=False,
        allow_missing=False
    )
    b = ColumnsPolicy(
        expected_column_names=["a", "b"], ordered=True, allow_extra=False,
        allow_missing=False
    )
    c = ColumnsPolicy(
        expected_column_names=["a", "b"], ordered=False, allow_extra=False,
        allow_missing=False
    )
    d = ColumnsPolicy(
        expected_column_names=["a", "c"], ordered=True, allow_extra=False,
        allow_missing=False
    )
    e = ColumnsPolicy(
        expected_column_names=["a", "b"], ordered=True, allow_extra=True,
        allow_missing=False
    )
    f = ColumnsPolicy(
        expected_column_names=["a", "b"], ordered=True, allow_extra=False,
        allow_missing=True
    )

    assert a == b
    assert a != c
    assert a != d
    assert a != e
    assert a != f
    assert a != object()
