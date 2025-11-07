# test_relationship_parser.py
import pytest

from paguro.validation.valid_relations.utils.relation_strings import (
    RelationshipParseError,
    parse_relationship_strings,
)

def get_entries(out: dict, a: str, b: str):
    """Return list of RelationEntry for key (a,b)."""
    return out[(a, b)]

def pairs(out: dict):
    """Return sorted list of keys for easy assertions."""
    return sorted(out.keys())


# -------------------------
# Happy paths / orientation
# -------------------------

def test_simple_one_edge_orientation_a_lt_b():
    out = parse_relationship_strings("A[a] < B[b]")
    assert pairs(out) == [("A", "B")]
    assert get_entries(out, "A", "B") == [
        {"left_on": ["a"], "right_on": ["b"], "relation": "<"}
    ]

def test_simple_one_edge_orientation_b_lt_a_flips_to_a_gt_b():
    out = parse_relationship_strings("B[b] < A[a]")
    assert pairs(out) == [("A", "B")]
    # flipped to orient from A -> B
    assert get_entries(out, "A", "B") == [
        {"left_on": ["a"], "right_on": ["b"], "relation": ">"}
    ]

def test_multi_edge_chain_normalization_and_order():
    # A[x] <> B[y] > C[z] yields edges (A,B) with '<>' and (B,C) with '>'
    out = parse_relationship_strings("A[x] <> B[y] > C[z]")
    assert pairs(out) == [("A", "B"), ("B", "C")]
    assert get_entries(out, "A", "B") == [
        {"left_on": ["x"], "right_on": ["y"], "relation": "<>"}
    ]
    assert get_entries(out, "B", "C") == [
        {"left_on": ["y"], "right_on": ["z"], "relation": ">"}
    ]

def test_parentheses_and_brackets_mixed():
    out = parse_relationship_strings("A(x, y) > B[x, y]")
    assert get_entries(out, "A", "B") == [
        {"left_on": ["x", "y"], "right_on": ["x", "y"], "relation": ">"}
    ]


# -------------------------
# Quoting / Escaping
# -------------------------

def test_quoted_columns_with_commas_and_escapes():
    # doubled quotes and backslash escapes inside quotes
    s = r"""A['last, first', "co""de", 'quo\'te'] <> B["last, first", "co""de", "quo'te"]"""
    out = parse_relationship_strings(s)
    e = get_entries(out, "A", "B")[0]
    assert e["relation"] == "<>"
    # Both sides should normalize to the same unquoted column literals
    expected = ["last, first", 'co"de', "quo'te"]
    assert e["left_on"] == expected
    assert e["right_on"] == expected

def test_quoted_and_qualified_table_names():
    s = r"""`My Schema`.`Tbl`(k1) > 'Other'."Tbl"(k1)"""
    out = parse_relationship_strings(s)
    # Identifiers are unquoted and re-joined by '.'
    assert pairs(out) == [("My Schema.Tbl", "Other.Tbl")]
    assert get_entries(out, "My Schema.Tbl", "Other.Tbl") == [
        {"left_on": ["k1"], "right_on": ["k1"], "relation": ">"}
    ]


# -------------------------
# Trailing commas option
# -------------------------

def test_trailing_commas_allowed_default_true():
    out = parse_relationship_strings("A[id, ] < B[id,]")
    assert get_entries(out, "A", "B") == [
        {"left_on": ["id"], "right_on": ["id"], "relation": "<"}
    ]

def test_trailing_commas_disallowed_raises():
    with pytest.raises(RelationshipParseError):
        parse_relationship_strings("A[id, ] < B[id,]", allow_trailing_commas=False)


# -------------------------
# Deterministic output & dedup/coalescing
# -------------------------

def test_deterministic_output_independent_of_input_order():
    s1 = [
        "B[b] < A[a]",
        "A[a] <> B[b]",
        "B[b] > A[a]",
    ]
    s2 = list(reversed(s1))
    out1 = parse_relationship_strings(s1, deterministic=True)
    out2 = parse_relationship_strings(s2, deterministic=True)
    assert out1 == out2  # stable keys and sorted entries

def test_deduplication_and_coalescing():
    # '<' plus '>' for same columns collapses to '<>'; exact duplicates are dropped
    s = [
        "A[a] < B[b]",
        "B[b] > A[a]",  # opposite direction
        "A[a] < B[b]",  # exact duplicate
    ]
    out = parse_relationship_strings(s)
    entries = get_entries(out, "A", "B")
    assert entries == [{"left_on": ["a"], "right_on": ["b"], "relation": "<>"}]

def test_coalescing_prefers_bidirectional_if_present():
    # If '<>' is present, it dominates regardless of also having '<' or '>'
    s = [
        "A[a] < B[b]",
        "A[a] <> B[b]",
        "B[b] > A[a]",
    ]
    out = parse_relationship_strings(s)
    entries = get_entries(out, "A", "B")
    assert entries == [{"left_on": ["a"], "right_on": ["b"], "relation": "<>"}]


# -------------------------
# Error paths
# -------------------------

def test_mismatched_column_counts_raises():
    with pytest.raises(RelationshipParseError):
        parse_relationship_strings("A[id, name] < B[id]")

def test_missing_operator_between_nodes_raises():
    with pytest.raises(RelationshipParseError):
        parse_relationship_strings("A[id]  B[id]")  # no operator

def test_unclosed_quote_in_columns_raises():
    with pytest.raises(RelationshipParseError):
        parse_relationship_strings("A['id] < B[id]")

def test_malformed_node_raises():
    with pytest.raises(RelationshipParseError):
        parse_relationship_strings("A < B[id]")  # left side missing column list


# -------------------------
# Multiple strings input
# -------------------------

def test_iterable_inputs_accumulate_pairs_and_edges_with_coalescing():
    inputs = [
        "customers[id] < orders[id] <> shipments[id]",
        "orders[id] <> shipments[id]",   # duplicate edge ignored
        "shipments[id] > orders[id]",    # opposite direction; with '<>' present, keep only '<>'
    ]
    out = parse_relationship_strings(inputs)
    assert pairs(out) == [("customers", "orders"), ("orders", "shipments")]
    assert get_entries(out, "customers", "orders") == [
        {"left_on": ["id"], "right_on": ["id"], "relation": "<"}
    ]
    assert get_entries(out, "orders", "shipments") == [
        {"left_on": ["id"], "right_on": ["id"], "relation": "<>"}
    ]


# -------------------------
# Mixed whitespace robustness
# -------------------------

@pytest.mark.parametrize(
    "s",
    [
        " A ( id ,  name )  >  B [ id , name ] ",
        "\tA(id,name)\n>\nB[id,name]\n",
        "A (id,name)>B[id,name]",
    ],
)
def test_whitespace_variants(s: str):
    out = parse_relationship_strings(s)
    assert get_entries(out, "A", "B") == [
        {"left_on": ["id", "name"], "right_on": ["id", "name"], "relation": ">"}
    ]
