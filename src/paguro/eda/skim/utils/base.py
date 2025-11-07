from __future__ import annotations

import polars as pl
from polars import selectors as cs


def _base_config() -> list[tuple]:
    out = [
        # --------------------------------------------------------------
        (
            "numeric",
            cs.numeric(),
            ["null_count", "n_unique", "mean", "std", "min", "max"],
        ),
        (
            "boolean",
            cs.boolean(),
            [
                "null_count",
                ("true_count", pl.all().sum()),
                ("true_share", pl.all().mean()),
            ],
        ),
        # --------------------------------------------------------------
        (
            "string",
            cs.string(),
            [
                "null_count",
                "n_unique",
                ("min_chars", pl.all().str.len_chars().min()),
                ("max_chars", pl.all().str.len_chars().max()),
                ("mean_chars", pl.all().str.len_chars().mean()),
            ],
        ),
        # --------------------------------------------------------------
        # (
        #     "list",
        #     cs.by_dtype(pl.List),
        #     [
        #         "null_count",
        #         "n_unique",
        #         ("min_chars", pl.all().list.len().min()),
        #         ("max_chars", pl.all().list.len().max()),
        #         ("mean_chars", pl.all().list.len().mean()),
        #     ],
        # ),
        # --------------------------------------------------------------
        (
            "categorical",
            cs.by_dtype(pl.Categorical, pl.Enum),
            [
                "null_count",
                "n_unique",
                (
                    "value_counts",
                    pl.all()
                    .value_counts(sort=True, name="value_counts")
                    # .struct.rename_fields(["value", "count"])
                    # .pipe(
                    #     lambda x: x.struct.field("value").cast(pl.String)
                    #               + ": "
                    #               + x.struct.field("count").cast(pl.String)
                    # )
                    .pipe(
                        lambda x: x.struct[0].cast(pl.String)
                        + ": "
                        + x.struct[1].cast(pl.String)
                        + " ["
                        + (x.struct[1] / x.struct[1].sum())
                        .mul(100)
                        .cast(pl.Int64)
                        .cast(pl.String)
                        + "%]"
                    )
                    .str.join("\n"),
                ),
            ],
        ),
        # --------------------------------------------------------------
        (
            "date",
            cs.date(),
            ["null_count", "n_unique", "mean", "min", "max"],
        ),
        # --------------------------------------------------------------
        (
            "datetime",
            cs.datetime(),
            ["null_count", "n_unique", "mean", "min", "max"],
        ),
        # --------------------------------------------------------------
        (
            "duration",
            cs.duration(),
            ["null_count", "n_unique", "mean", "min", "max"],
        ),
    ]

    remaining_columns = cs.all()
    for i in out:
        remaining_columns -= i[1]

    out.append(
        ("other", remaining_columns, ["null_count", "n_unique"]),
    )
    return out
